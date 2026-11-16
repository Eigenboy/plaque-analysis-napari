"""Segment bright plaque candidates with the existing ChemiDoc detector."""

import cv2
import numpy as np


def estimate_dish_diameter_px(agar_mask):
    """Estimate detector scale from the main agar outline when the dish fit fails."""
    outlines, _ = cv2.findContours(agar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = max((cv2.contourArea(outline) for outline in outlines), default=0.0)
    if area <= 0:
        raise ValueError("The agar mask must enclose an area to estimate the dish size.")

    # The exported agar boundary typically spans about 91% of the dish diameter.
    return float(np.sqrt(4 * area / np.pi) / 0.91)


def detect_plaques(image, agar_mask, dish_diameter_px):
    """Return filled plaque masks with positive integer IDs in image coordinates."""
    inside = agar_mask != 0
    if inside.shape != image.shape or not inside.any():
        raise ValueError("The agar boundary must enclose pixels in the plate image.")

    # Keep the filter scale tied to the dish, even when the user edits its agar ROI.
    scale = dish_diameter_px / 1240
    kernel_size = max(9, 2 * round(17 * scale) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,) * 2)
    fine = cv2.GaussianBlur(image.astype(np.float32), (0, 0), max(0.7, 1.05 * scale))
    response = fine - cv2.morphologyEx(fine, cv2.MORPH_OPEN, kernel)
    response -= cv2.GaussianBlur(response, (0, 0), max(6, 18 * scale))
    lawn_response = response[inside]
    baseline = np.median(lawn_response)
    noise = 1.4826 * np.median(abs(lawn_response - baseline))
    if noise <= 0:
        raise ValueError("The agar interior has insufficient intensity variation.")

    grow = baseline + 1.95 * noise
    peak = baseline + 4.35 * noise
    binary = ((response >= grow) & inside).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = inside.astype(np.uint8)
    boundary = mask - cv2.erode(mask, np.ones((3, 3), np.uint8))
    plaques = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not max(4, 9 * scale**2) <= area <= 950 * scale**2:
            continue
        xs, ys = contour[:, 0].T
        if boundary[ys, xs].any():
            continue
        perimeter = cv2.arcLength(contour, True)
        if 4 * np.pi * area / perimeter**2 < 0.32:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        pixels = np.zeros((h, w), np.uint8)
        cv2.drawContours(pixels, [contour - (x, y)], 0, 1, cv2.FILLED)
        roi = np.s_[y:y + h, x:x + w]
        pixels &= mask[roi]
        if cv2.minMaxLoc(response[roi], mask=pixels)[1] >= peak:
            moments = cv2.moments(contour)
            center = (moments["m01"] / moments["m00"], moments["m10"] / moments["m00"])
            plaques.append((center, roi, pixels != 0))

    # Give each contour its own ID, in top-to-bottom order.
    labels = np.zeros(image.shape, dtype=np.uint32)
    for label_id, (_, roi, pixels) in enumerate(sorted(plaques, key=lambda p: p[0]), 1):
        labels[roi][pixels] = label_id
    return labels
