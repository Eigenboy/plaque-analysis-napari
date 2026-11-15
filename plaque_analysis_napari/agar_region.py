"""Agar-region detection for full-plate 16-bit grayscale images."""

import cv2
import numpy as np


def detect_agar(image):
    """Return the agar mask in the original image's pixel coordinates."""
    dish = _dish_ellipse(image)
    gray = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 2.5)
    return _agar_mask(gray, dish)


def dish_diameter_px(image):
    """Measure the outer dish without computing an agar region."""
    _, (width, height), _ = _dish_ellipse(image)
    return float(np.sqrt(width * height))


def boundary_paths(mask):
    """Return editable closed outlines in Napari (row, column) coordinates."""
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for contour in contours:
        if len(contour) < 3:
            continue
        # Remove pixel stair steps so neighbouring handles can be dragged separately.
        outline = cv2.approxPolyDP(contour, 2.5, closed=True)
        if len(outline) < 3:
            outline = contour
        paths.append(outline[:, 0, ::-1].astype(np.float32))
    return paths


def mask_from_boundaries(shape, paths):
    """Fill the edited outlines, preserving holes by the even-odd fill rule."""
    contours = []
    for path in paths:
        if len(path) < 3 or not np.isfinite(path).all():
            continue
        # OpenCV uses (x, y). Fixed-point coordinates retain subpixel edits.
        contours.append(np.rint(path[:, ::-1] * 16).astype(np.int32))
    if not contours:
        raise ValueError("Add a boundary with at least three vertices.")
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, contours, 1, shift=4)
    if not mask.any():
        raise ValueError("The edited boundary does not enclose any image pixels.")
    return mask


def _dish_ellipse(raw):
    # Threshold the dark dish against the illuminated surround at quarter resolution.
    if min(raw.shape) < 48:
        raise ValueError("The image is too small to resolve the agar rim.")
    height, width = raw.shape
    small = cv2.resize(raw, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _, dark = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(dark.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if len(contour) < 5 or area < 0.035 * small.size:
            continue
        if x == 0 or y == 0 or x + w >= small.shape[1] or y + h >= small.shape[0]:
            continue
        perimeter = cv2.arcLength(contour, True)
        if 4 * np.pi * area / perimeter**2 >= 0.75:
            candidates.append(contour)
    if not candidates:
        raise ValueError("Could not locate a complete round dish in the image.")
    points = max(candidates, key=cv2.contourArea).astype(np.float32)
    # Reverse cv2.resize's pixel-centre transform.
    points = (points + 0.5) * np.array([width / small.shape[1], height / small.shape[0]],
                                       dtype=np.float32) - 0.5
    return cv2.fitEllipse(points)


def _agar_mask(gray, dish):
    # Find the inner dark rim along radial spokes. Ignore spokes where ink or texture
    # produces an edge away from the typical rim distance.
    (x, y), diameters, _ = dish
    radius = min(diameters) / 2
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    radii = np.arange(0.89 * radius, 0.947 * radius, dtype=np.float32)
    if radii.size < 12:
        raise ValueError("The dish is too small to resolve the inner agar rim.")
    xs = (x + np.cos(angles[:, None]) * radii).astype(np.float32)
    ys = (y + np.sin(angles[:, None]) * radii).astype(np.float32)
    values = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR)
    fall = values[:, :-6] - values[:, 6:]
    best = np.argmax(fall, axis=1) + 3
    edge = np.column_stack((xs[np.arange(best.size), best], ys[np.arange(best.size), best]))
    strong = fall.max(axis=1)
    usable = ((abs(radii[best] - np.median(radii[best])) < 0.017 * radius)
              & (strong > 0.35 * np.median(strong))
              & (best > 3) & (best < radii.size - 4))
    if np.count_nonzero(usable) < angles.size // 2:
        raise ValueError("Could not locate the inner agar rim in the image.")
    center, diameters, angle = cv2.fitEllipse(edge[usable])
    inset = max(3, round(radius * 0.013))
    axes = tuple(round(d / 2) - inset for d in diameters)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.ellipse(mask, tuple(round(v) for v in center), axes, angle, 0, 360, 1, -1)

    # Exclude very dark marks and a buffer around them, which can otherwise look bright
    # after local background subtraction. Estimate the darkness threshold from the lawn.
    values = gray[mask != 0]
    lawn = np.median(values)
    spread = 1.4826 * np.median(abs(values - lawn))
    dark = ((gray < lawn - 6.5 * spread) & (mask != 0)).astype(np.uint8)
    buffer = max(3, round(radius * 0.023))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buffer + 1,) * 2)
    mask[cv2.dilate(dark, kernel) != 0] = 0
    # Remove slivers of rim isolated by ink. Keep the connected lawn.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        raise ValueError("Could not isolate the agar interior.")
    return (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
