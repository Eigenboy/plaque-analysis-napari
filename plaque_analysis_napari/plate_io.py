"""File naming and image display helpers shared by the plate plugins."""

import cv2
import numpy as np

IMAGE_FILTER = "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)"


def load_grayscale(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Could not read the image.")
    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
        raise ValueError("Choose a single grayscale image with 8-bit or 16-bit pixels.")
    return image


def agar_mask_path(image_path):
    return image_path.parent / "out" / f"{image_path.stem}_agar_mask.tif"


def display_limits(image, mask=None):
    if mask is None:
        lo, hi = np.percentile(image, (1, 99))
        return float(lo), float(max(hi, lo + 1))
    lo, hi = np.percentile(image[mask != 0], (2, 99.95))
    padding = 0.12 * (hi - lo)
    return float(lo - padding), float(max(hi + padding, lo + 1))
