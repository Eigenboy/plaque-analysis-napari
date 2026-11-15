"""Suggest area-preserving splits of touching plaque pairs."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

_NEIGHBORS = np.ones((3, 3), dtype=bool)


@dataclass
class SplitProposal:
    label: int
    bounds: tuple[slice, slice]
    regions: np.ndarray


def _distance_markers(distance, foreground, prominence):
    peaks = h_maxima(distance, prominence, footprint=_NEIGHBORS).astype(bool)
    return ndimage.label(peaks & foreground, structure=_NEIGHBORS)


def _has_brightness_peaks(image, markers, distance, foreground):
    """Check that each of two weak geometric peaks has its own bright center."""
    intensity = ndimage.gaussian_filter(image.astype(np.float32), 0.8)
    low, high = np.percentile(intensity[foreground], (10, 95))
    if high <= low:
        return False
    interior = foreground & (distance >= 1.25)
    response = np.where(interior, np.maximum(intensity - low, 0), 0)
    peaks = np.argwhere(h_maxima(response, 0.18 * (high - low), footprint=_NEIGHBORS) & interior)
    supported = []
    for marker_id in (1, 2):
        coords = np.argwhere(markers == marker_id)
        center = coords.mean(axis=0)
        radius = max(1.5, 0.45 * distance[markers == marker_id].max())
        nearby = np.flatnonzero(np.linalg.norm(peaks - center, axis=1) <= radius)
        supported.append(set(nearby))
    return any(first != second for first in supported[0] for second in supported[1])


def suggest_split(mask, image=None):
    """Return two watershed regions, or None if the mask lacks two distinct lobes."""
    area = np.count_nonzero(mask)
    if area < 16:
        return None

    # Pad with background so the distance is correct at the crop boundary.
    foreground = np.pad(mask, 1)
    if ndimage.label(foreground, structure=_NEIGHBORS)[1] != 1:
        return None
    distance = ndimage.gaussian_filter(ndimage.distance_transform_edt(foreground), 0.65)

    # Strong shape peaks are enough. For a shallow waist, only relax the
    # geometric threshold if the grayscale image supports both lobes.
    prominence = max(0.75, 0.15 * distance.max())
    markers, count = _distance_markers(distance, foreground, prominence)
    if count == 1 and image is not None:
        markers, count = _distance_markers(distance, foreground, max(0.40, 0.07 * distance.max()))
        if count != 2 or not _has_brightness_peaks(
            image, markers[1:-1, 1:-1], distance[1:-1, 1:-1], mask,
        ):
            return None
    elif count != 2:
        return None

    regions = watershed(-distance, markers, mask=foreground, connectivity=2)
    sizes = np.bincount(regions[foreground], minlength=3)
    if sizes[1:].min() < max(8, 0.10 * area):
        return None

    return _order_regions(regions[1:-1, 1:-1], sizes)


def _order_regions(regions, sizes):
    # Keep the original plaque ID on the larger side when accepted.
    if sizes[2] > sizes[1]:
        return np.array([0, 2, 1], dtype=np.uint8)[regions]
    return regions.astype(np.uint8)


def split_from_stroke(labels, vertices):
    """Use a drawn divider to split one mask, retaining the pixels on the cut."""
    if len(vertices) < 2 or np.all(vertices == vertices[0]):
        raise ValueError("Drag a line all the way across the plaque.")
    low = np.maximum(vertices.min(axis=0), 0)
    high = np.minimum(vertices.max(axis=0) + 1, labels.shape)
    if np.any(low >= high):
        raise ValueError("Draw the divider on the plate image.")

    def draw_cut(start, stop):
        cut = np.zeros(tuple(stop - start), dtype=np.uint8)
        points = np.ascontiguousarray((vertices - start)[:, ::-1], dtype=np.int32)
        cv2.polylines(cut, [points], False, 1, lineType=cv2.LINE_4)
        return cut != 0

    stroke_bounds = tuple(slice(a, b) for a, b in zip(low, high))
    ids = np.unique(labels[stroke_bounds][draw_cut(low, high)])
    ids = ids[ids != 0]
    if len(ids) != 1:
        raise ValueError("Draw the divider across just one colored plaque mask.")

    label = int(ids[0])
    pixels = np.argwhere(labels == label)
    low, high = pixels.min(axis=0), pixels.max(axis=0) + 1
    bounds = tuple(slice(a, b) for a, b in zip(low, high))
    mask = labels[bounds] == label
    if ndimage.label(mask, structure=_NEIGHBORS)[1] != 1:
        raise ValueError("The selected plaque ID already has disconnected pieces.")
    cut = draw_cut(low, high)
    markers, count = ndimage.label(mask & ~cut, structure=_NEIGHBORS)
    if count != 2:
        raise ValueError("Draw one divider from outside the mask, across it, to outside again.")

    # Both sides are fixed markers. Watershed assigns only pixels touched by the line.
    distance = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    regions = watershed(-distance, markers, mask=mask, connectivity=2)
    sizes = np.bincount(regions[mask], minlength=3)
    return SplitProposal(label, bounds, _order_regions(regions, sizes))


def find_split_proposals(labels, image):
    """Scan every label, in ID order, using a compact index for sparse IDs."""
    rows, cols = np.nonzero(labels)
    if not rows.size:
        return []
    ids, compact_ids = np.unique(labels[rows, cols], return_inverse=True)
    compact = np.zeros(labels.shape, dtype=np.int32)
    compact[rows, cols] = compact_ids + 1
    proposals = []
    for label, bounds in zip(ids, ndimage.find_objects(compact, max_label=len(ids))):
        regions = suggest_split(labels[bounds] == label, image[bounds])
        if regions is not None:
            proposals.append(SplitProposal(int(label), bounds, regions))
    return proposals
