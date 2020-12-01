import six
import math
import numpy as np
from numba import jit
import pycocotools.mask as mask_util


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]
    return boxes_from_polys


@jit
def polys_to_mask_wrt_box(polygons, box, M):
    w = box[2] - box[0]
    h = box[3] - box[1]
    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    polygons_norm = []
    for poly in polygons:
        p = np.array(poly, dtype=np.float32)
        p[0::2] = (p[0::2] - box[0]) * M / w
        p[1::2] = (p[1::2] - box[1]) * M / h
        polygons_norm.append(p)
    rle = mask_util.frPyObjects(polygons_norm, M, M)
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask


#@jit
def expand_mask_targets(masks, mask_class_labels, resolution, num_classes):
    """Expand masks from shape (#masks, resolution ** 2)
    to (#masks, #classes * resolution ** 2) to encode class
    specific mask targets.
    """
    assert masks.shape[0] == mask_class_labels.shape[0]
    # Target values of -1 are "don't care" / ignore labels
    mask_targets = -np.ones(
        (masks.shape[0], num_classes * resolution**2), dtype=np.int32)
    for i in range(masks.shape[0]):
        cls = int(mask_class_labels[i])
        start = resolution**2 * cls
        end = start + resolution**2
        # Ignore background instance
        # (only happens when there is no fg samples in an image)
        if cls > 0:
            mask_targets[i, start:end] = masks[i, :]

    return mask_targets
