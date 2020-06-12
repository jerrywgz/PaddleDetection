import numpy as np
from numba import jit


@jit
def bbox2delta(bboxes1, bboxes2, weights):
    ex_w = bboxes1[:, 2] - bboxes1[:, 0] + 1
    ex_h = bboxes1[:, 3] - bboxes1[:, 1] + 1
    ex_ctr_x = bboxes1[:, 0] + 0.5 * ex_w
    ex_ctr_y = bboxes1[:, 1] + 0.5 * ex_h

    gt_w = bboxes2[:, 2] - bboxes2[:, 0] + 1
    gt_h = bboxes2[:, 3] - bboxes2[:, 1] + 1
    gt_ctr_x = bboxes2[:, 0] + 0.5 * gt_w
    gt_ctr_y = bboxes2[:, 1] + 0.5 * gt_h

    dx = (gt_ctr_x - ex_ctr_x) / ex_w / weights[0]
    dy = (gt_ctr_y - ex_ctr_y) / ex_h / weights[1]
    dw = (np.log(gt_w / ex_w)) / weights[2]
    dh = (np.log(gt_h / ex_h)) / weights[3]

    deltas = np.vstack([dx, dy, dw, dh]).transpose()
    return deltas


@jit
def delta2bbox(deltas, boxes, weights, bbox_clip=4.13):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] * wx
    dy = deltas[:, 1::4] * wy
    dw = deltas[:, 2::4] * ww
    dh = deltas[:, 3::4] * wh

    # Prevent sending too large values into np.exp()
    dw = np.minimum(dw, bbox_clip)
    dh = np.minimum(dh, bbox_clip)

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


@jit
def compute_targets(bboxes1, bboxes2, labels, bbox_reg_weights):
    assert bboxes1.shape[0] == bboxes2.shape[0]
    assert bboxes1.shape[1] == 4
    assert bboxes2.shape[1] == 4

    targets = np.zeros(bboxes1.shape)
    bbox_reg_weights = np.asarray(bbox_reg_weights)
    targets = bbox2delta(
        bboxes1=bboxes1, bboxes2=bboxes2, weights=bbox_reg_weights)

    return np.hstack([labels[:, np.newaxis], targets]).astype(
        np.float32, copy=False)


@jit
def expand_bbox_targets(bbox_targets_input,
                        class_nums=81,
                        is_cls_agnostic=False):
    class_labels = bbox_targets_input[:, 0]
    fg_inds = np.where(class_labels > 0)[0]
    if not is_cls_agnostic:
        class_nums = 2
    bbox_targets = np.zeros((class_labels.shape[0], 4 * class_nums))
    bbox_inside_weights = np.zeros(bbox_targets.shape)
    for ind in fg_inds:
        class_label = int(class_labels[ind]) if not is_cls_agnostic else 1
        start_ind = class_label * 4
        end_ind = class_label * 4 + 4
        bbox_targets[ind, start_ind:end_ind] = bbox_targets_input[ind, 1:]
        bbox_inside_weights[ind, start_ind:end_ind] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


@jit
def bbox_overlaps(bboxes1, bboxes2):
    w1 = np.maximum(bboxes1[:, 2] - bboxes1[:, 0] + 1, 0)
    h1 = np.maximum(bboxes1[:, 3] - bboxes1[:, 1] + 1, 0)
    w2 = np.maximum(bboxes2[:, 2] - bboxes2[:, 0] + 1, 0)
    h2 = np.maximum(bboxes2[:, 3] - bboxes2[:, 1] + 1, 0)
    area1 = w1 * h1
    area2 = w2 * h2

    overlaps = np.zeros((bboxes1.shape[0], bboxes2.shape[0]))
    for ind1 in range(bboxes1.shape[0]):
        for ind2 in range(bboxes2.shape[0]):
            inter_x1 = np.maximum(bboxes1[ind1, 0], bboxes2[ind2, 0])
            inter_y1 = np.maximum(bboxes1[ind1, 1], bboxes2[ind2, 1])
            inter_x2 = np.minimum(bboxes1[ind1, 2], bboxes2[ind2, 2])
            inter_y2 = np.minimum(bboxes1[ind1, 3], bboxes2[ind2, 3])
            inter_w = np.maximum(inter_x2 - inter_x1 + 1, 0)
            inter_h = np.maximum(inter_y2 - inter_y1 + 1, 0)
            inter_area = inter_w * inter_h
            iou = inter_area * 1.0 / (area1[ind1] + area2[ind2] - inter_area)
            overlaps[ind1, ind2] = iou
    return overlaps


@jit
def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return np.where(suppressed == 0)[0]


@jit
def expand_bbox(bboxes, scale):
    """Expand an array of bboxes by a given scale."""
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * .5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * .5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * .5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    bboxes_exp = np.zeros(bboxes.shape)
    bboxes_exp[:, 0] = x_c - w_half
    bboxes_exp[:, 2] = x_c + w_half
    bboxes_exp[:, 1] = y_c - h_half
    bboxes_exp[:, 3] = y_c + h_half

    return bboxes_exp


@jit
def clip_tiled_bbox(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(
        boxes.shape[1]
    )
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


@jit
def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    """Clip coordinates to an image with the given height and width."""
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2


@jit
def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple bboxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


@jit
def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple bboxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1)))
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')
