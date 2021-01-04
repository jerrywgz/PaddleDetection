import numpy as np
from numba import jit
import math
import paddle


def bbox2delta(src_boxes, tgt_boxes, weights):
    src_w = src_boxes[:, 2] - src_boxes[:, 0]
    src_h = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_h

    tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h

    wx, wy, ww, wh = weights
    dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
    dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
    dw = ww * paddle.log(tgt_w / src_w)
    dh = wh * paddle.log(tgt_h / src_h)

    deltas = paddle.stack((dx, dy, dw, dh), axis=1)
    return deltas


def delta2bbox(deltas, boxes, weights):
    clip_scale = math.log(1000.0 / 16)
    if boxes.shape[0] == 0:
        return paddle.zeros((0, deltas.shape[1]), dtype='float32')

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx, dy, dw, dh = paddle.tensor.split(deltas, 4, axis=1)
    dx = dx * wx
    dy = dy * wy
    dw = dw * ww
    dh = dh * wh
    # Prevent sending too large values into np.exp()
    dw = paddle.clip(dw, max=clip_scale)
    dh = paddle.clip(dh, max=clip_scale)

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = paddle.exp(dw) * widths.unsqueeze(1)
    pred_h = paddle.exp(dh) * heights.unsqueeze(1)

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = paddle.concat([x1, y1, x2, y2], axis=1)

    return pred_boxes


def expand_bbox(bboxes, scale):
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * .5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * .5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * .5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    bboxes_exp = np.zeros(bboxes.shape, dtype=np.float32)
    bboxes_exp[:, 0] = x_c - w_half
    bboxes_exp[:, 2] = x_c + w_half
    bboxes_exp[:, 1] = y_c - h_half
    bboxes_exp[:, 3] = y_c + h_half

    return bboxes_exp


def clip_bbox(boxes, im_shape):
    h, w = im_shape
    x1 = boxes[:, 0].clip(0, w)
    y1 = boxes[:, 1].clip(0, h)
    x2 = boxes[:, 2].clip(0, w)
    y2 = boxes[:, 3].clip(0, h)
    return paddle.stack([x1, y1, x2, y2], axis=1)


def nonempty_bbox(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = paddle.logical_and(w > min_size, w > min_size)
    keep = paddle.nonzero(mask).squeeze()
    return keep


def bbox_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def bbox_overlaps(boxes1, boxes2):
    area1 = bbox_area(boxes1)
    area2 = bbox_area(boxes2)

    xy_max = paddle.minimum(
        paddle.unsqueeze(boxes1, 1)[:, :, 2:], boxes2[:, 2:])
    xy_min = paddle.maximum(
        paddle.unsqueeze(boxes1, 1)[:, :, :2], boxes2[:, :2])
    width_height = xy_max - xy_min
    width_height = width_height.clip(min=0)
    inter = width_height.prod(axis=2)

    overlaps = paddle.where(
        inter > 0,
        inter / (paddle.unsqueeze(area1, 1) + area2 - inter),
        paddle.zeros(
            [1], dtype='float32'), )
    return overlaps


@jit
def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
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


def nms_with_decode(bboxes,
                    bbox_probs,
                    bbox_deltas,
                    im_info,
                    keep_top_k=100,
                    score_thresh=0.05,
                    nms_thresh=0.5,
                    class_nums=81,
                    bbox_reg_weights=[0.1, 0.1, 0.2, 0.2]):
    bboxes_num = [0, bboxes.shape[0]]
    bboxes_v = np.array(bboxes)
    bbox_probs_v = np.array(bbox_probs)
    bbox_deltas_v = np.array(bbox_deltas)
    variance_v = np.array(bbox_reg_weights)
    im_results = [[] for _ in range(len(bboxes_num) - 1)]
    new_bboxes_num = [0]
    for i in range(len(bboxes_num) - 1):
        start = bboxes_num[i]
        end = bboxes_num[i + 1]
        if start == end:
            continue

        bbox_deltas_n = bbox_deltas_v[start:end, :]  # box delta 
        rois_n = bboxes_v[start:end, :]  # box 
        rois_n = rois_n / im_info[i][2]  # scale 
        rois_n = delta2bbox(bbox_deltas_n, rois_n, variance_v)
        rois_n = clip_bbox(rois_n, np.round(im_info[i][:2] / im_info[i][2]))
        cls_boxes = [[] for _ in range(class_nums)]
        scores_n = bbox_probs_v[start:end, :]
        for j in range(1, class_nums):
            inds = np.where(scores_n[:, j] > score_thresh)[0]
            scores_j = scores_n[inds, j]
            rois_j = rois_n[inds, j * 4:(j + 1) * 4]
            dets_j = np.hstack((scores_j[:, np.newaxis], rois_j)).astype(
                np.float32, copy=False)
            keep = nms(dets_j, nms_thresh)
            nms_dets = dets_j[keep, :]
            #add labels
            label = np.array([j for _ in range(len(keep))])
            nms_dets = np.hstack((label[:, np.newaxis], nms_dets)).astype(
                np.float32, copy=False)
            cls_boxes[j] = nms_dets

        # Limit to max_per_image detections **over all classes**
        image_scores = np.hstack(
            [cls_boxes[j][:, 1] for j in range(1, class_nums)])
        if len(image_scores) > keep_top_k:
            image_thresh = np.sort(image_scores)[-keep_top_k]
            for j in range(1, class_nums):
                keep = np.where(cls_boxes[j][:, 1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]
        im_results_n = np.vstack([cls_boxes[j] for j in range(1, class_nums)])
        im_results[i] = im_results_n
        new_bboxes_num.append(len(im_results_n) + new_bboxes_num[-1])
        labels = im_results_n[:, 0]
        scores = im_results_n[:, 1]
        boxes = im_results_n[:, 2:]
    im_results = np.vstack([im_results[k] for k in range(len(bboxes_num) - 1)])
    new_bboxes_num = np.array(new_bboxes_num)
    return new_bboxes_num, im_results


#@jit
def compute_bbox_targets(bboxes1, bboxes2, labels, bbox_reg_weights):
    assert bboxes1.shape[0] == bboxes2.shape[0]
    assert bboxes1.shape[1] == 4
    assert bboxes2.shape[1] == 4

    targets = np.zeros(bboxes1.shape)
    bbox_reg_weights = np.asarray(bbox_reg_weights)
    targets = bbox2delta(
        bboxes1=bboxes1, bboxes2=bboxes2, weights=bbox_reg_weights)

    return np.hstack([labels[:, np.newaxis], targets]).astype(
        np.float32, copy=False)


#@jit
def expand_bbox_targets(bbox_targets_input,
                        class_nums=81,
                        is_cls_agnostic=False):
    class_labels = bbox_targets_input[:, 0]
    fg_inds = np.where(class_labels > 0)[0]
    if is_cls_agnostic:
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
