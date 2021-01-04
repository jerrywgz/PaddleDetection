import six
import math
import numpy as np
import paddle
from .bbox import *
from .mask import *
import copy


def generate_rpn_anchor_target(anchors,
                               gt_boxes,
                               rpn_batch_size_per_im,
                               rpn_positive_overlap,
                               rpn_negative_overlap,
                               rpn_fg_fraction,
                               use_random=True,
                               batch_size=1,
                               weights=[1., 1., 1., 1.]):
    tgt_labels = []
    tgt_bboxes = []

    tgt_deltas = []
    for i in range(batch_size):
        gt_bbox = gt_boxes[i]

        # Step1: match anchor and gt_bbox
        matches, match_labels, matched_vals = label_box(
            anchors, gt_bbox, rpn_positive_overlap, rpn_negative_overlap, True)
        # Step2: sample anchor 
        fg_inds, bg_inds = subsample_labels(match_labels, rpn_batch_size_per_im,
                                            rpn_fg_fraction, use_random)
        # Fill with the ignore label (-1), then set positive and negative labels
        labels = paddle.full(match_labels.shape, -1, dtype='int32')
        labels = paddle.scatter(labels, fg_inds, paddle.ones_like(fg_inds))
        labels = paddle.scatter(labels, bg_inds, paddle.zeros_like(bg_inds))
        # Step3: make output  
        if gt_bbox.shape[0] == 0:
            matched_gt_boxes = paddle.zeros_like(anchor)
        else:
            matched_gt_boxes = paddle.gather(gt_bbox, matches)

        tgt_delta = bbox2delta(anchors, matched_gt_boxes, weights)
        tgt_labels.append(labels)
        tgt_bboxes.append(matched_gt_boxes)
        tgt_deltas.append(tgt_delta)

    return tgt_labels, tgt_bboxes, tgt_deltas


def label_box(anchors, gt_boxes, positive_overlap, negative_overlap,
              allow_low_quality):
    iou = bbox_overlaps(gt_boxes, anchors)
    if iou.numel() == 0:
        default_matches = paddle.full((iou.shape[1], ), 0, dtype='int64')
        default_match_labels = paddle.full((iou.shape[1], ), -1, dtype='int32')
        return default_matches, default_match_labels

    matched_vals, matches = paddle.topk(iou, k=1, axis=0)
    match_labels = paddle.full(matches.shape, -1, dtype='int32')

    match_labels = paddle.where(
        matched_vals < negative_overlap,
        paddle.zeros(
            [1], dtype='int32'),
        match_labels)
    match_labels = paddle.where(
        matched_vals >= positive_overlap,
        paddle.ones(
            [1], dtype='int32'),
        match_labels)

    if allow_low_quality:
        highest_quality_foreach_gt = iou.max(axis=1, keepdim=True)
        pred_inds_with_highest_quality = (
            iou == highest_quality_foreach_gt).cast('int32').sum(0)

        match_labels = paddle.where(
            pred_inds_with_highest_quality > 0,
            paddle.ones(
                [1], dtype='int32'),
            match_labels)

    matches = matches.reshape([-1])
    match_labels = match_labels.reshape([-1])
    matched_vals = matched_vals.reshape([-1])
    return matches, match_labels, matched_vals


def subsample_labels(labels, num_samples, fg_fraction, use_random=True):
    positive = paddle.nonzero(labels > 0).cast('int32').reshape([-1])
    negative = paddle.nonzero(labels == 0).cast('int32').reshape([-1])

    fg_num = int(num_samples * fg_fraction)
    fg_num = min(positive.numel(), fg_num)
    bg_num = num_samples - fg_num
    bg_num = min(negative.numel(), bg_num)
    # randomly select positive and negative examples
    fg_perm = paddle.randperm(positive.numel(), dtype='int32')
    fg_perm = paddle.slice(fg_perm, axes=[0], starts=[0], ends=[fg_num])
    bg_perm = paddle.randperm(negative.numel(), dtype='int32')
    bg_perm = paddle.slice(bg_perm, axes=[0], starts=[0], ends=[bg_num])
    if use_random:
        fg_inds = paddle.gather(positive, fg_perm)
        bg_inds = paddle.gather(negative, bg_perm)
    else:
        fg_inds = paddle.slice(positive, axes=[0], starts=[0], ends=[fg_num])
        bg_inds = paddle.slice(negative, axes=[0], starts=[0], ends=[bg_num])
    return fg_inds, bg_inds


def filter_roi(rois, max_overlap):
    ws = rois[:, 2] - rois[:, 0]
    hs = rois[:, 3] - rois[:, 1]
    keep = paddle.nonzero((ws > 0) & (hs > 0) & (max_overlap < 1))
    if keep.numel() > 0:
        return rois[keep[:, 1]]
    return paddle.zeros((1, 4), dtype='float32')


def generate_proposal_target(rpn_rois,
                             gt_classes,
                             gt_boxes,
                             batch_size_per_im,
                             fg_fraction,
                             fg_thresh,
                             bg_thresh,
                             use_random=True,
                             is_cascade_rcnn=False,
                             max_overlaps=None):

    rois_with_gt = []
    tgt_labels = []
    tgt_bboxes = []
    sampled_max_overlaps = []
    tgt_gt_inds = []
    new_rois_num = []

    st_num = 0
    end_num = 0
    for i, rpn_roi in enumerate(rpn_rois):
        max_overlap = max_overlaps[i] if is_cascade_rcnn else None
        gt_bbox = gt_boxes[i]
        gt_classes = gt_classes[i]
        if is_cascade_rcnn:
            rpn_roi = filter_roi(rpn_roi, max_overlap)
        bbox = paddle.concat([rpn_roi, gt_bbox])

        # Step1: label bbox 
        matches, match_labels, matched_vals = label_box(
            bbox, gt_bbox, fg_thresh, bg_thresh, False)
        # Step2: sample bbox 
        sampled_inds, sampled_gt_classes = sample_bbox(
            matches, match_labels, gt_classes, batch_size_per_im, fg_fraction,
            use_random)

        # Step3: make output 
        rois_per_image = paddle.gather(bbox, sampled_inds)
        sampled_gt_ind = paddle.gather(matches, sampled_inds)
        sampled_bbox = paddle.gather(gt_bbox, sampled_gt_ind)
        sampled_overlap = paddle.gather(matched_vals.squeeze(), sampled_inds)

        tgt_labels.append(sampled_gt_classes)
        tgt_bboxes.append(sampled_bbox)
        rois_with_gt.append(rois_per_image)
        sampled_max_overlaps.append(sampled_overlap)
        tgt_gt_inds.append(sampled_gt_ind)
        new_rois_num.append(sampled_inds.shape[0])
    new_rois_num = paddle.to_tensor(new_rois_num, dtype='int32')
    return rois_with_gt, tgt_labels, tgt_bboxes, tgt_gt_inds, new_rois_num, sampled_max_overlaps


def sample_bbox(
        matches,
        match_labels,
        gt_classes,
        batch_size_per_im,
        fg_fraction,
        use_random=True, ):
    gt_classes = paddle.gather(gt_classes, matches)
    gt_classes = paddle.where(
        match_labels == 0, paddle.zeros(
            [1], dtype='int32'), gt_classes)
    gt_classes = paddle.where(
        match_labels == -1, paddle.ones(
            [1], dtype='int32') * -1, gt_classes)
    rois_per_image = int(batch_size_per_im)

    fg_inds, bg_inds = subsample_labels(gt_classes, rois_per_image, fg_fraction,
                                        use_random)
    sampled_inds = paddle.concat([fg_inds, bg_inds])
    sampled_gt_classes = paddle.gather(gt_classes, sampled_inds)
    return sampled_inds, sampled_gt_classes


def _strip_pad(gt_polys):
    new_gt_polys = []
    for i in range(gt_polys.shape[0]):
        gt_segs = []
        for j in range(gt_polys[i].shape[0]):
            new_poly = []
            polys = gt_polys[i][j]
            for ii in range(polys.shape[0]):
                x, y = polys[ii]
                if (x == -1 and y == -1):
                    continue
                elif (x >= 0 or y >= 0):
                    new_poly.extend([x, y])  # array, one poly
            if len(new_poly) > 6:
                gt_segs.append(np.array(new_poly).astype('float64'))
        new_gt_polys.append(gt_segs)
    return new_gt_polys


def polygons_to_mask(polygons, height, width):
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    import pycocotools.mask as mask_util
    assert len(polygons) > 0, "COCOAPI does not support empty polygons"
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(np.bool)


def rasterize_polygons_within_box(poly, box, resolution):
    w, h = box[2] - box[0], box[3] - box[1]

    polygons = copy.deepcopy(poly)
    for p in polygons:
        p[0::2] = p[0::2] - box[0]
        p[1::2] = p[1::2] - box[1]

    ratio_h = resolution / max(h, 0.1)
    ratio_w = resolution / max(w, 0.1)

    if ratio_h == ratio_w:
        for p in polygons:
            p *= ratio_h
    else:
        for p in polygons:
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h

    # 3. Rasterize the polygons with coco api
    mask = polygons_to_mask(polygons, resolution, resolution)
    mask = paddle.to_tensor(mask).cast('int32')
    return mask


def generate_mask_target(gt_segms, rois, rois_num, labels_int32,
                         sampled_gt_inds, resolution):
    mask_rois = []
    mask_rois_num = []
    tgt_masks = []
    tgt_classes = []
    mask_index = []
    tgt_weights = []
    for k in range(len(rois)):
        has_fg = True
        rois_per_im = rois[k]
        gt_segms_per_im = gt_segms[k]
        labels_per_im = labels_int32[k]
        fg_inds = paddle.nonzero(labels_per_im > 0)
        if fg_inds.numel() == 0:
            has_fg = False
            fg_inds = paddle.ones([1], dtype='int32')

        inds_per_im = sampled_gt_inds[k]
        inds_per_im = paddle.gather(inds_per_im, fg_inds)

        gt_segms_per_im = paddle.gather(gt_segms_per_im, inds_per_im)

        fg_rois = paddle.gather(rois_per_im, fg_inds)
        fg_classes = paddle.gather(labels_per_im, fg_inds)
        fg_segms = paddle.gather(gt_segms_per_im, fg_inds)
        weight = paddle.ones([fg_rois.shape[0]], dtype='float32')
        if not has_fg:
            weight = weight - 1
        # remove padding
        gt_polys = fg_segms.numpy()
        boxes = fg_rois.numpy()
        new_gt_polys = _strip_pad(gt_polys)
        results = [
            rasterize_polygons_within_box(poly, box, resolution)
            for poly, box in zip(new_gt_polys, boxes)
        ]
        tgt_mask = paddle.stack(results)
        mask_index.append(fg_inds)
        mask_rois.append(fg_rois)
        mask_rois_num.append(fg_rois.shape[0])
        tgt_classes.append(fg_classes)
        tgt_masks.append(tgt_mask)
        tgt_weights.append(weight)

    mask_index = paddle.nonzero(paddle.concat(labels_int32) > 0)
    mask_rois_num = paddle.to_tensor(mask_rois_num, dtype='int32')
    tgt_classes = paddle.concat(tgt_classes, axis=0)
    tgt_masks = paddle.concat(tgt_masks, axis=0)
    tgt_weights = paddle.concat(tgt_weights, axis=0)

    return mask_rois, mask_rois_num, tgt_classes, tgt_masks, mask_index, tgt_weights
