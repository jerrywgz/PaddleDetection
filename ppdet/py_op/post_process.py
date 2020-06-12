import six
import os
import numpy as np
from numba import jit
from .bbox import nms, delta2bbox, clip_tiled_bbox


#@jit 
def get_nmsed_bbox(bboxes,
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
        rois_n = clip_tiled_bbox(rois_n, im_info[i][:2] / im_info[i][2])
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


@jit
def get_dt_res(batch_size, bboxes_num, nmsed_out, data, num_id_to_cat_id_map):
    dts_res = []
    nmsed_out_v = np.array(nmsed_out)
    if nmsed_out_v.shape == (
            1,
            1, ):
        return dts_res
    assert (len(bboxes_num) == batch_size + 1), \
      "Error bboxes_num Tensor offset dimension. bboxes_num({}) vs. batch_size({})"\
                    .format(len(bboxes_num), batch_size)
    k = 0
    for i in range(batch_size):
        dt_num_this_img = bboxes_num[i + 1] - bboxes_num[i]
        image_id = int(data[i][-1])
        image_width = int(data[i][1][1])
        image_height = int(data[i][1][2])
        for j in range(dt_num_this_img):
            dt = nmsed_out_v[k]
            k = k + 1
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            category_id = num_id_to_cat_id_map[num_id]
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            dts_res.append(dt_res)
    return dts_res


@jit
def get_segms_res(batch_size, mask_nums, segms_out, data, num_id_to_cat_id_map):
    segms_res = []
    segms_out_v = np.array(segms_out)
    k = 0
    for i in range(batch_size):
        dt_num_this_img = mask_nums[i + 1] - mask_nums[i]
        image_id = int(data[i][-1])
        for j in range(dt_num_this_img):
            dt = segms_out_v[k]
            k = k + 1
            segm, num_id, score = dt.tolist()
            cat_id = num_id_to_cat_id_map[num_id]
            if six.PY3:
                if 'counts' in segm:
                    segm['counts'] = segm['counts'].decode("utf8")
            segm_res = {
                'image_id': image_id,
                'category_id': cat_id,
                'segmentation': segm,
                'score': score
            }
            segms_res.append(segm_res)
    return segms_res
