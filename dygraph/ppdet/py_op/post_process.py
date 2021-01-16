import six
import os
import numpy as np
from numba import jit

import paddle
import paddle.nn.functional as F
from .bbox import delta2bbox, clip_bbox, expand_bbox, nms, nonempty_bbox
import cv2


def bbox_post_process(det_res, inputs):
    bbox, bbox_num = det_res  # [N, 6], [N,]
    if bbox.shape[0] == 0:
        return det_res
    im_shape = inputs['im_shape']
    scale_factor = inputs['scale_factor']

    # scale_factor: scale_y, scale_x
    origin_shape = im_shape / scale_factor

    origin_shape_list = []
    scale_factor_list = []
    for i in range(bbox_num.shape[0]):
        expand_shape = paddle.expand(origin_shape[i:i + 1, :], [bbox_num[i], 2])
        scale_y, scale_x = scale_factor[i]
        scale = paddle.concat([scale_x, scale_y, scale_x, scale_y])
        expand_scale = paddle.expand(scale, [bbox_num[i], 4])
        origin_shape_list.append(expand_shape)
        scale_factor_list.append(expand_scale)

    origin_shape_list = paddle.concat(origin_shape_list)
    scale_factor_list = paddle.concat(scale_factor_list)

    pred_label = bbox[:, 0:1]
    pred_score = bbox[:, 1:2]
    pred_bbox = bbox[:, 2:]
    scaled_bbox = pred_bbox / scale_factor_list
    origin_h = origin_shape_list[:, 0]
    origin_w = origin_shape_list[:, 1]
    zeros = paddle.zeros_like(origin_h)
    x1 = paddle.maximum(paddle.minimum(scaled_bbox[:, 0], origin_w), zeros)
    y1 = paddle.maximum(paddle.minimum(scaled_bbox[:, 1], origin_h), zeros)
    x2 = paddle.maximum(paddle.minimum(scaled_bbox[:, 2], origin_w), zeros)
    y2 = paddle.maximum(paddle.minimum(scaled_bbox[:, 3], origin_h), zeros)
    pred_bbox = paddle.stack([x1, y1, x2, y2], axis=-1)

    keep_mask = nonempty_bbox(pred_bbox, return_mask=True)
    keep_mask = paddle.unsqueeze(keep_mask, [1])
    pred_label = paddle.where(keep_mask, pred_label,
                              paddle.ones_like(pred_label) * -1)

    pred_result = paddle.concat([pred_label, pred_score, pred_bbox], axis=1)
    return pred_result, bbox_num


def paste_mask(masks, boxes, im_h, im_w):
    # paste each mask on image
    x0, y0, x1, y1 = paddle.split(boxes, 4, axis=1)
    masks = paddle.unsqueeze(masks, [0, 1])
    img_y = paddle.arange(0, im_h, dtype='float32') + 0.5
    img_x = paddle.arange(0, im_w, dtype='float32') + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    img_x = paddle.unsqueeze(img_x, [1])
    img_y = paddle.unsqueeze(img_y, [2])
    N = boxes.shape[0]

    gx = paddle.expand(img_x, [N, img_y.shape[1], img_x.shape[2]])
    gy = paddle.expand(img_y, [N, img_y.shape[1], img_x.shape[2]])
    grid = paddle.stack([gx, gy], axis=3)
    img_masks = F.grid_sample(masks, grid, align_corners=False)
    return img_masks[:, 0]


def mask_post_process(head_out, bboxes, inputs, thresh=0.5):
    """
    Args:
        head_out (Tensor): output from mask_head, the shape is [N, M, M]
        bboxes (List): prediction bounding bbox, contains [bbox, bbox_num]
        inputs (Dict): input of network
    """
    bbox, bbox_num = bboxes
    if bbox.shape == 0:
        return head_out
    im_shape = inputs['im_shape']
    scale_factor = inputs['scale_factor']

    # scale_factor: scale_y, scale_x
    origin_shape = im_shape / scale_factor
    origin_shape_list = []
    for i in range(bbox_num.shape[0]):
        expand_shape = paddle.expand(origin_shape[i, :], [bbox_num[i], 2])
        origin_shape_list.append(expand_shape)
    origin_shape_list = paddle.floor(paddle.concat(origin_shape_list) + 0.5)

    num_mask = head_out.shape[0]
    # TODO: support bs > 1
    pred_result = paddle.zeros(
        [num_mask, origin_shape_list[0][0], origin_shape_list[0][1]],
        dtype='bool')
    # TODO: optimize chunk paste
    for i in range(bbox.shape[0]):
        im_h, im_w = origin_shape_list[i]
        pred_mask = paste_mask(head_out[i], bbox[i:i + 1, 2:], im_h, im_w)
        pred_mask = pred_mask >= thresh
        pred_result[i] = pred_mask

    return pred_result


def get_det_res(bboxes, scores, labels, bbox_nums, scale_factor, image_id,
                num_id_to_cat_id_map):
    det_res = []
    k = 0
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i][0])
        scale_y, scale_x = scale_factor[i]
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            box = bboxes[k]
            score = float(scores[k])
            label = int(labels[k])
            if label < 0: continue
            k = k + 1
            xmin, ymin, xmax, ymax = box.tolist()
            category_id = num_id_to_cat_id_map[label + 1]
            w = xmax - xmin
            h = ymax - ymin
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': cur_image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res


def get_seg_res(masks, scores, labels, mask_nums, image_id,
                num_id_to_cat_id_map):
    import pycocotools.mask as mask_util
    seg_res = []
    k = 0
    for i in range(len(mask_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = mask_nums[i]
        for j in range(det_nums):
            mask = masks[k]
            score = float(scores[k])
            label = int(labels[k])
            k = k + 1
            cat_id = num_id_to_cat_id_map[label + 1]
            rle = mask_util.encode(
                np.array(
                    mask[:, :, None], order="F", dtype="uint8"))[0]
            if six.PY3:
                if 'counts' in rle:
                    rle['counts'] = rle['counts'].decode("utf8")
            sg_res = {
                'image_id': cur_image_id,
                'category_id': cat_id,
                'segmentation': rle,
                'score': score
            }
            seg_res.append(sg_res)
    return seg_res
