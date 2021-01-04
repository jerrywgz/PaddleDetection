#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
from numbers import Integral

import paddle
from paddle import to_tensor
from ppdet.core.workspace import register, serializable
from ppdet.py_op.target import generate_rpn_anchor_target, generate_proposal_target, generate_mask_target
from ppdet.py_op.post_process import bbox_post_process
from . import ops
import paddle.nn.functional as F


def _to_list(l):
    if isinstance(l, (list, tuple)):
        return list(l)
    return [l]


@register
@serializable
class AnchorGeneratorRPN(object):
    def __init__(self,
                 anchor_sizes=[32, 64, 128, 256, 512],
                 aspect_ratios=[0.5, 1.0, 2.0],
                 strides=[16],
                 variance=[1.0, 1.0, 1.0, 1.0],
                 offset=0.):
        super(AnchorGeneratorRPN, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.variance = variance
        self.cell_anchors = self._calculate_anchors(len(strides))
        self.offset = offset

    def _broadcast_params(self, params, num_features):
        if not isinstance(params[0], (list, tuple)):  # list[float]
            return [params] * num_features
        if len(params) == 1:
            return list(params) * num_features
        return params

    def generate_cell_anchors(self, sizes, aspect_ratios):
        anchors = []
        for size in sizes:
            area = size**2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return paddle.to_tensor(anchors)

    def _calculate_anchors(self, num_features):
        sizes = self._broadcast_params(self.anchor_sizes, num_features)
        aspect_ratios = self._broadcast_params(self.aspect_ratios, num_features)
        cell_anchors = [
            self.generate_cell_anchors(s, a).cast('float32')
            for s, a in zip(sizes, aspect_ratios)
        ]
        return cell_anchors

    def _create_grid_offsets(self, size, stride, offset):
        grid_height, grid_width = size
        shifts_x = paddle.arange(
            offset * stride, grid_width * stride, step=stride, dtype='float32')
        shifts_y = paddle.arange(
            offset * stride, grid_height * stride, step=stride, dtype='float32')
        shift_y, shift_x = paddle.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape([-1])
        shift_y = shift_y.reshape([-1])
        return shift_x, shift_y

    def _grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides,
                                              self.cell_anchors):
            shift_x, shift_y = self._create_grid_offsets(size, stride,
                                                         self.offset)
            shifts = paddle.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

            anchors.append((shifts.reshape([-1, 1, 4]) + base_anchors.reshape(
                [1, -1, 4])).reshape([-1, 4]))

        return anchors

    def __call__(self, input):
        grid_sizes = [feature_map.shape[-2:] for feature_map in input]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return anchors_over_all_feature_maps


@register
@serializable
class AnchorTargetGeneratorRPN(object):
    def __init__(self,
                 batch_size_per_im=256,
                 fg_fraction=0.5,
                 positive_overlap=0.7,
                 negative_overlap=0.3,
                 use_random=True):
        super(AnchorTargetGeneratorRPN, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.use_random = use_random

    def __call__(self, cls_logits, bbox_pred, anchor_box, gt_boxes):
        batch_size = gt_boxes.shape[0]
        tgt_labels, tgt_bboxes, tgt_deltas = generate_rpn_anchor_target(
            anchor_box, gt_boxes, self.batch_size_per_im, self.positive_overlap,
            self.negative_overlap, self.fg_fraction, self.use_random,
            batch_size)

        cls_logits = paddle.reshape(x=cls_logits, shape=(-1, ))
        bbox_pred = paddle.reshape(x=bbox_pred, shape=(-1, 4))
        norm = self.batch_size_per_im * batch_size

        return cls_logits, bbox_pred, tgt_labels, tgt_bboxes, tgt_deltas, norm


@register
@serializable
class AnchorGeneratorSSD(object):
    def __init__(self,
                 steps=[8, 16, 32, 64, 100, 300],
                 aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
                 min_ratio=15,
                 max_ratio=90,
                 min_sizes=[30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
                 max_sizes=[60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
                 offset=0.5,
                 flip=True,
                 clip=False,
                 min_max_aspect_ratios_order=False):
        self.steps = steps
        self.aspect_ratios = aspect_ratios
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.offset = offset
        self.flip = flip
        self.clip = clip
        self.min_max_aspect_ratios_order = min_max_aspect_ratios_order

        self.num_priors = []
        for aspect_ratio, min_size, max_size in zip(aspect_ratios, min_sizes,
                                                    max_sizes):
            self.num_priors.append((len(aspect_ratio) * 2 + 1) * len(
                _to_list(min_size)) + len(_to_list(max_size)))

    def __call__(self, inputs, image):
        boxes = []
        for input, min_size, max_size, aspect_ratio, step in zip(
                inputs, self.min_sizes, self.max_sizes, self.aspect_ratios,
                self.steps):
            box, _ = ops.prior_box(
                input=input,
                image=image,
                min_sizes=_to_list(min_size),
                max_sizes=_to_list(max_size),
                aspect_ratios=aspect_ratio,
                flip=self.flip,
                clip=self.clip,
                steps=[step, step],
                offset=self.offset,
                min_max_aspect_ratios_order=self.min_max_aspect_ratios_order)
            boxes.append(paddle.reshape(box, [-1, 4]))
        return boxes


@register
@serializable
class ProposalGenerator(object):
    __append_doc__ = True

    def __init__(self,
                 train_pre_nms_top_n=12000,
                 train_post_nms_top_n=2000,
                 infer_pre_nms_top_n=6000,
                 infer_post_nms_top_n=1000,
                 nms_thresh=.5,
                 min_size=.1,
                 eta=1.,
                 topk_after_collect=False):
        super(ProposalGenerator, self).__init__()
        self.train_pre_nms_top_n = train_pre_nms_top_n
        self.train_post_nms_top_n = train_post_nms_top_n
        self.infer_pre_nms_top_n = infer_pre_nms_top_n
        self.infer_post_nms_top_n = infer_post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.eta = eta
        self.topk_after_collect = topk_after_collect

    def __call__(self, scores, bbox_deltas, anchors, im_shape, mode='train'):
        pre_nms_top_n = self.train_pre_nms_top_n if mode == 'train' else self.infer_pre_nms_top_n
        if self.topk_after_collect and mode == 'train':
            post_nms_top_n = self.train_pre_nms_top_n
        else:
            post_nms_top_n = self.train_post_nms_top_n if mode == 'train' else self.infer_post_nms_top_n
        variances = paddle.ones_like(anchors)
        rpn_rois, rpn_rois_prob, rpn_rois_num = ops.generate_proposals(
            scores,
            bbox_deltas,
            im_shape,
            anchors,
            variances,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=self.nms_thresh,
            min_size=self.min_size,
            eta=self.eta,
            return_rois_num=True)
        top_n = self.train_post_nms_top_n if mode == 'train' else self.infer_post_nms_top_n
        return rpn_rois, rpn_rois_prob, rpn_rois_num, top_n


@register
@serializable
class ProposalTargetGenerator(object):
    #__shared__ = ['num_classes']

    def __init__(
            self,
            batch_size_per_im=512,
            fg_fraction=.25,
            fg_thresh=[.5, ],
            bg_thresh_hi=[.5, ],
            #bg_thresh_lo=[0., ],
            #bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
            #num_classes=81,
            use_random=True,
            is_cls_agnostic=False):
        super(ProposalTargetGenerator, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        #self.bg_thresh_lo = bg_thresh_lo
        #self.bbox_reg_weights = bbox_reg_weights
        #self.num_classes = num_classes
        self.use_random = use_random

    def __call__(self,
                 rpn_rois,
                 gt_classes,
                 gt_boxes,
                 stage=0,
                 max_overlap=None):
        max_overlap = max_overlap if max_overlap is None else max_overlap
        #reg_weights = [i / (stage + 1) for i in self.bbox_reg_weights]
        is_cascade = True if stage > 0 else False
        #num_classes = 2 if is_cascade else self.num_classes
        outs = generate_proposal_target(
            rpn_rois, gt_classes, gt_boxes, self.batch_size_per_im,
            self.fg_fraction, self.fg_thresh[stage], self.bg_thresh_hi[stage],
            self.use_random, is_cascade, max_overlap)
        return outs


@register
@serializable
class MaskTargetGenerator(object):
    __shared__ = ['mask_resolution']

    def __init__(self, mask_resolution=14):
        super(MaskTargetGenerator, self).__init__()
        self.mask_resolution = mask_resolution

    def __call__(self, gt_segms, rois, rois_num, labels_int32, sampled_gt_inds):
        outs = generate_mask_target(gt_segms, rois, rois_num, labels_int32,
                                    sampled_gt_inds, self.mask_resolution)

        return outs


@register
@serializable
class RCNNBox(object):
    __shared__ = ['num_classes', 'batch_size']

    def __init__(self,
                 num_classes=81,
                 batch_size=1,
                 prior_box_var=[0.1, 0.1, 0.2, 0.2],
                 code_type="decode_center_size",
                 box_normalized=False,
                 axis=1,
                 var_weight=1.):
        super(RCNNBox, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.prior_box_var = prior_box_var
        self.code_type = code_type
        self.box_normalized = box_normalized
        self.axis = axis
        self.var_weight = var_weight

    def __call__(self, bbox_head_out, rois, im_shape, scale_factor):
        bbox_pred, cls_prob = bbox_head_out
        roi, rois_num = rois
        origin_shape = im_shape / scale_factor
        scale_list = []
        origin_shape_list = []
        for idx in range(self.batch_size):
            scale = scale_factor[idx, :][0]
            rois_num_per_im = rois_num[idx]
            expand_scale = paddle.expand(scale, [rois_num_per_im, 1])
            scale_list.append(expand_scale)
            expand_im_shape = paddle.expand(origin_shape[idx, :],
                                            [rois_num_per_im, 2])
            origin_shape_list.append(expand_im_shape)

        scale = paddle.concat(scale_list)
        origin_shape = paddle.concat(origin_shape_list)

        bbox = roi / scale
        prior_box_var = [i / self.var_weight for i in self.prior_box_var]
        bbox = ops.box_coder(
            prior_box=bbox,
            prior_box_var=prior_box_var,
            target_box=bbox_pred,
            code_type=self.code_type,
            box_normalized=self.box_normalized,
            axis=self.axis)
        # TODO: Updata box_clip
        origin_h = paddle.unsqueeze(origin_shape[:, 0] - 1, axis=1)
        origin_w = paddle.unsqueeze(origin_shape[:, 1] - 1, axis=1)
        zeros = paddle.zeros(origin_h.shape, 'float32')
        x1 = paddle.maximum(paddle.minimum(bbox[:, :, 0], origin_w), zeros)
        y1 = paddle.maximum(paddle.minimum(bbox[:, :, 1], origin_h), zeros)
        x2 = paddle.maximum(paddle.minimum(bbox[:, :, 2], origin_w), zeros)
        y2 = paddle.maximum(paddle.minimum(bbox[:, :, 3], origin_h), zeros)
        bbox = paddle.stack([x1, y1, x2, y2], axis=-1)

        bboxes = (bbox, rois_num)
        return bboxes, cls_prob


@register
@serializable
class DecodeClipNms(object):
    __shared__ = ['num_classes']

    def __init__(
            self,
            num_classes=81,
            keep_top_k=100,
            score_threshold=0.05,
            nms_threshold=0.5, ):
        super(DecodeClipNms, self).__init__()
        self.num_classes = num_classes
        self.keep_top_k = keep_top_k
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def __call__(self, bboxes, bbox_prob, bbox_delta, im_info):
        bboxes_np = (i.numpy() for i in bboxes)
        # bbox, bbox_num
        outs = bbox_post_process(bboxes_np,
                                 bbox_prob.numpy(),
                                 bbox_delta.numpy(),
                                 im_info.numpy(), self.keep_top_k,
                                 self.score_threshold, self.nms_threshold,
                                 self.num_classes)
        outs = [to_tensor(v) for v in outs]
        for v in outs:
            v.stop_gradient = True
        return outs


@register
@serializable
class MultiClassNMS(object):
    def __init__(self,
                 score_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 nms_threshold=.5,
                 normalized=False,
                 nms_eta=1.0,
                 background_label=0,
                 return_rois_num=True):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.background_label = background_label
        self.return_rois_num = return_rois_num

    def __call__(self, bboxes, score):
        kwargs = self.__dict__.copy()
        if isinstance(bboxes, tuple):
            bboxes, bbox_num = bboxes
            kwargs.update({'rois_num': bbox_num})
        return ops.multiclass_nms(bboxes, score, **kwargs)


@register
@serializable
class MatrixNMS(object):
    __op__ = ops.matrix_nms
    __append_doc__ = True

    def __init__(self,
                 score_threshold=.05,
                 post_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 use_gaussian=False,
                 gaussian_sigma=2.,
                 normalized=False,
                 background_label=0):
        super(MatrixNMS, self).__init__()
        self.score_threshold = score_threshold
        self.post_threshold = post_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.normalized = normalized
        self.use_gaussian = use_gaussian
        self.gaussian_sigma = gaussian_sigma
        self.background_label = background_label


@register
@serializable
class YOLOBox(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 conf_thresh=0.005,
                 downsample_ratio=32,
                 clip_bbox=True,
                 scale_x_y=1.):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.downsample_ratio = downsample_ratio
        self.clip_bbox = clip_bbox
        self.scale_x_y = scale_x_y

    def __call__(self,
                 yolo_head_out,
                 anchors,
                 im_shape,
                 scale_factor,
                 var_weight=None):
        boxes_list = []
        scores_list = []
        origin_shape = im_shape / scale_factor
        origin_shape = paddle.cast(origin_shape, 'int32')
        for i, head_out in enumerate(yolo_head_out):
            boxes, scores = ops.yolo_box(head_out, origin_shape, anchors[i],
                                         self.num_classes, self.conf_thresh,
                                         self.downsample_ratio // 2**i,
                                         self.clip_bbox, self.scale_x_y)
            boxes_list.append(boxes)
            scores_list.append(paddle.transpose(scores, perm=[0, 2, 1]))
        yolo_boxes = paddle.concat(boxes_list, axis=1)
        yolo_scores = paddle.concat(scores_list, axis=2)
        return yolo_boxes, yolo_scores


@register
@serializable
class SSDBox(object):
    def __init__(self, is_normalized=True):
        self.is_normalized = is_normalized
        self.norm_delta = float(not self.is_normalized)

    def __call__(self,
                 preds,
                 prior_boxes,
                 im_shape,
                 scale_factor,
                 var_weight=None):
        boxes, scores = preds['boxes'], preds['scores']
        outputs = []
        for box, score, prior_box in zip(boxes, scores, prior_boxes):
            pb_w = prior_box[:, 2] - prior_box[:, 0] + self.norm_delta
            pb_h = prior_box[:, 3] - prior_box[:, 1] + self.norm_delta
            pb_x = prior_box[:, 0] + pb_w * 0.5
            pb_y = prior_box[:, 1] + pb_h * 0.5
            out_x = pb_x + box[:, :, 0] * pb_w * 0.1
            out_y = pb_y + box[:, :, 1] * pb_h * 0.1
            out_w = paddle.exp(box[:, :, 2] * 0.2) * pb_w
            out_h = paddle.exp(box[:, :, 3] * 0.2) * pb_h

            if self.is_normalized:
                h = im_shape[:, 0] / scale_factor[:, 0]
                w = im_shape[:, 1] / scale_factor[:, 1]
                output = paddle.stack(
                    [(out_x - out_w / 2.) * w, (out_y - out_h / 2.) * h,
                     (out_x + out_w / 2.) * w, (out_y + out_h / 2.) * h],
                    axis=-1)
            else:
                output = paddle.stack(
                    [
                        out_x - out_w / 2., out_y - out_h / 2.,
                        out_x + out_w / 2. - 1., out_y + out_h / 2. - 1.
                    ],
                    axis=-1)
            outputs.append(output)
        boxes = paddle.concat(outputs, axis=1)

        scores = F.softmax(paddle.concat(scores, axis=1))
        scores = paddle.transpose(scores, [0, 2, 1])

        return boxes, scores


@register
@serializable
class AnchorGrid(object):
    """Generate anchor grid

    Args:
        image_size (int or list): input image size, may be a single integer or
            list of [h, w]. Default: 512
        min_level (int): min level of the feature pyramid. Default: 3
        max_level (int): max level of the feature pyramid. Default: 7
        anchor_base_scale: base anchor scale. Default: 4
        num_scales: number of anchor scales. Default: 3
        aspect_ratios: aspect ratios. default: [[1, 1], [1.4, 0.7], [0.7, 1.4]]
    """

    def __init__(self,
                 image_size=512,
                 min_level=3,
                 max_level=7,
                 anchor_base_scale=4,
                 num_scales=3,
                 aspect_ratios=[[1, 1], [1.4, 0.7], [0.7, 1.4]]):
        super(AnchorGrid, self).__init__()
        if isinstance(image_size, Integral):
            self.image_size = [image_size, image_size]
        else:
            self.image_size = image_size
        for dim in self.image_size:
            assert dim % 2 ** max_level == 0, \
                "image size should be multiple of the max level stride"
        self.min_level = min_level
        self.max_level = max_level
        self.anchor_base_scale = anchor_base_scale
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios

    @property
    def base_cell(self):
        if not hasattr(self, '_base_cell'):
            self._base_cell = self.make_cell()
        return self._base_cell

    def make_cell(self):
        scales = [2**(i / self.num_scales) for i in range(self.num_scales)]
        scales = np.array(scales)
        ratios = np.array(self.aspect_ratios)
        ws = np.outer(scales, ratios[:, 0]).reshape(-1, 1)
        hs = np.outer(scales, ratios[:, 1]).reshape(-1, 1)
        anchors = np.hstack((-0.5 * ws, -0.5 * hs, 0.5 * ws, 0.5 * hs))
        return anchors

    def make_grid(self, stride):
        cell = self.base_cell * stride * self.anchor_base_scale
        x_steps = np.arange(stride // 2, self.image_size[1], stride)
        y_steps = np.arange(stride // 2, self.image_size[0], stride)
        offset_x, offset_y = np.meshgrid(x_steps, y_steps)
        offset_x = offset_x.flatten()
        offset_y = offset_y.flatten()
        offsets = np.stack((offset_x, offset_y, offset_x, offset_y), axis=-1)
        offsets = offsets[:, np.newaxis, :]
        return (cell + offsets).reshape(-1, 4)

    def generate(self):
        return [
            self.make_grid(2**l)
            for l in range(self.min_level, self.max_level + 1)
        ]

    def __call__(self):
        if not hasattr(self, '_anchor_vars'):
            anchor_vars = []
            helper = LayerHelper('anchor_grid')
            for idx, l in enumerate(range(self.min_level, self.max_level + 1)):
                stride = 2**l
                anchors = self.make_grid(stride)
                var = helper.create_parameter(
                    attr=ParamAttr(name='anchors_{}'.format(idx)),
                    shape=anchors.shape,
                    dtype='float32',
                    stop_gradient=True,
                    default_initializer=NumpyArrayInitializer(anchors))
                anchor_vars.append(var)
                var.persistable = True
            self._anchor_vars = anchor_vars

        return self._anchor_vars
