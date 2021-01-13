import numpy as np
import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ppdet.py_op.bbox import delta2bbox, clip_bbox, nonempty_bbox
from . import ops


@register
class Anchor(object):
    __inject__ = ['anchor_generator', 'anchor_target_generator']

    def __init__(self, anchor_generator, anchor_target_generator):
        super(Anchor, self).__init__()
        self.anchor_generator = anchor_generator
        self.anchor_target_generator = anchor_target_generator

    def __call__(self, features):
        anchors = self.anchor_generator(features)
        return anchors

    def _get_target_input(self, rpn_scores, rpn_deltas, anchors):
        rpn_score_list = []
        rpn_delta_list = []
        anchor_list = []
        for rpn_score, rpn_delta, anchor in zip(rpn_scores, rpn_deltas,
                                                anchors):
            rpn_score = paddle.transpose(rpn_score, perm=[0, 2, 3, 1])
            rpn_delta = paddle.transpose(rpn_delta, perm=[0, 2, 3, 1])
            rpn_score = paddle.reshape(x=rpn_score, shape=(0, -1, 1))
            rpn_delta = paddle.reshape(x=rpn_delta, shape=(0, -1, 4))

            anchor = paddle.reshape(anchor, shape=(-1, 4))
            var = paddle.ones_like(anchor)
            rpn_score_list.append(rpn_score)
            rpn_delta_list.append(rpn_delta)
            anchor_list.append(anchor)

        rpn_scores = paddle.concat(rpn_score_list, axis=1)
        rpn_deltas = paddle.concat(rpn_delta_list, axis=1)
        anchors = paddle.concat(anchor_list)
        return rpn_scores, rpn_deltas, anchors

    def generate_loss_inputs(self, inputs, rpn_scores, rpn_deltas, anchors):
        if len(rpn_scores) != len(anchors):
            raise ValueError(
                "rpn_scores and anchors should have same length, "
                " but received rpn_scores' length is {} and anchors' "
                " length is {}".format(len(rpn_scores), len(anchors)))
        rpn_score, rpn_delta, anchors = self._get_target_input(
            rpn_scores, rpn_deltas, anchors)

        score_pred, roi_pred, tgt_labels, tgt_bboxes, tgt_deltas, norm = self.anchor_target_generator(
            bbox_pred=rpn_delta,
            cls_logits=rpn_score,
            anchor_box=anchors,
            gt_boxes=inputs['gt_bbox'])
        outs = {
            'rpn_score_pred': score_pred,
            'rpn_score_target': tgt_labels,
            'rpn_rois_pred': roi_pred,
            'rpn_rois_target': tgt_deltas,
            'norm': norm,
        }
        return outs


@register
class Proposal(object):
    __inject__ = ['proposal_generator', 'proposal_target_generator']

    #__inject__ = ['proposal_target_generator']

    def __init__(
            self,
            proposal_generator,
            proposal_target_generator,
            decode_weight=[1., 1., 1., 1.],
            train_pre_nms_top_n=12000.,
            train_post_nms_top_n=2000.,
            infer_pre_nms_top_n=6000.,
            infer_post_nms_top_n=1000,
            min_size=0.,
            nms_threshold=0.5, ):
        super(Proposal, self).__init__()
        self.proposal_generator = proposal_generator
        self.decode_weight = decode_weight
        self.train_pre_nms_top_n = train_pre_nms_top_n
        self.train_post_nms_top_n = train_post_nms_top_n
        self.infer_pre_nms_top_n = infer_pre_nms_top_n
        self.infer_post_nms_top_n = infer_post_nms_top_n
        self.min_size = min_size
        self.nms_threshold = nms_threshold

        self.proposal_target_generator = proposal_target_generator

    def _decode_proposals(self, anchors, deltas):
        batch_size = deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, deltas_i in zip(anchors, deltas):
            deltas_i = deltas_i.reshape([-1, 4])
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.unsqueeze(0).tile([batch_size, 1, 1]).reshape(
                [-1, 4])
            proposals_i = delta2bbox(deltas_i, anchors_i, self.decode_weight)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.reshape([batch_size, -1, 4]))
        return proposals

    def _nms_per_img(self, boxes, scores, lvl, post_nms_top_n):
        rois_num = paddle.shape(boxes)[0]

        max_value = boxes.max()
        offsets = lvl * (max_value + 1)
        boxes_for_nms = boxes + offsets.unsqueeze(1)

        boxes_for_nms = boxes_for_nms.unsqueeze(1)  # topk, 1, 4
        scores_for_nms = scores.unsqueeze(1)  # topk, 1 

        # TODO: add cuda implement for nms
        nms_result, _, index = ops.multiclass_nms(
            boxes_for_nms,
            scores_for_nms,
            score_threshold=-sys.maxsize,
            nms_top_k=-1,
            keep_top_k=post_nms_top_n,
            nms_threshold=self.nms_threshold,
            normalized=True,
            background_label=-1,
            return_index=True,
            rois_num=rois_num)

        boxes_nms = paddle.gather(boxes, index)
        scores_nms = paddle.gather(scores, index)
        topk_index = paddle.argsort(scores_nms, axis=0, descending=True)
        boxes_nms = paddle.gather(boxes_nms, topk_index)
        scores_nms = paddle.gather(scores_nms, topk_index)
        return boxes_nms, scores_nms

    def generate_proposal(self, inputs, rpn_scores, rpn_deltas, anchors):
        """
        ims_shape = inputs['im_shape']
        batch_size = ims_shape.shape[0]
        rpn_scores_list = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.transpose([0, 2, 3, 1]).flatten(1)
            for score in rpn_scores
        ]
        rpn_deltas_list = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.reshape([x.shape[0], -1, 4, x.shape[-2], x.shape[-1]])
            .transpose([0, 3, 4, 1, 2])
            .flatten(1, -2)
            for x in rpn_deltas
        ]

        proposals = self._decode_proposals(anchors, rpn_deltas_list)
       
        post_nms_top_n = self.train_post_nms_top_n if inputs['mode'] == 'train' else self.infer_post_nms_top_n
        # 1. Select top-k anchor for every level and every image
        topk_scores = []  
        topk_proposals = []
        level_ids = []  
        pre_nms_top_n = self.train_pre_nms_top_n if inputs['mode'] == 'train' else self.infer_pre_nms_top_n
        post_nms_top_n = self.train_post_nms_top_n if inputs['mode'] == 'train' else self.infer_post_nms_top_n
        batch_idx = paddle.arange(batch_size).unsqueeze([1,2])
        box_idx = paddle.arange(4)
        for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, rpn_scores_list)):

            Hi_Wi_A = logits_i.shape[1]
            num_proposals_i = min(pre_nms_top_n, Hi_Wi_A)

            topk_scores_i, topk_col_idx = logits_i.topk(num_proposals_i, axis=1)
            topk_row_idx = paddle.tile(batch_idx, [1, num_proposals_i, 4]).reshape([-1,1]) # N x topk x 4
            topk_col_idx = paddle.tile(topk_col_idx.unsqueeze(2), [1, 1, 4]).reshape([-1, 1]) # N x topk x 4
            box_idx_per_lvl = paddle.tile(box_idx, [batch_size * num_proposals_i]).unsqueeze(1)

            topk_idx = paddle.concat([topk_row_idx, topk_col_idx, box_idx_per_lvl], axis=1)
            topk_proposals_i = paddle.gather_nd(proposals_i, topk_idx).reshape([batch_size, num_proposals_i, 4]) # N x topk x 4

            topk_proposals.append(topk_proposals_i)
            topk_scores.append(topk_scores_i)
            level_ids.append(paddle.full([num_proposals_i], level_id, dtype='int64'))


        # 2. Concat all levels together
        topk_scores = paddle.concat(topk_scores, axis=1) if len(topk_scores) > 1 else topk_scores[0]
        topk_proposals = paddle.concat(topk_proposals, axis=1) if len(topk_proposals) > 1 else topk_proposals[0]
        level_ids = paddle.concat(level_ids) if len(level_ids) > 1 else level_ids[0]

        # 3. For each image, run a per-level NMS, and choose topk results.
        rois = []
        rois_num = []
        for n in range(batch_size):
            im_shape = ims_shape[n]
            boxes = topk_proposals[n]
            scores_per_img = topk_scores[n]
            lvl = level_ids
            boxes = clip_bbox(boxes, im_shape)

            # filter empty boxes
            keep = nonempty_bbox(boxes, self.min_size)
            if keep.shape[0] == 0:
                boxes = paddle.zeros([1, 4], dtype='float32')
                scores_per_img = paddle.zeros([1], dtype='float32')
                lvl = paddle.zeros([1], dtype='float32')
            elif keep.shape[0] != boxes.shape[0]:
                boxes = paddle.gather(boxes, keep)
                scores_per_img = paddle.gather(scores_per_img, keep)
                lvl = paddle.gather(lvl, keep)

            boxes, scores_per_img = self._nms_per_img(boxes, scores_per_img, lvl, post_nms_top_n)
            rois.append(boxes)
            rois_num.append(paddle.shape(boxes)[0])

        return rois, rois_num
        """
        im_shape = inputs['im_shape']
        batch_size = im_shape.shape[0]
        rpn_rois_list = [[] for i in range(batch_size)]
        rpn_prob_list = [[] for i in range(batch_size)]
        rpn_rois_num_list = [[] for i in range(batch_size)]
        for (rpn_score, rpn_delta, anchor) in zip(rpn_scores, rpn_deltas,
                                                  anchors):
            for i in range(batch_size):
                rpn_rois, rpn_rois_prob, rpn_rois_num, post_nms_top_n = self.proposal_generator(
                    scores=rpn_score[i:i + 1],
                    bbox_deltas=rpn_delta[i:i + 1],
                    anchors=anchor,
                    im_shape=im_shape[i:i + 1],
                    mode=inputs['mode'])
                if rpn_rois.shape[0] > 0:
                    rpn_rois_list[i].append(rpn_rois)
                    rpn_prob_list[i].append(rpn_rois_prob)
                    rpn_rois_num_list[i].append(rpn_rois_num)

        rois_collect = []
        rois_num_collect = []
        for i in range(batch_size):
            if len(rpn_scores) > 1:
                rpn_rois = paddle.concat(rpn_rois_list[i])
                rpn_prob = paddle.concat(rpn_prob_list[i]).flatten()
                if rpn_prob.shape[0] > post_nms_top_n:
                    #print('collect fpn: ', rpn_prob.shape, post_nms_top_n)
                    topk_prob, topk_inds = paddle.topk(rpn_prob, post_nms_top_n)
                    topk_rois = paddle.gather(rpn_rois, topk_inds)
                else:
                    topk_rois = rpn_rois
                    topk_prob = rpn_prob
            else:
                topk_rois = rpn_rois_list[0]
                topk_prob = rpn_prob_list[0].flatten()
            rois_collect.append(topk_rois)
            rois_num_collect.append(topk_rois.shape[0])
        #print('rois_collect: ', rois_collect[0].shape)
        return rois_collect, rois_num_collect

    def generate_proposal_target(self, inputs, rois, stage=0, max_overlap=None):
        outs = self.proposal_target_generator(
            rpn_rois=rois,
            gt_classes=inputs['gt_class'],
            gt_boxes=inputs['gt_bbox'],
            stage=stage,
            max_overlap=max_overlap)
        rois = outs[0]
        max_overlap = outs[-1]
        rois_num = outs[-2]
        targets = {
            'labels_int32': outs[1],
            'bbox_targets': outs[2],
            'sampled_gt_inds': outs[3],
        }
        return rois, rois_num, targets, max_overlap

    def refine_bbox(self, roi, bbox_delta, stage=1):
        out_dim = bbox_delta.shape[1] // 4
        bbox_delta_r = paddle.reshape(bbox_delta, (-1, out_dim, 4))
        bbox_delta_s = paddle.slice(
            bbox_delta_r, axes=[1], starts=[1], ends=[2])

        reg_weights = [
            i / stage for i in self.proposal_target_generator.bbox_reg_weights
        ]
        refined_bbox = ops.box_coder(
            prior_box=roi,
            prior_box_var=reg_weights,
            target_box=bbox_delta_s,
            code_type='decode_center_size',
            box_normalized=False,
            axis=1)
        refined_bbox = paddle.reshape(refined_bbox, shape=[-1, 4])
        return refined_bbox

    def __call__(self,
                 inputs,
                 rpn_scores,
                 rpn_deltas,
                 anchor,
                 stage=0,
                 proposal_out=None,
                 bbox_head_out=None,
                 max_overlap=None):
        if stage == 0:
            roi, rois_num = self.generate_proposal(inputs, rpn_scores,
                                                   rpn_deltas, anchor)
            self.targets_list = []
            self.max_overlap = None

        else:
            bbox_delta = bbox_head_out[1]
            roi = self.refine_bbox(proposal_out[0], bbox_delta, stage)
            rois_num = proposal_out[1]
        if inputs['mode'] == 'train':
            roi, rois_num, targets, self.max_overlap = self.generate_proposal_target(
                inputs, roi, stage, self.max_overlap)
            self.targets_list.append(targets)
        return roi, rois_num

    def get_targets(self):
        return self.targets_list

    def get_max_overlap(self):
        return self.max_overlap
