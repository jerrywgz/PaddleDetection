# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import ReLU
from paddle.nn.initializer import Normal, XavierUniform
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling import ops
from ppdet.py_op.bbox import bbox2delta


@register
class TwoFCHead(nn.Layer):

    __shared__ = ['roi_stages']

    def __init__(self, in_dim=256, mlp_dim=1024, resolution=7, roi_stages=1):
        super(TwoFCHead, self).__init__()
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim
        self.roi_stages = roi_stages
        fan = in_dim * resolution * resolution
        self.fc6_list = []
        self.fc6_relu_list = []
        self.fc7_list = []
        self.fc7_relu_list = []
        for stage in range(roi_stages):
            fc6_name = 'fc6_{}'.format(stage)
            fc7_name = 'fc7_{}'.format(stage)
            lr_factor = 2**stage
            fc6 = self.add_sublayer(
                fc6_name,
                nn.Linear(
                    in_dim * resolution * resolution,
                    mlp_dim,
                    weight_attr=ParamAttr(
                        learning_rate=lr_factor,
                        initializer=XavierUniform(fan_out=fan))))
            fc6_relu = self.add_sublayer(fc6_name + 'act', ReLU())
            fc7 = self.add_sublayer(
                fc7_name,
                nn.Linear(
                    mlp_dim,
                    mlp_dim,
                    weight_attr=ParamAttr(
                        learning_rate=lr_factor, initializer=XavierUniform())))
            fc7_relu = self.add_sublayer(fc7_name + 'act', ReLU())
            self.fc6_list.append(fc6)
            self.fc6_relu_list.append(fc6_relu)
            self.fc7_list.append(fc7)
            self.fc7_relu_list.append(fc7_relu)

    def forward(self, rois_feat, stage=0):
        rois_feat = paddle.flatten(rois_feat, start_axis=1, stop_axis=-1)
        fc6 = self.fc6_list[stage](rois_feat)
        fc6_relu = self.fc6_relu_list[stage](fc6)
        fc7 = self.fc7_list[stage](fc6_relu)
        fc7_relu = self.fc7_relu_list[stage](fc7)
        return fc7_relu


@register
class BBoxFeat(nn.Layer):
    __inject__ = ['roi_extractor', 'head_feat']

    def __init__(self, roi_extractor, head_feat):
        super(BBoxFeat, self).__init__()
        self.roi_extractor = roi_extractor
        self.head_feat = head_feat
        self.rois_feat_list = []

    def forward(self, body_feats, rois, spatial_scale, stage=0):
        rois_feat = self.roi_extractor(body_feats, rois, spatial_scale)
        bbox_feat = self.head_feat(rois_feat, stage)
        return rois_feat, bbox_feat


@register
class BBoxHead(nn.Layer):
    __shared__ = ['num_classes', 'roi_stages']
    __inject__ = ['bbox_feat']

    def __init__(self,
                 bbox_feat,
                 in_feat=1024,
                 num_classes=81,
                 cls_agnostic=False,
                 roi_stages=1,
                 with_pool=False,
                 score_stage=[0, 1, 2],
                 bbox_weight=[10., 10., 5., 5.],
                 delta_stage=[2]):
        super(BBoxHead, self).__init__()
        self.num_classes = num_classes
        self.cls_agnostic = cls_agnostic
        self.delta_dim = 2 if cls_agnostic else num_classes - 1
        self.bbox_feat = bbox_feat
        self.roi_stages = roi_stages
        self.bbox_score_list = []
        self.bbox_delta_list = []
        self.roi_feat_list = [[] for i in range(roi_stages)]
        self.with_pool = with_pool
        self.score_stage = score_stage
        self.bbox_weight = bbox_weight
        self.delta_stage = delta_stage
        for stage in range(roi_stages):
            score_name = 'bbox_score_{}'.format(stage)
            delta_name = 'bbox_delta_{}'.format(stage)
            lr_factor = 2**stage
            bbox_score = self.add_sublayer(
                score_name,
                nn.Linear(
                    in_feat,
                    1 * self.num_classes,
                    weight_attr=ParamAttr(
                        learning_rate=lr_factor,
                        initializer=Normal(
                            mean=0.0, std=0.01))))

            bbox_delta = self.add_sublayer(
                delta_name,
                nn.Linear(
                    in_feat,
                    4 * self.delta_dim,
                    weight_attr=ParamAttr(
                        learning_rate=lr_factor,
                        initializer=Normal(
                            mean=0.0, std=0.001))))
            self.bbox_score_list.append(bbox_score)
            self.bbox_delta_list.append(bbox_delta)

    def forward(self,
                body_feats=None,
                rois=None,
                spatial_scale=None,
                stage=0,
                roi_stage=-1):
        if rois is not None:
            rois_feat, bbox_feat = self.bbox_feat(body_feats, rois,
                                                  spatial_scale, stage)
            self.roi_feat_list[stage] = rois_feat
        else:
            rois_feat = self.roi_feat_list[roi_stage]
            bbox_feat = self.bbox_feat.head_feat(rois_feat, stage)
        if self.with_pool:
            bbox_feat_ = F.adaptive_avg_pool2d(bbox_feat, output_size=1)
            bbox_feat_ = paddle.squeeze(bbox_feat_, axis=[2, 3])
            scores = self.bbox_score_list[stage](bbox_feat_)
            deltas = self.bbox_delta_list[stage](bbox_feat_)
        else:
            scores = self.bbox_score_list[stage](bbox_feat)
            deltas = self.bbox_delta_list[stage](bbox_feat)
        bbox_head_out = (scores, deltas)
        return bbox_feat, bbox_head_out, self.bbox_feat.head_feat

    def _get_head_loss(self, score, delta, target, rois):
        # bbox cls  
        labels_int32 = target['labels_int32']
        labels_int32 = paddle.concat(labels_int32) if len(
            labels_int32) > 1 else labels_int32[0]
        label_mask = paddle.cast(labels_int32 == 0, 'int32')
        labels_int32 = labels_int32 + label_mask * 81
        labels_int32 = labels_int32 - 1
        labels_int64 = labels_int32.cast('int64')
        labels_int64.stop_gradient = True
        loss_bbox_cls = F.cross_entropy(
            input=score, label=labels_int64, reduction='mean')
        # bbox reg

        cls_agnostic_bbox_reg = delta.shape[1] == 4

        fg_inds = paddle.nonzero(
            paddle.logical_and(labels_int64 >= 0, labels_int64 < 80)).flatten()
        #fg_inds = paddle.nonzero(labels_int64 > 0).flatten()

        if cls_agnostic_bbox_reg:
            reg_delta = paddle.gather(delta, fg_inds)
        else:
            fg_gt_classes = paddle.gather(labels_int64, fg_inds)

            reg_row_inds = paddle.arange(fg_gt_classes.shape[0]).unsqueeze(1)
            reg_row_inds = paddle.tile(reg_row_inds, [1, 4]).reshape([-1, 1])

            #reg_col_inds = 4 * (fg_gt_classes-1).unsqueeze(1) + paddle.arange(4)
            reg_col_inds = 4 * fg_gt_classes.unsqueeze(1) + paddle.arange(4)

            reg_col_inds = reg_col_inds.reshape([-1, 1])
            reg_inds = paddle.concat([reg_row_inds, reg_col_inds], axis=1)

            reg_delta = paddle.gather(delta, fg_inds)
            reg_delta = paddle.gather_nd(reg_delta, reg_inds).reshape([-1, 4])
        rois = paddle.concat(rois) if len(rois) > 1 else rois[0]
        bbox_targets = target['bbox_targets']
        bbox_targets = paddle.concat(bbox_targets) if len(
            bbox_targets) > 1 else bbox_targets[0]

        reg_target = bbox2delta(rois, bbox_targets, self.bbox_weight)
        reg_target = paddle.gather(reg_target, fg_inds)
        reg_target.stop_gradient = True

        loss_box_reg = paddle.abs(reg_delta - reg_target).sum(
        ) / labels_int64.shape[0]
        return loss_bbox_cls, loss_box_reg

    def get_loss(self, bbox_head_out, targets, rois):
        roi, rois_num = rois
        loss_bbox = {}
        cls_name = 'loss_bbox_cls'
        reg_name = 'loss_bbox_reg'
        for lvl, (bboxhead, target) in enumerate(zip(bbox_head_out, targets)):
            score, delta = bboxhead
            if len(targets) > 1:
                cls_name = 'loss_bbox_cls_{}'.format(lvl)
                reg_name = 'loss_bbox_reg_{}'.format(lvl)
            loss_bbox_cls, loss_bbox_reg = self._get_head_loss(score, delta,
                                                               target, roi)
            loss_weight = 1. / 2**lvl
            loss_bbox[cls_name] = loss_bbox_cls * loss_weight
            loss_bbox[reg_name] = loss_bbox_reg * loss_weight
        return loss_bbox

    def get_prediction(self, bbox_head_out, rois):
        proposal, proposal_num = rois
        score, delta = bbox_head_out
        bbox_prob = F.softmax(score)
        delta = paddle.reshape(delta, (-1, self.delta_dim, 4))
        bbox_pred = (delta, bbox_prob)
        return bbox_pred, rois

    def get_cascade_prediction(self, bbox_head_out, rois):
        proposal_list = []
        prob_list = []
        delta_list = []
        for stage in range(len(rois)):
            proposals = rois[stage]
            bboxhead = bbox_head_out[stage]
            score, delta = bboxhead
            proposal, proposal_num = proposals
            if stage in self.score_stage:
                if stage < 2:
                    _, head_out, _ = self(stage=stage, roi_stage=-1)
                    score = head_out[0]

                bbox_prob = F.softmax(score)
                prob_list.append(bbox_prob)
            if stage in self.delta_stage:
                proposal_list.append(proposal)
                delta_list.append(delta)
        bbox_prob = paddle.mean(paddle.stack(prob_list), axis=0)
        delta = paddle.mean(paddle.stack(delta_list), axis=0)
        proposal = paddle.mean(paddle.stack(proposal_list), axis=0)
        delta = paddle.reshape(delta, (-1, self.delta_dim, 4))
        if self.cls_agnostic:
            N, C, M = delta.shape
            delta = delta[:, 1:2, :]
            delta = paddle.expand(delta, [N, self.num_classes, M])
        bboxes = (proposal, proposal_num)
        bbox_pred = (delta, bbox_prob)
        return bbox_pred, bboxes
