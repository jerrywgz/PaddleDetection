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
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, XavierUniform
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, create
from ppdet.modeling import ops

from .roi_extractor import RoIAlign
from ..shape_spec import ShapeSpec
from ..bbox_utils import bbox2delta


@register
class TwoFCHead(nn.Layer):
    def __init__(self, in_dim=256, mlp_dim=1024, resolution=7):
        super(TwoFCHead, self).__init__()
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim
        fan = in_dim * resolution * resolution
        self.fc6 = nn.Linear(
            in_dim * resolution * resolution,
            mlp_dim,
            weight_attr=paddle.ParamAttr(
                initializer=XavierUniform(fan_out=fan)))

        self.fc7 = nn.Linear(
            mlp_dim,
            mlp_dim,
            weight_attr=paddle.ParamAttr(initializer=XavierUniform()))

    @classmethod
    def from_config(cls, cfg, input_shape):
        s = input_shape
        s = s[0] if isinstance(s, (list, tuple)) else s
        return {'in_dim': s.channels}

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.mlp_dim, )]

    def forward(self, rois_feat):
        rois_feat = paddle.flatten(rois_feat, start_axis=1, stop_axis=-1)
        fc6 = self.fc6(rois_feat)
        fc6 = F.relu(fc6)
        fc7 = self.fc7(fc6)
        fc7 = F.relu(fc7)
        return fc7


@register
class BBoxHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['bbox_assigner']
    """
    head (nn.Layer): Extract feature in bbox head
    in_channel (int): Input channel after RoI extractor
    """

    def __init__(self,
                 head,
                 in_channel,
                 roi_extractor=RoIAlign().__dict__,
                 bbox_assigner='BboxAssigner',
                 with_pool=False,
                 num_classes=80,
                 bbox_weight=[10., 10., 5., 5.]):
        super(BBoxHead, self).__init__()
        self.head = head
        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)
        self.bbox_assigner = bbox_assigner

        self.with_pool = with_pool
        self.num_classes = num_classes
        self.bbox_weight = bbox_weight

        self.bbox_score = nn.Linear(
            in_channel,
            self.num_classes + 1,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0.0, std=0.01)))

        self.bbox_delta = nn.Linear(
            in_channel,
            4 * self.num_classes,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0.0, std=0.001)))
        self.assigned_label = None
        self.assigned_rois = None

    @classmethod
    def from_config(cls, cfg, input_shape):
        roi_pooler = cfg['roi_extractor']
        assert isinstance(roi_pooler, dict)
        kwargs = RoIAlign.from_config(cfg, input_shape)
        roi_pooler.update(kwargs)
        kwargs = {'input_shape': input_shape}
        head = create(cfg['head'], **kwargs)
        return {
            'roi_extractor': roi_pooler,
            'head': head,
            'in_channel': head.out_shape[0].channels
        }

    def forward(self, body_feats=None, rois=None, rois_num=None, inputs=None):
        """
        body_feats (list[Tensor]):
        rois (Tensor):
        rois_num (Tensor):
        inputs (dict{Tensor}):
        """
        if self.training:
            rois, rois_num, _, targets = self.bbox_assigner(rois, rois_num,
                                                            inputs)
            self.assigned_rois = (rois, rois_num)
            self.assigned_targets = targets

        rois_feat = self.roi_extractor(body_feats, rois, rois_num)
        bbox_feat = self.head(rois_feat)
        if len(bbox_feat.shape) > 2 and bbox_feat.shape[-1] > 1:
            feat = F.adaptive_avg_pool2d(bbox_feat, output_size=1)
            feat = paddle.squeeze(feat, axis=[2, 3])
        else:
            feat = bbox_feat
        scores = self.bbox_score(feat)
        deltas = self.bbox_delta(feat)

        if self.training:
            loss = self.get_loss(scores, deltas, targets, rois)
            return loss, bbox_feat
        else:
            pred = self.get_prediction(scores, deltas)
            return pred, self.head

    def get_loss(self, scores, deltas, targets, rois):
        """
        scores (Tensor): scores from bbox head outputs
        deltas (Tensor): deltas from bbox head outputs
        targets (list[List[Tensor]]): bbox targets containing tgt_labels, tgt_bboxes and tgt_gt_inds
        rois (List[Tensor]): RoIs generated in each batch
        """
        # TODO: better pass args
        tgt_labels, tgt_bboxes, tgt_gt_inds = targets
        tgt_labels = paddle.concat(tgt_labels) if len(
            tgt_labels) > 1 else tgt_labels[0]
        tgt_labels = tgt_labels.cast('int64')
        tgt_labels.stop_gradient = True
        loss_bbox_cls = F.cross_entropy(
            input=scores, label=tgt_labels, reduction='mean')
        # bbox reg

        cls_agnostic_bbox_reg = deltas.shape[1] == 4

        fg_inds = paddle.nonzero(
            paddle.logical_and(tgt_labels >= 0, tgt_labels <
                               self.num_classes)).flatten()

        if cls_agnostic_bbox_reg:
            reg_delta = paddle.gather(deltas, fg_inds)
        else:
            fg_gt_classes = paddle.gather(tgt_labels, fg_inds)

            reg_row_inds = paddle.arange(fg_gt_classes.shape[0]).unsqueeze(1)
            reg_row_inds = paddle.tile(reg_row_inds, [1, 4]).reshape([-1, 1])

            reg_col_inds = 4 * fg_gt_classes.unsqueeze(1) + paddle.arange(4)

            reg_col_inds = reg_col_inds.reshape([-1, 1])
            reg_inds = paddle.concat([reg_row_inds, reg_col_inds], axis=1)

            reg_delta = paddle.gather(deltas, fg_inds)
            reg_delta = paddle.gather_nd(reg_delta, reg_inds).reshape([-1, 4])
        rois = paddle.concat(rois) if len(rois) > 1 else rois[0]
        tgt_bboxes = paddle.concat(tgt_bboxes) if len(
            tgt_bboxes) > 1 else tgt_bboxes[0]

        reg_target = bbox2delta(rois, tgt_bboxes, self.bbox_weight)
        reg_target = paddle.gather(reg_target, fg_inds)
        reg_target.stop_gradient = True

        loss_bbox_reg = paddle.abs(reg_delta - reg_target).sum(
        ) / tgt_labels.shape[0]

        cls_name = 'loss_bbox_cls'
        reg_name = 'loss_bbox_reg'
        loss_bbox = {}
        loss_bbox[cls_name] = loss_bbox_cls
        loss_bbox[reg_name] = loss_bbox_reg

        return loss_bbox

    def get_prediction(self, score, delta):
        bbox_prob = F.softmax(score)
        return delta, bbox_prob

    def get_head(self, ):
        return self.head

    def get_assigned_targets(self, ):
        return self.assigned_targets

    def get_assigned_rois(self, ):
        return self.assigned_rois
