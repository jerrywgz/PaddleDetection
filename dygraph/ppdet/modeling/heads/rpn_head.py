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
from paddle import ParamAttr
from paddle.nn.initializer import Normal
from paddle.regularizer import L2Decay
from paddle.nn import Conv2D

from ppdet.core.workspace import register
from ppdet.modeling import ops


@register
class RPNHead(nn.Layer):
    def __init__(self, anchor_per_position=15, rpn_channel=1024):
        super(RPNHead, self).__init__()
        self.rpn_conv = Conv2D(
            in_channels=rpn_channel,
            out_channels=rpn_channel,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0., std=0.01)))
        # rpn head is shared with each level
        # rpn roi classification scores
        self.rpn_rois_score = Conv2D(
            in_channels=rpn_channel,
            out_channels=anchor_per_position,
            kernel_size=1,
            padding=0,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0., std=0.01)))

        # rpn roi bbox regression deltas
        self.rpn_rois_delta = Conv2D(
            in_channels=rpn_channel,
            out_channels=4 * anchor_per_position,
            kernel_size=1,
            padding=0,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0., std=0.01)))

    def forward(self, feats):
        rpn_feats = []
        rpn_rois_score = []
        rpn_rois_delta = []
        for feat in feats:
            rpn_feat = F.relu(self.rpn_conv(feat))
            rrs = self.rpn_rois_score(rpn_feat)
            rrd = self.rpn_rois_delta(rpn_feat)
            rpn_rois_score.append(rrs)
            rpn_rois_delta.append(rrd)
        return rpn_rois_score, rpn_rois_delta

    def get_loss(self, loss_inputs):
        # cls loss
        score_tgt = paddle.concat(x=loss_inputs['rpn_score_target'])
        score_tgt.stop_gradient = True

        pos_mask = score_tgt == 1
        pos_ind = paddle.nonzero(pos_mask)

        valid_mask = score_tgt >= 0
        valid_ind = paddle.nonzero(valid_mask)

        # cls loss
        score_pred = paddle.gather(loss_inputs['rpn_score_pred'], valid_ind)
        score_label = paddle.gather(score_tgt, valid_ind).cast('float32')
        loss_rpn_cls = F.binary_cross_entropy_with_logits(
            logit=score_pred, label=score_label, reduction="sum")

        # reg loss
        loc_pred = paddle.gather(loss_inputs['rpn_rois_pred'], pos_ind)
        loc_tgt = paddle.concat(x=loss_inputs['rpn_rois_target'])
        loc_tgt = paddle.gather(loc_tgt, pos_ind)
        loss_rpn_reg = paddle.abs(loc_pred - loc_tgt).sum()
        norm = loss_inputs['norm']
        return {
            'loss_rpn_cls': loss_rpn_cls / norm,
            'loss_rpn_reg': loss_rpn_reg / norm
        }
