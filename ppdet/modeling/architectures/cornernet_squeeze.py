# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

from paddle import fluid

from ppdet.core.workspace import register
from ppdet.modeling.ops import MultiClassSoftNMS
import numpy as np

__all__ = ['CornerNetSqueeze']


def rescale_bboxes(bboxes, ratios, borders, im):
    shape = fluid.layers.cast(fluid.layers.shape(im), 'float32')
    x1 = bboxes[:,:,0:1]
    y1 = bboxes[:,:,1:2]
    x2 = bboxes[:,:,2:3]
    y2 = bboxes[:,:,3:4]
    x1 = x1 / ratios[:,1] - borders[:, 2]
    zero = fluid.layers.assign(np.array([0], dtype='float32'))
    x1 = fluid.layers.elementwise_max(x1, zero)
    x1 = fluid.layers.elementwise_min(x1, shape[3])
    x2 = x2 / ratios[:,1] - borders[:, 2]
    x2 = fluid.layers.elementwise_max(x2, zero)
    x2 = fluid.layers.elementwise_min(x2, shape[3])

    y1 = y1 / ratios[:,0] - borders[:, 0]
    y1 = fluid.layers.elementwise_max(y1, zero)
    y1 = fluid.layers.elementwise_min(y1, shape[2])
    y2 = y2 / ratios[:,0] - borders[:, 0]
    y2 = fluid.layers.elementwise_max(y2, zero)
    y2 = fluid.layers.elementwise_min(y2, shape[2])

    return fluid.layers.concat([x1,y1,x2,y2], axis=2)
     

@register
class CornerNetSqueeze(object):
    """
    """
    __category__ = 'architecture'
    __inject__ = ['backbone', 'corner_head', 'nms']
    __shared__ = ['num_classes']

    def __init__(self, backbone, nms=MultiClassSoftNMS().__dict__, corner_head='CornerHead', num_classes=80):
        super(CornerNetSqueeze, self).__init__()
        self.backbone = backbone
        self.corner_head = corner_head
        self.nms = nms
        self.num_classes = num_classes
        if isinstance(nms, dict):
            self.nms = MultiClassSoftNMS(**nms)

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        body_feats = self.backbone(im)

        if mode == 'train':
            target_vars = [
                'tl_heatmaps', 'br_heatmaps', 'tag_nums', 'tl_regrs', 'br_regrs',
                'tl_tags', 'br_tags'
            ]
            target = {key: feed_vars[key] for key in target_vars}
            self.corner_head.get_output(body_feats)
            loss = self.corner_head.get_loss(target)
            return loss

        elif mode == 'test':
            ratios = feed_vars['ratios']
            borders = feed_vars['borders']
            bboxes, scores, tl_scores, br_scores, clses = self.corner_head.get_prediction(body_feats[-1])
            bboxes = rescale_bboxes(bboxes, ratios, borders, im)                   
            detections = fluid.layers.concat([bboxes, scores, tl_scores, br_scores, clses], axis=2)
            scores = fluid.layers.squeeze(scores, axes=[0,2]) 
            detections = detections[0]

            keep_inds = fluid.layers.squeeze(fluid.layers.where(scores > -1), axes=[-1])
            detections = fluid.layers.gather(detections, keep_inds)
            debug_box = detections[:, :4]

            total_res = self.nms(detections[:,:4], detections[:,4], detections[:,-1])
            
            return {'bbox': total_res}

    def train(self, feed_vars):
        return self.build(feed_vars, mode='train')

    def eval(self, feed_vars):
        return self.build(feed_vars, mode='test')

    def test(self, feed_vars):
        return self.build(feed_vars, mode='test')

