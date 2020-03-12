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
from ppdet.modeling.architectures.cornernet_squeeze import * 
import numpy as np

__all__ = ['CenterNetSqueeze']


def rescale_bboxes(bboxes, ratios, borders, im):
    shape = fluid.layers.cast(fluid.layers.shape(im), 'float32')
    x1 = bboxes[:, :, 0:1]
    y1 = bboxes[:, :, 1:2]
    x2 = bboxes[:, :, 2:3]
    y2 = bboxes[:, :, 3:4]
    x1 = x1 / ratios[:, 1] - borders[:, 2]
    zero = fluid.layers.assign(np.array([0], dtype='float32'))
    x1 = fluid.layers.elementwise_max(x1, zero)
    x1 = fluid.layers.elementwise_min(x1, shape[3])
    x2 = x2 / ratios[:, 1] - borders[:, 2]
    x2 = fluid.layers.elementwise_max(x2, zero)
    x2 = fluid.layers.elementwise_min(x2, shape[3])

    y1 = y1 / ratios[:, 0] - borders[:, 0]
    y1 = fluid.layers.elementwise_max(y1, zero)
    y1 = fluid.layers.elementwise_min(y1, shape[2])
    y2 = y2 / ratios[:, 0] - borders[:, 0]
    y2 = fluid.layers.elementwise_max(y2, zero)
    y2 = fluid.layers.elementwise_min(y2, shape[2])

    return fluid.layers.concat([x1, y1, x2, y2], axis=2)


@register
class CenterNetSqueeze(CornerNetSqueeze):
    """
    """
    __category__ = 'architecture'
    __inject__ = ['backbone', 'center_head', 'nms', 'fpn']
    __shared__ = ['num_classes']

    def __init__(self,
                 backbone,
                 nms=MultiClassSoftNMS().__dict__,
                 center_head='CornerHead',
                 num_classes=80,
                 fpn=None):
        super(CenterNetSqueeze, self).__init__(backbone,
            nms=nms, corner_head=center_head, num_classes=num_classes, fpn=fpn)


    def _inputs_def(self, image_shape, output_size, max_tag_len):
        im_shape = [None] + image_shape
        C = self.num_classes
        # yapf: disable
        inputs_def = {
            'image':        {'shape': im_shape,  'dtype': 'float32', 'lod_level': 0},
            'im_id':        {'shape': [None, 1], 'dtype': 'int64',   'lod_level': 0},
            'gt_bbox':       {'shape': [None, 4], 'dtype': 'float32', 'lod_level': 1},
            'gt_class':     {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'ratios':       {'shape': [None, 2],  'dtype': 'float32', 'lod_level': 0},
            'borders':      {'shape': [None, 4],  'dtype': 'float32', 'lod_level': 0},
            'tl_heatmaps':  {'shape': [None, C, output_size, output_size],  'dtype': 'float32', 'lod_level': 0},
            'br_heatmaps':  {'shape': [None, C, output_size, output_size],  'dtype': 'float32', 'lod_level': 0},
            'ct_heatmaps':  {'shape': [None, C, output_size, output_size],  'dtype': 'float32', 'lod_level': 0},
            'tl_regrs':     {'shape': [None, max_tag_len, 2], 'dtype': 'float32', 'lod_level': 0},
            'br_regrs':     {'shape': [None, max_tag_len, 2], 'dtype': 'float32', 'lod_level': 0},
            'ct_regrs':     {'shape': [None, max_tag_len, 2], 'dtype': 'float32', 'lod_level': 0},
            'tl_tags':      {'shape': [None, max_tag_len], 'dtype': 'int64', 'lod_level': 0},
            'br_tags':      {'shape': [None, max_tag_len], 'dtype': 'int64', 'lod_level': 0},
            'ct_tags':      {'shape': [None, max_tag_len], 'dtype': 'int64', 'lod_level': 0},
            'tag_masks':     {'shape': [None, max_tag_len], 'dtype': 'int32', 'lod_level': 0},
        }
        # yapf: enable
        return inputs_def

    def train(self, feed_vars):
        self.target_vars = [
            'tl_heatmaps', 'br_heatmaps', 'ct_heatmaps', 'tag_masks', 'tl_regrs',
            'br_regrs', 'ct_regrs', 'tl_tags', 'br_tags', 'ct_tags'
        ]
        return self.build(feed_vars, mode='train')

