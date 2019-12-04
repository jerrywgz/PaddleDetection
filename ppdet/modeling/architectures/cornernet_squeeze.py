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

__all__ = ['CornerNetSqueeze']


@register
class CornerNetSqueeze(object):
    """
    """
    __category__ = 'architecture'
    __inject__ = ['backbone', 'corner_head']

    def __init__(self, backbone, corner_head='CornerHead'):
        super(CornerNetSqueeze, self).__init__()
        self.backbone = backbone
        self.corner_head = corner_head

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        im.persistable = True
        print('image: ', im)
        body_feats = self.backbone(im)
        body_feats[0].persistable = True
        print('cnv: ', body_feats[0])

        if mode == 'train':
            target_vars = [
                'tl_heatmaps', 'br_heatmaps', 'tag_nums', 'tl_regrs', 'br_regrs',
                'tl_tags', 'br_tags'
            ]
            target = {key: feed_vars[key] for key in target_vars}
            self.corner_head.get_output(body_feats)
            loss = self.corner_head.get_loss(target)
            return {'loss': loss}

    def train(self, feed_vars):
        return self.build(feed_vars, mode='train')
