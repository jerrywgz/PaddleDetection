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

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant

from backbone.hourglass import _conv_norm
import cornerpool_lib

__all__ = ['CornerHead']


def corner_pool(x, dim, pool1, pool2, name=None):
    p1_conv1 = _conv_norm(x, 3, 128, pad=1, act='relu', name=name + '_p1_conv1')
    pool1 = pool1(p1_conv1)

    p2_conv1 = _conv_norm(x, 3, 128, pad=1, act='relu', name=name + '_p1_conv2')
    pool2 = pool2(p2_conv1)

    p1 = _conv_norm(pool1 + pool2, 3, dim, pad=1, name=name + '_p1')
    p2 = _conv_norm(x, 1, dim, pad=1, act='relu', name=name + '_p2')

    conv2 = _conv_norm(p2, 3, dim, name=name + '_conv2')


@register
class CornerHead(object):
    """
    """
    __shared__ = ['num_classes', 'stack']

    def __init__(self, batch_size, num_classes=80, stack=2):
        self.batch_size = batch_size
        self.stack = stack
        self.tl_heats = []
        self.br_heats = []
        self.tl_tags = []
        self.br_tags = []
        self.tl_offs = []
        self.br_offs = []

    def pred_mod(self, x, dim, name=None):
        conv0 = _conv_norm(x, 1, 256, with_bn=False, name=name + '_0')
        conv1 = fluid.layers.conv2d(
            input=conv0,
            filter_size=1,
            num_filters=dim,
            param_attr=ParamAttr(name=name + "_1_weights"),
            bias_attr=ParamAttr(
                name=name + "_1_bias", initializer=Constant(-2.19)),
            name=_name + '_1')
        return conv1

    def get_output(self, input, name=None):
        tl_heats = []
        br_heats = []
        tl_tags = []
        br_tags = []
        tl_offs = []
        br_offs = []
        for ind in range(self.stack):
            cnv = input[ind]
            tl_modules = corner_pool(
                cnv,
                256,
                cornerpool_lib.top_pool,
                cornerpool_lib.left_pool,
                name='tl_modules_' + str(ind))
            br_modules = corner_pool(
                cnv,
                256,
                cornerpool_lib.bottom_pool,
                cornerpool_lib.right_pool,
                name='br_modules_' + str(ind))

            tl_heat = self.pred_mod(
                tl_modules, num_classes, name=name + '_tl_heat_' + str(ind))
            br_heat = self.pred_mod(
                br_modules, num_classes, name=name + '_br_heat_' + str(ind))

            tl_tag = self.pred_mod(
                tl_modules, 1, name=name + '_tl_tag_' + str(ind))
            br_tag = self.pred_mod(
                br_modules, 1, name=name + '_tl_heat_' + str(ind))

            tl_off = self.pred_mod(
                tl_modules, 2, name=name + '_tl_off_' + str(ind))
            br_off = self.pred_mod(
                br_modules, 2, name=name + '_br_off_' + str(ind))

            self.tl_heats.append(tl_heat)
            self.br_heats.append(br_heat)
            self.tl_tags.append(tl_tag)
            self.br_tags.append(br_tag)
            self.tl_offs.append(tl_off)
            self.br_offs.append(br_off)

    def focal_loss(self, preds, gt):
        preds = [
            fluid.layers.clip(fluid.layers.sigmoid(pred), 1e-4, 1 - 1e-4)
            for pred in preds
        ]
        ones = fluid.layers.ones_like(gt)

        fg_map = fluid.layers.cast(gt == ones, 'float32')
        bg_map = fluid.layers.cast(gt < ones, 'float32')
        loss = 0
        for pred in preds:
            pos_loss = fluid.layers.log(pred) * fluid.layers.pow(1 - pred,
                                                                 2) * fg_map

            neg_weights = fluid.layers.pow(1 - gt, 4) * bg_map
            neg_loss = fluid.layers.log(1 - pred) * fluid.layers.pow(
                pred, 2) * neg_weights

            pos_loss = fluid.layers.reduce_sum(pos_loss)
            neg_loss = fluid.layers.reduce_sum(neg_loss)
            num_pos = fluid.layers.reduce_sum(fg_map)
            ones = fluid.layers.assign(np.array([1], dtype='float32'))
            num_pos = fluid.layers.elementwise_max(num_pos, ones)
            focal_loss_ = (neg_loss + pos_loss) / num_pos
            loss -= focal_loss_
        return loss

    def mask_feat(self, feat, ind):
        feat_t = fluid.layers.tranpose(feat, [0, 2, 3, 1])
        H, W, C = feat_t.shape[1], feat_t.shape[2], feat_t.shape[3]
        feat_r = fluid.layers.reshape(feat_t, [-1, C])
        feat_g = fluid.layers.gather(feat_r, inds)
        return fluid.layers.lod_reset(feat_g, ind)

    def as_loss(self, tl_tag, br_tag, gt_num):
        tag_mean = (tl_tag + br_tag) / 2
        num = fluid.layers.sequence_expand(gt_num, gt_tl_ind)
        tag0 = fluid.layers.pow(tag0 - tag_mean, 2) / (num + 1e-4)
        tag1 = fluid.layers.pow(tag1 - tag_mean, 2) / (num + 1e-4)

        tag0 = fluid.layers.reduce_sum(tag0)
        tag1 = fluid.layers.reduce_sum(tag1)
        pull = tag0 + tag1

        push = 0
        for ind in range(self.batch_size):
            num_ind = fluid.layers.slice(
                gt_num, axes=[0], starts=[ind], ends=[ind + 1])
            num_ind2 = (num_ind - 1) * num_ind

            offset = fluid.layers.assign(input=np.array(
                [[ind]]).astype('int32'))
            tag_mean_ind = fluid.layers.sequence_slice(tag_mean, offset,
                                                       num_ind)
            tag_mean_T = fluid.layers.tranpose(tag_mean_ind, [1, 0])
            dist = 1 - fluid.layers.abs(tag_mean_T - tag_mean_ind)
            dist = fluid.layers.relu(dist) - 1 / (num_ind + 1e-4)
            dist = dist / (num_ind2 + 1e-4)
            push += fluid.layers.reduce_sum(dist)
        return pull, push

    def off_loss(self, off, gt_off, gt_num):
        num = fluid.layers.reduce_sum(gt_num)
        off_loss = fluid.layers.smooth_l1(off, gt_off)
        off_loss = fluid.layers.reduce_sum(off_loss) / (num + 1e-4)
        return off_loss

    def get_loss(self, targets):
        gt_tl_heat = targets['tl_heatmaps']
        gt_br_heat = targets['br_heatmaps']
        gt_num = targets['tag_num']
        gt_tl_off = targets['tl_regrs']
        gt_br_off = targets['br_regrs']
        gt_tl_ind = targets['tl_tags']
        gt_br_ind = targets['br_tags']

        #gt_num = fluid.layers.reduce_sum(gt_num, dim=1, keep_dim=True)
        #gt_num = fluid.layers.sequence_expand(gt_num, gt_tl_ind)

        focal_loss = 0
        focal_loss += self.focal_loss(self.tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(self.br_heats, gt_br_heat)

        pull_loss = 0
        push_loss = 0

        ones = fluid.layers.assign(np.array([1], dtype='float32'))
        tl_tags = [self.mask_feat(tl_tag, gt_tl_ind) for tl_tag in self.tl_tags]
        br_tags = [self.mask_feat(br_tag, gt_br_ind) for br_tag in self.br_tags]

        pull_loss, push_loss = 0, 0
        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_num)
            pull_loss += pull
            push_loss += push

        tl_offs = [self.mask_feat(tl_off, gt_tl_ind) for tl_off in self.tl_offs]
        br_offs = [self.mask_feat(br_off, gt_br_ind) for br_off in self.br_offs]

        off_loss = 0
        for tl_off, br_off in zip(tl_offs, br_offs):
            off_loss += self.off_loss(tl_off, gt_tl_off)
            off_loss += self.off_loss(br_off, gt_br_off)

        loss = (focal_loss + pull_loss + push_loss + off_loss) / len(tl_heats)
        return loss
