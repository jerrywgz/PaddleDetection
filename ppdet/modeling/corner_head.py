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

from .backbones.hourglass import _conv_norm
from ppdet.core.workspace import register
import cornerpool_lib
import numpy as np

__all__ = ['CornerHead']


def corner_pool(x, dim, pool1, pool2, name=None):
    p1_conv1 = _conv_norm(x, 3, 128, pad=1, act='relu', name=name + '_p1_conv1')
    pool1 = pool1(p1_conv1, name=name+'_pool1')
    p2_conv1 = _conv_norm(x, 3, 128, pad=1, act='relu', name=name + '_p2_conv1')
    pool2 = pool2(p2_conv1, name=name+'_pool2')

    p_conv1 = fluid.layers.conv2d(
        pool1 + pool2, 
        filter_size=3, 
        num_filters=dim, 
        padding=1, 
        param_attr=ParamAttr(name=name + "_p_conv1_weight"), 
        bias_attr=False,
        name=name + '_p_conv1')
    p_bn1 = fluid.layers.batch_norm(
        p_conv1,
        param_attr=ParamAttr(name=name + '_p_bn1_weight'),
        bias_attr=ParamAttr(name=name + '_p_bn1_bias'),
        moving_mean_name=name + '_p_bn1_running_mean',
        moving_variance_name=name + '_p_bn1_running_var',
        name=name + '_p_bn1')
    conv1 = fluid.layers.conv2d(
        x, 
        filter_size=1, 
        num_filters=dim, 
        param_attr=ParamAttr(name=name + "_conv1_weight"), 
        bias_attr=False,
        name=name + '_conv1')
    bn1 = fluid.layers.batch_norm(
        conv1,
        param_attr=ParamAttr(name=name + '_bn1_weight'),
        bias_attr=ParamAttr(name=name + '_bn1_bias'),
        moving_mean_name=name + '_bn1_running_mean',
        moving_variance_name=name + '_bn1_running_var',
        name=name + '_bn1')

    relu1 = fluid.layers.relu(p_bn1 + bn1)
    conv2 = _conv_norm(relu1, 3, dim, pad=1, act='relu', name=name + '_conv2')
    return conv2
"""
def nms(heat):
    hmax = fluid.layers.pool2d(heat, 1)
    keep = fluid.layers.cast(hmax == heat, 'float32')
    return heat * keep

def topk(scores, K):
    C, H, W = scores.shape[1], scores.shape[2], scores.shape[3]
    topk_scores, topk_inds = fluid.layers.topk(fluid.layers.reshape(scores, [-1, C*H*W]), K)
    topk_clses = topk_inds / (H * W) 
    topk_inds = topk_inds % (H * W)
    topk_ys = topk_inds / W
    topk_xs = topk_inds % W
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def decode(tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, K=100, ae_threhold=1, num_dets=1000):
    tl_heat = fluid.layers.sigmoid(tl_heat)
    br_heat = fluid.layers.sigmoid(br_heat)

    tl_heat = nms(tl_heat) 
    br_heat = nms(br_heat)   

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    tl_ys = fluid.layers.expand(fluid.layers.reshape(tl_ys, [-1, K, 1]), [1, 1, K])
    tl_xs = fluid.layers.expand(fluid.layers.reshape(tl_xs, [-1, K, 1]), [1, 1, K])
    br_ys = fluid.layers.expand(fluid.layers.reshape(br_ys, [-1, 1, K]), [1, K, 1])
    br_xs = fluid.layers.expand(fluid.layers.reshape(br_xs, [-1, 1, K]), [1, K, 1])
"""
@register
class CornerHead(object):
    """
    """
    __shared__ = ['num_classes', 'stack']

    def __init__(self, batch_size, num_classes=80, stack=2, pull_weight=0.1, push_weight=0.1):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.stack = stack
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.tl_heats = []
        self.br_heats = []
        self.tl_tags = []
        self.br_tags = []
        self.tl_offs = []
        self.br_offs = []

    def pred_mod(self, x, dim, name=None):
        conv0 = _conv_norm(x, 1, 256, with_bn=False, act='relu', name=name + '_0')
        conv1 = fluid.layers.conv2d(
            input=conv0,
            filter_size=1,
            num_filters=dim,
            param_attr=ParamAttr(name=name + "_1_weight"),
            bias_attr=ParamAttr(
                name=name + "_1_bias", initializer=Constant(-2.19)),
            name=name + '_1')
        return conv1

    def get_output(self, input):
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
                tl_modules, self.num_classes, name='tl_heats_' + str(ind))
            br_heat = self.pred_mod(
                br_modules, self.num_classes, name='br_heats_' + str(ind))

            tl_tag = self.pred_mod(
                tl_modules, 1, name='tl_tags_' + str(ind))
            br_tag = self.pred_mod(
                br_modules, 1, name='br_tags_' + str(ind))

            tl_off = self.pred_mod(
                tl_modules, 2, name='tl_offs_' + str(ind))
            br_off = self.pred_mod(
                br_modules, 2, name='br_offs_' + str(ind))

            self.tl_heats.append(tl_heat)
            self.br_heats.append(br_heat)
            self.tl_tags.append(tl_tag)
            self.br_tags.append(br_tag)
            self.tl_offs.append(tl_off)
            self.br_offs.append(br_off)

    def focal_loss(self, preds, gt):
        preds_clip = []
        min = fluid.layers.assign(np.array([1e-4], dtype='float32'))
        max = fluid.layers.assign(np.array([1-1e-4], dtype='float32'))
        for pred in preds:
            pred_s = fluid.layers.sigmoid(pred)
            pred_min = fluid.layers.elementwise_max(pred_s, min)
            pred_max = fluid.layers.elementwise_min(pred_min, max)
            preds_clip.append(pred_max)

        ones = fluid.layers.ones_like(gt)

        fg_map = fluid.layers.cast(gt == ones, 'float32')
        fg_map.stop_gradient = True
        num_pos = fluid.layers.reduce_sum(fg_map)
        min_num = fluid.layers.ones_like(num_pos)
        num_pos = fluid.layers.elementwise_max(num_pos, min_num)
        num_pos.stop_gradient = True
        bg_map = fluid.layers.cast(gt < ones, 'float32')
        bg_map.stop_gradient = True
        neg_weights = fluid.layers.pow(1 - gt, 4) * bg_map
        neg_weights.stop_gradient = True
        loss = fluid.layers.assign(np.array([0], dtype='float32'))
        for ind, pred in enumerate(preds_clip):
            pos_loss = fluid.layers.log(pred) * fluid.layers.pow(1 - pred,
                                                                 2) * fg_map

            neg_loss = fluid.layers.log(1 - pred) * fluid.layers.pow(
                pred, 2) * neg_weights

            pos_loss = fluid.layers.reduce_sum(pos_loss)
            neg_loss = fluid.layers.reduce_sum(neg_loss)
            focal_loss_ = (neg_loss + pos_loss) / num_pos
            loss -= focal_loss_
        return loss

    def mask_feat(self, feat, ind):
        feat_t = fluid.layers.transpose(feat, [0, 2, 3, 1])
        H, W, C = feat_t.shape[1], feat_t.shape[2], feat_t.shape[3]
        feat_r = fluid.layers.reshape(feat_t, [-1, C])
        feat_g = fluid.layers.gather(feat_r, ind)
        return fluid.layers.lod_reset(feat_g, ind)

    def ae_loss(self, tl_tag, br_tag, gt_num, expand_num):
        tag_mean = (tl_tag + br_tag) / 2
        tag0 = fluid.layers.pow(tl_tag - tag_mean, 2) / (expand_num + 1e-4)
        tag1 = fluid.layers.pow(br_tag - tag_mean, 2) / (expand_num + 1e-4)

        tag0 = fluid.layers.reduce_sum(tag0)
        tag1 = fluid.layers.reduce_sum(tag1)
        pull = tag0 + tag1

        push = fluid.layers.assign(np.array([0], dtype='float32'))
        total_num = fluid.layers.assign(np.array([0], dtype='int32'))
        for ind in range(self.batch_size):
            num_ind = fluid.layers.slice(
                gt_num, axes=[0], starts=[ind], ends=[ind + 1])
            num_ind = fluid.layers.reduce_sum(num_ind)
            num_ind2 = (num_ind - 1) * num_ind
            num_ind2 = fluid.layers.cast(num_ind2, 'float32')
            num_ind2.stop_gradient = True

            tag_mean_ind = fluid.layers.slice(tag_mean, axes=[0], starts=[total_num], ends=[total_num + num_ind])
            total_num = total_num + num_ind
            num_ind = fluid.layers.cast(num_ind, 'float32')
            num_ind.stop_gradient = True
            tag_mean_T = fluid.layers.transpose(tag_mean_ind, [1, 0])
            shape = fluid.layers.shape(tag_mean_ind)
            shape.stop_gradient = True
            tag_mean_T = fluid.layers.expand(tag_mean_T, shape)
            dist = 1 - fluid.layers.abs(tag_mean_T - tag_mean_ind)
            dist = fluid.layers.relu(dist) - 1 / (num_ind + 1e-4)
            dist = dist / (num_ind2 + 1e-4)
            push += fluid.layers.reduce_sum(dist)
        return pull, push

    def off_loss(self, off, gt_off, gt_num):
        num = fluid.layers.reduce_sum(gt_num)
        off_loss = fluid.layers.smooth_l1(off, gt_off)
        num = fluid.layers.cast(num, 'float32')
        num.stop_gradient = True
        off_loss = fluid.layers.reduce_sum(off_loss) / (num + 1e-4)
        return off_loss

    def get_loss(self, targets):
        gt_tl_heat = targets['tl_heatmaps']
        gt_br_heat = targets['br_heatmaps']
        gt_num = targets['tag_nums']
        gt_tl_off = targets['tl_regrs']
        gt_br_off = targets['br_regrs']
        gt_tl_ind = targets['tl_tags']
        gt_br_ind = targets['br_tags']


        focal_loss = 0
        focal_loss_ = self.focal_loss(self.tl_heats, gt_tl_heat)
        focal_loss += focal_loss_
        focal_loss_ = self.focal_loss(self.br_heats, gt_br_heat)
        focal_loss += focal_loss_

        pull_loss = 0
        push_loss = 0

        ones = fluid.layers.assign(np.array([1], dtype='float32'))
        tl_tags = [self.mask_feat(tl_tag, gt_tl_ind) for tl_tag in self.tl_tags]
        br_tags = [self.mask_feat(br_tag, gt_br_ind) for br_tag in self.br_tags]

        pull_loss, push_loss = 0, 0
        expand_num = fluid.layers.sequence_expand(gt_num, gt_tl_ind)
        expand_num = fluid.layers.cast(expand_num, 'float32')
        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_num, expand_num)
            pull_loss += pull
            push_loss += push

        tl_offs = [self.mask_feat(tl_off, gt_tl_ind) for tl_off in self.tl_offs]
        br_offs = [self.mask_feat(br_off, gt_br_ind) for br_off in self.br_offs]

        off_loss = 0
        for tl_off, br_off in zip(tl_offs, br_offs):
            off_loss += self.off_loss(tl_off, gt_tl_off, gt_num)
            off_loss += self.off_loss(br_off, gt_br_off, gt_num)

        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        loss = (focal_loss + pull_loss + push_loss + off_loss) / len(self.tl_heats)
        return {'loss':loss}
    """
    def get_prediction(self, input):
        ind = self.stack - 1
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
                tl_modules, self.num_classes, name='tl_heats_' + str(ind))
        br_heat = self.pred_mod(
                br_modules, self.num_classes, name='br_heats_' + str(ind))

        tl_tag = self.pred_mod(
                tl_modules, 1, name='tl_tags_' + str(ind))
        br_tag = self.pred_mod(
                br_modules, 1, name='br_tags_' + str(ind))

        tl_off = self.pred_mod(
                tl_modules, 2, name='tl_offs_' + str(ind))
        br_off = self.pred_mod(
                br_modules, 2, name='br_offs_' + str(ind)) 
    """
