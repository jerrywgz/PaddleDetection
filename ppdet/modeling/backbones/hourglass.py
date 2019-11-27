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

import functools
from ppdet.core.workspace import register
from .resnet import ResNet

__all__ = ['Hourglass']


def _conv_norm(x,
               k,
               out_dim,
               stride=1,
               pad=0,
               with_bn=True,
               act=None,
               name=None):
    conv = fluid.layers.conv2d(
        input=input,
        filter_size=k,
        num_filters=out_dim,
        stride=stride,
        padding=pad,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=ParamAttr(name=name + "_bias") if not with_bn else False,
        name=_name + '_output')
    pattr = paramattr(name=name + '_bn_scale')
    battr = paramattr(name=name + '_bn_offset')
    out = fluid.layers.batch_norm(
        input=conv,
        act=act,
        name=name + '_bn_output',
        param_attr=pattr,
        bias_attr=battr,
        moving_mean_name=name + '_bn_mean',
        moving_variance_name=name + '_bn_variance') if with_bn else conv
    if not with_bn:
        out = fluid.layers.relu(out)
    return out


def residual_block(x, out_dim, k=3, stride=1, name=None):
    p = (k - 1) // 2
    conv1 = _conv_norm(x, k, out_dim, pad=p, act='relu', name=name + '_1_conv')
    conv2 = _conv_norm(conv1, k, out_dim, p=p, name=name + '_2_conv')

    skip = _conv_norm(
        x, 1, out_dim,
        name=name + '_skip_conv') if stride != 1 or x.shape[1] != out_dim else x
    return fluid.layers.elementwise_add(
        x=short, y=residual, act='relu', name=name + "_add")


def make_res_layer(x, in_dim, out_dim, modules, name=None):
    layers = residual_block(x, out_dim, name=name + '_res_block_0')
    for i in range(1, modules):
        layers = residual_block(
            layers, out_dim, name=name + '_res_block_' + str(i))
    return layers


def make_res_layer_revr(x, in_dim, out_dim, modules, name=None):
    for i in range(modules - 1):
        x = residual_block(x, in_dim, name=name + '_res_block_revr_' + str(i))
    layers = residual_block(
        x, out_dim, name=name + '_res_block_revr' + str(modules - 1))
    return layers


def fire_block(x, out_dim, sr=2, stride=1, name=None):
    conv1 = _conv_norm(x, 1, out_dim // sr, name=name + '_conv1')
    conv_1x1 = fluid.layers.conv2d(
        conv1,
        filter_size=1,
        num_filters=out_dim // 2,
        stride=stride,
        param_attr=ParamAttr(name=name + "_conv1x1_weights"),
        bias_attr=False,
        name=name + '_conv1x1')
    conv_3x3 = fluid.layers.conv2d(
        conv1,
        filter_size=3,
        num_filters=out_dim // 2,
        stride=stride,
        padding=padding,
        groups=out_dim // sr,
        param_attr=ParamAttr(name=name + "_conv3x3_weights"),
        bias_attr=False,
        name=name + '_conv3x3')
    conv2 = fluid.layers.concat(
        conv_1x1, conv_3x3, axis=1, name=name + '_conv2')
    pattr = paramattr(name=name + '_bn2_scale')
    battr = paramattr(name=name + '_bn2_offset')

    bn2 = fluid.layers.batch_norm(
        input=conv2,
        name=name + '_bn2',
        param_attr=pattr,
        bias_attr=battr,
        moving_mean_name=name + '_bn2_mean',
        moving_variance_name=name + '_bn2_variance')

    if strde == 1 and x.shape[1] == out_dim:
        return fluid.layers.elementwise_add(
            x=bn2, y=x, act='relu', name=name + "_add_relu")
    else:
        return fluid.layers.relu(bn2, name="_relu")


def make_fire_layer(x, in_dim, out_dim, modules, name):
    layers = fire_block(x, out_dim, name=name + '_0')
    for i in range(1, modules):
        layers = fire_block(layers, out_dim, name=name + '_' + str(i))
    return layers


def make_fire_layer_revr(x, in_dim, out_dim, modules, name=None):
    for i in range(modules - 1):
        x = fire_block(x, in_dim, name=name + '_' + str(i))
    layers = fire_block(x, out_dim, name=name + '_' + str(i))
    return layers


def make_unpool_layer(x, dim, name=None):
    pattr = paramattr(name=name + '_weight')
    battr = paramattr(name=name + '_bias')
    layer = fluid.layers.conv2d_transpose(
        input=x,
        num_fileters=dim,
        filter_size=4,
        stride=2,
        padding=1,
        param_attr=pattr,
        bias_attr=battr)
    return layer


@register
class Hourglass(object):
    """
    """

    def __init__(self, stack=2):
        super(Hourglass, self).__init__()
        self.stack = stack

    def __call__(self, input, name='hg'):
        inter = self.pre(x, name + '_pre')

        cnvs = []
        for ind in range(self.stack):
            hg = self.hg_module(inter, name='_hgs_' + str(ind))
            cnv = _conv_norm(
                hg, 3, 256, act='relu', pad=1, name=name + '_cnv_' + str(ind))
            cnvs.append(cnv)

            if ind < self.stack - 1:
                inter = residual_block(
                    inter, 256, name=name + '_inters_' + str(ind)) + _conv_norm(
                        cnv, 1, 256, name=name + '_cnvs_' + str(ind))
                inter = fluid.layers.relu(inter)
                inter = _conv_norm(
                    cnv, 1, 256, name=name + '_inter_' + str(ind))
        return cnvs

    def hg_module(self,
                  x,
                  n=4,
                  dims=[256, 256, 384, 384, 512],
                  modules=[2, 2, 2, 2, 4],
                  make_up_layer=make_fire_layer,
                  make_hg_layer=make_hg_layer,
                  make_low_layer=make_fire_layer,
                  make_hg_layer_revr=make_fire_layer_revr,
                  make_unpool_layer=make_unpool_layer,
                  name=None):
        curr_mod = modules[0]
        next_mod = modules[1]
        curr_dim = dims[0]
        next_dim = dims[1]

        up1 = make_up_layer(x, curr_dim, curr_dim, curr_mod, name=name + '_up1')
        max1 = x
        low1 = make_hg_layer(
            max1, curr_dim, next_dim, curr_mod, name=name + '_low1')
        low2 = self.hg_module(
            low1,
            n - 1,
            dims[1:],
            modules[1:],
            n - 1,
            dims[1:],
            modules[1:],
            make_up_layer=make_up_layer,
            make_hg_layer=make_fire_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_unpool_layer=make_unpool_layer,
            name=name + '_low2') if n > 1 else make_low_layer(
                low1, next_dim, next_dim, next_mod, name=name + '_low2')
        low3 = make_hg_layer_revr(
            low2, next_dim, curr_dim, curr_mod, name=name + '_low3')
        up2 = make_unpool_layer(low3, name=name + '_up2')
        merg = fluid.layers.elementwise_add(x=up1, y=up2, name=name + '_merg')

    def pre(self, x, name=None):
        conv = _conv_norm(x, 7, 128, stride=2, name=name + '_conv')
        res1 = residual_block(conv, 256, stride=2, name=name + '_res1')
        res2 = residual_block(res1, 256, stride=2, name=name + '_res2')
        return res2
