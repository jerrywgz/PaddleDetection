import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from ppdet.core.workspace import register, serializable
from numbers import Integral


class ConvBNLayer(Layer):
    def __init__(self,
                 name_scope,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 act='relu',
                 lr=1.0):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=1,
            act=act,
            param_attr=ParamAttr(
                name=name_scope + "_weights", learning_rate=lr),
            bias_attr=ParamAttr(name=name_scope + "_bias"))
        if name_scope == "conv1":
            bn_name = "bn_" + name_scope
        else:
            bn_name = "bn" + name_scope[3:]
        self.bn = BatchNorm(
            num_channels=ch_out,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(name=bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.bn(out)
        return out


class ConvAffineLayer(Layer):
    def __init__(self,
                 name_scope,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 lr=1.0,
                 act='relu'):
        super(ConvAffineLayer, self).__init__()

        self.conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(
                name=name_scope + "_weights", learning_rate=lr),
            bias_attr=False)
        if name_scope == "conv1":
            bn_name = "bn_" + name_scope
        else:
            bn_name = "bn" + name_scope[3:]
        self.scale = fluid.layers.create_parameter(
            shape=[ch_out],
            dtype='float32',
            attr=ParamAttr(
                name=bn_name + '_scale', learning_rate=0.),
            default_initializer=Constant(1.))

        self.offset = fluid.layers.create_parameter(
            shape=[ch_out],
            dtype='float32',
            attr=ParamAttr(
                name=bn_name + '_offset', learning_rate=0.),
            default_initializer=Constant(0.))

        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = fluid.layers.affine_channel(
            out, scale=self.scale, bias=self.offset)
        if self.act == 'relu':
            out = fluid.layers.relu(out)
        return out


class BottleNeck(Layer):
    def __init__(self,
                 name_scope,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut=True,
                 lr=1.0,
                 norm_type='bn'):
        super(BottleNeck, self).__init__()
        self.name_scope = name_scope
        if norm_type == 'bn':
            atom_block = ConvBNLayer
        elif norm_type == 'affine':
            atom_block = ConvAffineLayer
        else:
            atom_block = None
        assert atom_block != None, 'NormType only support BatchNorm and Affine!'

        self.shortcut = shortcut
        if not shortcut:
            self.branch1 = atom_block(
                name_scope + "_branch1",
                ch_in=ch_in,
                ch_out=ch_out * 4,
                filter_size=1,
                stride=stride,
                padding=0,
                act=None,
                lr=lr)

        self.branch2a = atom_block(
            name_scope + "_branch2a",
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=1,
            stride=stride,
            padding=0,
            lr=lr)

        self.branch2b = atom_block(
            name_scope + "_branch2b",
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            padding=1,
            lr=lr)

        self.branch2c = atom_block(
            name_scope + "_branch2c",
            ch_in=ch_out,
            ch_out=ch_out * 4,
            filter_size=1,
            stride=1,
            padding=0,
            lr=lr,
            act=None)

    def forward(self, inputs):
        if self.shortcut:
            short = inputs
        else:
            short = self.branch1(inputs)

        out = self.branch2a(inputs)
        out = self.branch2b(out)
        out = self.branch2c(out)

        out = fluid.layers.elementwise_add(
            x=short, y=out, act='relu', name=self.name_scope + ".add.output.5")

        return out


class Blocks(Layer):
    def __init__(self,
                 name_scope,
                 ch_in,
                 ch_out,
                 count,
                 stride,
                 lr=1.0,
                 norm_type='bn'):
        super(Blocks, self).__init__()

        self.blocks = []
        for i in range(count):
            if i == 0:
                name = name_scope + "a"
                self.stride = stride
                self.shortcut = False
            else:
                name = name_scope + chr(ord("a") + i)
                self.stride = 1
                self.shortcut = True

            block = self.add_sublayer(
                name,
                BottleNeck(
                    name,
                    ch_in=ch_in if i == 0 else ch_out * 4,
                    ch_out=ch_out,
                    stride=self.stride,
                    shortcut=self.shortcut,
                    lr=lr,
                    norm_type=norm_type))
            self.blocks.append(block)
            shortcut = True

    def forward(self, inputs):
        res_out = self.blocks[0](inputs)
        for block in self.blocks[1:]:
            res_out = block(res_out)
        return res_out


ResNet_cfg = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}


@register
@serializable
class ResNet(Layer):
    def __init__(self,
                 depth=50,
                 norm_type='bn',
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3],
                 num_blocks=4):
        super(ResNet, self).__init__()
        self.depth = depth
        self.norm_type = norm_type
        self.freeze_at = freeze_at
        if isinstance(return_idx, Integral):
            return_idx = [return_idx]
        assert max(return_idx) < num_stages, \
            'the maximum return index must smaller than num_stages, ' \
            'but received maximum return index is {} and num_stages ' \
            'is {}'.format(max(return_idx), num_stages)
        self.return_idx = return_idx
        self.num_stages = num_stages

        block_nums = ResNet_cfg[depth]
        if self.norm_type == 'bn':
            atom_block = ConvBNLayer
        elif self.norm_type == 'affine':
            atom_block = ConvAffineLayer
        else:
            raise 'NormType only supports bn and affine!'

        self.conv1 = atom_block(
            'conv1', ch_in=3, ch_out=64, filter_size=7, stride=2, padding=3)

        self.pool = Pool2D(
            pool_type='max', pool_size=3, pool_stride=2, pool_padding=1)

        ch_in_list = [64, 256, 512, 1024]
        ch_out_list = [64, 128, 256, 512]

        self.res_layers = []
        for i in range(num_stages):
            res_name = "res{}".format(i + 2)
            res_layer = self.add_sublayer(
                res_name,
                Blocks(
                    res_name,
                    ch_in_list[i],
                    ch_out_list[i],
                    count=blocks[i],
                    stride=2 if i == 0 else 1))
            self.res_layers.append(res_layer)

    def forward(self, inputs):
        x = inputs['image']

        conv1 = self.conv1(x)
        x = self.pool(conv1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx == self.freeze_at:
                x.stop_gradient = True
            if idx in self.return_idx:
                outs.append(x)
        return outs
