from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import paddle.nn as nn
from ppdet.core.workspace import register
from ppdet.utils.data_structure import BufferDict

__all__ = ['BaseArch']


@register
class BaseArch(nn.Layer):
    def __init__(self):
        super(BaseArch, self).__init__()

    def forward(self, data, input_def, mode):
        self.inputs = self.build_inputs(data, input_def)
        self.inputs['mode'] = mode
        self.model_arch()

        if mode == 'train':
            out = self.loss()
        elif mode == 'infer':
            out = self.infer()
        elif mode == 'export_model':
            out = self.export_model()
        else:
            raise "Now, only support train, infer and export mode!"
        return out

    def build_inputs(self, data, input_def):
        inputs = {}
        for name in input_def:
            inputs[name] = []
        batch_size = len(data)
        for bs in range(batch_size):
            for name, input in zip(input_def, data[bs]):
                input_v = np.array(input)[np.newaxis, ...]
                inputs[name].append(input_v)
        for name in input_def:
            inputs[name] = paddle.to_tensor(np.concatenate(inputs[name]))
        return inputs

    def model_arch(self):
        raise NotImplementedError("Should implement model_arch method!")

    def loss(self, ):
        raise NotImplementedError("Should implement loss method!")

    def infer(self, ):
        raise NotImplementedError("Should implement infer method!")

    def export_model(self, exclude_nms=False):
        raise NotImplementedError("Should implement infer method!")
