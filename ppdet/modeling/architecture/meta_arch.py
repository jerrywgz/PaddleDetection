from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph.base import to_variable
from ppdet.core.workspace import register
from ppdet.utils.data_structure import BufferDict

__all__ = ['BaseArch']


@register
class BaseArch(Layer):
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
        else:
            raise "Now, only support train or infer mode!"
        return out

    def build_inputs(self, data, input_def):
        inputs = {}
        for name, input in zip(input_def, data):
            v = to_variable(np.array(input))
            inputs[name] = v
        return inputs

    def model_arch(self, mode):
        raise NotImplementedError("Should implement model_arch method!")

    def loss(self, ):
        raise NotImplementedError("Should implement loss method!")

    def infer(self, ):
        raise NotImplementedError("Should implement infer method!")
