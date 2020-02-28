import os
import paddle.fluid as fluid

__all__ = [
    'bottom_pool',
    'top_pool',
    'right_pool',
    'left_pool',
]


def bottom_pool(input, is_test=False, name=None):
    """
    This layer calculates the bottom pooling output based on the input.
    Scan the input from top to bottm for the vertical max-pooling.
    The output has the same shape with input.

    Args:
        input(Variable): This input is a Tensor with shape [N, C, H, W].
            The data type is float32 or float64.
    Returns:
        Variable(Tensor): The output of bottom_pool, with shape [N, C, H, W].
        The data type is float32 or float64.

    Examples:

        ..code-block:: python

            import paddle.fluid as fluid
            import cornerpool_lib
            input = fluid.data(
                name='input', shape=[2, 64, 10, 10], dtype='float32')
            output = corner_pool.bottom_pool(input)
    """
    if is_test:
        zero = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
        H = fluid.layers.shape(input)[2]
        cond = fluid.layers.less_than(x=i, y=H)
        bottom_out = fluid.layers.create_array('float32')
        output = fluid.layers.assign(input)
        while_op = fluid.layers.While(cond=cond)
        with while_op.block():
            idx = fluid.layers.assign(i)
            Hx = fluid.layers.assign(H)
            cur = fluid.layers.slice(output, [2], [idx], [Hx])
            next = fluid.layers.slice(output, [2], [0], [Hx - idx])
            max_v = fluid.layers.elementwise_max(cur, next)
            orig = fluid.layers.slice(output, [2], [0], [idx])
            output = fluid.layers.concat([orig, max_v], axis=2)
            fluid.layers.array_write(output, zero, bottom_out)
            i = fluid.layers.increment(x=i, value=1, in_place=True)
            #i = i * 2
            fluid.layers.less_than(x=i, y=H, cond=cond)
            #return [i, output]
        output, _ = fluid.layers.tensor_array_to_tensor(input=bottom_out, axis=0)
        
        return output

    H = input.shape[2]
    i = 1
    output = input
    while i < H:
        cur = output[:, :, i:, :]
        next = output[:, :, :H - i, :]
        max_v = fluid.layers.elementwise_max(cur, next)
        output = fluid.layers.concat([output[:, :, :i, :], max_v], axis=2)
        i *= 2
    return output


def top_pool(input, is_test=False, name=None):
    """
    This layer calculates the top pooling output based on the input.
    Scan the input from bottom to top for the vertical max-pooling.
    The output has the same shape with input.

    Args:
        input(Variable): This input is a Tensor with shape [N, C, H, W].
            The data type is float32 or float64.
    Returns:
        Variable(Tensor): The output of top_pool, with shape [N, C, H, W].
        The data type is float32 or float64.

    Examples:

        ..code-block:: python

            import paddle.fluid as fluid
            import cornerpool_lib
            input = fluid.data(
                name='input', shape=[2, 64, 10, 10], dtype='float32')
            output = corner_pool.top_pool(input)

    """
    if is_test:
        zero = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
        H = fluid.layers.shape(input)[2]
        cond = fluid.layers.less_than(x=i, y=H)
        top_out = fluid.layers.create_array('float32')
        output = fluid.layers.assign(input)
        while_op = fluid.layers.While(cond=cond)
        with while_op.block():
            idx = fluid.layers.assign(i)
            Hx = fluid.layers.assign(H)
            cur = fluid.layers.slice(output, [2], [0], [Hx - idx])
            next = fluid.layers.slice(output, [2], [idx], [Hx])
            max_v = fluid.layers.elementwise_max(cur, next)
            orig = fluid.layers.slice(output, [2], [Hx - idx], [Hx])
            output = fluid.layers.concat([max_v, orig], axis=2)
            fluid.layers.array_write(output, zero, top_out)
            i = fluid.layers.increment(x=i, value=1, in_place=True)
            fluid.layers.less_than(x=i, y=H, cond=cond)
        output, _ = fluid.layers.tensor_array_to_tensor(input=top_out, axis=0)
        return output

    H = input.shape[2]
    i = 1
    output = input
    while i < H:
        cur = output[:, :, :H - i, :]
        next = output[:, :, i:, :]
        max_v = fluid.layers.elementwise_max(cur, next)
        output = fluid.layers.concat([max_v, output[:, :, H - i:, :]], axis=2)
        i *= 2
    return output


def right_pool(input, is_test=False, name=None):
    """
    This layer calculates the right pooling output based on the input.
    Scan the input from left to right for the horizontal max-pooling.
    The output has the same shape with input.

    Args:
        input(Variable): This input is a Tensor with shape [N, C, H, W].
            The data type is float32 or float64.
    Returns:
        Variable(Tensor): The output of right_pool, with shape [N, C, H, W].
        The data type is float32 or float64.

    Examples:

        ..code-block:: python

            import paddle.fluid as fluid
            import cornerpool_lib
            input = fluid.data(
                name='input', shape=[2, 64, 10, 10], dtype='float32')
            output = corner_pool.right_pool(input)

    """
    if is_test:
        zero = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
        W = fluid.layers.shape(input)[3]
        cond = fluid.layers.less_than(x=i, y=W)
        right_out = fluid.layers.create_array('float32')
        output = fluid.layers.assign(input)
        while_op = fluid.layers.While(cond=cond)
        with while_op.block():
            idx = fluid.layers.assign(i)
            Wx = fluid.layers.assign(W)
            cur = fluid.layers.slice(output, [3], [idx], [Wx])
            next = fluid.layers.slice(output, [3], [0], [Wx - idx])
            max_v = fluid.layers.elementwise_max(cur, next)
            orig = fluid.layers.slice(output, [3], [0], [idx])
            output = fluid.layers.concat([orig, max_v], axis=-1)
            fluid.layers.array_write(output, zero, right_out)
            i = fluid.layers.increment(x=i, value=1, in_place=True)
            fluid.layers.less_than(x=i, y=W, cond=cond)
        output, _ = fluid.layers.tensor_array_to_tensor(input=right_out, axis=0)

        return output

    W = input.shape[3]
    i = 1
    output = input
    while i < W:
        cur = output[:, :, :, i:]
        next = output[:, :, :, :W - i]
        max_v = fluid.layers.elementwise_max(cur, next)
        output = fluid.layers.concat([output[:, :, :, :i], max_v], axis=-1)
        i *= 2
    return output


def left_pool(input, is_test=False, name=None):
    """
    This layer calculates the left pooling output based on the input.
    Scan the input from right to left for the horizontal max-pooling.
    The output has the same shape with input.

    Args:
        input(Variable): This input is a Tensor with shape [N, C, H, W].
            The data type is float32 or float64.
    Returns:
        Variable(Tensor): The output of left_pool, with shape [N, C, H, W].
        The data type is float32 or float64.

    Examples:

        ..code-block:: python

            import paddle.fluid as fluid
            import cornerpool_lib
            input = fluid.data(
                name='input', shape=[2, 64, 10, 10], dtype='float32')
            output = corner_pool.left_pool(input)

    """
    if is_test:
        zero = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
        W = fluid.layers.shape(input)[3]
        cond = fluid.layers.less_than(x=i, y=W)
        left_out = fluid.layers.create_array('float32')
        output = fluid.layers.assign(input)
        while_op = fluid.layers.While(cond=cond)
        with while_op.block():
            idx = fluid.layers.assign(i)
            Wx = fluid.layers.assign(W)
            cur = fluid.layers.slice(output, [3], [0], [Wx - idx])
            next = fluid.layers.slice(output, [3], [idx], [Wx])
            max_v = fluid.layers.elementwise_max(cur, next)
            orig = fluid.layers.slice(output, [3], [Wx - idx], [Wx])
            output = fluid.layers.concat([max_v, orig], axis=-1)
            fluid.layers.array_write(output, zero, left_out)
            i = fluid.layers.increment(x=i, value=1, in_place=True)
            #i = i * 2
            fluid.layers.less_than(x=i, y=W, cond=cond)
        output, _ = fluid.layers.tensor_array_to_tensor(input=left_out, axis=0)
        #print('left: ', output)
        return output

    W = input.shape[3]
    i = 1
    output = input
    while i < W:
        cur = output[:, :, :, :W - i]
        next = output[:, :, :, i:]
        max_v = fluid.layers.elementwise_max(cur, next)
        output = fluid.layers.concat([max_v, output[:, :, :, W - i:]], axis=-1)
        i *= 2
    return output
