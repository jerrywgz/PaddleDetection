import os
import paddle.fluid as fluid

__all__ = [
    'bottom_pool',
    'top_pool',
    'right_pool',
    'left_pool',
]


def bottom_pool(input, name=None):
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
    H = input.shape[2]
    i = 1
    output = input
    while i < H:
        cur = output[:,:,i:,:]
        next = output[:,:,:H-i,:]
        max_v = fluid.layers.elementwise_max(cur,next)
        output = fluid.layers.concat([output[:,:,:i,:], max_v], axis=2)
        i *= 2
    return output

def top_pool(input, name=None):
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
    H = input.shape[2]
    i = 1
    output = input
    while i < H:
        cur = output[:,:,:H-i,:]
        next = output[:,:,i:,:]
        max_v = fluid.layers.elementwise_max(cur,next)
        output = fluid.layers.concat([max_v, output[:,:,H-i:,:]], axis=2)
        i *= 2
    return output

def right_pool(input, name=None):
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
    W = input.shape[3]
    i = 1
    output = input
    while i < W:
        cur = output[:,:,:,i:]
        next = output[:,:,:,:W-i]
        max_v = fluid.layers.elementwise_max(cur,next)
        output = fluid.layers.concat([output[:,:,:,:i], max_v], axis=-1)
        i *= 2
    return output

def left_pool(input, name=None):
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
    W = input.shape[3]
    i = 1
    output = input
    while i < W:
        cur = output[:,:,:,:W-i]
        next = output[:,:,:,i:]
        print(cur)
        print(next)
        max_v = fluid.layers.elementwise_max(cur,next)
        output = fluid.layers.concat([max_v, output[:,:,:,W-i:]], axis=-1)
        print(output)
        i *= 2
    return output


