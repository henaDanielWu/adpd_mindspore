import mindspore
import x2ms_adapter

def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.shape
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view( batch_size, groups, channels_per_group, height, width)
    x = mindspore.ops.swapaxes(x, 1, 2)
    x = x.view(batch_size, -1, height, width)

    return x
