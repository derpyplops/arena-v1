import torch as t
import utils
import einops
from fancy_einsum import einsum

def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    arr3 = rearrange(arr2, "c (h2 h) w -> c h (h2 w)", h2=2)
    '''
    # input_arr = einops.rearrange(t.arange(0,10), 'arr -> 1 arr')
    # # rearrange(ims, '(b1 b2) h w c -> (b1 h) (b2 w) c ', b1=2)
    # input_arr = einops.rearrange(input_arr, 'h w -> ')
    xsB, xsI, xsWi = x.stride()
    x_new_stride = (xsB, xsI, xsWi, xsWi)

    batch, in_channels, width = x.shape
    out_channels, in_channels_2, kernel_width = weights.shape
    output_width = width - kernel_width + 1
    
    x_new_shape = (batch, in_channels, output_width, kernel_width)


    return einsum(
        "batch in_channels output_width kernel_width, out_channels in_channels kernel_width -> batch out_channels output_width", 
        x.as_strided(x_new_shape, x_new_stride), weights
    )

    

    # print(input_arr)

utils.test_conv1d_minimal(conv1d_minimal)

def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    xsB, xsI, xsHi, xsWi = x.stride()
    x_new_stride = (xsB, xsI, xsHi, xsWi, xsHi, xsWi)

    batch, in_channels, height, width = x.shape
    out_channels, in_channels_2, kernel_height, kernel_width = weights.shape
    output_width = width - kernel_width + 1
    output_height = height - kernel_height + 1
    
    x_new_shape = (batch, in_channels, output_height, output_width, kernel_height, kernel_width)


    return einsum(
        """batch in_channels output_height output_width kernel_height kernel_width, 
        out_channels in_channels kernel_height kernel_width 
        -> 
        batch out_channels output_height output_width""", 
        x.as_strided(x_new_shape, x_new_stride), weights
    )

utils.test_conv2d_minimal(conv2d_minimal)