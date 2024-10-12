from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_extra.typing import _int, _size


def conv4d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[_int, _size] = 1,
    padding: Union[_int, _size] = 0,
    dilation: Union[_int, _size] = 1,
    groups: _int = 1,
) -> Tensor:
    """Applies a 4D convolution over an input signal composed of several input
    planes. Based on  https://github.com/ZhengyuLiang24/Conv4d-PyTorch
    """
    # Define shortcut names for dimensions of input and kernel
    (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape)
    (l_k, d_k, h_k, w_k) = (
        weight.size(3),
        weight.size(4),
        weight.size(5),
        weight.size(6),
    )
    (l_p, d_p, h_p, w_p) = padding
    (l_d, d_d, h_d, w_d) = dilation
    (l_s, d_s, h_s, w_s) = stride

    # Compute the size of the output tensor based on the zero padding
    l_o = (l_i + 2 * l_p - (l_k) - (l_k - 1) * (l_d - 1)) // l_s + 1
    d_o = (d_i + 2 * d_p - (d_k) - (d_k - 1) * (d_d - 1)) // d_s + 1
    h_o = (h_i + 2 * h_p - (h_k) - (h_k - 1) * (h_d - 1)) // h_s + 1
    w_o = (w_i + 2 * w_p - (w_k) - (w_k - 1) * (w_d - 1)) // w_s + 1
    out_channels = weight.size(1)
    # Pre-define output tensors
    out = torch.zeros(Batch, out_channels, l_o, d_o, h_o, w_o).to(input.device)

    # Convolve each kernel frame i with each input frame j
    for i in range(l_k):
        # Calculate the zero-offset of kernel frame i
        zero_offset = -l_p + (i * l_d)
        # Calculate the range of input frame j corresponding to kernel frame i
        j_start = max(zero_offset % l_s, zero_offset)
        j_end = min(l_i, l_i + l_p - (l_k - i - 1) * l_d)
        # Convolve each kernel frame i with corresponding input frame j
        for j in range(j_start, j_end, l_s):
            # Calculate the output frame
            out_frame = (j - zero_offset) // l_s
            # Add results to this output frame
            out[:, :, out_frame, :, :, :] += F.conv3d(
                input[:, :, j, :, :],
                weight[:, :, i, :, :, :],
                bias=None,
                stride=stride[1::],
                padding=padding[1::],
                dilation=dilation[1::],
                groups=groups,
            )

    # Add bias to output
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1, 1)

    return out


def conv_transpose4d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[_int, _size] = 1,
    padding: Union[_int, _size] = 0,
    output_padding: Union[_int, _size] = 0,
    groups: _int = 1,
    dilation: Union[_int, _size] = 1,
) -> Tensor:
    if dilation != 1 or dilation != (1, 1, 1, 1):
        raise NotImplementedError("Dilation other than 1 not yet implemented!")
    # Define shortcut names for dimensions of input and kernel
    (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape)
    (l_k, d_k, h_k, w_k) = (
        weight.size(3),
        weight.size(4),
        weight.size(5),
        weight.size(6),
    )
    (l_p, d_p, h_p, w_p) = padding
    (l_op, d_op, h_op, w_op) = output_padding
    (l_d, d_d, h_d, w_d) = dilation
    (l_s, d_s, h_s, w_s) = stride

    # Compute the size of the output tensor based on the zero padding
    l_o = (l_i - 1) * l_s - 2 * l_p + l_d * (l_k - 1) + l_op + 1
    d_o = (d_i - 1) * d_s - 2 * d_p + d_d * (d_k - 1) + d_op + 1
    h_o = (h_i - 1) * h_s - 2 * h_p + h_d * (h_k - 1) + h_op + 1
    w_o = (w_i - 1) * w_s - 2 * w_p + w_d * (w_k - 1) + w_op + 1

    out_channels = weight.size(1)
    # Pre-define output tensors
    out = torch.zeros(Batch, out_channels, l_o, d_o, h_o, w_o).to(input.device)

    # Convolve each kernel frame i with each input frame j
    for i in range(l_k):
        # Calculate the zero-offset of kernel frame i
        zero_offset = -(l_p) + i
        # Calculate the range of input frame j corresponding to kernel frame i
        # Convolve each kernel frame i with corresponding input frame j
        for j in range(0, l_i):
            # Calculate the output frame
            out_frame = l_s * j + zero_offset
            if out_frame < 0 or out_frame >= out.shape[2]:
                # print("{} -> {} (no)".format((i,l_s * j), out_frame))
                continue
            # Add results to this output frame
            out[:, :, out_frame, :, :, :] += F.conv_transpose3d(
                input[:, :, j, :, :],
                weight[:, :, i, :, :, :],
                bias=None,
                stride=stride[1::],
                padding=padding[1::],
                output_padding=output_padding[1::],
                dilation=dilation[1::],
                groups=groups,
            )

    # Add bias to output
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1, 1)

    return out
