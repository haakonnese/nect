import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple


def _calculate_output_size_single_dim(input_size, kernel_size, stride, padding):
    return (input_size + 2 * padding - kernel_size) // stride + 1


def avg_pool4d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
) -> torch.Tensor:
    if ceil_mode:
        raise NotImplementedError("ceil_mode is not implemented yet")
    if stride is None:
        stride = _quadruple(kernel_size)
    else:
        stride = _quadruple(stride)
    padding = _quadruple(padding)
    if padding[0] * 2 > kernel_size[0]:
        raise ValueError("Padding length should be at most half of kernel size")
    kernel_size = _quadruple(kernel_size)
    batch_size, channels, time, depth, height, width = input.size()
    time_dim = _calculate_output_size_single_dim(time, kernel_size[0], stride[0], padding[0])
    depth_dim = _calculate_output_size_single_dim(depth, kernel_size[1], stride[1], padding[1])
    height_dim = _calculate_output_size_single_dim(height, kernel_size[2], stride[2], padding[2])
    width_dim = _calculate_output_size_single_dim(width, kernel_size[3], stride[3], padding[3])
    out = torch.zeros(batch_size, channels, time, depth_dim, height_dim, width_dim)
    for t in range(time):
        out[:, :, t] = torch.nn.functional.avg_pool3d(
            input[:, :, t],
            kernel_size=kernel_size[1:],
            stride=stride[1:],
            padding=padding[1:],
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
    out = out.movedim(2, -1)
    out = F.pad(out, (padding[0], padding[0]), mode="constant", value=0)
    out = out.movedim(-1, 2)
    out_time = torch.zeros(batch_size, channels, time_dim, depth_dim, height_dim, width_dim)
    for i, t in enumerate(range(0, time_dim * stride[0], stride[0])):
        out_time[:, :, i, :] = torch.mean(out[:, :, t : t + kernel_size[0]], dim=2)
    return out_time


def max_pool4d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
) -> torch.Tensor:
    if ceil_mode:
        raise NotImplementedError("ceil_mode is not implemented yet")
    if stride is None:
        stride = _quadruple(kernel_size)
    else:
        stride = _quadruple(stride)
    padding = _quadruple(padding)
    if padding[0] * 2 > kernel_size[0]:
        raise ValueError("Padding length should be at most half of kernel size")
    kernel_size = _quadruple(kernel_size)
    batch_size, channels, time, depth, height, width = input.size()
    time_dim = _calculate_output_size_single_dim(time, kernel_size[0], stride[0], padding[0])
    depth_dim = _calculate_output_size_single_dim(depth, kernel_size[1], stride[1], padding[1])
    height_dim = _calculate_output_size_single_dim(height, kernel_size[2], stride[2], padding[2])
    width_dim = _calculate_output_size_single_dim(width, kernel_size[3], stride[3], padding[3])
    out = torch.zeros(batch_size, channels, time, depth_dim, height_dim, width_dim)
    for t in range(time):
        out[:, :, t] = torch.nn.functional.max_pool3d(
            input[:, :, t],
            kernel_size=kernel_size[1:],
            stride=stride[1:],
            padding=padding[1:],
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
    out = out.movedim(2, -1)
    out = F.pad(out, (padding[0], padding[0]), mode="constant", value=0)
    out = out.movedim(-1, 2)
    out_time = torch.zeros(batch_size, channels, time_dim, depth_dim, height_dim, width_dim)
    for i, t in enumerate(range(0, time_dim * stride[0], stride[0])):
        out_time[:, :, i, :] = torch.max(out[:, :, t : t + kernel_size[0]], dim=2)
    return out_time
