import pytest
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

from torch_extra.nn.functional import _calculate_output_size_single_dim


def avg_pool3d_test_function(
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
        stride = _triple(kernel_size)
    else:
        stride = _triple(stride)
    padding = _triple(padding)
    kernel_size = _triple(kernel_size)
    if padding[0] * 2 > kernel_size[0]:
        raise ValueError("Padding length should be at most half of kernel size")
    batch_size, channels, depth, height, width = input.size()
    depth_dim = _calculate_output_size_single_dim(depth, kernel_size[0], stride[0], padding[0])
    height_dim = _calculate_output_size_single_dim(height, kernel_size[1], stride[1], padding[1])
    width_dim = _calculate_output_size_single_dim(width, kernel_size[2], stride[2], padding[2])
    out = torch.zeros(batch_size, channels, depth, height_dim, width_dim)
    for d in range(depth):
        out[:, :, d] = torch.nn.functional.avg_pool2d(
            input[:, :, d],
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
    out_depth = torch.zeros(batch_size, channels, depth_dim, height_dim, width_dim)
    for i, d in enumerate(range(0, depth_dim * stride[0], stride[0])):
        out_depth[:, :, i, :] = torch.mean(out[:, :, d : d + kernel_size[0]], dim=2)
    return out_depth


@pytest.mark.parametrize(
    "kernel_size, stride, padding",
    [
        (2, 2, 0),
        (2, (1, 2, 2), 0),
        ((1, 2, 2), (1, 2, 2), 0),
        (2, 2, 1),
        ((5, 2, 2), (1, 2, 2), 1),
    ],
    ids=[
        "Equal stride length",
        "Different stride length",
        "Different stride and kernel_size length",
        "Zero padding",
        "Zero padding, kernel_size, stride",
    ],
)
def test_avg_pool3d(kernel_size, stride, padding):
    test_input = torch.randn(10, 20, 51, 50, 50)
    avg_3d_self = avg_pool3d_test_function(test_input, kernel_size, stride=stride, padding=padding)
    avg_3d_torch = torch.nn.functional.avg_pool3d(test_input, kernel_size, stride=stride, padding=padding)
    assert torch.allclose(avg_3d_self, avg_3d_torch, atol=1e-6)
