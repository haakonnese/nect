from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_3_t
from torch.nn.modules.utils import _triple
from torch.optim import Adam


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Applies a 2D convolution over an input signal composed of several input
    planes.
    """
    # Define shortcut names for dimensions of input and kernel
    (Batch, _, l_i, h_i, w_i) = tuple(input.shape)
    (l_k, h_k, w_k) = (weight.size(2), weight.size(3), weight.size(4))
    (l_p, h_p, w_p) = padding
    (l_d, h_d, w_d) = dilation
    (l_s, h_s, w_s) = stride

    # Compute the size of the output tensor based on the zero padding
    l_o = (l_i + 2 * l_p - (l_k) - (l_k - 1) * (l_d - 1)) // l_s + 1
    w_o = (w_i + 2 * w_p - (w_k) - (w_k - 1) * (w_d - 1)) // w_s + 1
    h_o = (h_i + 2 * h_p - (h_k) - (h_k - 1) * (h_d - 1)) // h_s + 1
    out_channels = weight.size(1)
    # Pre-define output tensors
    out = torch.zeros(Batch, out_channels, l_o, h_o, w_o).to(input.device)

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
            out[:, :, out_frame, :, :] += F.conv2d(
                input[:, :, j, :, :],
                weight[:, :, i, :, :],
                bias=None,
                stride=stride[1::],
                padding=padding[1::],
                dilation=dilation[1::],
                groups=groups,
            )

    # Add bias to output
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)

    return out


def conv_transpose3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, _size_3_t] = 1,
    padding: Union[int, _size_3_t] = 0,
    output_padding: Union[int, _size_3_t] = 0,
    groups: int = 1,
    dilation: Union[int, _size_3_t] = 1,
) -> torch.Tensor:
    # Define shortcut names for dimensions of input and kernel
    (Batch, _, l_i, h_i, w_i) = tuple(input.shape)
    (l_k, h_k, w_k) = (weight.size(2), weight.size(3), weight.size(4))
    (l_p, h_p, w_p) = padding
    (l_op, h_op, w_op) = output_padding
    (l_d, h_d, w_d) = dilation
    (l_s, h_s, w_s) = stride

    # Compute the size of the output tensor based on the zero padding
    l_o = (l_i - 1) * l_s - 2 * l_p + l_d * (l_k - 1) + l_op + 1
    h_o = (h_i - 1) * h_s - 2 * h_p + h_d * (h_k - 1) + h_op + 1
    w_o = (w_i - 1) * w_s - 2 * w_p + w_d * (w_k - 1) + w_op + 1
    out_channels = weight.size(1)

    # Pre-define output tensors
    out = torch.zeros(Batch, out_channels, l_o, h_o, w_o).to(input.device)

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
            out[:, :, out_frame, :, :] += F.conv_transpose2d(
                input[:, :, j, :, :],
                weight[:, :, i, :, :],
                bias=None,
                stride=stride[1::],
                padding=padding[1::],
                output_padding=output_padding[1::],
                dilation=dilation[1::],
                groups=groups,
            )

    # Add bias to output
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)

    return out


class Conv3d(nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        valid_padding_modes = {"zeros"}
        assert groups == 1, "Groups other than 1 not yet implemented!"
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be 'zeros', but got padding_mode='{}'. Other modes not yet implemented".format(
                    padding_mode
                )
            )
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        return conv3d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)


class ConvTranspose3d(nn.modules.conv._ConvTransposeNd):
    "From https://github.com/matheusja/ConvTranspose4d-PyTorch"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_3_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose4d")

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 3
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,
        )  # type: ignore[arg-type]

        return conv_transpose3d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


def test_conv3d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(100):
        input = torch.randn(2, 16, 50, 50, 50).to(device)
        kernel_size = (
            np.random.randint(1, 5),
            np.random.randint(1, 5),
            np.random.randint(1, 5),
        )
        padding = (
            np.random.randint(0, 10),
            np.random.randint(0, 10),
            np.random.randint(0, 10),
        )
        stride = (
            np.random.randint(1, 10),
            np.random.randint(1, 10),
            np.random.randint(1, 10),
        )
        dilation = (
            np.random.randint(1, 3),
            np.random.randint(1, 3),
            np.random.randint(1, 3),
        )
        bias = np.random.choice([True, False])
        net = Conv3d(
            16,
            16,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=1,
            bias=bias,
        ).to(device)

        official_conv3d = nn.Conv3d(16, 16, kernel_size, stride, padding, dilation, 1, bias=bias)
        official_conv3d.load_state_dict(net.state_dict())
        official_conv3d.to(device)
        out = net(input)
        out_official = official_conv3d(input)
        assert torch.allclose(out, out_official, atol=1e-5)


def test_conv3d_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(10):
        input_torch = torch.randn(2, 16, 50, 50, 50).to(device).requires_grad_(True)
        input_custom = input_torch.clone().detach().requires_grad_(True)
        optim_torch = Adam([input_torch], lr=1e-3)
        optim_custom = Adam([input_custom], lr=1e-3)
        kernel_size = (
            np.random.randint(1, 5),
            np.random.randint(1, 5),
            np.random.randint(1, 5),
        )
        padding = (
            np.random.randint(0, 10),
            np.random.randint(0, 10),
            np.random.randint(0, 10),
        )
        stride = (
            np.random.randint(1, 10),
            np.random.randint(1, 10),
            np.random.randint(1, 10),
        )
        dilation = (
            np.random.randint(1, 3),
            np.random.randint(1, 3),
            np.random.randint(1, 3),
        )
        bias = np.random.choice([True, False])
        net = Conv3d(
            16,
            16,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=1,
            bias=bias,
        ).to(device)

        official_conv3d = nn.Conv3d(16, 16, kernel_size, stride, padding, dilation, 1, bias=bias)
        official_conv3d.load_state_dict(net.state_dict())
        official_conv3d.to(device)
        out = net(input_torch)
        out_official = official_conv3d(input_custom)
        out.sum().backward()
        out_official.sum().backward()
        optim_torch.step()
        optim_custom.step()
        assert torch.allclose(input_torch.grad, input_custom.grad, atol=1e-5)


def test_conv_transpose3d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(100):
        input = torch.randn(2, 16, 50, 50, 50).to(device)
        kernel_size = (
            np.random.randint(1, 5),
            np.random.randint(1, 5),
            np.random.randint(1, 5),
        )
        padding = (
            np.random.randint(0, 10),
            np.random.randint(0, 10),
            np.random.randint(0, 10),
        )
        stride = (
            np.random.randint(1, 10),
            np.random.randint(1, 10),
            np.random.randint(1, 10),
        )
        dilation = (
            1,
            1,
            1,
        )  # dilation not yet implemented for transposed conv3d or transposed conv4d
        output_padding = (
            np.random.randint(0, max(stride[0], dilation[0])),
            np.random.randint(0, max(stride[1], dilation[1])),
            np.random.randint(0, max(stride[2], dilation[2])),
        )
        bias = np.random.choice([True, False])
        net = ConvTranspose3d(
            16,
            16,
            kernel_size=kernel_size,
            padding=padding,
            output_padding=output_padding,
            stride=stride,
            dilation=dilation,
            groups=1,
            bias=bias,
        ).to(device)

        official_conv_transpose3d = nn.ConvTranspose3d(
            in_channels=16,
            out_channels=16,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=1,
            bias=bias,
        )
        official_conv_transpose3d.load_state_dict(net.state_dict())
        official_conv_transpose3d.to(device)
        out = net(input)
        out_official = official_conv_transpose3d(input)
        assert torch.allclose(out, out_official, atol=1e-5)


def test_transposed_conv3d_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(10):
        input_torch = torch.randn(2, 16, 50, 50, 50).to(device).requires_grad_(True)
        input_custom = input_torch.clone().detach().requires_grad_(True)
        optim_torch = Adam([input_torch], lr=1e-3)
        optim_custom = Adam([input_custom], lr=1e-3)
        kernel_size = (
            np.random.randint(1, 5),
            np.random.randint(1, 5),
            np.random.randint(1, 5),
        )
        padding = (
            np.random.randint(0, 10),
            np.random.randint(0, 10),
            np.random.randint(0, 10),
        )
        stride = (
            np.random.randint(1, 10),
            np.random.randint(1, 10),
            np.random.randint(1, 10),
        )
        dilation = (
            1,
            1,
            1,
        )  # dilation not yet implemented for transposed conv3d or transposed conv4d
        output_padding = (
            np.random.randint(0, max(stride[0], dilation[0])),
            np.random.randint(0, max(stride[1], dilation[1])),
            np.random.randint(0, max(stride[2], dilation[2])),
        )
        bias = np.random.choice([True, False])
        net = ConvTranspose3d(
            16,
            16,
            kernel_size=kernel_size,
            padding=padding,
            output_padding=output_padding,
            stride=stride,
            dilation=dilation,
            groups=1,
            bias=bias,
        ).to(device)

        official_conv_transpose3d = nn.ConvTranspose3d(
            in_channels=16,
            out_channels=16,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=1,
            bias=bias,
        )
        official_conv_transpose3d.load_state_dict(net.state_dict())
        official_conv_transpose3d.to(device)
        out = net(input_custom)
        out_official = official_conv_transpose3d(input_torch)
        out.sum().backward()
        out_official.sum().backward()
        optim_torch.step()
        optim_custom.step()
        assert torch.allclose(input_torch.grad, input_custom.grad, atol=1e-5)
