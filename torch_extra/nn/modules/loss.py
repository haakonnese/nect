from __future__ import annotations
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

import torch_extra.nn.functional as F_extra


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        win: Optional[Tensor] = None,
        nonnegative_ssim: bool = False,
        stride: int = 1,
        device: Optional[torch.device | int | str] = None,
    ) -> None:
        r"""class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            win (Tensor, optional): window for ssim. The size should be 1xwin_size
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
            stride (int, optional): stride of the sliding window
            device (torch.device, int or str, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor type (default: None). If provided, it will send kernel to device before forward.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        if win is not None:
            if len(win.size()) == 1:
                win = win.repeat([channel, 1] + [1] * spatial_dims)
            self.win = win
        else:
            self.win = F_extra._fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        if device:
            self.win = self.win.to(device=device, dtype=torch.float32)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim
        self.stride = stride

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return F_extra.ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
            stride=self.stride,
        )


class ST_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        nonnegative_ssim: bool = False,
        temporal_win_size: Optional[int] = None,
        stride: int = 1,
    ) -> None:
        r"""class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
            temporal_win_size (int, optional): the size of gauss kernel for temporal dimension
            stride (int, optional): stride of the sliding window
        """

        super(ST_SSIM, self).__init__()
        self.win_size = win_size
        self.temporal_win_size = temporal_win_size
        temporal_dim = 1 if temporal_win_size is not None else 0
        self.win = F_extra._fspecial_gauss_1d(win_size, win_sigma).repeat(
            [channel, 1] + [1] * (spatial_dims + temporal_dim)
        )
        if temporal_dim:
            self.temporal_win = (
                F_extra._fspecial_gauss_1d(temporal_win_size, win_sigma)
                .repeat([channel, 1] + [1] * (spatial_dims + temporal_dim))
                .transpose(-1, 2)
            )
        else:
            self.temporal_win = None
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim
        self.stride = stride

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return F_extra.st_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
            temporal_win=self.temporal_win,
            temporal_win_size=self.temporal_win_size,
            stride=self.stride,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        stride: int = 1,
    ) -> None:
        r"""class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            stride (int, optional): stride of the sliding window
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = F_extra._fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K
        self.stride = stride

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return F_extra.ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
            stride=self.stride,
        )


class ST_MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        temporal_win_size: Optional[int] = None,
    ) -> None:
        r"""class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            temporal_win_size (int, optional): the size of gauss kernel for temporal dimension
        """

        super(ST_MS_SSIM, self).__init__()
        self.win_size = win_size
        self.temporal_win_size = temporal_win_size
        temporal_dim = 1 if temporal_win_size is not None else 0
        self.win = F_extra._fspecial_gauss_1d(win_size, win_sigma).repeat(
            [channel, 1] + [1] * (spatial_dims + temporal_dim)
        )
        if temporal_dim:
            self.temporal_win = (
                F_extra._fspecial_gauss_1d(temporal_win_size, win_sigma)
                .repeat([channel, 1] + [1] * (spatial_dims + temporal_dim))
                .transpose(-1, 2)
            )
        else:
            self.temporal_win = None
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.weights = weights

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return F_extra.st_ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
            temporal_win=self.temporal_win,
            temporal_win_size=self.temporal_win_size,
        )


class FourierShellCorrelation(torch.nn.Module):
    """Fourier Shell Correlation (FSC) loss function.

    Args:
        shape (tuple[int, ...]): shape of the input tensors.
        size_average (bool): if size_average=True, FSC will be averaged over radii, channel and batch. Else, FSC will be averaged over channel and batch. Default: True.
        delta (int): delta for the shell size. Default: 1.
    """

    def __init__(self, shape: Tuple[int, ...], size_average: bool = True, delta: int = 1) -> None:
        super(FourierShellCorrelation, self).__init__()
        self.size_average = size_average
        self.radial_masks = F_extra._create_radial_masks(shape, delta=delta, dims=3)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return F_extra.fourier_shell_correlation(X, Y, size_average=self.size_average, radial_masks=self.radial_masks)


class FourierRingCorrelation(torch.nn.Module):
    """Fourier Ring Correlation (FRC) loss function.

    Args:
        shape (tuple[int, ...]): shape of the input tensors.
        size_average (bool): if size_average=True, FRC will be averaged over radii, channel and batch. Else, FRC will be averaged over channel and batch. Default: True.
        delta (int): delta for the ring size. Default: 1.
    """

    def __init__(self, shape: Tuple[int, ...], size_average: bool = True, delta: int = 1) -> None:
        super(FourierRingCorrelation, self).__init__()
        self.size_average = size_average
        self.radial_masks = F_extra._create_radial_masks(shape, delta=delta, dims=2)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return F_extra.fourier_ring_correlation(X, Y, size_average=self.size_average, radial_masks=self.radial_masks)
