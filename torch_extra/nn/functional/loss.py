import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import torch_extra.nn.functional as F_extra


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor, temporal_win: Optional[Tensor] = None, stride: int = 1) -> Tensor:
    r"""Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    elif len(input.shape) == 6:
        conv = F_extra.conv4d
    else:
        raise NotImplementedError(input.shape)
    C = input.shape[1]
    out = input
    temporal_dim = 0
    if temporal_win is not None:
        temporal_dim = 1
    # print("Input shape", input.shape[2+temporal_dim:])
    for i, s in enumerate(input.shape[2 + temporal_dim :]):
        # print("Window size", win.transpose(2 + temporal_dim + i, -1).size())
        if s >= win.shape[-1]:
            out = conv(
                out,
                weight=win.transpose(2 + temporal_dim + i, -1),
                stride=stride,
                padding=0,
                groups=C,
            )
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )
    if temporal_win is not None:
        if input.shape[2] >= temporal_win.shape[-1]:
            # print("Temporal window size", temporal_win.size())

            out = conv(out, weight=temporal_win, stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at time-dimension 2 for input: {input.shape} and temporal win size: {temporal_win.shape[-1]}"
            )
    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    temporal_win: Optional[Tensor] = None,
    luminance: bool = True,
    stride: int = 1,
) -> Tuple[Tensor, Tensor]:
    r"""Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        temporal_win (torch.Tensor, optional): 1-D gauss kernel for temporal dimension

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)
    temporal_win = temporal_win.to(X.device, dtype=X.dtype) if temporal_win is not None else None
    mu1 = gaussian_filter(X, win, temporal_win, stride)
    # print(mu1)
    mu2 = gaussian_filter(Y, win, temporal_win, stride)
    # print(mu2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win, temporal_win, stride) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win, temporal_win, stride) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win, temporal_win, stride) - mu1_mu2)
    # print(sigma1_sq, sigma2_sq, sigma12)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    if luminance:
        ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
        ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    else:
        ssim_per_channel = None
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
    stride: int = 1,
) -> Tensor:
    r"""interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,D,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,D,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
        temporal_win (torch.Tensor, optional): 1-D gauss kernel for temporal dimension. if None, a new kernel will be created according to temporal_win_size and win_sigma
        temporal_win_size (int, optional): the size of gauss kernel for temporal dimension
        stride (int, optional): stride for the convolution

    Returns:
        torch.Tensor: ssim results
    """
    return _ssim_wrapper(
        X=X,
        Y=Y,
        data_range=data_range,
        size_average=size_average,
        win_size=win_size,
        win_sigma=win_sigma,
        win=win,
        K=K,
        nonnegative_ssim=nonnegative_ssim,
        stride=stride,
    )


def st_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
    temporal_win: Optional[Tensor] = None,
    temporal_win_size: int = 5,
    stride: int = 1,
) -> Tensor:
    r"""interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,D,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,D,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
        temporal_win (torch.Tensor, optional): 1-D gauss kernel for temporal dimension. if None, a new kernel will be created according to temporal_win_size and win_sigma
        temporal_win_size (int, optional): the size of gauss kernel for temporal dimension
        stride (int, optional): stride for the convolution

    Returns:
        torch.Tensor: ssim results
    """
    return _ssim_wrapper(
        X=X,
        Y=Y,
        data_range=data_range,
        size_average=size_average,
        win_size=win_size,
        win_sigma=win_sigma,
        win=win,
        K=K,
        nonnegative_ssim=nonnegative_ssim,
        temporal_win=temporal_win,
        temporal_win_size=temporal_win_size,
        stride=stride,
    )


def _ssim_wrapper(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
    temporal_win: Optional[Tensor] = None,
    temporal_win_size: Optional[int] = None,
    stride: int = 1,
) -> Tensor:
    r"""interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,D,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,D,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
        temporal_win (torch.Tensor, optional): 1-D gauss kernel for temporal dimension. if None, a new kernel will be created according to temporal_win_size and win_sigma
        temporal_win_size (int, optional): the size of gauss kernel for temporal dimension
        stride (int, optional): stride for the convolution

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5, 6):
        raise ValueError(f"Input images should be 4-d, 5-d or 6-d tensors, but got {X.shape}")

    # if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")
    temporal_dim = 1 if temporal_win_size is not None else 0

    if win is not None:  # set win_size
        win_size = win.shape[-1]
        if len(win.size()) == 1:
            win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
        if temporal_win is None and temporal_dim:
            temporal_win = _fspecial_gauss_1d(temporal_win_size, win_sigma).repeat(
                [X.shape[1]] + [1] * (len(X.shape) - 1)
            )

    if win is None:
        if not (win_size % 2 == 1):
            raise ValueError("Window size should be odd.")
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
        if temporal_win is None and temporal_dim:
            temporal_win = _fspecial_gauss_1d(temporal_win_size, win_sigma).repeat(
                [X.shape[1]] + [1] * (len(X.shape) - 1)
            )

    ssim_per_channel, cs = _ssim(
        X,
        Y,
        data_range=data_range,
        win=win,
        size_average=False,
        K=K,
        temporal_win=temporal_win,
        stride=stride,
    )
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    stride: int = 1,
) -> Tensor:
    r"""interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,D,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,D,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        stride (int, optional): stride for the convolution
    Returns:
        torch.Tensor: ms-ssim results
    """
    return _ms_ssim_wrapper(
        X=X,
        Y=Y,
        data_range=data_range,
        size_average=size_average,
        win_size=win_size,
        win_sigma=win_sigma,
        win=win,
        weights=weights,
        K=K,
        stride=stride,
    )


def st_ms_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    temporal_win: Optional[Tensor] = None,
    temporal_win_size: Optional[int] = None,
    stride: int = 1,
) -> Tensor:
    r"""interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,D,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,D,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        temporal_win (torch.Tensor, optional): 1-D gauss kernel for temporal dimension. if None, a new kernel will be created according to temporal_win_size and win_sigma
        temporal_win_size (int, optional): the size of gauss kernel for temporal dimension
        stride (int, optional): stride for the convolution
    Returns:
        torch.Tensor: ms-ssim results
    """
    return _ms_ssim_wrapper(
        X=X,
        Y=Y,
        data_range=data_range,
        size_average=size_average,
        win_size=win_size,
        win_sigma=win_sigma,
        win=win,
        weights=weights,
        K=K,
        temporal_win=temporal_win,
        temporal_win_size=temporal_win_size,
        stride=stride,
    )


def _ms_ssim_wrapper(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    temporal_win: Optional[Tensor] = None,
    temporal_win_size: Optional[int] = None,
    stride: int = 1,
) -> Tensor:
    r"""interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,D,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,D,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        temporal_win (torch.Tensor, optional): 1-D gauss kernel for temporal dimension. if None, a new kernel will be created according to temporal_win_size and win_sigma
        temporal_win_size (int, optional): the size of gauss kernel for temporal dimension
        stride (int, optional): stride for the convolution
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    # if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    elif len(X.shape) == 6:
        avg_pool = F_extra.avg_pool4d
    else:
        raise ValueError(f"Input images should be 4-d, 5-d or 6-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]
    if temporal_win is not None:  # set win_size
        temporal_win_size = temporal_win.shape[2]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[2 if temporal_win_size is None else 3 :])
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_tensor = X.new_tensor(weights)
    downsample = len(weights) - 1
    assert smaller_side > (win_size - 1) * (2**downsample), (
        "Image size should be larger than %d due to the %i downsamplings in ms-ssim"
        % ((win_size - 1) * (2**downsample), downsample)
    )
    if temporal_win_size is not None and X.shape[2] <= (temporal_win_size - 1) * (2**downsample):
        warnings.warn(
            "Temporal dimension size should be larger than %d due to the %i downsamplings in ms-ssim"
            % ((temporal_win_size - 1) * (2**downsample)),
            downsample,
        )

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights_tensor.shape[0]
    mcs = []
    max_temporal_downsample = (
        int(torch.log2(torch.tensor(X.shape[2] / (temporal_win_size - 1)))) if temporal_win_size is not None else 10000
    )  # just set to large number if not temporal
    # print(max_temporal_downsample)
    for i in range(levels):
        # don't calculate luminance when it is not used. Only calculate it for the last level.
        luminance = False
        if i == levels - 1:
            luminance = True
        ssim_per_channel, cs = _ssim(
            X,
            Y,
            win=win,
            data_range=data_range,
            size_average=False,
            K=K,
            temporal_win=temporal_win,
            luminance=luminance,
            stride=stride,
        )

        if i < levels - 1:
            if max_temporal_downsample < i + 1:
                stride = tuple([1] + [2] * (len(X.shape) - 3))
            else:
                stride = None
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, stride=stride, padding=padding)
            Y = avg_pool(Y, kernel_size=2, stride=stride, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


def _radial_mask(
    r,
    cx=128,
    cy=128,
    cz=None,
    sx=torch.arange(0, 256),
    sy=torch.arange(0, 256),
    sz=None,
    delta=1,
):
    if sz is None:
        ind = (sx[None, :] - cx) ** 2 + (sy[:, None] - cy) ** 2
    else:
        ind = (sx[None, None, :] - cx) ** 2 + (sy[:, None, None] - cy) ** 2 + (sz[None, :, None] - cz) ** 2
    ind1 = ind <= ((r[0] + delta) ** 2)  # one liner for this and below?
    ind2 = ind > (r[0] ** 2)
    return ind1 * ind2


def _create_radial_masks(shape: Tuple[int, ...], delta=1, dims: int = 2) -> torch.Tensor:
    """Create radial masks for Fourier Shell Correlation (FSC) calculation.

    Args:
        shape (tuple[int, ...]): Shape of the image.
        delta (int): Ring size in pixels. Defaults to 1.
        dims (int): Number of dimensions. Defaults to 2.

    Returns:
        torch.Tensor: Radial masks.
    """
    if shape[-1] != shape[-2]:
        raise NotImplementedError(f"Only square images are supported, but got {shape}.")
    freq_nyq = int(np.floor(int(shape[-1]) / 2.0))
    radii = torch.arange(freq_nyq).reshape(freq_nyq, 1)  # image size 256, binning = 3
    if dims == 2:
        sz = None
        cz = None
    elif dims == 3:
        sz = torch.arange(0, shape[-1])
        cz = freq_nyq
    else:
        raise ValueError(f"Only 2D and 3D images are supported, but got {dims}.")
    radial_masks = torch.stack(
        [
            _radial_mask(
                r=radii[i],
                cx=freq_nyq,
                cy=freq_nyq,
                cz=cz,
                sx=torch.arange(0, shape[-1]),
                sy=torch.arange(0, shape[-1]),
                sz=sz,
                delta=delta,
            )
            for i in range(radii.shape[0])
        ],
        dim=0,
    )
    radial_masks = radial_masks.unsqueeze(1).unsqueeze(1)
    return radial_masks


def fourier_shell_correlation(
    X: torch.Tensor,
    Y: torch.Tensor,
    radial_masks: Optional[torch.Tensor] = None,
    size_average: bool = True,
    delta: int = 1,
) -> Union[int, torch.Tensor]:
    """Calculate Fourier Shell Correlation (FSC) between two images.

    Args:
        X (torch.Tensor): First image. Shape (batch, channel, depth, height, width).
        Y (torch.Tensor): Second image. Must have the same shape as X.
        radial_masks (Optional[torch.Tensor]): Radial masks. Defaults to None. If None, radial masks are created with ring size of delta pixel.
        size_average (bool): If True, average FSC over r, batch and channel else averages over batch and channels. Defaults to True.
        delta (int): Ring size in pixels. Defaults to 1.

    Returns:
        Union[int, torch.Tensor]: FSC value(s)."""
    return _fourier_shell_ring_correlation(
        X,
        Y,
        radial_masks=radial_masks,
        size_average=size_average,
        dims=[-3, -2, -1],
        delta=delta,
    )


def fourier_ring_correlation(
    X: torch.Tensor,
    Y: torch.Tensor,
    radial_masks: Optional[torch.Tensor] = None,
    size_average: bool = True,
    delta: int = 1,
) -> Union[int, torch.Tensor]:
    """Calculate Fourier Ring Correlation (FRC) between two images.

    Args:
        X (torch.Tensor): First image. Shape (batch, channel, height, width).
        Y (torch.Tensor): Second image. Must have the same shape as X.
        radial_masks (Optional[torch.Tensor]): Radial masks. Defaults to None. If None, radial masks are created with ring size of 1 pixel.
        size_average (bool): If True, average FRC over r, batch and channel else averages over batch and channels. Defaults to True.
        delta (int): Ring size in pixels. Defaults to 1.

    Returns:
        Union[int, torch.Tensor]: FRC value(s)."""
    return _fourier_shell_ring_correlation(
        X,
        Y,
        radial_masks=radial_masks,
        size_average=size_average,
        dims=[-2, -1],
        delta=delta,
    )


def _fourier_shell_ring_correlation(
    X: torch.Tensor,
    Y: torch.Tensor,
    dims: List[int],
    radial_masks: Optional[torch.Tensor] = None,
    size_average: bool = True,
    delta: int = 1,
) -> Union[int, torch.Tensor]:
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")
    if radial_masks is None:
        radial_masks = _create_radial_masks(X.shape, delta=delta, dims=len(dims))
    F_1 = torch.fft.fftshift(
        torch.fft.fftn(X, dim=dims)
    )  # fourier transform of X shifted so that low frequencies are in the center
    F_2 = torch.fft.fftshift(torch.fft.fftn(Y, dim=dims))
    numerator = (F_1 * F_2.conj()).real  # F_1(r) * F_2(r)*
    denominator = torch.sqrt(torch.abs(F_1) ** 2 * torch.abs(F_2) ** 2)  # sqrt(|F_1(r)|^2 * |F_2(r)|^2)
    numerator = torch.sum(
        numerator * radial_masks, dim=dims
    )  # sum(F_1(r_i) * F_2(r_i)*) over r_i for all r_i in r for all r
    denominator = torch.sum(
        denominator * radial_masks, dim=dims
    )  # sum(sqrt(|F_1(r_i)|^2 * |F_2(r_i)|^2)) over r_i for all r_i in r for all r
    fsc = numerator / denominator  # gives FSC(r) for all r
    fsc = fsc.transpose(0, -1)  # move r to the end, new size (batch, channel, r)
    return (
        fsc.mean() if size_average else torch.mean(fsc, dim=[0, 1])
    )  # if size_average, average FSC(r) over r, batch and channel, else average over batch and channel
