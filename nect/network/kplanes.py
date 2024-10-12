import itertools
from typing import Collection, Iterable, Optional, Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F
from nect.config import KPlanesEncoderConfig


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(
            f"Grid-sample was called with {grid_dim}D data but is only " f"implemented for 2 and 3D data."
        )

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode="bilinear",
        padding_mode="border",
    )
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp


def init_grid_param(
    grid_nd: int,
    in_dim: int,
    out_dim: int,
    reso: Sequence[int],
    a: float = 0.1,
    b: float = 0.5,
):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(
    pts: torch.Tensor,
    ms_grids: Collection[Iterable[nn.Module]],
    grid_dimensions: int,
    concat_features: bool,
    num_levels: Optional[int],
) -> torch.Tensor:
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.0
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.0
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feature_dim)
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


def regularize_k_planes(config: KPlanesEncoderConfig, model):
    reg_time = config.regularization.time_type
    loss = 0
    loss += config.regularization.space_lambda * plane_tv_regularization(model)
    if reg_time is not None:
        if reg_time.casefold() == "smoothness".casefold():
            loss += config.regularization.time_lambda * time_smoothness_regularization(model)
        elif reg_time.casefold() == "l1".casefold():
            loss += config.regularization.time_lambda * time_tv_l1_regularization(model)
        else:
            raise NotImplementedError(f"Time regularization {reg_time} is not implemented.")


def plane_tv_regularization(model):
    multi_res_grids = model.grids
    total = 0
    # Note: input to compute_plane_tv should be of shape [batch_size, c, h, w]
    for grids in multi_res_grids:
        if len(grids) == 3:
            spatial_grids = [0, 1, 2]
        else:
            spatial_grids = [
                0,
                1,
                3,
            ]  # These are the spatial grids; the others are spatiotemporal
        for grid_id in spatial_grids:
            total += compute_plane_tv(grids[grid_id])
        for grid in grids:
            # grid: [1, c, h, w]
            total += compute_plane_tv(grid)
    return total


def time_tv_l1_regularization(model):
    multi_res_grids = model.grids
    total = 0.0
    for grids in multi_res_grids:
        if len(grids) == 3:
            continue
        else:
            # These are the spatiotemporal grids
            spatiotemporal_grids = [2, 4, 5]
        for grid_id in spatiotemporal_grids:
            total += torch.abs(1 - grids[grid_id]).mean()
    return torch.as_tensor(total)


def time_smoothness_regularization(model):
    multi_res_grids = model.grids
    total = 0
    # model.grids is 6 x [1, rank * F_dim, reso, reso]
    for grids in multi_res_grids:
        if len(grids) == 3:
            time_grids = []
        else:
            time_grids = [2, 4, 5]
        for grid_id in time_grids:
            total += compute_plane_smoothness(grids[grid_id])
    return torch.as_tensor(total)


def compute_plane_tv(t):
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)  # This is summing over batch and c instead of avg


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., : h - 2, :]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()
