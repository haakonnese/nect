import numpy as np
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from torch.optim import Adam

from torch_extra.nn import MS_SSIM, ST_MS_SSIM


def test_st_ms_ssim():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    image = shepp_logan_phantom()

    # use float64 to avoid allclose error
    # must use at least 161x161x161 to avoid MS scaling error
    h_w_d = 162
    video = np.zeros((h_w_d, h_w_d, h_w_d), dtype=np.float64)

    # create a 3D image that is a 2D image with different scales
    for i in range(h_w_d):
        vid_i = rescale(image, scale=0.21 + round(i * 0.001, 2), mode="reflect", channel_axis=None)
        video[i] = np.pad(
            vid_i,
            ((h_w_d - vid_i.shape[0]) // 2, (h_w_d - vid_i.shape[1]) // 2),
            "constant",
            constant_values=(0, 0),
        )
    video_with_noise = video + np.random.normal(0, 0.1, size=video.shape)

    video = torch.from_numpy(video).unsqueeze(0).unsqueeze(0).to(device).clone().requires_grad_()
    video_2_plus_1 = torch.from_numpy(video_with_noise).unsqueeze(0).unsqueeze(0).to(device).clone().requires_grad_()
    image_3d = torch.from_numpy(video_with_noise).unsqueeze(0).unsqueeze(0).to(device).clone().requires_grad_()

    assert video_2_plus_1.shape == (1, 1, h_w_d, h_w_d, h_w_d)
    assert video_2_plus_1.size() == image_3d.size()
    assert video_2_plus_1.dtype == torch.float64

    optimizer_video = Adam([video_2_plus_1], lr=0.1)
    optimizer_3d_image = Adam([image_3d], lr=0.1)
    ms_ssim_func = MS_SSIM(data_range=1.0, spatial_dims=3, channel=1)
    st_ms_ssim_func = ST_MS_SSIM(data_range=1.0, spatial_dims=2, temporal_win_size=11, channel=1)

    optimizer_video.zero_grad()
    optimizer_3d_image.zero_grad()
    ms_ssim = ms_ssim_func(video, image_3d)
    st_ms_ssim = st_ms_ssim_func(video, video_2_plus_1)
    assert torch.isclose(ms_ssim, st_ms_ssim)
    ms_ssim.backward()
    st_ms_ssim.backward()
    optimizer_video.step()
    optimizer_3d_image.step()
    assert torch.allclose(
        video_2_plus_1, image_3d, atol=1e-6
    )  # atol=1e-6 to avoid allclose error due to floating point precision.
