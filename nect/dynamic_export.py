from __future__ import annotations
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tifffile as tif
import torch
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt

from nect.config import Config, get_cfg
from nect.utils import is_fourcc_available, setup_logger
from nect.data import NeCTDataset
from nect.sampling import Geometry

def get_number_of_projections_per_revolution(
    angles: list[float] | np.ndarray | torch.Tensor,
) -> int:
    projs_per_revolution = len(angles)
    for i in range(len(angles) - 2):
        angle_difference = (angles[i + 1] - angles[i]) % (2 * np.pi)
        next_angle_difference = (angles[i + 2] - angles[i + 1]) % (2 * np.pi)
        if abs(next_angle_difference - angle_difference) > 0.01:
            projs_per_revolution = i + 2
            break
    return projs_per_revolution


def get_scale(target_scale_mm):
    multiplier = 10 ** (-np.floor(np.log10(target_scale_mm)))
    target_scale_mm = round(target_scale_mm * multiplier)
    unit = "mm"
    target_scale = target_scale_mm
    eps = 1e-6
    if target_scale == 0:
        target_scale = 1
    elif target_scale == 3:
        target_scale = 2
    elif target_scale in [4, 6, 7]:
        target_scale = 5
    elif target_scale in [8, 9]:
        target_scale = 1
        multiplier /= 10

    if multiplier - eps > 1000:
        target_scale *= 1_000_000
        unit = "nm"
    elif multiplier + eps > 10:
        target_scale *= 1000
        unit = "μm"
    if multiplier + eps > 1:
        pass
    elif multiplier - eps <= 0.001:
        target_scale *= 0.001
        unit = "m"
    elif multiplier - eps <= 0.1:
        target_scale *= 0.1
        unit = "cm"
    target_scale = target_scale / multiplier
    if target_scale < 1:
        decimals = int(np.log10(multiplier)) - 6
        target_scale = round(target_scale, decimals)
    else:
        target_scale = int(target_scale)
    target_scale_mm = target_scale
    if unit == "nm":
        target_scale_mm /= 1_000_000
    elif unit == "μm":
        target_scale_mm /= 1000
    elif unit == "m":
        target_scale_mm *= 1000
    elif unit == "cm":
        target_scale_mm *= 10
    return target_scale, unit, target_scale_mm


def add_scale_bar_and_text(frame, time: float, config: Config, acquisition_time_minutes: float):
    acquisition_seconds = acquisition_time_minutes * 60
    frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (100, 100)
    fontScale = 2
    fontColor = (255, 255, 255)
    lineThickness = 4
    lineType = 1
    if time * acquisition_seconds // 60 >= 60:
        hours = (time * acquisition_seconds) // 3600
        minutes = ((time * acquisition_seconds) % 3600) // 60
        seconds = (time * acquisition_seconds) % 60
        text = f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"
    else:
        minutes = (time * acquisition_seconds) // 60
        seconds = (time * acquisition_seconds) % 60
        text = f"{int(minutes)}:{int(seconds):02}"
    cv2.putText(
        frame,
        text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineThickness,
        lineType,
    )
    # add scalebar to the image
    height, width = frame.shape
    TOTAL_WIDTH_MM = config.geometry.sVoxel[2]
    target_scale_mm = TOTAL_WIDTH_MM * 0.3
    target_scale, unit, target_scale_mm = get_scale(target_scale_mm)
    TARGET_SCALE_LENGTH_PX = width * (target_scale_mm / TOTAL_WIDTH_MM)  # Length of the scale bar in pixels

    SCALE_BAR_COLOR = (255, 255, 255)  # White color for the scale bar
    SCALE_BAR_THICKNESS = 6  # Thi

    # Position of the scale bar
    bar_x = width - 50 - int(TARGET_SCALE_LENGTH_PX)
    bar_y = int(0.9 * height)

    # Draw scale bar
    cv2.line(
        frame,
        (bar_x, bar_y),
        (bar_x + int(TARGET_SCALE_LENGTH_PX), bar_y),
        SCALE_BAR_COLOR,
        SCALE_BAR_THICKNESS,
    )

    # Add text indicating the scale
    cv2.putText(
        frame,
        f"{target_scale}{unit}",
        (bar_x + int(TARGET_SCALE_LENGTH_PX) // 4, bar_y + int(height / 20)),
        font,
        fontScale,
        fontColor,
        lineThickness,
        lineType,
    )

    return frame


def export_video(
    base_path: str | Path,
    add_scale_bar: bool = False,
    acquisition_time_minutes: float | None = None,
    plot_slice: str = "XZ",
    fps: int = 5,
    difference: bool = True,
    difference_revolutions: int = 1,
    video_name: str = "video",
) -> Path:
    """
    Exports a video of the dynamic model output. The video will be saved in the base_path directory.

    Args:
        base_path (str | Path): Path to the directory containing the config.yaml and checkpoints folder.
        add_scale_bar (bool, optional): Whether to add a scale bar to the video. Defaults to False.
        acquisition_time_minutes (float, optional): Acquisition time in minutes. Required if add_scale_bar is True. Defaults to None.
        plot_slice (str, optional): Slice to plot. Must be one of "XY", "XZ", "YZ". Defaults to "XZ".
        fps (int, optional): Frames per second. Defaults to 5.
        difference (bool, optional): Export a difference video. Defaults to True.
        difference_revolutions (int, optional): Number of revolutions for the background in the difference video. Defaults to 1.
        video_name (str, optional): Name of the video. Defaults to "video".

    Returns:
        Path to the saved video.
    """
    setup_logger()
    base_path = Path(base_path)
    with torch.no_grad():  # use torch.no_grad() to disable gradient computation and avoid retaining graph
        config = get_cfg(base_path / "config.yaml")
        assert config.geometry is not None
        model = config.get_model()
        device = torch.device(0)
        assert config.mode == "dynamic", "Only dynamic mode is supported for video creation"
        logger.info("Starting to load model")
        checkpoints = torch.load(base_path / "checkpoints" / "last.ckpt", map_location="cpu")
        video_path = base_path.parent / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
        model.load_state_dict(checkpoints["model"])
        model = model.to(device)
        logger.info("Model loading finished")
        height, width = config.geometry.nVoxel[0], config.geometry.nVoxel[1]
        if plot_slice.lower() not in ["xy", "xz", "yz"]:
            raise ValueError("Invalid plot_slice. Must be one of 'XY', 'XZ', 'YZ'")
        plot_slice = plot_slice.lower()
        if plot_slice == "xy":
            height = width
        z, y, x = torch.meshgrid(
            [
                torch.tensor(0.5, device=device)
                if plot_slice == "xy"
                else torch.linspace(0.0, 1.0, steps=height, device=device),
                torch.tensor(0.5, device=device)
                if plot_slice == "xz"
                else torch.linspace(0.0, 1.0, steps=width, device=device),
                torch.tensor(0.5, device=device)
                if plot_slice == "yz"
                else torch.linspace(0.0, 1.0, steps=width, device=device),
            ],
            indexing="ij",
        )
        grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t()
        angles = config.geometry.angles
        projs_per_revolution = get_number_of_projections_per_revolution(angles)

        # avc1 is not available on all systems, so we use mp4v as a fallback. avc1 is preferred because it uses the h.264 codec which can be viewed on most devices
        if is_fourcc_available("avc1"):
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if difference:
            avg_img = torch.zeros((height, width), device=device)
            n_percent = projs_per_revolution * difference_revolutions / len(angles)
            n_steps = 30
            for t in torch.linspace(0.00, n_percent, n_steps):
                avg_img += model(grid, t).view(height, width) / n_steps
            out_diff = cv2.VideoWriter(
                str(video_path / f"{video_name}_diff.mp4"),
                fourcc,
                fps,
                (width, height),
                isColor=False,
            )
            out_merge = cv2.VideoWriter(
                str(video_path / f"{video_name}_merge.mp4"),
                fourcc,
                fps,
                (width * 2, height),
                isColor=False,
            )
        out = cv2.VideoWriter(
            str(video_path / f"{video_name}.mp4"),
            fourcc,
            fps,
            (width, height),
            isColor=False,
        )

        for t in tqdm(torch.linspace(0, 1, len(angles))):
            output: torch.Tensor = model(grid, t).view(height, width)
            if difference:
                output_diff = output - avg_img
            output = output
            output = output / 3 * 255
            output = output.clamp(0, 255)
            output = output.cpu().numpy().astype(np.uint8)
            output = np.rot90(output, 2)
            if add_scale_bar:
                assert (
                    acquisition_time_minutes is not None
                ), "acquisition_time_minutes must be provided if add_scale_bar is True"
                output = add_scale_bar_and_text(
                    frame=output,
                    time=t.item(),
                    config=config,
                    acquisition_time_minutes=acquisition_time_minutes,
                )
            out.write(output)
            if difference:
                output_diff = output_diff + 4
                output_diff = output_diff / 6 * 255
                output_diff = output_diff.clamp(0, 255)

                output_diff = output_diff.cpu().numpy().astype(np.uint8)
                output_diff = np.rot90(output_diff, 2)
                if add_scale_bar:
                    assert (
                        acquisition_time_minutes is not None
                    ), "acquisition_time_minutes must be provided if add_scale_bar is True"
                    output_diff = add_scale_bar_and_text(
                        frame=output_diff,
                        time=t.item(),
                        config=config,
                        acquisition_time_minutes=acquisition_time_minutes,
                    )
                # add text to the image in the top left corner saying the time

                out_diff.write(output_diff)
                out_merge.write(np.concatenate([output, output_diff], axis=1))

        out.release()
        if difference:
            out_diff.release()
            out_merge.release()
    logger.info(f"Video saved to {base_path/f'{video_name}.mp4'}")
    if difference:
        logger.info(f"Difference video saved to {base_path/f'{video_name}_diff.mp4'}")
        logger.info(f"Merged video saved to {base_path/f'{video_name}_merge.mp4'}")
    return base_path / f"{video_name}.mp4"


def export_volumes(
    base_path: str | Path,
    binning: int = 1,
    avg_timesteps: int = 1,
    timesteps_per_revolution: int | Literal["all"] = "all",
    export_revolutions: list | str = "all",
    show_slices: bool = False,
    ROIx: list[int] | None = None,
    ROIy: list[int] | None = None,
    ROIz: list[int] | None = None,
    dtype: np.dtype = np.float32
) -> Path:
    """
    Exports volumes from the dynamic model output. The volumes will be saved in the base_path/volumes directory.

    Args:
        base_path (str | Path): Path to the directory containing the config.yaml and checkpoints folder.
        binning (int, optional): Binning factor. Defaults to 1.
        avg_timesteps (int, optional): Number of timesteps to average together. Defaults to 1.
        timesteps_per_revolution (int | Literal["all"], optional): Number of timesteps to export per revolution. Defaults to "all".
        export_revolutions (list | str, optional): List of revolutions to export. Defaults to "all".

    Returns:
        Path to the saved volumes.
    """
    setup_logger()
    base_path = Path(base_path)
    with torch.no_grad():  # use torch.no_grad() to disable gradient computation and avoid retaining graph
        config = get_cfg(base_path / "config.yaml")
        assert config.geometry is not None
        model = config.get_model()
        dataset = NeCTDataset(
            config=config,
            device="cpu",  # if gpu memory is less than 50 GB, load to cpu
        )
        geometry = Geometry.from_cfg(
            config.geometry,
            reconstruction_mode=config.reconstruction_mode,
            sample_outside=config.sample_outside,
        )
        device = torch.device(0)
        checkpoints = torch.load(base_path / "checkpoints" / "last.ckpt", map_location="cpu")
        model.load_state_dict(checkpoints["model"])
        model = model.to(device)
        assert config.mode == "dynamic", "Only dynamic mode is supported for video creation"
        height, width = config.geometry.nVoxel[0], config.geometry.nVoxel[1]
        z_h = height // binning
        y_w = width // binning
        x_w = width // binning
        base_output_path = base_path / "volumesfloat32"
        base_output_path.mkdir(exist_ok=True, parents=True)
        angles = config.geometry.angles
        linspace = torch.linspace(0, 1, steps=len(angles), device=device)
        projs_per_revolution = get_number_of_projections_per_revolution(angles)
        if export_revolutions == "all":
            export_revolutions = [i for i in range(len(angles) // projs_per_revolution)]
        if avg_timesteps > 1:
            logger.info(f"Averaging {avg_timesteps} timesteps together")
        total_volumes_saved = 0
        nVoxels = config.geometry.nVoxel
        rm = config.sample_outside
        nVoxels = [nVoxels[0], nVoxels[1]+2*rm, nVoxels[2]+2*rm]
        start_x = 0
        end_x = 1
        if ROIx is not None:
            start_x = (ROIx[0] - rm) / nVoxels[2]
            end_x = (ROIx[1] - rm) / nVoxels[2]
            x_w = (ROIx[1]-ROIx[0]) // binning
            
        start_y = 0
        end_y = 1
        if ROIy is not None:
            start_y = (ROIy[0] - rm) / nVoxels[1]
            end_y = (ROIy[1] - rm) / nVoxels[1]
            y_w = (ROIy[1]-ROIy[0]) // binning
            
        start_z = 0
        end_z = 1
        if ROIz is not None:
            start_z = (ROIz[0]) / nVoxels[0]
            end_z = (ROIz[1]) / nVoxels[0]
            z_h = (ROIz[1]-ROIz[0]) // binning
        if show_slices:
            for slice_idx in ["z", "y", "x"]:
                if slice_idx == "z":
                    size = (y_w, x_w)
                elif slice_idx == "y":
                    size = (z_h, x_w)
                elif slice_idx == "x":
                    size = (z_h, y_w)
                default_tensor = torch.tensor(0.5, device=device)
                z_l = torch.linspace(start_z, end_z, steps=z_h, device=device) if slice_idx != "z" else default_tensor
                y_l = torch.linspace(start_y, end_y, steps=y_w, device=device) if slice_idx != "y" else default_tensor
                x_l = torch.linspace(start_x, end_x, steps=x_w, device=device) if slice_idx != "x" else default_tensor
                z, y, x = torch.meshgrid([z_l, y_l, x_l], indexing="ij")
                grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t()
                output = model(grid, torch.tensor(0.5, device=device)).view(size).cpu().numpy()
                plt.imshow(output, cmap="gray")
                (base_path / "imgs").mkdir(parents=True, exist_ok=True)
                plt.savefig(base_path / "imgs" / f"{slice_idx}.png")
            return base_path / "imgs"
                
        else: 
            for i in tqdm(export_revolutions, leave=True, desc="Exporting revolutions"):
                revolution_start = i * projs_per_revolution
                revolution_end = (i + 1) * projs_per_revolution
                if timesteps_per_revolution != "all":
                                    
                    sub_linspace = torch.linspace(linspace[revolution_start], linspace[revolution_end], timesteps_per_revolution)  # skip timesteps to speed up export
                else:
                    sub_linspace = linspace[revolution_start:revolution_end]
                for j, t in tqdm(
                    enumerate(sub_linspace),
                    total=len(sub_linspace),
                    leave=False,
                    desc="Projection",
                ):
                    output = torch.zeros((z_h, y_w, x_w), device=device)
                    for avg in range(avg_timesteps):
                        for ii, z_ in enumerate(
                            torch.linspace(start_z, end_z, steps=z_h, device=device)
                        ):  # progress through as we don't have enough memory to compute all at once
                            z, y, x = torch.meshgrid(
                                [
                                    z_,
                                    torch.linspace(start_y, end_y, steps=y_w, device=device),
                                    torch.linspace(start_x, end_x, steps=x_w, device=device),
                                ],
                                indexing="ij",
                            )
                            grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t()
                            output[ii] += model(grid, t + avg/len(angles)).view(y_w, x_w) / avg_timesteps
                    output = output / geometry.max_distance_traveled
                    output = output * (dataset.maximum.item() - dataset.minimum.item()) + dataset.minimum.item()
                    output = output.cpu().numpy()
                    output = output.astype(dtype)
                    if timesteps_per_revolution == "all":
                        proj_n = j
                    else:
                        proj_n = ((j * projs_per_revolution)//timesteps_per_revolution)
                    tif.imsave(base_output_path / f"T{i:04}_{proj_n:04}.tiff", output)
                    total_volumes_saved += 1
            logger.info(f"{total_volumes_saved} volumes saved to {base_output_path}")
            return base_output_path

def export_volumes_at_timesteps(
    base_path: str | Path,
    timesteps: list[float],
    binning: int = 1,
    ROIx: list[int] | None = None,
    ROIy: list[int] | None = None,
    ROIz: list[int] | None = None,
    dtype: np.dtype = np.float32
) -> Path:
    """
    Exports volumes from the dynamic model output. The volumes will be saved in the base_path/volumes directory.

    Args:
        base_path (str | Path): Path to the directory containing the config.yaml and checkpoints folder.
        timesteps (list[float]): List of timesteps to export.
        binning (int, optional): Binning factor. Defaults to 1.
        ROIx (list[int] | None, optional): Region of interest in the x direction. Defaults to None.
        ROIy (list[int] | None, optional): Region of interest in the y direction. Defaults to None.
        ROIz (list[int] | None, optional): Region of interest in the z direction. Defaults to None.
        dtype (np.dtype, optional): Data type of the exported volumes. Defaults to np.float32.
        
    Returns:
        Path to the saved volumes.
    """
    setup_logger()
    base_path = Path(base_path)
    with torch.no_grad():  # use torch.no_grad() to disable gradient computation and avoid retaining graph
        config = get_cfg(base_path / "config.yaml")
        assert config.geometry is not None
        model = config.get_model()
        dataset = NeCTDataset(
            config=config,
            device="cpu",  # if gpu memory is less than 50 GB, load to cpu
        )
        geometry = Geometry.from_cfg(
            config.geometry,
            reconstruction_mode=config.reconstruction_mode,
            sample_outside=config.sample_outside,
        )
        device = torch.device(0)
        checkpoints = torch.load(base_path / "checkpoints" / "last.ckpt", map_location="cpu")
        model.load_state_dict(checkpoints["model"])
        model = model.to(device)
        assert config.mode == "dynamic", "Only dynamic mode is supported for video creation"
        height, width = config.geometry.nVoxel[0], config.geometry.nVoxel[1]
        z_h = height // binning
        y_w = width // binning
        x_w = width // binning
        base_output_path = base_path / "volumesfloat32"
        base_output_path.mkdir(exist_ok=True, parents=True)
        angles = config.geometry.angles
        linspace = torch.linspace(0, 1, steps=len(angles), device=device)
        projs_per_revolution = get_number_of_projections_per_revolution(angles)
        total_volumes_saved = 0
        nVoxels = config.geometry.nVoxel
        rm = config.sample_outside
        nVoxels = [nVoxels[0], nVoxels[1]+2*rm, nVoxels[2]+2*rm]
        start_x = 0
        end_x = 1
        if ROIx is not None:
            start_x = (ROIx[0] - rm) / nVoxels[2]
            end_x = (ROIx[1] - rm) / nVoxels[2]
            x_w = (ROIx[1]-ROIx[0]) // binning
            
        start_y = 0
        end_y = 1
        if ROIy is not None:
            start_y = (ROIy[0] - rm) / nVoxels[1]
            end_y = (ROIy[1] - rm) / nVoxels[1]
            y_w = (ROIy[1]-ROIy[0]) // binning
            
        start_z = 0
        end_z = 1
        if ROIz is not None:
            start_z = (ROIz[0]) / nVoxels[0]
            end_z = (ROIz[1]) / nVoxels[0]
            z_h = (ROIz[1]-ROIz[0]) // binning
        if max(timesteps) > 1:
            timesteps = [t/len(angles) for t in timesteps]
        for t in tqdm(timesteps, leave=True, desc="Exporting revolutions"):
            output = torch.zeros((z_h, y_w, x_w), device=device)
            for ii, z_ in enumerate(
                torch.linspace(start_z, end_z, steps=z_h, device=device)
            ):  # progress through as we don't have enough memory to compute all at once
                z, y, x = torch.meshgrid(
                    [
                        z_,
                        torch.linspace(start_y, end_y, steps=y_w, device=device),
                        torch.linspace(start_x, end_x, steps=x_w, device=device),
                    ],
                    indexing="ij",
                )
                grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t()
                output[ii] = model(grid, t).view(y_w, x_w)
            output = output / geometry.max_distance_traveled
            output = output * (dataset.maximum.item() - dataset.minimum.item()) + dataset.minimum.item()
            output = output.cpu().numpy()
            output = output.astype(dtype)
                
            tif.imsave(base_output_path / f"T{t:04}.tiff", output)
            total_volumes_saved += 1
        logger.info(f"{total_volumes_saved} volumes saved to {base_output_path}")
        return base_output_path
