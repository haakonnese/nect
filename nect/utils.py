from __future__ import annotations

import logging
import os
import sys

import cv2
import torch
import yaml
from loguru import logger
import shutil
from pathlib import Path

import torch
from nect.config import get_cfg

def setup_logger(level=logging.INFO):
    """Set up the logger.

    Args:
        level (int, optional): The logging level. Defaults to logging.INFO.
    """
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
        level=level,
    )


def load_config(config) -> dict:
    """Load the configuration file.

    Args:
        config (str): The path to the configuration file.

    Returns:
        dict: The loaded configuration.
    """
    with open(config, "r") as f:
        return yaml.safe_load(f)


def create_sub_folders(output_directory) -> tuple[str, str]:
    """Create the sub-folders for the output directory.

    Args:
        output_directory (str): The output directory.

    Returns:
        Tuple[str, str]: The checkpoint and image directories.
    """
    logger.info(f"Using output directory: {output_directory}")
    checkpoint_directory = os.path.join(output_directory, "model", "checkpoints")
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)
    image_directory = os.path.join(output_directory, "images")
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
    return checkpoint_directory, image_directory


def total_variation_3d(input, weight: float = 1.0) -> torch.Tensor:
    """Calculate the total variation in 3D.

    Args:
        input (torch.Tensor): The input tensor. It may have an arbitrary number of dimensions.
        weight (float, optional): The weight for the total variation. Defaults to 1.0.

    Returns:
        torch.Tensor: The total variation.
    """
    diff_i = input[..., 1:, :, :] - input[..., :-1, :, :]
    diff_j = input[..., :, 1:, :] - input[..., :, :-1, :]
    diff_k = input[..., :, :, 1:] - input[..., :, :, :-1]
    tv = (torch.sum(torch.abs(diff_i)) + torch.sum(torch.abs(diff_j)) + torch.sum(torch.abs(diff_k))) / (
        input.size(0) * input.size(1) * input.size(2)
    )
    return tv * weight


def is_fourcc_available(codec) -> bool:
    """
    Check if the codec is available for video writing.

    Args:
        codec (str): Codec to check.

    Returns:
        True if the codec is available, False otherwise.
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        temp_video = cv2.VideoWriter("temp.mp4", fourcc, 30, (640, 480), isColor=False)
        is_open = temp_video.isOpened()
        temp_video.release()
        os.remove("temp.mp4")
        return is_open
    except Exception as e:
        logger.warning(e)
        return False


def prune_from_path(base_path: str | Path):
    """
    Prunes model to remove optimizer.

    Args:
        base_path (str | Path): Path to the directory containing the config.yaml and checkpoints folder.

    """
    base_path = Path(base_path)
    config = get_cfg(base_path / "config.yaml")
    assert config.geometry is not None
    setup_logger()
    logger.info("LOADING CHECKPOINT. This might take a while...")
    checkpoints = torch.load(base_path / "checkpoints" / "last.ckpt", map_location="cpu")
    state = {
        "model": checkpoints["model"],
    }
    pruned_path = base_path.parent / "pruned"
    model_path = pruned_path / "checkpoints" / "last.ckpt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("FINISHED LOADING. Starting to export pruned model...")
    
    torch.save(state, model_path)
    shutil.copy(base_path / "config.yaml", pruned_path / "config.yaml")
    shutil.copy(base_path / "geometry.yaml", pruned_path / "geometry.yaml")
    logger.info(f"Pruned model saved to {pruned_path}")
    
def prune_model(model: torch.nn.Module,
                base_path: str | Path):
    """
    Prunes model to remove optimizer.
    
    Args:
        model (torch.nn.Module): Model to prune.
        path (str | Path): Path to save the pruned model.
    """
    state = {
        "model": model.state_dict(),
    }
    base_path = Path(base_path)
    pruned_path = base_path.parent / "pruned"
    model_path = pruned_path / "checkpoints" / "last.ckpt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, model_path)
    shutil.copy(base_path / "config.yaml", pruned_path / "config.yaml")
    shutil.copy(base_path / "geometry.yaml", pruned_path / "geometry.yaml")
    