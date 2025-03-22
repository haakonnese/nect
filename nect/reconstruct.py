from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal, cast, overload

import numpy as np
import torch
from loguru import logger

from nect.config import (
    get_cfg,
    get_dynamic_cfg,
    get_static_cfg,
    setup_cfg,
)
from nect.sampling.geometry import Geometry
from nect.trainers import BaseTrainer, ProjectionsLoadedTrainer

_list = list[float] | np.ndarray | torch.Tensor


@overload
def reconstruct(
    geometry: Geometry,
    projections: str | Path | torch.Tensor | np.ndarray,
    mode: Literal["static"] = ...,
    angles: _list | None = None,
    radians: bool = True,
    quality: Literal["poor", "low", "medium", "high", "higher", "highest"] = "high",
    niter: int | None = None,
    lr: float | None = None,
    timesteps: _list | None = None,
    verbose: bool = True,
    log: bool = False,
    exp_name: str | None = None,
    flip_projections: bool = False,
    channel_order: str | None = None,
    config_override: dict | None = None,
) -> np.ndarray:
    """
    Create a 3D reconstruction from a set of 2D projections.

    Args:
        geometry (Geometry): Geometry object containing the acquisition geometry.
        projections (str | Path | torch.Tensor | np.ndarray): Path to the projections file or folder or the projections themselves.
        mode (Literal["static", "dynamic"]): "static" for 3D reconstruction.
        angles (list | np.ndarray | torch.Tensor | None): Projection angles in radians. If None, the angles from the geometry object are used.
        radians (bool): If angles are provided, this defines the angle type either "radians" (`True`) or "degrees" (`False`). If angles are None, it is ignored. Defaults to `True` (radians).
        quality (Literal["poor", "low", "medium", "high", "higher", "highest"]): Time used for reconstructing and getting approximately the given quality
        niter (int | str | None): Override the number of iterations from quality.
        lr (float | None): Learning rate. Default depends on reconstruction type.
        timesteps (list | np.ndarray | torch.Tensor | None): An array of timesteps. Do not need to be normalized.
            If the order of the angles and corresponding projections does not equal the acqustition order, this parameter needs to be set to get the timesteps correct.
            Only important for dynamic reconstruction. Overrides the timestep of the Geometry if not None.
        verbose (bool): Verbosity. Default is True.
        flip_projections (bool): Flip the projections. Default is False.
        channel_order (str | None): Channel order. This is only used if the projections are a path to file ("NHW", "NWH") or files ("HW" or "WH").
        config_override (dict | None): Dictionary containing the configuration that overrides all the other arguments.

    Returns:
        A np.ndarray of the reconstructed object.
    """
    ...


@overload
def reconstruct(
    geometry: Geometry,
    projections: str | Path | torch.Tensor | np.ndarray,
    mode: Literal["dynamic"] = ...,
    angles: _list | None = None,
    radians: bool = True,
    quality: Literal["poor", "low", "medium", "high", "higher", "highest"] = "high",
    niter: int | None = None,
    lr: float | None = None,
    timesteps: _list | None = None,
    verbose: bool = True,
    log: bool = False,
    exp_name: str | None = None,
    flip_projections: bool = False,
    channel_order: str | None = None,
    config_override: dict | None = None,
) -> Path:
    """
    Create a 4D-CT reconstruction from a set of 2D projections.

    Args:
        geometry (Geometry): Geometry object containing the acquisition geometry.
        projections (str | Path | torch.Tensor | np.ndarray): Path to the projections file or folder or the projections themselves.
        mode (Literal["static", "dynamic"]): "dynamic" for 4D-CT reconstruction.
        angles (list | np.ndarray | torch.Tensor | None): Projection angles in radians. If None, the angles from the geometry object are used.
        radians (bool): If angles are provided, this defines the angle type either "radians" (`True`) or "degrees" (`False`). If angles are None, it is ignored. Defaults to `True` (radians).
        quality (Literal["poor", "low", "medium", "high", "higher", "highest"]): Time used for reconstructing and getting approximately the given quality
        niter (int | str | None): Override the number of iterations from quality.
        lr (float | None): Learning rate. Default depends on reconstruction type.
        timesteps (list | np.ndarray | torch.Tensor | None): An array of timesteps. Do not need to be normalized.
            If the order of the angles and corresponding projections does not equal the acqustition order, this parameter needs to be set to get the timesteps correct.
            Only important for dynamic reconstruction. Overrides the timestep of the Geometry if not None.
        verbose (bool): Verbosity. Default is True.
        flip_projections (bool): Flip the projections. Default is False.
        channel_order (str | None): Channel order. This is only used if the projections are a path to file ("NHW", "NWH") or files ("HW" or "WH").
        config_override (dict | None): Dictionary containing the configuration that overrides all the other arguments.

    Returns:
        The path to the reconstruction directory.
    """
    ...


def reconstruct(
    geometry: Geometry,
    projections: str | Path | torch.Tensor | np.ndarray,
    mode="static",
    angles: _list | None = None,
    radians: bool = True,
    quality: Literal["poor", "low", "medium", "high", "higher", "highest"] = "high",
    niter: int | None = None,
    lr: float | None = None,
    timesteps: _list | None = None,
    verbose: bool = True,
    log: bool = False,
    exp_name: str | None = None,
    flip_projections: bool = False,
    channel_order: str | None = None,
    config_override: dict | None = None,
    save_ckpt: bool = False
) -> np.ndarray | Path:
    """
    Create a 3D or 4D-CT reconstruction from a set of 2D projections.

    Args:
        geometry (Geometry): Geometry object containing the acquisition geometry.
        projections (str | Path | torch.Tensor | np.ndarray): Path to the projections file or folder or the projections themselves.
        mode (Literal["static", "dynamic"]): "dynamic" for 4D-CT reconstruction and "static" for 3D.
        angles (list | np.ndarray | torch.Tensor | None): Projection angles in radians. If None, the angles from the geometry object are used.
        radians (bool): If angles are provided, this defines the angle type either "radians" (`True`) or "degrees" (`False`). If angles are None, it is ignored. Defaults to `True` (radians).
        quality (Literal["poor", "low", "medium", "high", "higher", "highest"]): Time used for reconstructing and getting approximately the given quality
        niter (int | str | None): Override the number of iterations from quality.
        lr (float | None): Learning rate. Default depends on reconstruction type.
        timesteps (list | np.ndarray | torch.Tensor | None): An array of timesteps. Do not need to be normalized.
            If the order of the angles and corresponding projections does not equal the acqustition order, this parameter needs to be set to get the timesteps correct.
            Only important for dynamic reconstruction. Overrides the timestep of the Geometry if not None.
        verbose (bool): Verbosity. Default is True.
        flip_projections (bool): Flip the projections. Default is False.
        channel_order (str | None): Channel order. This is only used if the projections are a path to file ("NHW", "NWH") or files ("HW" or "WH").
        config_override (dict | None): Dictionary containing the configuration that overrides all the other arguments.

    Returns:
        The path to the reconstruction directory if mode is dynamic. If static, a np.ndarray is returned.
    """
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
        level="INFO" if verbose else "WARNING",
    )

    if mode == "static":
        cfg = get_static_cfg(name="hash_grid")
    elif mode == "dynamic":
        cfg = get_dynamic_cfg(name="quadcubes")
    if channel_order is not None:
        cfg["channel_order"] = channel_order
    cfg["flip"] = flip_projections
    if isinstance(projections, (str, Path)):
        cfg["img_path"] = projections
    else:
        cfg["img_path"] = "RECONSTRUCTING_FROM_ARRAY"
    if quality in ["poor", "low"]:
        cfg["loss"] = "L2"
        cfg["encoder"]["log2_hashmap_size"] = 19
        if quality == "poor":
            cfg["epochs"] = "0.02x"
            cfg["base_lr"] *= 10
        elif quality == "low":
            cfg["epochs"] = "0.1x"
            cfg["base_lr"] *= 5
    elif quality == "medium":
        cfg["epochs"] = "0.3x"
        cfg["base_lr"] *= 2
    elif quality == "high":
        cfg["epochs"] = "1x"
        cfg["base_lr"] /= 2
        cfg["warmup"]["steps"] = 500
    elif quality in ["higher", "highest"]:
#        cfg["net"]["n_neurons"] = 64
#        cfg["net"]["n_hidden_layers"] = 6
        if quality == "higher":
            cfg["epochs"] = "2x"
            cfg["base_lr"] /= 2
            cfg["warmup"]["steps"] = 3000
            cfg["lr_scheduler"]["lrf"] = 0.05
            cfg["points_per_ray"]["end"] = "1.5x"
        elif quality == "highest":
            cfg["epochs"] = "4x"
            cfg["base_lr"] /= 5
            cfg["warmup"]["steps"] = 5000
            cfg["lr_scheduler"]["lrf"] = 0.01
            cfg["points_per_ray"]["end"] = "1.5x"
            cfg["encoder"]["log2_hashmap_size"] = 23
    if niter is not None:
        cfg["epochs"] = niter
    if lr is not None:
        cfg["base_lr"] = lr
    if angles is not None:
        geometry.set_angles(angles, radians)
    elif geometry.angles is None:
        raise ValueError("angles must be provided, either as an argument or in the `Geometry` object.")
    if timesteps is not None:
        geometry.set_timesteps(timesteps)
    if config_override is not None:
        cfg.update(config_override)
    cfg["geometry"] = geometry.to_dict()
    config = setup_cfg(cfg)
    if exp_name is None:
        log_path = Path("outputs")
    else:
        log_path = Path("outputs") / exp_name
    if mode == "dynamic":
        log = True
    if isinstance(projections, (str, Path)):
        trainer = BaseTrainer(
            config=config,
            output_directory=log_path if log else None,
            save_ckpt=save_ckpt,
            save_last=False if mode == "static" else True,
            save_optimizer=False,
            verbose=verbose,
            log=log,
        )
    else:
        trainer = ProjectionsLoadedTrainer(
            config=config,
            projections=projections,
            output_directory=log_path if log else None,
            save_ckpt=save_ckpt,
            save_last=False if mode == "static" else True,
            save_optimizer=False,
            verbose=verbose,
            log=log,
        )
    trainer.fit()
    if mode == "static":
        return torch.rot90(cast(torch.Tensor, trainer.create_volume(save=False, cpu=True)), 2, (1, 2)).cpu().numpy()
    else:
        return Path(trainer.checkpoint_directory_base).parent


def reconstruct_from_config_file(
    cfg: str | Path,
    log_path: str = "outputs",
    save_ckpt: bool = True,
    checkpoint: str | None = None,
    save_volume: bool = False,
    save_last: bool = True,
    save_optimizer: bool = True,
    cancel_at: str | None = None,
    prune: bool = True,
    keep_two: bool = True
):
    """
    Create a 3D or 4D-CT reconstruction from a set of 2D projections from config file.

    Args:
        cfg (str | Path): Path to the configuration file.
        log_path (str): Base path to the output directory.
        save_ckpt (bool): Save the checkpoint during reconstruction.
        checkpoint (str | None): Load a checkpoint file to continue training.
        save_volume (bool): Save the final volume to a file.
        save_last (bool): Save the last checkpoint.
        save_optimizer (bool): Save the optimizer state.
        cancel_at (str, optional): Cancel training at the specified ISO-datetime. Save the model before canceling.
        prune (bool): Prune the model to remove the optimizer.
    """
    config_file_path_splitted = str(cfg).split(os.sep)
    config = get_cfg(cfg)

    output_folder = os.path.join(*config_file_path_splitted[-2:-1])
    exp_name = os.path.join(log_path, output_folder, config.mode)
    if checkpoint:
        exp_name = Path(checkpoint).parent.parent.parent
    trainer = BaseTrainer
    if config.evaluation is not None:
        if config.evaluation.gt_path_mode.upper() == "SCIVIS":
            from nect.trainers.scivis_trainer import SciVisTrainer

            trainer = SciVisTrainer
        elif config.evaluation.gt_path_mode.upper() == "POROUSMEDIUM":
            from nect.trainers.porous_medium_trainer import PorousMediumTrainer

            trainer = PorousMediumTrainer
    if config.continous_scanning is True:
        from nect.trainers.continous_scanning_trainer import ContinousScanningTrainer

        trainer = ContinousScanningTrainer
    trainer = trainer(
        config=config,
        checkpoint=checkpoint,
        output_directory=exp_name,
        save_ckpt=save_ckpt,
        save_last=save_last,
        save_optimizer=save_optimizer,
        verbose=True,
        cancel_at=cancel_at,
        prune=prune,
        keep_two=keep_two
    )
    if save_volume:
        trainer.save_volume()
    else:
        trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file.")
    parser.add_argument("--log_path", type=str, default="outputs", help="outputs path")
    parser.add_argument("--no-save", action="store_true", help="outputs path")
    parser.add_argument("--ckpt", type=str, help="load checkpoint file to continue training")
    parser.add_argument("--ckpt-prior", type=str, help="load checkpoint file for prior optimization")
    parser.add_argument("--save-volume", action="store_true", help="Save the final volume to a file")
    parser.add_argument("--no-save-last", action="store_true", help="Save the last checkpoint")
    parser.add_argument("--no-prune", action="store_true", help="Save the last checkpoint")
    parser.add_argument("--no-keep-two", action="store_true", help="Save the last checkpoint")
    parser.add_argument(
        "--no-save-optimizer",
        action="store_true",
        help="Do not save the optimizer state",
    )
    parser.add_argument("--cancel-at", type=str, help="Save the model weights at the specified time")

    # Load experiment setting
    args = parser.parse_args()
    reconstruct_from_config_file(
        cfg=args.config,
        log_path=args.log_path,
        save_volume=args.save_volume,
        checkpoint=args.ckpt,
        save_ckpt=not args.no_save,
        save_last=not args.no_save_last,
        save_optimizer=not args.no_save_optimizer,
        cancel_at=args.cancel_at,
        prune=not args.no_prune,
        keep_two=not args.no_keep_two
    )
