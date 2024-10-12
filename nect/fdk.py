from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from nect.config import Config
from nect.config import Geometry as ConfigGeometry
from nect.config import GeometryCone as ConfigGeometryCone
from nect.data import NeCTDataset


# from nect.sampling import Geometry
class Geometry:
    pass


if TYPE_CHECKING:
    import tigre


def fdk(
    projections: torch.Tensor | np.ndarray,
    geometry: Geometry | ConfigGeometry | ConfigGeometryCone | tigre.geometry.Geometry,
    angles: list[float] | np.ndarray | torch.Tensor,
) -> np.ndarray:
    """
    Reconstruct object using FDK

    Args:
        projections (np.ndarray, torch.Tensor): The projections for reconstruction with shape (nProjections, height, width). The order of the projections must match the order of the angles.
        geometry (nect.Geometry, nect.config.Geometry, nect.config.GeometryCone, tigre.geometry.Geometry): Geometry of the system
        angles (list[float], np.ndarray, torch.Tensor): The acqusition angles. Must match the order of the projections.

    Returns:
        np.ndarray: The reconstructed volume
    """
    import tigre

    if isinstance(angles, list):
        angles = np.array(angles, dtype=np.float32)
    elif isinstance(angles, torch.Tensor):
        angles = angles.detach().cpu().numpy().astype(np.float32)
    argsort_idx = np.argsort(angles)
    angles = np.copy(angles[argsort_idx])
    if isinstance(projections, torch.Tensor):
        projections = projections.detach().cpu().numpy()
    projs = np.copy(projections[argsort_idx])
    if isinstance(geometry, (Geometry, ConfigGeometryCone, ConfigGeometry)):
        geo = tigre_geometry_from_geometry(geometry)
    elif isinstance(geometry, tigre.geometry.Geometry):
        geo = geometry
    else:
        raise NotImplementedError(
            f"geometry must be of type 'nect.config.Geometry', 'nect.config.GeometryCone' or 'tigre.geometry.Geometry', got {type(geometry)}"
        )
    volume = tigre.algorithms.fdk(projs, geo, angles)
    return volume


def fdk_from_config(config: Config, output_directory: str | Path | None = None) -> np.ndarray:
    """
    Reconstruct the object with FDK as a validation step using the config object with the same dataloading as done in NeCT reconstruction.

    Args:
        config (nect.config.Config): The config object for reconstruction
        output_directory (str, Path, optional): Save volume and image slices to the specified folder

    Returns:
        np.ndarray: The reconstructed volume
    """
    import tigre

    dataset = NeCTDataset(config=config, device="cpu")
    projs = np.zeros((len(dataset), config.geometry.nDetector[0], config.geometry.nDetector[1]), dtype=np.float32)
    angles = np.zeros((len(dataset)), dtype=np.float32)
    for i, (proj, angle, _) in tqdm(enumerate(dataset), total=len(dataset), desc="Loading projections"):
        projs[i] = proj
        angles[i] = angle
    geo = tigre_geometry_from_geometry(config.geometry)
    argsort_idx = np.argsort(angles)
    angles = np.copy(angles[argsort_idx])
    projs = np.copy(projs[argsort_idx])
    volume = tigre.algorithms.fdk(projs, geo, angles)
    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplot(1, 3)
        axes[0].imshow(volume[geo.nVoxel[0] // 2])
        axes[1].imshow(volume[:, geo.nVoxel[1] // 2])
        axes[2].imshow(volume[:, :, geo.nVoxel[2] // 2])
        plt.savefig(output_directory / "slices.png", dpi=300)
        if config.save_volume is True:
            np.save(output_directory / "volume.npy", volume)
    return volume


def tigre_geometry_from_geometry(geo: Geometry | ConfigGeometry | ConfigGeometryCone) -> tigre.geometry.Geometry:
    """
    Create a tigre Geometry form nect geometry.

    Args:
        geo (nect.Geometry, nect.config.Geometry, nect.config.GeometryCone): nect geometry to convert to tigre geometry

    Returns:
        tigre.geometry.Geometry: The tigre geometry
    """
    import tigre

    tigre_geo = tigre.geometry()
    tigre_geo.DSD = geo.DSD
    tigre_geo.DSO = geo.DSO
    tigre_geo.dDetector = np.array(geo.dDetector)
    tigre_geo.nDetector = np.array(geo.nDetector).astype(np.int32)
    tigre_geo.sDetector = tigre_geo.dDetector * tigre_geo.nDetector
    tigre_geo.nVoxel = np.array(geo.nVoxel).astype(np.int32)
    tigre_geo.dVoxel = np.array(geo.dVoxel)
    tigre_geo.sVoxel = tigre_geo.nVoxel * tigre_geo.dVoxel
    tigre_geo.offOrigin = np.array(geo.offOrigin)
    tigre_geo.offDetector = np.array(geo.offDetector)
    tigre_geo.accuracy = 0.5
    tigre_geo.mode = geo.mode
    tigre_geo.COR = geo.COR
    tigre_geo.rotDetector = np.array(geo.rotDetector)
    tigre_geo.filter = None
    return tigre_geo
