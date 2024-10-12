import numpy as np
import tigre
from tigre.utilities.geometry import Geometry
from tqdm import tqdm

from nect.src.sampling.methods import equidistant, golden_angle
from nect.src.simulator.configuration.config import TigreGeometry


def equidistant_sampling(
    image, nprojs: int, nrevs: int, geo: Geometry = None, radians=True, *args, **kwargs
) -> np.ndarray:
    """Sample a static image using the TIGRE toolbox.

    Args:
        image (np.ndarray): Static 3D image.
        nprojs (int): Number of projections.
        nrevs (int): Number of revolutions.
        geo (Geometry): Geometry object from TIGRE.
        radians (bool, optional): Wheter to use radians. Defaults to True.

    Returns:
        np.ndarray: An array containing the sinogram.
    """
    if isinstance(geo, TigreGeometry):
        geo = (
            geo.geo
        )  # If the user forgets to pass the actual Tigre geometry, but rather the object, set geo to the geometry
    theta = equidistant(nprojs=nprojs, nrevs=nrevs, radians=radians)
    return tigre.Ax(image.get_phantom(0).astype(np.float32), geo, theta, **kwargs), theta


def golden_angle_sampling(image, nprojs: int, geo: Geometry = None, radians=True, *args, **kwargs) -> np.ndarray:
    """Sample a static image using the TIGRE toolbox and the golden ratio sampling.

    Args:
        image (np.ndarray): Static 3D image.
        nprojs (int): Number of projections.
        geo (Geometry): Geometry object from TIGRE.
        radians (bool, optional): Wheter to use radians. Defaults to True.

    Returns:
        np.ndarray: An array containing the sinogram.
    """
    if isinstance(geo, TigreGeometry):
        geo = (
            geo.geo
        )  # If the user forgets to pass the actual Tigre geometry, but rather the object, set geo to the geometry
    theta = golden_angle(nprojs=nprojs, radians=radians)
    return tigre.Ax(image.astype(np.float32), geo, theta, **kwargs), theta


def dynamic_equidistant_sampling(
    dynamic_image,
    scheduler,
    nprojs: int,
    nrevs: int,
    geo: Geometry = None,
    radians=True,
    *args,
    **kwargs,
) -> np.ndarray:
    """Sample a dynamic image using the TIGRE toolbox and the equidistant sampling.

    Args:
        dynamic_image (DynamicImage): Dynamic image.
        scheduler (Scheduler): Scheduler object.
        nprojs (int): Number of projections.
        nrevs (int): Number of revolutions.
        geo (Geometry): Geometry object from TIGRE.
        radians (bool, optional): Wheter to use radians. Defaults to True.

    Returns:
        np.ndarray: An array containing the sinogram.
    """
    if isinstance(geo, TigreGeometry):
        geo = (
            geo.geo
        )  # If the user forgets to pass the actual Tigre geometry, but rather the object, set geo to the geometry
    time_series = np.zeros((nrevs * nprojs, *dynamic_image.size[:-1]))  # Initialize sinogram
    theta = equidistant(nprojs=nprojs, nrevs=nrevs, radians=radians)
    time = 0
    for i in tqdm(range(len(theta)), desc="Sampling"):
        phantom = dynamic_image.get_phantom(time)
        time_series[i, ...] = tigre.Ax(phantom.astype(np.float32), geo, angles=np.array([theta[i]]), **kwargs).squeeze(
            axis=0
        )
        time = scheduler.rotate(theta[i], theta[i + 1]) if i < len(theta) - 1 else time
    return time_series, theta


def dynamic_golden_angle_sampling(
    dynamic_image,
    scheduler,
    nprojs: int,
    geo: Geometry = None,
    radians=True,
    *args,
    **kwargs,
) -> np.ndarray:
    """Sample a dynamic image using the TIGRE toolbox and the golden ratio sampling.

    Args:
        dynamic_image (DynamicImage): Dynamic image.
        scheduler (Scheduler): Scheduler object.
        nprojs (int): Number of projections.
        geo (Geometry): Geometry object from TIGRE.
        radians (bool, optional): Wheter to use radians. Defaults to True.

    Returns:
        np.ndarray: An array containing the sinogram.
    """
    if isinstance(geo, TigreGeometry):
        geo = (
            geo.geo
        )  # If the user forgets to pass the actual Tigre geometry, but rather the object, set geo to the geometry
    time_series = np.zeros((nprojs, *dynamic_image.size[:-1]))  # Initialize sinogram
    theta = golden_angle(nprojs=nprojs, radians=radians)
    time = 0
    for i in tqdm(range(len(theta)), desc="Sampling"):
        phantom = dynamic_image.get_phantom(time)
        time_series[i, ...] = tigre.Ax(phantom.astype(np.float32), geo, angles=np.array([theta[i]]), **kwargs).squeeze(
            axis=0
        )
        time = scheduler.rotate(theta[i], theta[i + 1]) if i < len(theta) - 1 else time
    return time_series, theta
