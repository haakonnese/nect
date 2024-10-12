import numpy as np
import tigre.algorithms as algs

from nect.src.simulator.configuration.config import TigreGeometry


def fdk(sinogram, theta, geo=None, *args, **kwargs) -> np.ndarray:
    """FDK reconstruction using the TIGRE toolbox.

    Args:
        sinogram (np.ndarray): A sinogram.
        geo (Geometry): A TIGRE Geometry object.
        theta (np.ndarray): The angles of the sinogram.

    Returns:
        np.ndarray: The reconstructed image.
    """
    if geo is None:
        raise ValueError("Geometry is None")
    elif isinstance(geo, TigreGeometry):
        geo = geo.geo
    return algs.fdk(sinogram, geo, angles=theta, *args, **kwargs)


def ossart(sinogram, geo, theta, niter, *args, **kwargs) -> np.ndarray:
    """OSSART reconstruction using the TIGRE toolbox.

    Args:
        sinogram (np.ndarray): A sinogram.
        geo (Geometry): A TIGRE Geometry object.
        theta (np.ndarray): The angles of the sinogram.
        niter (int): Number of iterations.

    Returns:
        np.ndarray: The reconstructed image.
    """
    if geo is None:
        raise ValueError("Geometry is None")
    elif isinstance(geo, TigreGeometry):
        geo = geo.geo
    return algs.ossart(sinogram, geo, angles=theta, niter=niter, *args, **kwargs)
