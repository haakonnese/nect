import numpy as np
from scipy import ndimage

from nect.src.phantom.geometric import CustomGeometry


def create_random_closed_geometry(size=(100, 100)) -> CustomGeometry:
    """Creates a random closed geometry.

    Args:
        size (tuple[int,int], optional): The size of the geometry. Defaults to (100,100).

    Returns:
        CustomGeometry: The random closed geometry.
    """
    # create a random geometry with more 0 than 1
    geometry = np.random.choice([0, 1], size=size, p=[0.7, 0.3])
    # close the geometry
    geometry = ndimage.binary_closing(geometry, border_value=0, iterations=1)
    # create a custom geometry
    return CustomGeometry(mask=geometry)


def eoi_simple_dynamic_pore(intensity: np.ndarray, t: float) -> np.ndarray:
    t = int(t)
    if t == 0:
        intensity[0] = np.where(
            intensity[0] == 1,
            np.random.randint(1, 3, size=len(intensity[0])),
            np.where(intensity[0] == 2, 2, 0),
        )
        return intensity
    intensity[:t, ...] = np.where(
        intensity[:t, ...] == 1,
        np.random.randint(1, 3, size=len(intensity[0])),
        np.where(intensity[:t, ...] == 2, 2, 0),
    )
    return intensity
