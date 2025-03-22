import numpy as np
import scipy.constants


def equidistant(nprojs: int, nrevs: int, radians: bool = True) -> np.ndarray:
    """Generates a set of angles using the equidistant method

    Args:
        nprojs (int): The number of projections per revolution.
        nrevs (int): The number of revolutions.
        radians (bool, optional): Whether to return angles in radians. Defaults to True.

    Returns:
        np.ndarray: The equiditant angles.
    """
    start = 0
    end = 360 * nrevs
    angles = np.linspace(start, end, nprojs * nrevs, endpoint=False)

    if radians:
        return angles * np.pi / 180
    else:
        return angles


def golden_angle(nprojs: int, radians: bool = True) -> np.ndarray:
    """Generates a set of angles using the golden angle method

    Args:
        nprojs (int): The number of projections per revolution.
        radians (bool, optional): Whether to return angles in radians. Defaults to True.

    Returns:
        np.ndarray: The angles sampled by the golden angle method.
    """
    angles = np.arange(nprojs) * 360 / scipy.constants.golden_ratio
    if radians:
        return angles * np.pi / 180
    else:
        return angles


def hybrid_golden_angle(nprojs, nrevs, radians=True, starting=0):
    golden_angle_sampling = lambda n, inc: np.mod((n * 1 / ((np.sqrt(5) - 1) / 2) * inc), inc)

    startings = golden_angle_sampling(np.arange(starting, nrevs), 360 / nprojs)

    linear_sampling = lambda s: np.linspace(s, s + 360, nprojs, endpoint=False)

    angles = linear_sampling(startings).T
    if radians:
        return angles * np.pi / 180

    angles[1::2, ...] = angles[
        1::2, ::-1
    ]
    angles = np.array(angles).flatten()
    return angles
