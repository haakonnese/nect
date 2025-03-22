import numpy as np
from skimage.transform import radon
from tqdm import tqdm

from nect.src.sampling.methods import equidistant, golden_angle
from nect.src.simulator.scheduler import Scheduler


def equidistant_sampling(image, nprojs, nrevs, radians=False, *args, **kwargs):
    """
    Equidistant sampling of the image.
    :param image: 2D image
    :param n_samples: number of samples. Use at least 360 for "full" sampling.
    :return: sinogram of sampled image
    """
    theta = equidistant(nprojs=nprojs, nrevs=nrevs, radians=radians)
    return radon(image, theta=theta, circle=True), theta


def golden_angle_sampling(image, nprojs, radians=False, *args, **kwargs):
    """
    Golden angle sampling of the image.
    :param image: 2D image
    :param n_samples: number of samples.
    :return: sinogram of sampled image
    """
    theta = golden_angle(nprojs=nprojs, radians=radians)
    return radon(image, theta=theta, circle=True), theta


def dynamic_equidistant_sampling(dynamic_image, scheduler: Scheduler, nprojs, nrevs, radians=False, *args, **kwargs):
    time_series = np.zeros((dynamic_image.size[0], nrevs * nprojs))  # Initialize sinogram
    theta = equidistant(nprojs=nprojs, nrevs=nrevs, radians=radians)
    time = 0
    for i in tqdm(range(len(theta)), desc="Sampling"):
        phantom = dynamic_image.get_phantom(time)
        time_series[:, i] = radon(phantom, theta=[theta[i]], **kwargs).squeeze(axis=1)
        time = scheduler.rotate(theta[i], theta[i + 1]) if i < len(theta) - 1 else time
    return time_series, theta


def dynamic_golden_angle_sampling(dynamic_image, scheduler: Scheduler, nprojs, radians=False, *args, **kwargs):
    time_series = np.zeros((dynamic_image.size[0], nprojs))  # Initialize sinogram
    theta = golden_angle(nprojs=nprojs, radians=radians)
    time = 0
    for i in tqdm(range(len(theta)), desc="Sampling"):
        phantom = dynamic_image.get_phantom(time)
        time_series[:, i] = radon(phantom, theta=[theta[i]], **kwargs).squeeze(axis=1)
        time = scheduler.rotate(theta[i], theta[i + 1]) if i < len(theta) - 1 else time
    return time_series, theta


if __name__ == "__main__":
    theta = equidistant(10, 2, radians=False)
    rots = np.diff(theta)
    print(rots)
    print(theta)
    print(len(rots))
    print(len(theta))
