import numpy as np
import torch
from leaptorch import Projector
from tqdm import tqdm

from nect.src.sampling.methods import equidistant, hybrid_golden_angle
from nect.src.simulator.configuration.config import LeapGeometry


def equidistant_sampling(
    image,
    nprojs: int,
    nrevs: int,
    proj: Projector = None,
    radians=False,
    device=torch.device("cpu"),
    t: float = 0,
    *args,
    **kwargs,
) -> np.ndarray:
    """Sample a static image using the TIGRE toolbox.

    Args:
        image (np.ndarray): Static 3D image.
        nprojs (int): Number of projections.
        nrevs (int): Number of revolutions.
        radians (bool, optional): Wheter to use radians. Defaults to True.
        proj (Projector, optional): Projector object from LEAP. Defaults to None.
        device (torch.device, optional): Device to use. Defaults to torch.device("cpu").
        t (float, optional): Time of the image. Defaults to 0.

    Returns:
        np.ndarray: An array containing the sinogram.
    """
    print("[WARNING] LeapTorch arrays are shifted by -n_projs//4. This might change in the future")
    theta = torch.from_numpy(equidistant(nprojs=nprojs, nrevs=nrevs, radians=radians)).float()
    proj.update_phi(theta)
    img = torch.from_numpy(image.get_phantom(t)).to(device).unsqueeze(0).float()
    return proj(img).squeeze(0).detach().cpu().numpy(), theta.detach().cpu().numpy()


def dynamic_equidistant_sampling(
    dynamic_image,
    scheduler,
    nprojs: int,
    nrevs: int,
    radians: bool = False,
    proj: LeapGeometry = None,
    device: torch.device = None,
    *args,
    **kwargs,
) -> np.ndarray:
    """Sample a dynamic image using the TIGRE toolbox and the equidistant sampling.

    Args:
        dynamic_image (DynamicImage): Dynamic image.
        scheduler (Scheduler): Scheduler object.
        nprojs (int): Number of projections.
        nrevs (int): Number of revolutions.
        radians (bool, optional): Wheter to use radians. Defaults to True.
        proj (LeapGeometry, optional): Geometry object from LEAP. Defaults to None.
        device (torch.device, optional): Device to use. Defaults to None.

    Returns:
        np.ndarray: An array containing the sinogram.
    """
    print("[WARNING] LeapTorch arrays are shifted by -n_projs//4. This might change in the future")
    time_series = np.zeros((nrevs * nprojs, proj.numRows, proj.numCols))  # Initialize sinogram
    time = 0
    theta = torch.from_numpy(equidistant(nprojs=nprojs, nrevs=nrevs, radians=radians)).float()
    for i in tqdm(range(len(theta)), desc="Sampling"):
        phantom = torch.from_numpy(dynamic_image.get_phantom(float(time))).to(device).unsqueeze(0).float()
        proj.update_phi(theta[[i]])
        time_series[i, ...] = proj(phantom).squeeze(0).detach().cpu().numpy()
        time = scheduler.rotate(theta[i], theta[i + 1]) if i < len(theta) - 1 else time
    return time_series, theta


def dynamic_hybrid_golden_angle_sampling(
    dynamic_image,
    scheduler,
    nprojs: int,
    nrevs: int,
    radians: bool = False,
    proj: LeapGeometry = None,
    device: torch.device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu"),
    *args,
    **kwargs,
) -> np.ndarray:
    """Sample a dynamic image using the TIGRE toolbox and the golden ratio sampling.

    Args:
        dynamic_image (DynamicImage): Dynamic image.
        scheduler (Scheduler): Scheduler object.
        nprojs (int): Number of projections.
        nrevs (int): Number of revolutions
        proj (Geometry): Geometry object from LEAP.
        radians (bool, optional): Wheter to use radians. Defaults to True.

    Returns:
        np.ndarray: An array containing the sinogram.
    """
    print("[WARNING] LeapTorch arrays are shifted by -n_projs//4. This might change in the future")
    time_series = np.zeros((nrevs * nprojs, proj.numRows, proj.numCols))  # Initialize sinogram
    time = 0
    theta = torch.from_numpy(hybrid_golden_angle(nprojs=nprojs, nrevs=nrevs, radians=radians)).float()
    for i in tqdm(range(len(theta)), desc="Sampling: "):
        phantom = torch.from_numpy(dynamic_image.get_phantom(float(time))).to(device).unsqueeze(0).float()
        proj.update_phi(theta[[i]])
        time_series[i, ...] = proj(phantom).squeeze(0).detach().cpu().numpy()
        time = scheduler.rotate(theta[i], theta[i + 1]) if i < len(theta) - 1 else time
    return time_series, theta.detach().cpu().numpy()


def dynamic_hybrid_golden_angle_sampling_linear_time(
    dynamic_image,
    scheduler,
    nprojs: int,
    nrevs: int,
    radians: bool = False,
    proj: LeapGeometry = None,
    device: torch.device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu"),
    *args,
    **kwargs,
) -> np.ndarray:
    """Sample a dynamic image using the TIGRE toolbox and the golden ratio sampling.

    Args:
        dynamic_image (DynamicImage): Dynamic image.
        scheduler (Scheduler): Scheduler object.
        nprojs (int): Number of projections.
        nrevs (int): Number of revolutions
        proj (Geometry): Geometry object from LEAP.
        radians (bool, optional): Wheter to use radians. Defaults to True.

    Returns:
        np.ndarray: An array containing the sinogram.
    """
    print("[WARNING] LeapTorch arrays are shifted by -n_projs//4. This might change in the future")
    time_series = np.zeros((nrevs * nprojs, proj.numRows, proj.numCols))  # Initialize sinogram
    time = np.linspace(0, 1, num=nprojs * nrevs, endpoint=True)
    theta = torch.from_numpy(hybrid_golden_angle(nprojs=nprojs, nrevs=nrevs, radians=radians)).float()
    for i in tqdm(range(len(theta)), desc="Sampling: "):
        phantom = (
            torch.from_numpy(dynamic_image.get_phantom(float(time[i]), blinking_cubes=kwargs.get("blinking_cubes")))
            .to(device)
            .unsqueeze(0)
            .float()
        )
        proj.update_phi(theta[[i]])
        time_series[i, ...] = proj(phantom).squeeze(0).detach().cpu().numpy()
    return time_series, theta.detach().cpu().numpy()
