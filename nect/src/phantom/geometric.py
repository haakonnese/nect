from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from nect.src.utils.video import save_video


class Rectangle:
    def __init__(self, x, y) -> None:
        self.mask = torch.ones((x, y))


class Circle:
    def __init__(self, x, y, r) -> None:
        self.mask = torch.zeros((x, y))
        for i in range(x):
            for j in range(y):
                if (i - x / 2) ** 2 + (j - y / 2) ** 2 <= r**2:
                    self.mask[i, j] = 1


class Triangle:
    def __init__(self, x, y) -> None:
        self.mask = torch.zeros((x, y))
        for i in range(x):
            self.mask[i, : i + 1] = 1


class Ellipse:
    def __init__(self, x, y, a, b) -> None:
        self.mask = torch.zeros((x, y))
        for i in range(x):
            for j in range(y):
                if (i - a) ** 2 / a**2 + (j - b) ** 2 / b**2 <= 1:
                    self.mask[i, j] = 1


class EllipseV2:
    def __init__(self, a, b, r, random_rotation=True):
        self.mask = torch.zeros((2 * r, 2 * r))
        for i in range(2 * r):
            for j in range(2 * r):
                if (i - r) ** 2 / a**2 + (j - r) ** 2 / b**2 <= 1:
                    self.mask[i, j] = 1
        if random_rotation:
            # Rotate the mask by a random degree, but keep the mask size
            self.mask = torch.from_numpy(
                np.array(Image.fromarray(self.mask.numpy()).rotate(np.random.randint(0, 360), resample=Image.NEAREST))
            )


class Cuboid:
    def __init__(self, x, y, z) -> None:
        self.mask = torch.ones((x, y, z))


class Sphere:
    def __init__(self, x, y, z, r) -> None:
        self.mask = torch.zeros((x, y, z))
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if (i - x / 2) ** 2 + (j - y / 2) ** 2 + (k - z / 2) ** 2 <= r**2:
                        self.mask[i, j, k] = 1


class CustomGeometry:
    def __init__(self, mask) -> None:
        self.mask = mask.astype(np.uint8)


class PhantomObject:
    def __init__(
        self,
        eom: Callable,
        eoi: Callable,
        tl: torch.Tensor,
        geometry,
        intensity: int | torch.Tensor,
    ) -> None:
        """Creates an object in a phantom that moves according to the equation of motion (eom) and changes its intensity according to the equation of intensity (eoi).

        Args:
            eom (Callable): Takes the initial tlbr coordinates and the time t as input and returns the new tlbr coordinates.
            eoi (Callable): Takes the initial intensity and the time t as input and returns the new intensity.
            tl (torch.Tensor): The initial tl (top-left) coordinates of the bounding box.
            geometry (): An object containing a boolean mask of the shape of the object.
            intensity (int | torch.Tensor): The intensity of the object, either as a scalar or as a grid of the same shape as the geometry mask.
        """
        self.eom = eom  # equation of motion
        self.eoi = eoi  # equation of intensity
        assert len(tl) == len(
            geometry.mask.shape
        ), "Initialization coordinates and geometry must have the same dimensionality."
        self.tlbr_init = torch.cat(
            (tl, tl + torch.tensor(geometry.mask.shape)), 0
        )  # initial tlbr coordinates of bounding box
        self.tlbr = self.tlbr_init  # tlbr coordinates of bounding box
        self.geometry = geometry  # shape of object (boolean mask)
        self.intensity = intensity  # intensity of object (scalar or grid)
        self.intensity_grid = self.geometry.mask * self.intensity  # intensity grid of object

    def update(self, t: float) -> None:
        """Updates the position and the intensity of the object.

        Args:
            t (float): The current time.
        """
        self.tlbr = self.eom(self.tlbr_init, t)
        self.intensity_grid = self.geometry.mask * self.eoi(self.intensity, t)


class Phantom:
    def __init__(self, size: tuple[int, int], background_method: str = "zeros") -> None:
        """Creates a phantom image with a given background. Phantom objects can be added to the phantom.

        Args:
            size (tuple[int,int]): The size of the phantom. Defined as [x, y, [z]]
            background_method (str, optional): How to create the background. Defaults to "zeros".
        """
        self.objects = []
        self.size = size
        assert len(size) >= 2 or len(size) <= 3, "Size must be a tuple of length 2 or 3."
        self.is_3d = len(size) == 3
        self.create_background(size=size, method=background_method)

    def create_background(self, size: tuple[int, int], method: str = "zeros") -> torch.Tensor:
        if method == "zeros":
            self.background = torch.zeros(size)
        else:
            raise NotImplementedError(f"Method {method} not implemented.")

    def add_phantom_object(self, obj: PhantomObject | list[PhantomObject]) -> None:
        if isinstance(obj, list):
            for object in obj:
                self.validate(object)
            self.objects.extend(obj)
        else:
            self.validate(obj)
            self.objects.append(obj)

    def get_phantom(self, t: float) -> torch.Tensor:
        """Gets the state of the phantom at time t.

        Args:
            t (float): The current time.

        Returns:
            torch.Tensor: The background with the object intensities added.
        """
        phantom = self.background.clone()
        t_ = t
        if self.is_3d:
            for obj in self.objects:
                obj.update(t_)
                top = int(max(0, obj.tlbr[0]))  # Top
                left = int(max(0, obj.tlbr[1]))  # Left
                high = int(max(0, obj.tlbr[2]))  # Upper
                bottom = int(min(obj.tlbr[3], self.size[0]))
                right = int(min(obj.tlbr[4], self.size[1]))
                low = int(min(obj.tlbr[5], self.size[2]))
                phantom[top:bottom, left:right, high:low] += obj.intensity_grid[
                    0 : bottom - top, 0 : right - left, 0 : low - high
                ]
        else:
            for obj in self.objects:
                obj.update(t_)
                top = int(max(0, obj.tlbr[0]))
                left = int(max(0, obj.tlbr[1]))
                bottom = int(min(obj.tlbr[2], self.size[0]))
                right = int(min(obj.tlbr[3], self.size[1]))
                phantom[top:bottom, left:right] += obj.intensity_grid[0 : bottom - top, 0 : right - left]
        return phantom.detach().cpu().numpy()

    def save_video(self, time_steps: torch.Tensor, filename: str, fps: int = 5):
        if self.is_3d:
            raise NotImplementedError("Saving videos is not supported for 3D phantoms")
        vid = []
        for t in tqdm(time_steps, desc="Creating phantom images"):
            vid.append(self.get_phantom(t))
        save_video(vid, filename, fps=fps)

    def create_video_as_tensor(self, time_steps: np.ndarray) -> torch.Tensor:
        """
        Creates a video of the phantom as a tensor.

        Args:
            time_steps (np.ndarray): The time steps of the video.

        Returns:
            torch.Tensor: The video as a tensor. Has shape (time_steps, height, width)."""
        vid = np.zeros(shape=(len(time_steps), *self.background.shape))
        for i in tqdm(range(len(time_steps)), desc="Creating phantom images"):
            vid[i] = self.get_phantom(time_steps[i])
        return torch.from_numpy(np.array(vid)).float()

    def save_as_array(self, time_steps: torch.Tensor, filename: str):
        vid = torch.zeros((len(time_steps), *self.background.shape))
        for i in tqdm(range(len(time_steps)), desc="Creating phantom images"):
            vid[i] = self.get_phantom(time_steps[i])
        np.save(Path(filename), vid.numpy())

    def validate(self, obj):
        if len(obj.geometry.mask.shape) != len(self.size):
            raise ValueError("Object must have the same dimensionality as the phantom.")
