from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import porespy as ps
import torch
from tqdm import tqdm

from nect.src.phantom.phantom_config import IntensityConfig
from nect.src.utils.video import save_video


class RandomCube:
    def __init__(self, cube_position: tuple[int, int, int], size: int, start_time: int, on_time) -> None:
        """Creates a blinking cube. The cube will blink between air and water with a given period.

        Args:
            cube_position (tuple[int, int, int]): The position of the cube in the image. (top-left corner, zyx)
            size (int): The size of the cube.
            start_time (int): The time when the cube starts blinking.
            on_time (int): The time the cube is on.
        """

        self.cube_position = cube_position
        self.size = size
        self.attenuation = IntensityConfig.AIR
        self.start_time = start_time
        self.on_time = on_time
        self.time = 0

    def place_cube(self, img: np.ndarray) -> np.ndarray:
        """Places the cube in the image.

        Args:
            img (np.ndarray): The image to place the cube in.

        Returns:
            np.ndarray: The image with the cube placed.
        """
        self.attenuation = IntensityConfig.AIR

        if self.time >= self.start_time and self.time < self.start_time + self.on_time:
            self.attenuation = IntensityConfig.WATER
        self.time += 1
        cube = np.ones((self.size, self.size, self.size)) * self.attenuation
        z, y, x = self.cube_position
        img[z : z + self.size, y : y + self.size, x : x + self.size] = cube
        return img


class BlinkingCube:
    def __init__(self, cube_position: tuple[int, int, int], size: int, period: int) -> None:
        """Creates a blinking cube. The cube will blink between air and water with a given period.

        Args:
            cube_position (tuple[int, int, int]): The position of the cube in the image. (top-left corner, zyx)
            size (int): The size of the cube.
            period (int): The period of the blinking. If the period is 10, then the cube will be air for 5 projections and water for 5 projections.
        """

        self.cube_position = cube_position
        self.size = size
        self.period = period
        assert period % 2 == 0, "The period must be an even number."
        self.half_period = period // 2
        self.projs_since_last_blink = 0
        self.blinking = False
        self.attenuation = IntensityConfig.AIR

    def place_cube(self, img: np.ndarray) -> np.ndarray:
        """Places the cube in the image.

        Args:
            img (np.ndarray): The image to place the cube in.

        Returns:
            np.ndarray: The image with the cube placed.
        """
        # print(f"Since blink: {self.projs_since_last_blink}")
        if self.projs_since_last_blink == self.half_period:
            # print("Blinking")
            self.projs_since_last_blink = 1
            self.blinking = not self.blinking
        else:
            self.projs_since_last_blink += 1
        if self.blinking:
            self.attenuation = IntensityConfig.WATER
            cube = np.ones((self.size, self.size, self.size)) * self.attenuation
            z, y, x = self.cube_position
            img[z : z + self.size, y : y + self.size, x : x + self.size] = cube
        return img


class EmptyCanvas:
    def __init__(self, size: tuple[int, int, int], cubes: list[BlinkingCube]) -> None:
        """Creates an empty canvas with blinking cubes.

        Args:
            cubes (list[BlinkingCube]): A list of blinking cubes.
        """
        self.cubes = cubes
        self.size = size
        self.base_img = np.ones(size)

    def get_phantom(self, t: float) -> np.ndarray:
        dynamic_img = self.base_img.copy()
        for cube in self.cubes:
            dynamic_img = cube.place_cube(dynamic_img)

        dynamic_img += dynamic_img.min()
        dynamic_img /= dynamic_img.max()
        return dynamic_img


class PorousMedium:
    def __init__(
        self,
        shape: np.ndarray = (50, 50, 50),
        blobiness: list[int] = [1],
        porosity: float = 0.5,
        divs: int = 1,
    ) -> None:
        """Creates a porous medium geometry. Calling with divs > 1 processes the porous medium in parallel.

        Args:
            shape (np.ndarray, optional): The shape of the porous medium. Defaults to (50,50,50).
            blobiness (list[int], optional): The blobiness of the porous medium. Defaults to [1].
            porosity (float, optional): The porosity of the porous medium. Defaults to 0.5.
            divs (int, optional): The number of divisions to use in parallel computing. Defaults to 1.
        """
        self.mask = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness, divs=divs)


class MultiScalePorousMedium:
    def __init__(self, medium1: PorousMedium, medium2: PorousMedium) -> None:
        """Creates a porous medium geometry that is the intersection of two porous medium geometries.

        Args:
            medium1 (PorousMedium): The first porous medium.
            medium2 (PorousMedium): The second porous medium.
        """
        self.mask = ~(~medium1.mask * medium2.mask)


class PorousMediumPhantom:
    def __init__(
        self,
        geometry: np.ndarray,
        inlet: str = "xyt",
        cylindrical: bool = False,
        poisson_noise: bool = False,
        dynamic: bool = True,
    ) -> None:
        """Creates a porous medium phantom based on the geometry (shape=(z, y, x)). The inlet defines the direction of the flow.
        "xyt" should be read as "in the direction of the xy-plane from the top" and "yb" as "in the direction of the y-axis from the bottom".

        Args:
            geometry (np.ndarray): The porous medium geometry. Can be created with the PorousMedium class.
            inlet (str, optional): A string defining the direction of inlet flow. Defaults to "xyt".
            cylindrical (bool, optional): Whether to create a cylindrical phantom. Defaults to False.
        """
        self.is_3d = len(geometry.shape) == 3
        self.inlet = inlet
        self.size = geometry.shape
        self.geometry = geometry
        self.bd = self.create_inlet(inlet=inlet)
        self.target = self.get_target(
            scale=True
        )  # When scale is True, the timescale of porous medium filling will always be between 0 and 1.
        self.base_img = np.zeros_like(self.target)
        self.base_img[~geometry] = IntensityConfig.ROCK  # Setting rock to 2 for contrast
        self.steps = np.sort(np.unique(self.target)[1:])
        self.cylindrical = cylindrical
        self.poisson_noise = poisson_noise
        self.dynamic = dynamic
        # self.base_img[~self.geometry] = IntensityConfig.ROCK
        if cylindrical:
            self.cylindrical_geometry = ps.generators.cylindrical_plug(shape=self.size, r=self.size[-1] // 2, axis=0)

    def get_target(self, scale=False) -> np.ndarray:
        """Returns an array which specifies when a portion of space should be filled.
        See https://porespy.org/examples/simulations/tutorials/drainage_with_gravity_advanced.html for details.
        Returns an array of saturation between 0 and 1. If there is unfilled pores in the phantom, the maximum
        saturation will be less than 1. Scaling between 0 and 1 can be forced by passing scale=True to the function."""
        # self.geometry = ps.filters.trim_disconnected_blobs(im=self.geometry, inlets=self.bd)  # Removes disconnected blobs, but we probably don't want this
        out = ps.filters.ibip(im=self.geometry, inlets=self.bd, maxiter=15000)
        inv_seq = out.inv_sequence
        inv_satn = ps.filters.seq_to_satn(seq=inv_seq)
        if scale:
            inv_satn = inv_satn / np.max(inv_satn)
        return np.around(inv_satn, decimals=3)

    def get_phantom(
        self,
        t: float,
        poisson_noise: bool = False,
        scaled: bool = True,
        blinking_cubes: list[BlinkingCube | RandomCube] | None = None,
    ) -> np.ndarray:
        """Fills the pores with fluid according to simulation results at time t.

        Args:
            t (float): Simulation time

        Returns:
            np.ndarray: An array with pores filled according to simulation results at time t.
        """
        dynamic_img = self.base_img.copy()
        # if self.dynamic:
        if not self.dynamic:
            t = 0
        dynamic_img[np.logical_and(self.target < t, self.target > 0)] = (
            IntensityConfig.WATER
        )  # All cells with a value less than t are filled at time t
        dynamic_img[~self.geometry] = IntensityConfig.ROCK  # Set the rock to 2 for contrast
        if self.cylindrical:
            dynamic_img *= self.cylindrical_geometry
        if poisson_noise or self.poisson_noise:
            noise_mask = np.random.poisson(
                dynamic_img
            )  # Poisson noise is not additive, but depends on the signal strength. This should be the correct way.
            dynamic_img = dynamic_img + noise_mask

        if blinking_cubes is not None:
            for cube in blinking_cubes:
                dynamic_img = cube.place_cube(dynamic_img)

        if scaled:
            dynamic_img += dynamic_img.min()
            dynamic_img /= dynamic_img.max()
        return dynamic_img

    def create_inlet(self, inlet: str = "xyt") -> np.ndarray:
        """Creates the inlet boundary condition based on the "inlet" string.
        It should be read as "in the direction of the xy-plane from the top" for "xyt"
        or "in the direction of the y-axis from the bottom" for "yb".

        Args:
            inlet (str, optional): String defining the orientation. Defaults to "xyt".

        Raises:
            NotImplementedError: If the inlet string is not defined.

        Returns:
            np.ndarray: An array of the same shape as the geometry with True values at the inlet.
        """
        bd = np.zeros_like(self.geometry, dtype=bool)
        if self.is_3d:
            if inlet == "xyt":  # xy-plane top
                bd[0, :, :] = 1
            elif inlet == "xyb":  # xy-plane bottom
                bd[-1, :, :] = 1
            else:
                raise NotImplementedError(f"Inlet {inlet} is not implemented.")
        else:
            if inlet == "xl":  # x-axis left
                bd[:, 0] = 1
            elif inlet == "xr":  # x-axis right
                bd[:, -1] = 1
            elif inlet == "yt":  # y-axis top
                bd[0, :] = 1
            elif inlet == "yb":  # y-axis bottom
                bd[-1, :] = 1
            else:
                raise NotImplementedError(f"Inlet {inlet} is not implemented.")

        bd *= self.geometry
        return bd

    def save_video(self, time_steps: np.ndarray, filename: str, fps: int = 5):
        vid = []
        if self.is_3d:
            logging.info("The phantom is 3D. The video displays the center slice in the xz-plane.")
            for t in tqdm(time_steps, desc="Creating phantom images"):
                vid.append(self.get_phantom(t)[:, self.geometry.shape[1] // 2, :])
        else:
            for t in tqdm(time_steps, desc="Creating phantom images"):
                vid.append(self.get_phantom(t))
        save_video(vid, filename, fps=fps)

    def create_video_as_tensor(self, time_steps: np.ndarray, save_frames=None, save_video=None) -> torch.Tensor:
        """
        Creates a video of the phantom as a tensor.

        Args:
            time_steps (np.ndarray): The time steps of the video.
            save_frames (str, optional): Path to a directory where the frames should be saved. Defaults to None.
            save_video (str, optional): Path to a file where the video should be saved. Defaults to None.

        Returns:
            torch.Tensor: The video as a tensor. Has shape (time_steps, height, width)."""
        vid = np.zeros(shape=(len(time_steps), *self.size))
        for i in tqdm(range(len(time_steps)), desc="Creating phantom images"):
            vid[i] = self.get_phantom(time_steps[i])
            if save_frames is not None:
                np.save(Path(save_frames) / f"frame_{i}", vid[i])
        if save_video is not None:
            np.save(Path(save_video), vid)
        return torch.from_numpy(np.array(vid)).float()
