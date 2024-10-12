from __future__ import annotations

import nect.sampling.ct_sampling
import torch

from nect.sampling.geometry import Geometry


def _check_and_return_cuda_device(device: torch.device | str | int) -> torch.device:
    if torch.cuda.is_available() is False:
        raise ValueError("CUDA is not available")
    available_devices = torch.cuda.device_count()
    max_gpu_id = available_devices - 1
    if isinstance(device, torch.device):
        if device.index > max_gpu_id:
            raise ValueError(f"GPU ID {device.index} is greater than the maximum available GPU ID {max_gpu_id}")
    if isinstance(device, int):
        if device > max_gpu_id:
            raise ValueError(f"GPU ID {device} is greater than the maximum available GPU ID {max_gpu_id}")
        device = torch.device(device)
    # check if device is a string with cuda:0, cuda:1, etc above the maximum available GPU ID
    if isinstance(device, str):
        if device.startswith("cuda:"):
            gpu_id = int(device.split(":")[1])
            if gpu_id > max_gpu_id:
                raise ValueError(f"GPU ID {gpu_id} is greater than the maximum available GPU ID {max_gpu_id}")
            device = torch.device(device)
        elif device == "cuda":
            device = torch.device("cuda:0")
        else:
            raise ValueError(f"'{device}' is not supported")
    return device


class Projector:
    def __init__(
        self,
        geometry: Geometry,
        points_per_batch: int,
        points_per_ray: int,
        device: torch.device | str | int,
        random_offset_detector: float = 0.0,
        uniform_ray_spacing: bool = True,
    ):
        self.device = _check_and_return_cuda_device(device)
        self.points_per_batch = points_per_batch
        self.points_per_ray = points_per_ray
        if points_per_ray < 1:
            raise ValueError(f"points_per_ray ({points_per_ray}) must be greater than or equal to 1")
        if points_per_batch < points_per_ray:
            raise ValueError(
                f"points_per_batch ({points_per_batch}) must be greater than or equal to points_per_ray ({points_per_ray})"
            )
        self.geometry = geometry
        self.random_offset_detector = random_offset_detector
        self.uniform_ray_spacing = uniform_ray_spacing
        self.c_geometry = self.geometry.get_c_geometry()

    def set_geometry(self, geometry: Geometry):
        self.geometry = geometry
        self.c_geometry = self.geometry.get_c_geometry()

    def to_dict(self):
        return {
            "geometry": self.geometry.to_dict(),
            "points_per_batch": self.points_per_batch,
            "reconstruction_mode": self.geometry.reconstruction_mode,
            "points_per_ray": self.points_per_ray,
            "uniform_ray_spacing": self.uniform_ray_spacing,
        }

    def update(
        self,
        angle: float,
        detector_binning: int | None = None,
        points_per_ray: int | None = None,
        random_offset_detector: float | None = None,
        uniform_ray_spacing: bool | None = None,
    ):
        if detector_binning is not None:
            self.geometry.set_detector_binning(detector_binning)
            self.c_geometry = self.geometry.get_c_geometry()
        if points_per_ray is not None:
            self.points_per_ray = points_per_ray
        if random_offset_detector is not None:
            self.random_offset_detector = random_offset_detector
        if uniform_ray_spacing is not None:
            self.uniform_ray_spacing = uniform_ray_spacing
        self.total_detector_pixels = (self.geometry.nDetector[0] // self.geometry.detector_binning) * (
            self.geometry.nDetector[1] // self.geometry.detector_binning
        )
        self.batch_size = self.points_per_batch // self.points_per_ray
        self.random_indexes = torch.randperm(self.total_detector_pixels, dtype=torch.int64, device=self.device)
        self.angle = angle
        self.distance_between_points = self.geometry.max_distance_traveled / self.points_per_ray
        self.batch_per_epoch = (self.total_detector_pixels) // self.batch_size
        self.batch_per_epoch += 1 if (self.total_detector_pixels) % self.batch_size != 0 else 0

    def update_angle(self, angle: float):
        self.angle = angle

    def __call__(self, batch_num: int, proj: torch.Tensor):
        """
        Sample points along the rays, and return the points and the corresponding projection values.
        """
        starting_ray_index = batch_num * self.batch_size
        batch_size = self.batch_size
        if (batch_num + 1) * batch_size > self.random_indexes.size(0):
            if batch_num == 0:
                batch_size = self.random_indexes.size(0) % batch_size
            else:
                return None, None
        ray_points, distances = nect.sampling.ct_sampling.sample(
            random_ray_index=self.random_indexes,
            geometry=self.c_geometry,
            angle_rad=self.angle,
            num_points_per_ray=self.points_per_ray,
            num_rays=batch_size,
            starting_ray_index=starting_ray_index,
            max_ray_distance_per_point=self.distance_between_points,
            uniform_ray_spacing=self.uniform_ray_spacing,
            # random_detector_offset=False,
            random_detector_offset=self.random_offset_detector,
            device=self.device.index,
        )
        self.distances = distances
        ray_points.clamp_(min=0, max=1)
        return (
            ray_points,
            proj[self.random_indexes[starting_ray_index : starting_ray_index + batch_size]],
        )
