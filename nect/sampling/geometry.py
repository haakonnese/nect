from __future__ import annotations

from pathlib import Path
from typing import cast
from warnings import warn

import nect.sampling.ct_sampling
import numpy as np
import torch
import yaml

import nect.config

_tuple_3_float = tuple[float, float, float]
_tuple_3_int = tuple[int, int, int]
_tuple_2_float = tuple[float, float]
_tuple_2_int = tuple[int, int]
_list_2_float = _tuple_2_float | list[float] | torch.Tensor | np.ndarray
_list_2_int = _tuple_2_int | list[int] | torch.Tensor | np.ndarray
_list_3_float = _tuple_3_float | list[float] | torch.Tensor | np.ndarray
_list_3_int = _tuple_3_int | list[int] | torch.Tensor | np.ndarray
_list_float = list[float] | torch.Tensor | np.ndarray
_list = list[float | int] | torch.Tensor | np.ndarray


class Geometry:
    def __init__(
        self,
        nDetector: _list_2_int,
        dDetector: _list_2_float,
        mode: str,
        DSD: float | None = None,
        DSO: float | None = None,
        nVoxel: _list_3_int | None = None,
        dVoxel: _list_3_float | None = None,
        radius: float | None = None,
        height: float | None = None,
        offOrigin: _list_3_float = (0.0, 0.0, 0.0),
        COR: float = 0.0,
        offDetector: _list_2_float = (0.0, 0.0),
        rotDetector: _list_3_float = (0.0, 0.0, 0.0),
        reconstruction_mode: str = "voxel",
        detector_binning: int = 1,
        angles: _list_float | None = None,
        radians: bool = True,
        timesteps: _list | None = None,
    ):
        """
        Set up the geometry for the CT system.

        Args:
            nDetector (tuple[int, int] | list[int] | torch.Tensor | np.ndarray):
                Number of pixels `[height, width]` of the detector
            dDetector (tuple[float, float] | list[float] | torch.Tensor | np.ndarray):
                Height and width of the detector (mm)
            mode (str):
                Type of geometry. Supported modes are `cone` and `parallel`
            DSD (float):
                Distance Source Detector (mm)
            DSO (float):
                Distance Source Origin (mm)
            nVoxel (tuple[int, int, int] | list[int] | torch.Tensor | np.ndarray):
                Number of voxels `[z, y, x]` of the volume
            dVoxel (tuple[float, float, float] | list[float] | torch.Tensor | np.ndarray):
                Size of a voxel `[z, y, x]` (mm)
            radius (float):
                Radius of the object (mm)
            height (float):
                Height of the object (mm)
            offOrigin (tuple[float, float, float] | list[float] | torch.Tensor | np.ndarray):
                Offset of the object from the origin `[z, y, x]` (mm)
            COR (float):
                Center of rotation (mm)
            offDetector (tuple[float, float] | list[float] | torch.Tensor | np.ndarray):
                Offset of the detector from the center `[height, width]` (mm)
            rotDetector (tuple[float, float, float] | list[float] | torch.Tensor | np.ndarray):
                Rotation of the detector `[roll, pitch, yaw]` (radians).
            reconstruction_mode (str):
                Type of reconstruction. Supported modes are `'voxel'` and `'cylindrical'`. Default is `'voxel'`
            detector_binning (int):
                Binning factor of the detector. Default is 1
            angles (list[float] | torch.Tensor | np.ndarray | None):
                List of angles.
            radians (bool):
                Unit of angles. If `True`, the unit is radians, if `False` the unit is degrees. Default is `True`
            timesteps (list[float | int] | torch.Tensor | np.ndarray | None):
                An array of timesteps. Do not need to be normalized.
                If the order of the angles and corresponding projections does not equal the acqustition order, this parameter needs to be set to get the timesteps correct.
                Only important for dynamic reconstruction. Overrides the timestep of the Geometry if not `None`.
        """
        if mode not in ["cone", "parallel"]:
            raise ValueError(f"Unsupported mode '{mode}'")
        if mode == "cone":
            if DSD is None:
                raise ValueError("DSD is required for cone geometry")
            if DSO is None:
                raise ValueError("DSO is required for cone geometry")
            self.DSD = float(DSD)
            self.DSO = float(DSO)
        else:
            if DSD is not None:
                warn("DSD is not required for parallel geometry. Ignoring the value of DSD")
            if DSO is not None:
                warn("DSO is not required for parallel geometry. Ignoring the value of DSO")
            self.DSD = None
            self.DSO = None
        if reconstruction_mode not in ["voxel", "cylindrical"]:
            raise ValueError(f"Unsupported reconstruction mode '{reconstruction_mode}'")
        self.mode = mode
        self.reconstruction_mode = reconstruction_mode
        self.detector_binning = detector_binning
        self.dDetector = cast(_tuple_2_float, self._check_and_return_float(dDetector, 2, "dDetector"))
        self.nDetector = cast(_tuple_2_int, self._check_and_return_int(nDetector, 2, "nDetector"))
        self.offDetector = cast(_tuple_2_float, self._check_and_return_float(offDetector, 2, "offDetector"))
        self.rotDetector = cast(_tuple_3_float, self._check_and_return_float(rotDetector, 3, "rotDetector"))
        self.COR = COR
        self.set_angles(angles, radians)
        self.set_timesteps(timesteps)
        if radius is None or height is None:
            self.nVoxel = cast(_tuple_3_int, self._check_and_return_int(nVoxel, 3, "nVoxel"))
            self.dVoxel = cast(_tuple_3_float, self._check_and_return_float(dVoxel, 3, "dVoxel"))
            self.sVoxel = (
                self.nVoxel[0] * self.dVoxel[0],
                self.nVoxel[1] * self.dVoxel[1],
                self.nVoxel[2] * self.dVoxel[2],
            )
            self.offOrigin = cast(_tuple_3_float, self._check_and_return_float(offOrigin, 3, "offOrigin"))
            if self.reconstruction_mode == "cylindrical":
                if radius is None:
                    self.radius = max(self.sVoxel[1], self.sVoxel[2]) / 2
                if height is None:
                    self.height = self.sVoxel[0]
            else:
                self.radius = None
                self.height = None
        else:
            self.radius = radius
            self.height = height
            if nVoxel is not None:
                self.nVoxel = cast(_tuple_3_int, self._check_and_return_int(nVoxel, 3, "nVoxel"))
            else:
                self.nVoxel = (self.nDetector[0], self.nDetector[1], self.nDetector[1])
            if dVoxel is not None:
                self.dVoxel = cast(_tuple_3_float, self._check_and_return_float(dVoxel, 3, "dVoxel"))
            else:
                self.dVoxel = (
                    self.nDetector[0] / height,
                    self.nDetector[1] / (2 * radius),
                    self.nDetector[1] / (2 * radius),
                )
            self.sVoxel = (
                self.nVoxel[0] * self.dVoxel[0],
                self.nVoxel[1] * self.dVoxel[1],
                self.nVoxel[2] * self.dVoxel[2],
            )
        if self.mode == "cone":
            if (nVoxel is None or dVoxel is None) and reconstruction_mode == "cylindrical":
                max_length = self.radius * 2
                triangle_theta = np.arctan((self.height / 2) / (self.DSO + max_length / 2))
                self.max_distance_traveled = max_length / np.cos(triangle_theta)
            else:
                max_length = ((self.sVoxel[1] - self.dVoxel[1]) ** 2 + (self.sVoxel[2] - self.dVoxel[2]) ** 2) ** 0.5
                triangle_theta = np.arctan(((self.sVoxel[0] - self.dVoxel[0]) / 2) / (self.DSO + max_length / 2))
                self.max_distance_traveled = max_length / np.cos(triangle_theta)
        else:
            if self.reconstruction_mode == "voxel":
                self.max_distance_traveled = (
                    (self.sVoxel[1] - self.dVoxel[1]) ** 2 + (self.sVoxel[2] - self.dVoxel[2]) ** 2
                ) ** 0.5
            elif self.reconstruction_mode == "cylindrical":
                if nVoxel is None or dVoxel is None:
                    self.max_distance_traveled = self.radius * 2
                else:
                    self.max_distance_traveled = max(self.sVoxel[1], self.sVoxel[2])
        self.sDetector = (
            self.nDetector[0] * self.dDetector[0],
            self.nDetector[1] * self.dDetector[1],
        )

    @classmethod
    def from_yaml(cls, path: str | Path, reconstruction_mode: str | None = None) -> "Geometry":
        """
        Load the geometry from a YAML file.

        Args:
            path (str | Path): The path to the YAML file.
            reconstruction_mode (str | None): The reconstruction mode. Supported strings are `'voxel'` and `'cylindrical'`. Default is `None`.

        Returns:
            Geometry: The geometry object."""
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        if reconstruction_mode is None:
            reconstruction_mode = cfg.get("reconstruction_mode", "voxel")
        elif reconstruction_mode not in ("voxel", "cylindrical"):
            raise NotImplementedError(
                f"Only reconstruction mode 'voxel' and 'cylindrical' is implemented, got '{reconstruction_mode}'"
            )
        return cls(
            nDetector=cfg["nDetector"],
            dDetector=cfg["dDetector"],
            mode=cfg["mode"],
            DSD=cfg.get("DSD", None),
            DSO=cfg.get("DSO", None),
            nVoxel=cfg.get("nVoxel", None),
            dVoxel=cfg.get("dVoxel", None),
            radius=cfg.get("radius", None),
            height=cfg.get("height", None),
            offOrigin=cfg.get("offOrigin", (0.0, 0.0, 0.0)),
            COR=cfg.get("COR", 0.0),
            offDetector=cfg.get("offDetector", (0.0, 0.0)),
            rotDetector=cfg.get("rotDetector", (0.0, 0.0, 0.0)),
            reconstruction_mode=reconstruction_mode,
            detector_binning=1,
            angles=cfg.get("angles", None),
            timesteps=cfg.get("timesteps", None),
            radians=cfg.get("radians", True),
        )

    @classmethod
    def from_cfg(
        cls,
        cfg: nect.config.GeometryCone | nect.config.Geometry,
        reconstruction_mode: str = "voxel",
        sample_outside: int = 0,
    ) -> "Geometry":
        """
        Load the geometry from a configuration object.

        Args:
            cfg (nect.config.GeometryCone | nect.config.Geometry): The configuration object.
            reconstruction_mode (str): The reconstruction mode. Default is `'voxel'`.
            sample_outside (int): The number of voxels to sample outside the object. Default is `0`.

        Returns:
            Geometry: The geometry object.
        """
        nVoxel = cfg.nVoxel
        if sample_outside > 0:
            nVoxel = [s + 2 * sample_outside for s in nVoxel]
        return cls(
            nDetector=cfg.nDetector,
            dDetector=cfg.dDetector,
            mode=cfg.mode,
            DSD=cfg.DSD if hasattr(cfg, "DSD") else None,
            DSO=cfg.DSO if hasattr(cfg, "DSO") else None,
            nVoxel=nVoxel,
            dVoxel=cfg.dVoxel,
            radius=cfg.radius,
            height=cfg.height,
            offOrigin=cfg.offOrigin,
            COR=cfg.COR,
            offDetector=cfg.offDetector,
            rotDetector=cfg.rotDetector if cfg.rotDetector is not None else (0.0, 0.0, 0.0),
            reconstruction_mode=reconstruction_mode,
            detector_binning=1,
            angles=cfg.angles,
            timesteps=cfg.timesteps,
        )

    def set_angles(self, angles: _list_float | None, radians: bool):
        """
        Set the angles for the geometry.

        Args:
            angles (list[float] | torch.Tensor | np.ndarray | None):
                List of angles.
            radians (bool):
                Unit of angles. If `True`, the unit is radians, if `False` the unit is degrees. Default is `True`"""
        self.angles = angles
        if self.angles is not None:
            if not isinstance(self.angles, np.ndarray):
                if isinstance(self.angles, list):
                    self.angles = np.array(self.angles)
                elif isinstance(self.angles, torch.Tensor):
                    self.angles = self.angles.cpu().numpy()
            if radians is False:
                self.angles = np.radians(self.angles)

    def set_detector_binning(self, detector_binning: int):
        """
        Set the detector binning factor.

        Args:
            detector_binning (int): The binning factor of the detector"""
        self.detector_binning = detector_binning

    def set_timesteps(self, timesteps: _list | None):
        """
        Set the timesteps for dynamic reconstruction.

        Args:
            timesteps (list[float | int] | torch.Tensor | np.ndarray | None):
                An array of timesteps. Do not need to be normalized.
                If the order of the angles and corresponding projections does not equal the acqustition order, this parameter needs to be set to get the timesteps correct.
                Only important for dynamic reconstruction. Overrides the timestep of the Geometry if not `None`.
        """
        self.timesteps = timesteps

    def to_dict(self):
        return {
            "nDetector": self.nDetector,
            "dDetector": self.dDetector,
            "sDetector": self.sDetector,
            "mode": self.mode,
            "DSD": self.DSD,
            "DSO": self.DSO,
            "nVoxel": self.nVoxel,
            "dVoxel": self.dVoxel,
            "sVoxel": self.sVoxel,
            "radius": self.radius,
            "height": self.height,
            "angles": self.angles,
            "radians": True,
            "offOrigin": self.offOrigin,
            "COR": self.COR,
            "offDetector": self.offDetector,
            "rotDetector": self.rotDetector,
            "timesteps": self.timesteps,
        }

    def get_c_geometry(self):
        if self.mode == "cone":
            if self.DSD <= self.DSO:
                raise ValueError("DSD should be greater than DSO")
            if self.DSO <= 0:
                raise ValueError("DSO should be greater than 0")
            if self.DSD <= 0:
                raise ValueError("DSD should be greater than 0")
            if self.reconstruction_mode == "cylindrical":
                c_geometry = nect.sampling.ct_sampling.ConeGeometryCylindrical(
                    distance_source_to_origin=self.DSO,
                    distance_origin_to_detector=self.DSD - self.DSO,
                    nDetector=(
                        self.nDetector[0] // self.detector_binning,
                        self.nDetector[1] // self.detector_binning,
                    ),
                    dDetector=(
                        self.dDetector[0] * self.detector_binning,
                        self.dDetector[1] * self.detector_binning,
                    ),
                    offOrigin=self.offOrigin,
                    offDetector=self.offDetector,
                    rotDetector=self.rotDetector,
                    COR=self.COR,
                    object_radius=self.radius,
                    object_height=self.height,
                    remove_factor_top=0,
                    remove_factor_bottom=0,
                )
            else:
                c_geometry = nect.sampling.ct_sampling.ConeGeometryVoxel(
                    distance_source_to_origin=self.DSO,
                    distance_origin_to_detector=self.DSD - self.DSO,
                    nDetector=(
                        self.nDetector[0] // self.detector_binning,
                        self.nDetector[1] // self.detector_binning,
                    ),
                    dDetector=(
                        self.dDetector[0] * self.detector_binning,
                        self.dDetector[1] * self.detector_binning,
                    ),
                    offOrigin=self.offOrigin,
                    offDetector=self.offDetector,
                    rotDetector=self.rotDetector,
                    COR=self.COR,
                    sVoxel=self.sVoxel,
                    dVoxel=self.dVoxel,
                )
        else:
            if self.reconstruction_mode == "cylindrical":
                c_geometry = nect.sampling.ct_sampling.ParallelGeometryCylindrical(
                    nDetector=(
                        self.nDetector[0] // self.detector_binning,
                        self.nDetector[1] // self.detector_binning,
                    ),
                    dDetector=(
                        self.dDetector[0] * self.detector_binning,
                        self.dDetector[1] * self.detector_binning,
                    ),
                    offOrigin=self.offOrigin,
                    offDetector=self.offDetector,
                    rotDetector=self.rotDetector,
                    COR=self.COR,
                    object_radius=self.radius,
                    object_height=self.height,
                    remove_factor_top=0,
                    remove_factor_bottom=0,
                )
            else:
                c_geometry = nect.sampling.ct_sampling.ParallelGeometryVoxel(
                    nDetector=(
                        self.nDetector[0] // self.detector_binning,
                        self.nDetector[1] // self.detector_binning,
                    ),
                    dDetector=(
                        self.dDetector[0] * self.detector_binning,
                        self.dDetector[1] * self.detector_binning,
                    ),
                    offOrigin=self.offOrigin,
                    offDetector=self.offDetector,
                    rotDetector=self.rotDetector,
                    COR=self.COR,
                    sVoxel=self.sVoxel,
                    dVoxel=self.dVoxel,
                )
        return c_geometry

    def _check_and_return_float(self, t, length: int, var_name: str):
        t = self._check_and_return_tuple(t, length, var_name)
        for i in range(length):
            if not isinstance(t[i], (float, int)):
                raise ValueError(f"{var_name}[{i}] should be a float")
        return tuple(float(t[i]) for i in range(length))

    def _check_and_return_int(self, t, length: int, var_name: str):
        t = self._check_and_return_tuple(t, length, var_name)
        for i in range(length):
            if not isinstance(t[i], int):
                if isinstance(t[i], float):
                    if not t[i].is_integer():
                        raise ValueError(f"{var_name}[{i}] should be an integer")
                else:
                    raise ValueError(f"{var_name}[{i}] should be an integer")
            return tuple(int(t[i]) for i in range(length))

    def _check_and_return_tuple(self, t, length: int, var_name: str):
        if t is None:
            if var_name in "Detector":
                raise ValueError(f"{var_name} is required")
            raise ValueError(f"{var_name} is required for {self.reconstruction_mode} reconstruction")
        else:
            if isinstance(t, (list, tuple)):
                if len(t) != length:
                    raise ValueError(f"{var_name} should have {length} elements")
            elif isinstance(t, (np.ndarray, torch.Tensor)):
                if t.shape != (length,):
                    raise ValueError(f"{var_name} should have {length} elements")
            return tuple(t[i] for i in range(length))
