from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

from nect.utils import load_config

if TYPE_CHECKING:
    import tigre

load_dotenv()


def str_dtype_to_np_dtype(dtype: str) -> np.dtype:
    """Convert a string dtype to a numpy dtype.

    Args:
        dtype (str): The string dtype.

    Returns:
        np.dtype: The numpy dtype.
    """
    if "int8" in dtype:
        return np.dtype(dtype)
    elif "int" in dtype:
        return np.dtype(f"<{dtype}")
    else:
        raise NotImplementedError(f"The dtype {dtype} is not implemented yet.")


def geo_to_yaml(geo, angles, path: Path, radians: bool):
    """Save the geometry and angles to a yaml file.

    Args:
        geo (tigre.geometry): The geometry object.
        angles (np.ndarray): The angles.
        path (Path): The path to save the yaml file.
        radians (bool): Whether the angles are in radians.
    """
    data = {
        "DSD": geo.DSD,
        "DSO": geo.DSO,
        "nDetector": geo.nDetector.tolist(),
        "dDetector": geo.dDetector.tolist(),
        "sDetector": geo.sDetector.tolist(),
        "nVoxel": geo.nVoxel.tolist(),
        "sVoxel": geo.sVoxel.tolist(),
        "dVoxel": geo.dVoxel.tolist(),
        "offOrigin": geo.offOrigin.tolist(),
        "offDetector": geo.offDetector.tolist(),
        "rotDetector": geo.rotDetector.tolist(),
        "COR": geo.COR,
        "accuracy": geo.accuracy,
        "mode": geo.mode,
        "filter": geo.filter,
        "radians": radians,
        "angles": angles.tolist(),
    }

    with open(path, "w") as file:
        for key, value in data.items():
            # Write each key-value pair without indentation
            file.write(f"{key}: ")
            if isinstance(value, list):
                # If value is a list, write it in square brackets without indentation
                file.write("[")
                file.write(", ".join([str(item) for item in value]))
                file.write("]\n")
            else:
                # If value is not a list, write it directly followed by a newline
                file.write(f"{value}\n")


def yaml_to_geo(path: Path) -> tigre.geometry.Geometry:
    """Load the geometry from a yaml file.

    Args:
        path (Path): The path to the yaml file.

    Returns:
        tigre.geometry.Geometry: The geometry object.
    """
    import tigre

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    geo = tigre.geometry()
    geo.DSD = data["DSD"]
    geo.DSO = data["DSO"]
    geo.nDetector = np.array(data["nDetector"]).astype(np.int32)
    geo.dDetector = np.array(data["dDetector"])
    geo.sDetector = np.array(data["sDetector"])
    geo.nVoxel = np.array(data["nVoxel"]).astype(np.int32)
    geo.sVoxel = np.array(data["sVoxel"])
    geo.dVoxel = np.array(data["dVoxel"])
    geo.offOrigin = np.array(data["offOrigin"])
    geo.offDetector = np.array(data["offDetector"])
    geo.rotDetector = np.array(data["rotDetector"])
    geo.COR = data["COR"]
    geo.accuracy = data["accuracy"]
    geo.mode = data["mode"]
    geo.filter = data["filter"]
    return geo


class SciVisDataset:
    PROJECTION_BASE_PATH = os.environ.get("PROJECTION_BASE_PATH")
    assert PROJECTION_BASE_PATH is not None
    PROJECTION_BASE_PATH_STATIC = PROJECTION_BASE_PATH + "/Static"

    def __init__(
        self,
        base_path: Path = Path(PROJECTION_BASE_PATH_STATIC),
        dataset: str = "Teapot",
        scale: bool = False,
    ):
        r"""We read the data from https://klacansky.com/open-scivis-datasets/category-ct.html. Their format is (depth, height, width).
        We reshape it to (width, height, depth), then transpose it to (height, width, depth). Sometimes the data from open scivis is flipped
        upside down. Then we flip it the right way.

        Args:
            base_path (Path, optional): The base path to the dataset. Defaults to Path(PROJECTION_BASE_PATH_STATIC).
            dataset (str, optional): The dataset to load. Defaults to "Teapot".
            scale (bool, optional): Whether to scale the data between 0 and 1. Defaults to False.
        """

        dataset = dataset.lower().capitalize()
        self.base_path = base_path
        self.dataset_name = dataset
        self.cfg = load_config(self.base_path / dataset / "GT" / "config.yaml")
        self.gt_dtype = np.dtype(self.cfg["dtype"])
        self.gt = (
            np.fromfile(
                self.base_path / self.dataset_name / "GT" / self.cfg["file_name"],
                dtype=self.gt_dtype,
            )
            .reshape(tuple(self.cfg["size"]))
            .transpose((1, 0, 2))
        )
        self.gt = self.gt.astype(np.float32)
        self.geo = None
        self.angles = None
        if scale:
            print("Scaling")
            maximum = np.max(self.gt)
            minimum = np.min(self.gt)
            self.gt = (self.gt - minimum) / (maximum - minimum)
            print("Finished scaling")
        self.flip_z = self.cfg.get("flip_z")

        if not self.cfg.get("flip_z"):
            self.gt = np.flipud(self.gt)

    def print_all_datasets(self):
        """Print all the datasets in the base path."""
        for dataset in os.listdir(self.base_path):
            cfg = load_config(self.base_path / dataset / "GT" / "config.yaml")
            print(f"Name: {dataset} Size: {str(cfg['size'])}")

    def generate_projections(
        self,
        method: str = "tigre",
        nangles: int = 49,
        save: bool = False,
        detector_mag: float = 1.0,
        scale: bool = True,
        radians: bool = True,
    ) -> np.ndarray:
        """Generate the projections using the specified method.

        Args:
            method (str, optional): The method to use. Defaults to "tigre". Supported methods are "tigre" and "leap".
            nangles (int, optional): The number of angles. Defaults to 49.
            save (bool, optional): Whether to save the projections. Defaults to False.
            detector_mag (float, optional): The detector magnification. Defaults to 1.0.
            scale (bool, optional): Whether to scale the data between 0 and 1. Defaults to True.
            radians (bool, optional): Whether the angles are in radians. Defaults to True.

        Returns:
            np.ndarray: The projections.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        if method == "tigre":
            print("Generating projections using tigre")
            return self._generate_projections_tigre(
                nangles=nangles,
                save=save,
                detector_mag=detector_mag,
                scale=scale,
                radians=radians,
            )
        elif method == "leap":
            print("Generating projections using leap")
            return self._generate_projections_leap(nangles=nangles, save=save, detector_mag=detector_mag, scale=scale)
        else:
            raise NotImplementedError(
                f"The method {method} is not implemented. Supported methods are 'tigre' and 'leap'."
            )

    def remove_projections(self, nangles_to_remove: int, dataset_to_remove: str = None):
        """Helper function to quickly remove one or all projections with a specific number of angles.

        Args:
            nangles_to_remove (int): Marks the projections with this number of angles to be removed.
            dataset_to_remove (str, optional): If provided, only this projections will be removed. Defaults to None.
        """
        if dataset_to_remove is not None:
            dataset_to_remove = dataset_to_remove.lower().capitalize()
            shutil.rmtree(self.base_path / dataset_to_remove / f"sinogram_nangles_{nangles_to_remove}")
        else:
            for dataset_to_remove in os.listdir(self.base_path):
                path = self.base_path / dataset_to_remove / f"sinogram_nangles_{nangles_to_remove}"
                if path.exists():
                    shutil.rmtree(path)
                else:
                    continue

    def get_scaled_gt(self) -> np.ndarray:
        """Get the scaled ground truth.

        Returns:
            np.ndarray: The scaled ground truth."""
        return (self.gt - np.min(self.gt)) / (np.max(self.gt) - np.min(self.gt))

    def scale_0_to_1(self):
        """Scales the ground truth between 0 and 1."""
        self.gt = (self.gt - np.min(self.gt)) / (np.max(self.gt) - np.min(self.gt))

    def _generate_projections_tigre(
        self,
        nangles: int,
        save: bool = False,
        detector_mag: float = 1.0,
        scale: bool = True,
        radians: bool = True,
    ):
        import pickle

        import tigre

        angles = np.linspace(0, 360, num=nangles, endpoint=False) * (np.pi / 180)
        self.angles = angles
        geo = tigre.geometry()
        circle_radius = (
            self.gt.shape[2] ** 2 + self.gt.shape[1] ** 2
        ) ** 0.5 / 2  # circle radius, to make sure the object is inside the detector
        print(self.gt.shape)
        geo.DSO = max(999, circle_radius) + 1
        geo.DSD = geo.DSO + max(499, circle_radius) + 1
        # tangent_length = (geo.DSO**2 + circle_radius**2)**0.5
        wsDetector = np.tan(np.arcsin(circle_radius / geo.DSO)) * geo.DSD * 2
        print(circle_radius, wsDetector)
        geo.nDetector = (
            np.array([self.gt.shape[0], np.max(self.gt.shape[1:])]) * detector_mag
        )  # multiply by detector_mag to increase size
        dDetector = wsDetector / geo.nDetector[1]
        geo.nDetector = geo.nDetector.astype(np.int32)

        geo.dDetector = np.array([dDetector, dDetector])
        geo.sDetector = geo.nDetector * geo.dDetector
        geo.nVoxel = np.array(self.gt.shape)
        geo.dVoxel = np.array([1.0, 1.0, 1.0])
        geo.sVoxel = geo.nVoxel * geo.dVoxel
        geo.offOrigin = np.array([0, 0, 0])
        geo.offDetector = np.array([0, 0])
        geo.rotDetector = np.array([0.00, 0.0, 0.0])
        geo.accuracy = 0.01
        geo.COR = 0
        geo.mode = "cone"
        geo.filter = None
        self.geo = geo
        print(geo)
        if scale:
            self.scale_0_to_1()

        projections = tigre.Ax(
            self.gt.astype(np.float32),
            geo=geo,
            angles=angles,
            projection_type="interpolated",
        )
        if save:
            folder_path = self.base_path / self.dataset_name / f"sinogram_nangles_{nangles}_interpolated"
            folder_path.mkdir(exist_ok=True, parents=True)
            np.save(folder_path / f"sinogram_nangles_{nangles}.npy", projections)
            with open(folder_path / "geometry.pkl", "wb") as handle:
                pickle.dump(geo, handle, protocol=pickle.HIGHEST_PROTOCOL)
            info = str(geo) + "\n" + "Angles: " + str(angles)
            with open(folder_path / "geometry_info.txt", "w") as text_file:
                text_file.write(info)
            geo_to_yaml(
                geo=geo,
                angles=angles,
                path=folder_path / "geometry.yaml",
                radians=radians,
            )
            print(f"The files were saved to {folder_path}")
        return projections

    def _generate_projections_leap(
        self,
        nangles: int,
        save: bool = False,
        detector_mag: float = 1.0,
        scale: bool = True,
    ):
        import pickle

        import tigre

        from nect.src.simulator.configuration.config import LeapGeometry

        angles = np.linspace(0, 360, num=nangles, endpoint=False)
        self.angles = angles
        geo = tigre.geometry()
        circle_radius = (
            self.gt.shape[2] ** 2 + self.gt.shape[1] ** 2
        ) ** 0.5 / 2  # circle radius, to make sure the object is inside the detector
        print(self.gt.shape)
        geo.DSO = max(999, circle_radius) + 1
        geo.DSD = geo.DSO + max(499, circle_radius) + 1
        # tangent_length = (geo.DSO**2 + circle_radius**2)**0.5
        wsDetector = np.tan(np.arcsin(circle_radius / geo.DSO)) * geo.DSD * 2
        print(circle_radius, wsDetector)
        geo.nDetector = (
            np.array([self.gt.shape[0], np.max(self.gt.shape[1:])]) * detector_mag
        )  # multiply by detector_mag to increase size
        dDetector = wsDetector / geo.nDetector[1]
        geo.nDetector = geo.nDetector.astype(np.int32)

        geo.dDetector = np.array([dDetector, dDetector])
        geo.sDetector = geo.nDetector * geo.dDetector
        geo.nVoxel = np.array(self.gt.shape)
        geo.dVoxel = np.array([1.0, 1.0, 1.0])
        geo.sVoxel = geo.nVoxel * geo.dVoxel
        geo.offOrigin = np.array([0, 0, 0])
        geo.offDetector = np.array([0, 0])
        geo.rotDetector = np.array([0.0, 0.0, 0.0])
        geo.accuracy = 0.01
        geo.COR = 0
        geo.mode = "cone"
        geo.filter = None
        self.geo = geo
        proj = LeapGeometry()
        proj.from_tigre_geometry(geo)
        proj.update_phi(torch.from_numpy(angles).float())
        angles = angles * (np.pi / 180)
        self.angles = angles
        if scale:
            self.scale_0_to_1()
        img = torch.tensor(self.gt, dtype=(torch.float32), device="cuda:0").unsqueeze(0)
        img = img.permute(0, 3, 1, 2).contiguous()
        img = img.rot90(3, (1, 2)).contiguous()
        print(img.shape)
        # plt.imsave("test2.png", img[0,:,100,...].cpu().numpy(), cmap="gray")

        sinogram = proj(img).squeeze(0).detach().cpu().numpy()

        if save:
            folder_path = self.base_path / self.dataset_name / f"sinogram_nangles_{nangles}_LEAP2"
            folder_path.mkdir(exist_ok=True, parents=True)
            np.save(folder_path / f"sinogram_nangles_{nangles}.npy", sinogram)
            with open(folder_path / "geometry.pkl", "wb") as handle:
                pickle.dump(geo, handle, protocol=pickle.HIGHEST_PROTOCOL)
            info = str(geo) + "\n" + "Angles: " + str(angles)
            with open(folder_path / "geometry_info.txt", "w") as text_file:
                text_file.write(info)
            geo_to_yaml(geo=geo, angles=angles, path=folder_path / "geometry.yaml", radians=True)
            print(f"The files were saved to {folder_path}")
        return sinogram

    def save_as_quadrants(self):
        """Save the ground truth as 8 quadrants. Used for large datasets to save memory"""
        # save the gt into 8 quadrants
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    quad = self.gt[
                        i * self.gt.shape[0] // 2 : (i + 1) * self.gt.shape[0] // 2,
                        j * self.gt.shape[1] // 2 : (j + 1) * self.gt.shape[1] // 2,
                        k * self.gt.shape[2] // 2 : (k + 1) * self.gt.shape[2] // 2,
                    ]
                    print("Finished indexing")
                    np.save(
                        self.base_path / self.dataset_name / "GT/quadrants" / f"quadrant_z_{i}_y_{j}_x_{k}.npy",
                        quad,
                    )
                    print(f"Finished {i} {j} {k}")


if __name__ == "__main__":
    datasets = ["Engine", "Colon", "Stagbeetle"]
    for dataset in datasets:
        data = SciVisDataset(dataset=dataset, scale=True)
        for angle in [5, 9, 15, 25, 35, 45]:
            projections = data.generate_projections(save=True, nangles=angle, method="tigre")
