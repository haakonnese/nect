from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from nect.config import get_cfg
from nect.utils import is_fourcc_available

if TYPE_CHECKING:
    from nect.config import Config


def load_prior(img_path: str | Path, img_dim: int | tuple[int, int, int]) -> torch.Tensor:
    """
    Loading prior 3D volume from file.
    Args:
        img_path (str | Path): path to the prior image file
        img_dim (int | tuple): dimension of the image

    """
    img_dim = (img_dim, img_dim, img_dim) if isinstance(img_dim, int) else tuple(img_dim)
    suffix = Path(img_path).suffix
    if suffix == ".npy":
        image = np.load(img_path)
    elif suffix == ".raw":
        npimg = np.fromfile(img_path, dtype=np.single)
        image = npimg.reshape(img_dim)
    else:
        raise NotImplementedError(f"Only npy, npz and raw files implemented, was {suffix}")

    return torch.tensor(image, dtype=torch.float32)


class NeCTDataset(Dataset):
    def __init__(self, config: Config, device: int | str | torch.device = "cpu"):
        """
        Dataset for loading projections from file.
        Args:
            config (Config): configuration object
            device (int | str): device to load the projections to. If the whole dataset is contained in a single file, the device specifies where the projections is loaded to.
                                If the dataset is contained in multiple files, the device specifies where the projections is loaded to when it is accessed.
        """
        self.device = device
        self.config = config
        if config.channel_order is not None and config.channel_order.lower() not in [
            "nwh",
            "nhw",
            "hw",
            "wh",
        ]:
            raise ValueError(f"Only NWH, NHW, WH and HW supported, got {config.channel_order}.")
        self.channel_order = config.channel_order.lower() if config.channel_order is not None else None
        self.geometry = config.geometry
        self.img_files = []
        self.data_suffix = [".npy", ".raw"]
        self.image_suffixes = [".tiff", ".png", ".jpg", ".bmp"]
        self.all_valid_suffixes = [".npy", ".raw", ".tiff", ".png", ".jpg", ".bmp"]

        # the path may either be a single file or a directory
        self.setup_dataset()

        # timesteps can either be defined as a list in the same way as angles, or it is assumed to be linearly spaced
        if isinstance(config.geometry.timesteps, (list, np.ndarray, torch.Tensor )):
            self.timesteps = torch.tensor(config.geometry.timesteps)
            if self.timesteps.size(0) != self.num_timesteps:
                raise ValueError("Number of timesteps must match number of images")
            if torch.max(self.timesteps) > 1:
                self.timesteps = self.timesteps / torch.max(self.timesteps)
        elif config.geometry.timesteps is not None:
            raise ValueError(f"If timesteps is given it must be a list, but got {type(config.geometry.timesteps)}")
        else:
            self.timesteps = torch.linspace(0, 1, steps=self.num_timesteps)

        # angles can either be defined as radians or degrees. Default is radians
        self.angles = torch.tensor(config.geometry.angles)
        if self.config.geometry.radians is False:
            self.angles = self.angles / 180 * np.pi
        if self.config.geometry.invert_angles is True:
            self.angles = np.pi * 2 - self.angles
        if self.config.sparse_view is not None:
            self.angles = self.angles[self.config.sparse_view[0] : self.config.sparse_view[1]]
            self.timesteps = self.timesteps[self.config.sparse_view[0] : self.config.sparse_view[1]]
            if len(self.img_files) == 1:
                self.proj = self.proj[self.config.sparse_view[0] : self.config.sparse_view[1]]
            else:
                self.img_files = self.img_files[self.config.sparse_view[0] : self.config.sparse_view[1]]

    def setup_dataset(self):
        if os.path.isfile(self.config.img_path):
            if Path(self.config.img_path).suffix.lower() in self.data_suffix:
                self.img_files.append(self.config.img_path)
                self._load_projections(self.img_files[0])
                self.num_timesteps = self.proj.size(0)
        else:
            current_max = -float("inf")
            current_min = float("inf")
            min_max_saved = False
            if os.path.exists(os.path.join(self.config.img_path, "_min_max.yaml")):
                min_max_saved = True
                with open(os.path.join(self.config.img_path, "_min_max.yaml"), "r") as f:
                    logger.info("Loading min max")
                    min_max = yaml.safe_load(f)
                    current_max = min_max["max"]
                    current_min = min_max["min"]
            for root, _, files in sorted(os.walk(self.config.img_path)):
                for file in tqdm(sorted(files), desc="Loading files for scaling"):
                    file_path = os.path.join(root, file)
                    # allow image files, npy and raw files
                    if os.path.isfile(file_path) and Path(file_path).suffix.lower() in self.all_valid_suffixes:
                        self.img_files.append(file_path)
                        if min_max_saved is False:
                            # print("Error")
                            self._load_projections(file_path, scale=False, use_torch=False)
                            im = self.proj
                            # print(im.shape)
                            max_val = im.max()
                            min_val = im.min()
                            if max_val > current_max:
                                current_max = max_val
                            if min_val < current_min:
                                current_min = min_val
            self.maximum = torch.tensor(current_max)
            self.minimum = torch.tensor(current_min)
            if min_max_saved is False:
                with open(os.path.join(self.config.img_path, "_min_max.yaml"), "w") as f:
                    yaml.dump({"max": float(current_max), "min": float(current_min)}, f)
            self.num_timesteps = len(self.img_files)

    def scale_proj(self):
        self.proj = self.proj - self.minimum
        self.proj = self.proj / (self.maximum - self.minimum)
        return self.proj

    def _load_projections(self, img_path, scale=True, use_torch=True):
        suffix = Path(img_path).suffix.lower()
        N = int(-1)
        W = int(self.config.geometry.nDetector[1])
        H = int(self.config.geometry.nDetector[0])
        order = {"n": N, "h": H, "w": W}
        if len(self.img_files) == 1:
            if self.channel_order is None:
                self.channel_order = "nhw"
        else:
            if self.channel_order is None:
                self.channel_order = "hw"
        shape = [order[c] for c in self.channel_order]
        if suffix == ".raw":
            proj = np.fromfile(img_path, dtype=np.single)
            if use_torch:
                proj = torch.from_numpy(proj).to(device=self.device)
            proj = proj.reshape(shape)
        elif suffix == ".npy":
            proj = np.load(img_path)
            if use_torch:
                proj = torch.tensor(proj, dtype=torch.float32, device=self.device)

        else:
            image_pil = Image.open(img_path)
            if use_torch:
                transform = transforms.ToTensor()
                proj = transform(image_pil)
                proj = proj.to(device=self.device)
            else:
                proj = np.array(image_pil)

        if suffix not in self.image_suffixes and scale:
            minimum = proj.min()
            proj = proj - minimum
            maximum = proj.max()
            proj = proj / maximum
            self.maximum = maximum
            self.minimum = minimum

        if self.geometry.flip:
            if use_torch:
                if isinstance(proj, torch.Tensor):
                    proj = torch.flip(proj, [-2, -1])
                else:
                    raise ValueError("Sino is not a torch tensor")
            else:
                proj = np.flip(proj, axis=(-2, -1))
        if self.channel_order == "nwh" or self.channel_order == "wh":
            proj = proj.transpose(-1, -2)
        self.proj = proj

    def get_full_projections(self, downsample_projections_factor: int = 1):
        assert downsample_projections_factor is not None, "Downsample factor must be provided"
        if len(self.img_files) == 1:
            self._load_projections(self.img_files[0], use_torch=False, scale=False)
            return self.proj.copy().astype(np.float32)
        else:
            projs = []
            for i, img in enumerate(sorted(self.img_files)):
                if not (i) % downsample_projections_factor == 0:
                    continue
                self._load_projections(img, use_torch=False, scale=False)
                projs.append(self.proj)
            projs = np.array(projs).astype(np.float32)
            logger.info(f"Full projections shape: {projs.shape} with downsample factor {downsample_projections_factor}")
            return projs

    def _get(self, idx):
        if len(self.img_files) == 1:
            return self.proj[idx]
        else:
            self._load_projections(self.img_files[idx], scale=False)
            self.proj = self.scale_proj()
            return self.proj

    def __getitem__(self, idx):
        projections = self._get(idx)
        if self.config.continous_scanning is True:
            if idx == len(self.angles) - 1:
                return (
                    projections,
                    self.angles[idx],
                    self.angles[idx] + (self.angles[idx] - self.angles[idx - 1]),
                    self.timesteps[idx],
                )
            else:
                return (
                    projections,
                    self.angles[idx],
                    self.angles[idx + 1],
                    self.timesteps[idx],
                )
        return projections, self.angles[idx], self.timesteps[idx]

    def __len__(self):
        return len(self.img_files) if len(self.img_files) != 1 else self.proj.size(0)

    def export_video(self, file: str | Path, skip_first: int = 0, num_projections: int = -1):
        """Export all projections as a video, for data analysis. The order of the projections is determined by the time of acqusition.

        Args:
            file (str | Path): The path to the video file.
                               If the file does not exist, it will be created. If the file exists, it will be overwritten.
                               A .mp4 extension will be added if not already present.
            skip_first (int, optional): Number of first projections to skip. Defaults to 0.
            num_projections (int, optional): Number of projections to export. If -1, all projections are exported. Defaults to -1.

        Raises:
            ValueError: If no suitable codec is found for video writing. acv1 and mp4v are checked in that order.
        """
        file = Path(file)
        file = file.with_suffix(".mp4")
        file.parent.mkdir(parents=True, exist_ok=True)
        if is_fourcc_available("avc1"):
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
        elif is_fourcc_available("mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            raise ValueError("No suitable codec found for video writing")
        width = int(self.config.geometry.nDetector[1])
        height = int(self.config.geometry.nDetector[0])
        video = cv2.VideoWriter(
            str(file),
            fourcc,
            10,
            (width, height),
            isColor=False,
        )
        total_projections = len(self)
        if num_projections == -1:
            num_projections = total_projections - skip_first
        num_projections = min(num_projections, total_projections - skip_first)
        logger.info(f"Exporting {num_projections}/{total_projections} projections")
        for idx in tqdm(
            torch.argsort(self.timesteps)[skip_first : num_projections + skip_first], desc="Exporting video"
        ):
            projection = self._get(idx)
            projection = (projection * 255).cpu().numpy().astype(np.uint8)
            video.write(projection)
        video.release()
        logger.info(f"Video saved to {file.absolute()}")


class NeCTDatasetLoaded(NeCTDataset):
    def __init__(
        self,
        config: Config,
        projections: torch.Tensor | np.ndarray,
        device: int | str | torch.device = "cpu",
    ):
        """
        Dataset for loading projections from file.
        Args:
            config (Config): configuration object
            projections (torch.Tensor | np.ndarray): projections already loaded
            device (int | str): device to load the projections to. If the whole dataset is contained in a single file, the device specifies where the projections is loaded to.
                If the dataset is contained in multiple files, the device specifies where the projections is loaded to when it is accessed.
        """
        self.proj = projections
        super().__init__(config, device)

    def setup_dataset(self):
        if isinstance(self.proj, np.ndarray):
            self.proj = torch.tensor(self.proj, dtype=torch.float32)
        self.proj = self.proj.to(device=self.device)
        self.num_timesteps = self.proj.size(0)
        if len(self.proj.size()) != 3:
            raise ValueError("Projections must have shape (N, H, W)")
        self.img_files = ["placeholder_file"]
        self.maximum = self.proj.max()
        self.minimum = self.proj.min()


def export_video_projections(config_path: str | Path, file: str | Path, skip_first: int = 0, num_projections: int = -1):
    """Export all projections as a video, for data analysis. The order of the projections is determined by the time of acqusition.

    Args:
        config_path (str | Path): The path to the configuration file.
        file (str | Path): The path to the video file.
                           If the file does not exist, it will be created. If the file exists, it will be overwritten.
                           A .mp4 extension will be added if not already present.
        skip_first (int, optional): Number of first projections to skip. Defaults to 0.
        num_projections (int, optional): Number of projections to export. If -1, all projections are exported. Defaults to -1.

    Raises:
        ValueError: If no suitable codec is found for video writing. acv1 and mp4v are checked in that order.
    """
    config = get_cfg(config_path)
    dataset = NeCTDataset(config)
    dataset.export_video(file, skip_first, num_projections)
