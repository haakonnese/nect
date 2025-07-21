from pathlib import Path

import numpy as np
import tifffile as tif
import torch
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt

from nect.config import get_cfg
from nect.utils import setup_logger
from nect.sampling import Geometry
from nect.data import NeCTDataset

def export_volume(
    base_path: str | Path,
    binning: int = 1,
    show_slices: bool = False,
    ROIx: list[int] | None = None,
    ROIy: list[int] | None = None,
    ROIz: list[int] | None = None,
) -> Path:
    """
    Exports volume from the static model output. The volume will be saved in the base_path/volumes directory.

    Args:
        base_path (str | Path): Path to the directory containing the config.yaml and checkpoints folder.
        binning (int, optional): Binning factor. Defaults to 1.
        show_slices (bool, optional): If True, will show slices of the volume instead of saving it. Defaults to False.
        ROIx (list[int] | None, optional): Region of interest in x direction. Defaults to None. If None, the ROI will be the full range.
        ROIy (list[int] | None, optional): Region of interest in y direction. Defaults to None. If None, the ROI will be the full range.
        ROIz (list[int] | None, optional): Region of interest in z direction. Defaults to None. If None, the ROI will be the full range.

    Returns:
        Path to the saved volumes.
    """
    setup_logger()
    base_path = Path(base_path)
    with torch.no_grad():  # use torch.no_grad() to disable gradient computation and avoid retaining graph
        config = get_cfg(base_path / "config.yaml")
        assert config.geometry is not None
        model = config.get_model()
        dataset = NeCTDataset(
            config=config,
            device="cpu",  # if gpu memory is less than 50 GB, load to cpu
        )
        geometry = Geometry.from_cfg(
            config.geometry,
            reconstruction_mode=config.reconstruction_mode,
            sample_outside=config.sample_outside,
        )
        device = torch.device(0)
        checkpoints = torch.load(base_path / "checkpoints" / "last.ckpt", map_location="cpu")
        model.load_state_dict(checkpoints["model"])
        model = model.to(device)
        height, width = config.geometry.nVoxel[0], config.geometry.nVoxel[1]
        z_h = height // binning
        y_w = width // binning
        x_w = width // binning
        base_output_path = base_path / "volumefloat32"
        base_output_path.mkdir(exist_ok=True, parents=True)
        total_volumes_saved = 0
        nVoxels = config.geometry.nVoxel
        rm = config.sample_outside
        nVoxels = [nVoxels[0], nVoxels[1]+2*rm, nVoxels[2]+2*rm]
        start_x = 0
        end_x = 1
        if ROIx is not None:
            start_x = (ROIx[0] - rm) / nVoxels[2]
            end_x = (ROIx[1] - rm) / nVoxels[2]
            x_w = (ROIx[1]-ROIx[0]) // binning
            
        start_y = 0
        end_y = 1
        if ROIy is not None:
            start_y = (ROIy[0] - rm) / nVoxels[1]
            end_y = (ROIy[1] - rm) / nVoxels[1]
            y_w = (ROIy[1]-ROIy[0]) // binning
            
        start_z = 0
        end_z = 1
        if ROIz is not None:
            start_z = (ROIz[0]) / nVoxels[0]
            end_z = (ROIz[1]) / nVoxels[0]
            z_h = (ROIz[1]-ROIz[0]) // binning
        if show_slices:
            for slice_idx in ["z", "y", "x"]:
                if slice_idx == "z":
                    size = (y_w, x_w)
                elif slice_idx == "y":
                    size = (z_h, x_w)
                elif slice_idx == "x":
                    size = (z_h, y_w)
                default_tensor = torch.tensor(0.5, device=device)
                z_l = torch.linspace(start_z, end_z, steps=z_h, device=device) if slice_idx != "z" else default_tensor
                y_l = torch.linspace(start_y, end_y, steps=y_w, device=device) if slice_idx != "y" else default_tensor
                x_l = torch.linspace(start_x, end_x, steps=x_w, device=device) if slice_idx != "x" else default_tensor
                z, y, x = torch.meshgrid([z_l, y_l, x_l], indexing="ij")
                grid = torch.stack((z.flatten(), y.flatten(), x.flatten()))
                output = model(grid, torch.tensor(0.5, device=device)).view(size).cpu().numpy()
                plt.imshow(output, cmap="gray")
                (base_path / "imgs").mkdir(parents=True, exist_ok=True)
                plt.savefig(base_path / "imgs" / f"{slice_idx}.png")
            return base_path / "imgs"
                
        else: 
            output = torch.zeros((z_h, y_w, x_w), device="cpu", dtype=torch.float32)
            output = output.flatten()
            z_lin = torch.linspace(start_z, end_z, steps=z_h, dtype=torch.float32, device="cpu")
            y_lin = torch.linspace(start_y, end_y, steps=y_w, dtype=torch.float32, device="cpu")
            x_lin = torch.linspace(start_x, end_x, steps=x_w, dtype=torch.float32, device="cpu")

            batch_size = 5_000_000

            # Calculate total number of points
            total_points = z_h * y_w * x_w

            # Create a tensor of indices
            indices = torch.arange(total_points, dtype=torch.int64)

            # Split indices into batches
            batches = torch.split(indices, batch_size)

            # Process each batch
            for batch in tqdm(batches):
                # Calculate the z, y, x coordinates using vectorized operations
                z_indices = batch // (y_w * x_w)
                y_indices = (batch % (y_w * x_w)) // x_w
                x_indices = batch % x_w

                z = z_lin[z_indices]
                y = y_lin[y_indices]
                x = x_lin[x_indices]

                grid = torch.stack((z, y, x), dim=1).cuda()                
                batch_output = model(grid).flatten().float().detach().cpu()
                output.view(-1)[batch] = batch_output
            output = output.view((z_h, y_w, x_w))
            output = output / geometry.max_distance_traveled
            output = output * (dataset.maximum.item() - dataset.minimum.item()) + dataset.minimum.item()       
            tif.imsave(base_output_path / f"Static_Volume.tiff", output.numpy())
            total_volumes_saved += 1
            logger.info(f"{total_volumes_saved} volumes saved to {base_output_path}")
            return base_output_path
