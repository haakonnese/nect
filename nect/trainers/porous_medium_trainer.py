from __future__ import annotations

import pickle
import time
from typing import cast

import numpy as np
import torch
import torch.utils.data
from nect.src.evaluation.evaluator import Evaluator
import sys
import matplotlib.pyplot as plt
from nect.trainers.base_trainer import BaseTrainer

torch.autograd.set_detect_anomaly(True)
from nect import src
sys.modules['src'] = src


def load_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)


class PorousMediumTrainer(BaseTrainer):
    def setup_evaluator(self):
        if self.fabric.is_global_zero:
            if self.config.evaluation is None or self.config.evaluation.gt_path is None:
                raise ValueError("GT path must be provided for evaluation")
            porous_medium = load_pickle(self.config.evaluation.gt_path)
            porous_medium.dynamic = True
            self.gt = []
            self.mask = None
            for i in range(11):
                volume = np.rot90(porous_medium.get_phantom(i / 10, scaled=True), axes=(1, 2)).copy()
                if self.mask is None:
                    self.mask = np.zeros_like(volume)
                    size = volume.shape[1]
                    radius = size // 2
                    for j in range(size):
                        for k in range(size):
                            if (j - radius)**2 + (k - radius)**2 < radius**2:
                                self.mask[:, j, k] = 1
                    self.mask = torch.from_numpy(self.mask).to(self.fabric.device).float()
                self.gt.append(volume)
            self.gt = torch.from_numpy(np.array(self.gt)).to(self.fabric.device).float()
            metrics = ["PSNR"]
            # if torch.cuda.get_device_properties(self.fabric.device).total_memory > 100 * 1024**3:
            #     metrics.append("SSIM")
            self.evaluator = Evaluator(metrics=metrics, spatial_dims=3)

    def evaluate(self):
        if self.fabric.is_global_zero:
            evaluation_time = time.perf_counter()
            eval_dict_list = []
            for i in range(10):
                volume = self.create_volume(save=False, timestep=i / 10)
                eval_dict_list.append(self.evaluator.evaluate(cast(torch.Tensor, volume), self.gt[i]))
            evaluation_dict = {}
            for key in eval_dict_list[0].keys():
                evaluation_dict[key] = sum([d[key].item() for d in eval_dict_list]) / len(eval_dict_list)
            evaluation_time = time.perf_counter() - evaluation_time
            self.training_time += evaluation_time
            evaluation_dict["training_time"] = time.perf_counter() - self.training_time
            self.fabric.log_dict(evaluation_dict, step=self.step)
            self.last_evaluation_time = time.perf_counter()
            return evaluation_dict
    
    def generate_image(self, prior: bool = False):
        with torch.no_grad():
            if self.config.points_per_batch == "auto":
                return
            plot = self.config.plot_type
            if plot is None:
                return
            if self.fabric.is_global_zero:
                size = [*self.config.geometry.nVoxel]
                sample_size = [*size]
                rm = self.config.sample_outside
                if rm > 0:
                    sample_size = [size[0], size[1] + 2 * rm, size[2] + 2 * rm]
                if plot == "XZ":
                    size[1] = 1
                elif plot == "YZ":
                    size[2] = 1
                elif plot == "XY":
                    size[0] = 1
                if size[0] * size[1] * size[2] > self.config.points_per_batch:
                    sample_size = [sample_size[i] // 3 for i in range(3)]
                    sample_size = [s if s > 0 else 1 for s in sample_size]
                z, y, x = torch.meshgrid(
                    [
                        torch.linspace(0, 1, steps=sample_size[0]) if plot != "XY" else torch.tensor(0.5),
                        torch.linspace(0, 1, steps=sample_size[1])[slice(rm, -rm) if rm > 0 else slice(None)] if plot != "XZ" else torch.tensor(0.5),
                        torch.linspace(0, 1, steps=sample_size[2])[slice(rm, -rm) if rm > 0 else slice(None)] if plot != "YZ" else torch.tensor(0.5),
                    ],
                    indexing="ij",
                )
                grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t().to(self.fabric.device)
                self.model.eval()
                fig, axes = plt.subplots(2, 3, figsize=(24, 10))
                for i, t in enumerate([0, 0.5, 1]):
                    slicing = (
                        slice(None) if plot != "XY" else self.config.geometry.nVoxel[0] // 2, 
                        slice(None) if plot != "XZ" else self.config.geometry.nVoxel[1] // 2,
                        slice(None) if plot != "YZ" else self.config.geometry.nVoxel[2] // 2
                    )
                    dynamic = (
                        self.model(grid, torch.tensor(t))
                        .squeeze()
                        .reshape(size)
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    dynamic = dynamic / (self.geometry.max_distance_traveled * 2)
                    dynamic = dynamic * (self.dataset.maximum.item() - self.dataset.minimum.item())
                    dynamic = dynamic + self.dataset.minimum.item()
                    dynamic *= self.mask[slicing].cpu().numpy()
                    axes[0, i].imshow(dynamic - self.gt[i*5][slicing].cpu().numpy(), cmap="gray", interpolation="none")
                    axes[1, i].imshow(dynamic, cmap="gray", interpolation="none")
                for ax in axes.ravel():
                    ax.set_axis_off()
                fig.tight_layout()
                save_path = f"{self.image_directory_base}/{self.current_epoch:04}_{self.current_angle:04}.png"
                plt.savefig(save_path, dpi=300)
                plt.close()
            self.last_image_time = time.perf_counter()
