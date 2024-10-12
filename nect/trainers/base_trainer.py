from __future__ import annotations

import datetime
import logging
import math
import os
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast
import shutil

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from loguru import logger
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from torchinfo import summary
from tqdm import tqdm

import nect
from nect.data import NeCTDataset
from nect.network import KPlanes
from nect.network.kplanes import regularize_k_planes
from nect.utils import create_sub_folders, setup_logger, total_variation_3d, prune_model

if TYPE_CHECKING:
    from nect.config import Config
torch.autograd.set_detect_anomaly(True)


class BaseTrainer:
    def __init__(
        self,
        config: Config,
        output_directory: str | Path | None = None,
        checkpoint: str | Path | None = None,
        save_ckpt: bool = True,
        save_last: bool = True,
        save_optimizer: bool = True,
        verbose: bool = True,
        log: bool = True,
        cancel_at: str | None = None,
        keep_two: bool = True,
        prune: bool = True,
    ):
        setup_logger(level=logging.INFO if verbose else logging.WARNING)
        if checkpoint:
            output_directory = Path(checkpoint).parent.parent.parent
        elif output_directory is not None:
            specific_run = datetime.datetime.now().replace(microsecond=0).isoformat().replace(":","-")
            output_directory = os.path.join(output_directory, config.get_str(), specific_run)
        self.use_prior = config.use_prior
        self.save_optimizer = save_optimizer
        self.keep_two = keep_two
        if self.use_prior:
            prior_path = Path(config.img_path)
            self.prior = np.load(prior_path).astype(np.float32)
        L.seed_everything(42)
        self.logger = logger
        self.logger.info("-------------------")
        self.logger.info(f"SEEING NUMBER OF GPUS {torch.cuda.device_count()}")
        self.logger.info("-------------------")
        torch.set_float32_matmul_precision("high")
        self.cancel_at = None
        self.prune = prune
        if cancel_at is not None:
            try:
                self.cancel_at = datetime.datetime.fromisoformat(cancel_at)
            except ValueError as e:
                raise ValueError(f"Tried set cancel of job at '{cancel_at}', which is an invalid ISO-datetime.")
        self.config = config
        self.setup_dataset()
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=1, shuffle=True, num_workers=config.num_workers
        )
        if isinstance(self.config.epochs, str):
            fraction = self.config.epochs.split("x")
            self.config.epochs = math.ceil(49 / len(self.dataset) * max(config.geometry.nDetector))
            if len(fraction) == 2:
                self.config.epochs = math.ceil(self.config.epochs * float(fraction[0]))
        self.loss_fn = config.get_loss_fn()

        if output_directory is None:
            if save_ckpt or save_last or log:
                raise ValueError("Output directory must be provided if logging or saving checkpoints")
        if log:
            if output_directory is None:
                # raise a error that is due to coding error and not the user
                raise ValueError(
                    "Output directory must be provided if logging or saving checkpoints. This is a bug in the code."
                )
            tensorboard_logger = TensorBoardLogger(root_dir=output_directory, name="logs")
        else:
            self.config.image_interval = -1
            tensorboard_logger = None
        self.verbose = verbose
        if verbose is False:
            self.tqdm = lambda x, *args, **kwargs: x
        else:
            self.tqdm = tqdm
        self.fabric = L.Fabric(
            accelerator="cuda",
            devices="auto",
            strategy="auto",
            precision="16-mixed",
            loggers=tensorboard_logger,
        )

        # supress warnings from self.fabric.launch()
        warnings.filterwarnings("ignore")
        self.fabric.launch()
        warnings.resetwarnings()
        self.model = config.get_model()
        self.optim = config.get_optimizer(self.model)
        (
            self.lr_scheduler_warmup,
            self.lr_scheduler,
            self.lr_scheduler_warmup_downsample,
        ) = config.get_lr_schedulers(self.optim)
        self.current_epoch = 0
        self.current_angle = 0
        self.angle = 0.0
        self.current_projection = 0
        if config.s3im:
            self.s3im_loss = config.get_s3im_loss()

        self.downsample_detector_factor = config.downsampling_detector.start
        self.points_per_ray = config.points_per_ray.start
        self.geometry = nect.Geometry.from_cfg(
            config.geometry,
            reconstruction_mode=config.reconstruction_mode,
            sample_outside=config.sample_outside,
        )
        if config.points_per_batch == "auto":
            raise ValueError("`points_per_batch` should already have been calculated at this point.")
        self.projector = nect.sampling.Projector(
            geometry=self.geometry,
            points_per_batch=config.points_per_batch // 2,
            points_per_ray=self.points_per_ray,
            device=self.fabric.device,
            uniform_ray_spacing=config.uniform_ray_spacing,
        )

        self.model, self.optim = self.fabric.setup(self.model, self.optim)
        if hasattr(self.model, "encoder"):
            self.model.encoder.B = self.fabric.to_device(self.model.encoder.B)

        self.dataloader = cast(torch.utils.data.DataLoader, self.fabric.setup_dataloaders(self.dataloader))
        if config.points_per_ray.end == "auto":
            config.points_per_ray.end = math.ceil(max(config.geometry.nDetector) * 1.5)
        elif isinstance(config.points_per_ray.end, str):
            end_str = config.points_per_ray.end.split("x")
            if len(end_str) == 2:
                config.points_per_ray.end = math.ceil(max(config.geometry.nDetector) * float(end_str[0]))
            else:
                raise ValueError("Invalid format for `points_per_ray.end`")
        if isinstance(config.points_per_ray.update_interval, str):
            if config.points_per_ray.update_interval == "auto":
                factor = 0.9
            else:
                update_interval = config.points_per_ray.update_interval.split("x")
                if len(update_interval) == 2:
                    factor = float(update_interval[0])
                else:
                    raise ValueError(f"Invalid format for `points_per_ray.update_interval`, got '{config.points_per_ray.update_interval}'")
            projections = len(self.dataset)
            epochs = self.config.epochs * factor
            total_updates = projections * epochs / torch.cuda.device_count()
            number_of_updates_needed = config.points_per_ray.end - config.points_per_ray.start
            config.points_per_ray.update_interval = math.ceil(total_updates / number_of_updates_needed)

        if self.fabric.is_global_zero and output_directory is not None:
            self.checkpoint_directory_base, self.image_directory_base = create_sub_folders(output_directory)
        else:
            self.checkpoint_directory_base = "needs_to_be_defined_but_not_used"  # must be defined
            self.image_directory_base = "needs_to_be_defined_but_not_used"
        self.last_checkpoint_time = time.perf_counter()
        self.last_image_time = time.perf_counter()
        self.last_evaluation_time = time.perf_counter()
        self.save = save_ckpt
        self.save_last = save_last
        self.use_checkpoint = False
        self.downsample_warmup_iteration = 0
        if checkpoint:
            self.logger.info("LOADING CHEKPOINT")
            checkpoint_data = self.fabric.load(checkpoint, strict=True)
            self.current_angle = checkpoint_data["angle"]
            self.current_epoch = checkpoint_data["epoch"]
            self.model.load_state_dict(checkpoint_data["model"])
            self.optim.load_state_dict(checkpoint_data["optim"])
            self.logger.info(f"Starting from epoch {self.current_epoch} and angle number {self.current_angle}")
            self.dataloader._num_iter_calls = self.current_epoch
            self.use_checkpoint = True
            self.current_projection = self.current_epoch * config.geometry.num_angles + self.current_angle
            self.downsample_detector_factor = max(1, self.config.downsampling_detector.start // 2 ** (self.current_projection // self.config.downsampling_detector.update_interval)) 
            if self.config.points_per_ray.linear:
                self.points_per_ray = min(self.config.points_per_ray.end, self.config.points_per_ray.start + self.current_projection // self.config.points_per_ray.update_interval)
            else:
                self.points_per_ray = min(self.config.points_per_ray.end, self.config.points_per_ray.start * 2 ** (self.current_projection // self.config.points_per_ray.update_interval))
            self.projector.update(
                angle=self.angle,
                detector_binning=self.downsample_detector_factor,
                points_per_ray=self.points_per_ray,
                random_offset_detector=0.5 if self.downsample_detector_factor > 1 else 0,
            )
            self.step = 0
            self.lr_scheduler.load_state_dict(checkpoint_data["lr_scheduler"])
        self.setup_evaluator()
        self.batch_per_proj = config.batch_per_proj
        if self.batch_per_proj == "all":
            self.batch_per_proj = 1000000000

        if config.mode == "static" and log:
            model_summary = summary(self.model, input_size=((1, 3)))
            # save the model summary to a txt file
            with open(f"{Path(self.image_directory_base).parent/'model_summary.txt'}", "w") as file:
                file.write(str(model_summary))
        if self.fabric.is_global_zero and checkpoint is None and output_directory is not None:
            config.save(output_directory)

    def setup_dataset(self):
        self.dataset = NeCTDataset(
            config=self.config,
            device="cpu",  # if gpu memory is less than 50 GB, load to cpu
        )

    def setup_evaluator(self):
        pass

    def set_projection(self, projection):
        """
        Set the projection. This is used to update the downsampling factor and points per ray.

        Args:
            projection (int): current projection
        """
        if isinstance(self.config.points_per_ray.end, str):
            raise ValueError("`points_per_ray.end` should already have been calculated at this point.")
        self.projection = projection
        if (
            projection != 0
            and projection % self.config.downsampling_detector.update_interval == 0
            and self.downsample_detector_factor > self.config.downsampling_detector.end
        ):
            self.downsample_detector_factor //= 2
        if (
            projection != 0
            and projection % self.config.points_per_ray.update_interval == 0
            and self.points_per_ray < self.config.points_per_ray.end
            and self.config.points_per_ray.linear is False
        ):
            self.points_per_ray *= 2
        elif (
            projection != 0
            and projection % self.config.points_per_ray.update_interval == 0
            and self.points_per_ray < self.config.points_per_ray.end
            and self.config.points_per_ray.linear is True
        ):
            self.points_per_ray += 1
        self.projector.update(
            angle=self.angle,
            detector_binning=self.downsample_detector_factor,
            points_per_ray=self.points_per_ray,
            random_offset_detector=0.5 if self.downsample_detector_factor > 1 else 0,
        )

    def on_train_epoch_start(self):
        if self.use_checkpoint:
            self.use_checkpoint = False
        else:
            self.current_angle = 0

    def on_train_epoch_end(self):
        self.current_epoch = self.current_epoch + 1

    def on_angle_end(self):
        if self.cancel_at is not None and datetime.datetime.now(datetime.UTC) > self.cancel_at:
            self.save_model(last=True)
            exit()
        self.current_projection = self.current_projection + 1
        self.lr_scheduler.step()
        if self.current_projection <= self.config.warmup.steps:
            self.lr_scheduler_warmup.step()

    def on_angle_start(self, proj, angle):
        self.proj = proj
        self.angle = angle
        self.set_projection(self.current_projection)
        time_since_last_image = time.perf_counter() - self.last_image_time
        if time_since_last_image > self.config.image_interval and self.config.image_interval > 0:
            self.generate_image()
        time_since_last_checkpoint = time.perf_counter() - self.last_checkpoint_time
        if time_since_last_checkpoint > self.config.checkpoint_interval and self.config.checkpoint_interval > 0:
            self.save_model()
        time_last_evaluation = time.perf_counter() - self.last_evaluation_time
        if (
            self.config.evaluation is not None
            and self.config.evaluation.evaluate_interval > 0
            and time_last_evaluation > self.config.evaluation.evaluate_interval
        ):
            self.evaluate()
        if self.downsample_detector_factor != 1:
            self.proj = F.avg_pool2d(
                self.proj.unsqueeze(0),
                kernel_size=self.downsample_detector_factor,
                stride=self.downsample_detector_factor,
            ).squeeze(0)
        self.proj = self.proj.flatten()
        self.current_angle = self.current_angle + 1

    def evaluate(self):
        pass

    def save_volume(self, save_path: str | None = None) -> None:
        if self.fabric.is_global_zero:
            self.create_volume(save=True, save_path=save_path)

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
                if self.config.mode == "dynamic":
                    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
                    avg = self.model(grid, torch.tensor(0)).squeeze().reshape(size).squeeze().detach().cpu().numpy()
                    for i in range(3):
                        dynamic = (
                            self.model(grid, torch.tensor((i + 1) / 4))
                            .squeeze()
                            .reshape(size)
                            .squeeze()
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        axes[0, i].imshow(dynamic - avg, cmap="gray", interpolation="none")
                        dynamic = dynamic / (self.geometry.max_distance_traveled * 2)
                        dynamic = dynamic * (self.dataset.maximum.item() - self.dataset.minimum.item())
                        dynamic = dynamic + self.dataset.minimum.item()
                        axes[1, i].imshow(dynamic, cmap="gray", interpolation="none")
                    for ax in axes.ravel():
                        ax.set_axis_off()
                    fig.tight_layout()
                else:
                    if size[0] * size[1] * size[2] < self.config.points_per_batch:
                        output = self.model(grid).squeeze().reshape(size).squeeze().detach().cpu().numpy()
                    else:
                        output = torch.zeros(size).numpy()
                        for i in range(size[0]):
                            output[i] = (
                                self.model(grid[i * size[1] * size[2] : (i + 1) * size[1] * size[2]])
                                .squeeze(0)
                                .reshape(size[1], size[2])
                                .squeeze(1)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                    output = output / self.geometry.max_distance_traveled
                    output = output * (self.dataset.maximum.item() - self.dataset.minimum.item())
                    output = output + self.dataset.minimum.item()
                    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
                    axes[0].hist(output.flatten(), bins=100)
                    axes[1].imshow(output, cmap="gray", interpolation="none")
                save_path = f"{self.image_directory_base}/{self.current_epoch:04}_{self.current_angle:04}.png"
                plt.savefig(save_path, dpi=300)
                plt.close()
            self.last_image_time = time.perf_counter()

    def create_volume(self, save=True, save_path: str | None = None, timestep: float | None = None, cpu=False):
        if self.config.mode == "dynamic" and timestep is None:
            return
        with torch.no_grad():
            if self.fabric.is_global_zero:
                size = tuple([*self.config.geometry.nVoxel])
                rm = self.config.sample_outside
                sample_size = [size[0], size[1] + 2 * rm, size[2] + 2 * rm]
                output = torch.zeros(size, dtype=torch.float32, device=self.fabric.device if not cpu else "cpu")
                z_lin = torch.linspace(
                    0.5 / size[0],
                    1 - 0.5 / size[0],
                    steps=size[0],
                    device=self.fabric.device,
                )
                for i in range(size[0]):
                    z, y, x = torch.meshgrid(
                        [
                            z_lin[i],
                            torch.linspace(
                                0.5 / sample_size[1],
                                1 - 0.5 / sample_size[1],
                                steps=sample_size[1],
                                device=self.fabric.device,
                            )[slice(rm, -rm) if rm > 0 else slice(None)],
                            torch.linspace(
                                0.5 / sample_size[2],
                                1 - 0.5 / sample_size[2],
                                steps=sample_size[2],
                                device=self.fabric.device,
                            )[slice(rm, -rm) if rm > 0 else slice(None)],
                        ],
                        indexing="ij",
                    )

                    grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t()
                    if self.config.mode == "static":
                        output_slice = self.model(grid).reshape(size[1], size[2])
                    else:
                        output_slice = self.model(grid, timestep).reshape(size[1], size[2])
                    output_slice = output_slice / self.geometry.max_distance_traveled
                    output_slice = output_slice * (self.dataset.maximum.item() - self.dataset.minimum.item())
                    output_slice = output_slice + self.dataset.minimum.item()
                    output[i] = output_slice if not cpu else output_slice.cpu()
                if save:
                    if save_path is None:
                        save_path = f"{self.image_directory_base}/final.npy"
                    np.save(f"{self.image_directory_base}/final.npy", output.cpu().numpy())
                return output.float()

    def save_model(self, last=False):
        if self.save or (last and self.save_last):
            if self.save_optimizer:
                state = {
                    "model": self.model,
                    "optim": self.optim,
                    "epoch": self.current_epoch,
                    "angle": self.current_angle,
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                }
            else:
                state = {
                    "model": self.model,
                    "epoch": self.current_epoch,
                    "angle": self.current_angle,
                }
            if self.keep_two:
                last_ckpt = os.path.join(self.checkpoint_directory_base, "last.ckpt")
                if os.path.exists(last_ckpt):
                    second_last_ckpt = os.path.join(self.checkpoint_directory_base, "2nd_last.ckpt")
                    if os.path.exists(second_last_ckpt):
                        os.remove(second_last_ckpt)
                    shutil.move(last_ckpt, second_last_ckpt)
            self.logger.info("Saving model - time might take some time")
            self.fabric.save(os.path.join(self.checkpoint_directory_base, "last.ckpt"), state)
            self.logger.info("Saving model finished")
            self.create_volume(self.config.save_volume)
            self.last_checkpoint_time = time.perf_counter()
        elif last:
            self.create_volume(self.config.save_volume)
            self.last_checkpoint_time = time.perf_counter()
        if last and self.prune and self.fabric.is_global_zero:
            prune_model(self.model, Path(self.checkpoint_directory_base).parent)
        
    def fit(self):
        try:
            self.step = 0
            self.training_time = time.perf_counter()
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(0)
            for epoch in self.tqdm(
                range(self.current_epoch, self.config.epochs),
                total=self.config.epochs,
                initial=self.current_epoch,
                leave=True,
                desc="Epochs",
            ):
                self.on_train_epoch_start()
                tqdm_bar = self.tqdm(
                    enumerate(self.dataloader),
                    total=len(self.dataloader),
                    leave=False,
                    desc="Projections",
                )
                for i, (proj, angle, timestep) in tqdm_bar:
                    if i < self.current_angle:
                        continue
                    self.on_angle_start(proj, angle)

                    memory_info = nvmlDeviceGetMemoryInfo(h)
                    if self.verbose:
                        tqdm_bar.set_postfix(
                            {"GPU mem%": f"{round(int(memory_info.used)/1024**3, 1)}/{int(memory_info.total)/1024**3}G"}
                        )
                        tqdm_bar.refresh()
                    for batch_num in range(min(cast(int, self.batch_per_proj), self.projector.batch_per_epoch)):
                        self.optim.zero_grad()
                        self.model.train()
                        points, y = self.projector(batch_num=batch_num, proj=self.proj)
                        if points is None or y is None:
                            continue
                        zero_points_mask = torch.all(points.view(-1, 3) == 0, dim=-1)
                        points_shape = points.size()
                        if points_shape[1] == 0:
                            self.logger.warning("No points in the batch")
                            continue
                        points = points.view(-1, 3)[~zero_points_mask]
                        if points.size(0) == 0:
                            continue
                        atten_hats = []
                        points_per_batch = 5000000  # 5 million points per batch is about the maximum that can be processed at once with tinycudann
                        for points_num in range(0, points.size(0), points_per_batch):
                            if self.config.mode == "dynamic":
                                atten_hat = self.model(
                                    points[points_num : points_num + points_per_batch],
                                    float(timestep),
                                ).squeeze(0)  # .view((points.size(0), points.size(1)))
                            else:
                                atten_hat = self.model(points[points_num : points_num + points_per_batch]).squeeze(
                                    0
                                )  # .view((points.size(0), points.size(1)))
                            atten_hats.append(atten_hat)
                        atten_hat = torch.cat(atten_hats)
                        processed_tensor = torch.zeros(
                            (points_shape[0], points_shape[1], 1),
                            dtype=torch.float32,
                            device=self.fabric.device,
                        ).view(-1, 1)
                        processed_tensor[~zero_points_mask] = atten_hat
                        atten_hat = processed_tensor.view(points_shape[0], points_shape[1])
                        y_pred = torch.sum(atten_hat, dim=1) * (
                            self.projector.distances / (self.geometry.max_distance_traveled)
                        )  # * (self.ct_sampler.distance_between_points / self.geometry.max_distance_traveled)
                        if self.config.add_poisson:
                            y_pred = (y_pred + torch.poisson(y_pred * 1e5) / 1e5) / 2
                        if self.config.s3im and self.current_projection > self.config.warmup.steps:
                            loss = 0
                            patch_size = min(
                                math.floor(math.sqrt(self.projector.total_detector_pixels)),
                                math.floor(math.sqrt(self.batch_size)),
                            )  # 25x25 patch size, add a parameter later

                            self.fabric.log_dict({"patch_size": patch_size}, step=self.step)
                            loss += self.loss_fn(y_pred, y, i)
                            loss += self.s3im_loss(y_pred, y, patch_size=patch_size)
                        else:
                            loss = self.loss_fn(y_pred, y, i)
                            """
                        if self.use_prior:
                            z, y, x = torch.meshgrid(
                                [
                                    torch.linspace(
                                        0, 1, steps=1000,
                                        device=self.fabric.device,
                                    ),
                                    torch.linspace(
                                        0, 1, steps=1000,
                                        device=self.fabric.device,
                                    ),
                                    torch.linspace(
                                        0, 1, steps=1000,
                                        device=self.fabric.device,
                                    ),
                                ],
                                indexing="ij",
                            )
                            grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).view(3, -1).t()
                            points_dry = self.model(grid, 0).view(1000, 1000 ,1000)
                            points_wet = self.model(grid, 1).view(1000, 1000 ,1000)
                            loss += torch.functional.loss.l1(self.prior_volume, points_dry)
                            loss += torch.functional.loss.l1(self.prior_volume, points_wet)
                            """
                        if isinstance(self.model, KPlanes):
                            regularize_k_planes(self.config.encoder, self.model)
                        self.fabric.log_dict(
                            {
                                "loss": loss,
                                "max_mem": torch.cuda.max_memory_allocated(),
                                "current_mem": int(memory_info.used),
                                "epoch": epoch,
                                "downsample_detector_factor": self.downsample_detector_factor,
                                "points_per_ray": self.points_per_ray,
                                "distance_between_points": self.projector.distance_between_points,
                                "lr": self.optim.param_groups[0]["lr"],
                                "num_proj_processed": self.current_projection,
                            },
                            step=self.step,
                        )
                        if hasattr(self.model, "skip_alpha"):
                            self.fabric.log_dict(
                                {"skip_alpha_value": self.model.skip_alpha.item()},
                                step=self.step,
                            )
                        if (
                            self.config.mode == "static"
                            and self.current_projection > self.config.warmup.steps
                            and self.config.tv > 0
                        ):
                            rand_zyx = np.random.rand(3) * 0.8
                            z, y, x = torch.meshgrid(
                                [
                                    torch.linspace(
                                        rand_zyx[0],
                                        rand_zyx[0] + 0.2,
                                        steps=100,
                                        device=self.fabric.device,
                                    ),
                                    torch.linspace(
                                        rand_zyx[1],
                                        rand_zyx[1] + 0.2,
                                        steps=100,
                                        device=self.fabric.device,
                                    ),
                                    torch.linspace(
                                        rand_zyx[2],
                                        rand_zyx[2] + 0.2,
                                        steps=100,
                                        device=self.fabric.device,
                                    ),
                                ],
                                indexing="ij",
                            )
                            grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).view(3, -1).t()
                            grid = grid.to(self.fabric.device)
                            atten_hat = self.model(grid).squeeze(0)  # .view((points.size(0), points.size(1)))
                            atten_hat = atten_hat.view(100, 100, 100)
                            tv_loss = total_variation_3d(atten_hat, weight=self.config.tv)
                            loss += tv_loss
                        self.fabric.backward(loss)
                        if self.config.clip_grad_value is not None:
                            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip_grad_value)
                        self.optim.step()
                        self.step += 1
                    self.on_angle_end()
                self.on_train_epoch_end()
            self.evaluate()
            self.save_model(last=True)
        except KeyboardInterrupt:
            if self.step > 3000:
                self.logger.info("Please wait before canceling again as the model now is beeing saved")
                self.save_model(last=True)
            else:
                self.logger.info(
                    f"The model was trained for {self.step} steps, so it will not save the model before exiting."
                )
