from __future__ import annotations

import time
from typing import cast

import numpy as np
import torch
import torch.utils.data

from nect.scivis_data import SciVisDataset
from nect.src.evaluation.evaluator import Evaluator
from nect.trainers.base_trainer import BaseTrainer

torch.autograd.set_detect_anomaly(True)


class SciVisTrainer(BaseTrainer):
    def setup_evaluator(self):
        if self.config.evaluation is None or self.config.evaluation.gt_path is None:
            raise ValueError("GT path must be provided for evaluation")
        gt = SciVisDataset(dataset=self.config.evaluation.gt_path).get_scaled_gt()
        self.gt = torch.from_numpy(np.rot90(gt, 2, (1, 2)).copy()).cuda().float()
        metrics = ["PSNR"]
        if torch.cuda.get_device_properties(self.fabric.device).total_memory > 50 * 1024**3:
            metrics.append("SSIM")
        self.evaluator = Evaluator(metrics=metrics, spatial_dims=3)

    def evaluate(self):
        if self.fabric.is_global_zero:
            evaluation_time = time.perf_counter()
            volume = self.create_volume(save=False)
            evaluation_dict = self.evaluator.evaluate(cast(torch.Tensor, volume), self.gt)
            evaluation_time = time.perf_counter() - evaluation_time
            self.training_time += evaluation_time
            evaluation_dict["training_time"] = time.perf_counter() - self.training_time
            self.fabric.log_dict(evaluation_dict, step=self.step)
            self.last_evaluation_time = time.perf_counter()
            return evaluation_dict
