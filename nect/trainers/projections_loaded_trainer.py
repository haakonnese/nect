from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from nect.config import Config
from nect.data import NeCTDatasetLoaded
from nect.trainers.base_trainer import BaseTrainer

torch.autograd.set_detect_anomaly(True)


class ProjectionsLoadedTrainer(BaseTrainer):
    def __init__(
        self,
        config: Config,
        projections: torch.Tensor | np.ndarray,
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
        self.projections = projections
        super().__init__(
            config=config,
            output_directory=output_directory,
            checkpoint=checkpoint,
            save_ckpt=save_ckpt,
            save_last=save_last,
            save_optimizer=save_optimizer,
            verbose=verbose,
            log=log,
            cancel_at=cancel_at,
            keep_two=keep_two,
            prune=prune,
        )

    def setup_dataset(self):
        self.dataset = NeCTDatasetLoaded(
            config=self.config,
            projections=self.projections,
            device="cpu",  # if gpu memory is less than 50 GB, load to cpu
        )
