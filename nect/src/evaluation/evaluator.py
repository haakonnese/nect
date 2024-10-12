import torch

from nect.src.evaluation.config import (
    BASE_2D_EVALUATION_CONFIG,
    BASE_3D_EVALUATION_CONFIG,
)
from torch_extra.nn import (
    MS_SSIM,
    SSIM,
    ST_MS_SSIM,
    ST_SSIM,
    FourierRingCorrelation,
    FourierShellCorrelation,
)


class PSNR:
    def forward(self, X, Y):
        return -10 * torch.log10(torch.nn.functional.mse_loss(X, Y))


METRICS = {
    "ssim": SSIM,
    "ms_ssim": MS_SSIM,
    "st_ssim": ST_SSIM,
    "st_ms_ssim": ST_MS_SSIM,
    "frc": FourierRingCorrelation,
    "fsc": FourierShellCorrelation,
    "psnr": PSNR,
}


class Evaluator:
    def __init__(self, metrics: list[str], metrics_config: dict = {}, spatial_dims: int = 2) -> None:
        """Evaluator class for evaluating metrics on tensors.

        Args:
            metrics (list[str]): Defines which metrics to use for image quality evaluation.
            metrics_config (dict, optional): Optional configuration to override base configuration. Defaults to {}.
            spatial_dims (int, optional): Determines whether to use 2D or 3D base configuration. Defaults to 2.
        """
        self.metrics = [metric.capitalize().lower() for metric in metrics]
        assert spatial_dims in [
            2,
            3,
        ], f"Spatial dimensions must be either 2 or 3, got {spatial_dims}"
        self.base_config = BASE_2D_EVALUATION_CONFIG if spatial_dims == 2 else BASE_3D_EVALUATION_CONFIG
        self.config = self.update_config(metrics_config)
        self.spatial_dims = spatial_dims

    def evaluate(self, X: torch.Tensor, Y: torch.Tensor) -> dict:
        """Evaluates the similarity between two tensors using the metrics defined in self.metrics.
        The optional configuration defined in self.config is used to override the base configuration.

        Args:
            X (torch.Tensor):
            Y (torch.Tensor):

        Raises:
            ValueError: If a provided metric is not implemented.

        Returns:
            dict: Dictionary containing the results of the evaluation. The scores are accessible via the metric names.
        """
        results = {}
        for metric_name in self.metrics:
            # print(f"Evaluating {metric_name}...")
            if metric_name not in METRICS:
                raise ValueError(f"Metric {metric_name} not implemented")
            metric = METRICS[metric_name](**self.config[metric_name])
            while self.spatial_dims == 3 and X.dim() < 5:
                X = X.unsqueeze(0)
            while self.spatial_dims == 3 and Y.dim() < 5:
                Y = Y.unsqueeze(0)
            while self.spatial_dims == 4 and X.dim() < 6:
                X = X.unsqueeze(0)
            while self.spatial_dims == 4 and Y.dim() < 6:
                Y = Y.unsqueeze(0)
            results[metric_name] = metric.forward(X, Y)
        return results

    def update_config(self, new_config: dict) -> None:
        """Used to override the base configuration with the optional configuration defined in self.config."""
        for key, value in new_config.items():
            for k, v in value.items():
                self.base_config[key][k] = v
        return self.base_config
