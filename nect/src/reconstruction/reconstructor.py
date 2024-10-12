from __future__ import annotations

import numpy as np
import nect.src.reconstruction.leap as leap
import nect.src.reconstruction.scikit_image as scikit_image
import nect.src.reconstruction.tigre_toolbox as tigre_toolbox


class Reconstructor:
    def __init__(
        self,
        n_projs_per_frame: int,
        method: str = "fbp_skimage",
        sample_size=tuple[int],
        *args,
        **kwargs,
    ):
        """A wrapper class for all reconstruction methods. Reconstruction method specific arguments can be passed in *args and **kwargs.

        Args:
            n_projs_per_frame (int): How many projections to use for each reconstructed frame.
            method (str, optional): The reconstruction method. Defaults to "fbp_skimage".
            sample_size (tuple[int], optional): The size/shape of the phantom. Defaults to tuple[int].
        """
        self.supported_frameworks = ["skimage", "tigre", "leaptorch"]
        self.framework = self.resolve_framework(method)
        self.supported_methods = [
            "fbp_skimage",
            "fdk_tigre",
            "ossart_tigre",
            "fbp_leaptorch",
        ]
        self.method = self.resolve_method(method)
        self.n_projs_per_frame = n_projs_per_frame
        self.sample_size = sample_size
        self.args = args
        self.kwargs = kwargs

    def reconstruct(self, sinogram, theta) -> np.ndarray:
        """Divides the sinogram into frames and reconstructs each frame using the specified reconstruction method.

        Args:
            sinogram (np.ndarray | torch.Tensor): The sinogram to reconstruct. Shape=[nprojs, ...]
            theta (np.ndarray | torch.Tensor): The angles of the sinogram.

        Returns:
            np.ndarray: The reconstructed time series.
        """
        steps = np.arange(0, sinogram.shape[0], self.n_projs_per_frame)
        if (np.max(steps) + self.n_projs_per_frame) > sinogram.shape[0]:
            print(
                f"The {(np.max(steps) + self.n_projs_per_frame) - sinogram.shape[0]} last projections will be ignored. Consider adjusting the number of projections per frame."
            )
            reconstruction = np.zeros((len(steps) - 1, *self.sample_size))
            steps = steps[:-1]
        else:
            reconstruction = np.zeros((len(steps), *self.sample_size))
        if self.framework == "skimage":  # Get sinogram on skimage format: [..., nprojs]
            sinogram = np.transpose(sinogram, axes=(1, 0))
            for i, step in enumerate(steps):
                reconstruction[i] = self.method(
                    sinogram[..., step : step + self.n_projs_per_frame],
                    theta[step : step + self.n_projs_per_frame],
                    *self.args,
                    **self.kwargs,
                )
        else:
            for i, step in enumerate(steps):
                reconstruction[i] = self.method(
                    sinogram[step : step + self.n_projs_per_frame, ...],
                    theta[step : step + self.n_projs_per_frame],
                    *self.args,
                    **self.kwargs,
                )
        return reconstruction

    def resolve_method(self, method: str):
        """Returns the correct method based on the provided method name."""
        if method == "fbp_skimage":
            return scikit_image.fbp
        elif method == "fdk_tigre":
            return tigre_toolbox.fdk
        elif method == "ossart_tigre":
            return tigre_toolbox.ossart
        elif method == "fbp_leaptorch":
            return leap.fbp
        else:
            raise NotImplementedError(f"Method {method} not supported. Supported methods are: {self.supported_methods}")

    def resolve_framework(self, method: str) -> str:
        framework = method.split("_")[-1]
        if framework not in self.supported_frameworks:
            raise NotImplementedError(
                f"The framework {framework} is not supported. Supported frameworks are {self.supported_frameworks}"
            )
        return framework
