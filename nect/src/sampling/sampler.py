import numpy as np

import nect.src.sampling.leap as leap
import nect.src.sampling.scikit_image as scikit_image
import nect.src.sampling.tigre_toolbox as tigre_toolbox
from nect.src.simulator.scheduler import Scheduler


class Sampler:
    def __init__(
        self,
        method: str,
        scheduler: Scheduler,
        reset_time: bool = True,
        ct_noise: bool = False,
        *args,
        **kwargs,
    ):
        """A wrapper class for sampling a phantom. Sampling method specific arguments can be passed in *args and **kwargs.

        Args:
            method (str): The sampling method.
            scheduler (Scheduler): A Scheduler defining how time and thus the phantom evolves during sampling.
            reset_time (bool, optional): Whether to reset the internal scheduler time between each sampling. Defaults to True.
        """
        self.supported_frameworks = ["skimage", "tigre", "leaptorch"]
        self.framework = self.resolve_framework(method)
        self.supported_sampling_methods = [
            "dynamic_equidistant_skimage",
            "dynamic_golden_angle_skimage",
            "equidistant_skimage",
            "golden_angle_skimage",
            "dynamic_equidistant_tigre",
            "dynamic_golden_angle_tigre",
            "equidistant_tigre",
            "golden_angle_tigre",
            "dynamic_equidistant_leaptorch",
            "equidistant_leaptorch",
            "dynamic_golden_angle_v3_leaptorch",
        ]
        self.method = self.resolve_method(method)
        self.scheduler = scheduler
        self.reset_time = reset_time
        self.ct_noise = ct_noise
        self.args = args
        self.kwargs = kwargs

    def sample(self, phantom):
        """Samples the phantom using the specified sampling method. Returns the sinogram on the format [nproj, ...]"""
        if self.reset_time:
            self.scheduler.reset_time()
        sinogram, theta = self.method(phantom, scheduler=self.scheduler, *self.args, **self.kwargs)
        if self.framework == "skimage":  # skimage uses the convension [shape, nprojs]. We want [nprojs, shape]
            sinogram = np.transpose(sinogram, axes=(1, 0))  # skimage sinograms are always 2D
        if self.ct_noise:
            sinogram = self.apply_ct_noise(sinogram.astype(np.float32))
        return sinogram, theta

    def resolve_method(self, method: str):
        """Returns the correct method based on the provided method name."""
        # Skimage methods (2D)
        if method == "dynamic_equidistant_skimage":
            return scikit_image.dynamic_equidistant_sampling
        elif method == "dynamic_golden_angle_skimage":
            return scikit_image.dynamic_golden_angle_sampling
        elif method == "equidistant_skimage":
            return scikit_image.equidistant_sampling
        elif method == "golden_angle_skimage":
            return scikit_image.golden_angle_sampling

        # Tigre methods (3D)
        elif method == "dynamic_equidistant_tigre":
            return tigre_toolbox.dynamic_equidistant_sampling
        elif method == "dynamic_golden_angle_tigre":
            return tigre_toolbox.dynamic_golden_angle_sampling
        elif method == "equidistant_tigre":
            return tigre_toolbox.equidistant_sampling
        elif method == "golden_angle_tigre":
            return tigre_toolbox.golden_angle_sampling

        elif method == "dynamic_equidistant_leaptorch":
            return leap.dynamic_equidistant_sampling
        elif method == "equidistant_leaptorch":
            return leap.equidistant_sampling
        elif method == "dynamic_golden_angle_v3_leaptorch":
            return leap.dynamic_golden_angle_v3_sampling
        elif method == "dynamic_golden_angle_v3_linear_time_leaptorch":
            return leap.dynamic_golden_angle_v3_sampling_linear_time
        else:
            raise NotImplementedError(
                f"Method {method} not supported. Supported methods are: {self.supported_sampling_methods}"
            )

    def resolve_framework(self, method: str) -> str:
        framework = method.split("_")[-1]
        if framework not in self.supported_frameworks:
            raise NotImplementedError(
                f"The framework {framework} is not supported. Supported frameworks are {self.supported_frameworks}"
            )
        return framework

    def apply_ct_noise(self, sinogram):
        from tigre.utilities import CTnoise

        return CTnoise.add(sinogram, Poisson=1e5, Gaussian=np.array([0, 10]))
