"""
Demo 04: Reconstruct a dynamic volume and export volumes of the reconstruction.""
"""

from pathlib import Path

from nect.download_demo_data import download_demo_data, get_demo_data_path

import nect

download_demo_data("SimulatedFluidInvasion")
demo_dir = get_demo_data_path("SimulatedFluidInvasion")
geometry = nect.Geometry.from_yaml(demo_dir / "geometry.yaml")
reconstruction_path = nect.reconstruct(
    geometry=geometry,
    projections=demo_dir / "projections.npy",
    quality="high",
    mode="dynamic",
    config_override={
        "epochs": "3x",  # a multiplier of base-epochs. Base-epochs is: floor(49 / num_projections * max(nDetector))
        "checkpoint_interval": 1800,  # How often to save the model in seconds
        "image_interval": 600,  # How often to save images in seconds
        "plot_type": "XZ", # XZ or XY, YZ
    },
)
nect.export_volumes(reconstruction_path, binning=3, avg_timesteps=5)
