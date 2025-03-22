"""
Demo 03: Reconstruct a dynamic volume from an array and export a video of the reconstruction."""
from pathlib import Path

from nect.download_demo_data import download_demo_data, get_demo_data_path

import nect

demo_dir = get_demo_data_path("SimulatedFluidInvasion")
download_demo_data("SimulatedFluidInvasion")
geometry = nect.Geometry.from_yaml(demo_dir / "geometry.yaml")
reconstruction_path = nect.reconstruct(
    geometry=geometry,
    projections=demo_dir / "projections.npy",
    quality="high",
    mode="dynamic",
    exp_name="SimulatedFluidInvasion", # optional, name of the experiment
    config_override={
        "epochs": "1x",  # a multiplier of base-epochs. Base-epochs is: floor(49 / num_projections * max(nDetector))
        "checkpoint_interval": 1800,  # How often to save the model in seconds
        "image_interval": 30,  # How often to save images in seconds
        "plot_type": "XZ", # XZ or XY, YZ
    },
)
nect.export_video(reconstruction_path, add_scale_bar=True, acquisition_time_minutes=60)
