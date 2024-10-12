"""
Demo 04: Reconstruct a dynamic volume and export volumes of the reconstruction.""
"""

from pathlib import Path

from .download_demo_data import download_demo_data

import nect

demo_dir = Path(__file__).parent
data_dir = demo_dir / "NeCT-data" / "dynamic" / "cone"
download_demo_data(mode="dynamic", folder="cone")
geometry = nect.Geometry.from_yaml(data_dir / "geometry.yaml")
reconstruction_path = nect.reconstruct(
    geometry=geometry,
    projections=data_dir / "projections.npy",
    quality="high",
    mode="dynamic",
    config_override={
        "epochs": "1x",  # a multiplier of base-epochs. Base-epochs is: floor(49 / num_projections * max(nDetector))
        "checkpoint_interval": 1800,  # How often to save the model in seconds
        "image_interval": 600,  # How often to save images in seconds
    },
)
nect.export_volumes(reconstruction_path, binning=3, avg_timesteps=5)
