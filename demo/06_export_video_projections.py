"""
Demo 06: export video of the projections for data analysis, when one does not have a configuration file.
"""

from pathlib import Path

import numpy as np
from nect.download_demo_data import download_demo_data, get_demo_data_path

from nect import Geometry
from nect.config import get_config
from nect.data import NeCTDataset

demo_dir = get_demo_data_path("SimulatedFluidInvasion")
download_demo_data("SimulatedFluidInvasion")
geometry = Geometry.from_yaml(demo_dir / "geometry.yaml")
projections = demo_dir / "projections.npy"

config = get_config(geometry, projections, mode="dynamic")
NeCTDataset(config).export_video(file="video_projections.mp4")
