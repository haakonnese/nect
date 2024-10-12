"""
Demo 07: export video of the projections for data analysis from a configuration file.
"""

from pathlib import Path
from .download_demo_data import download_demo_data

import nect.data

download_demo_data(mode="dynamic", folder="cone")
config_path = Path(__file__).parent / "NeCT-data" / "dynamic" / "cone" / "config.yaml"
nect.data.export_video_projections(config_path, file=Path(__file__).parent / "video_projections.mp4")
