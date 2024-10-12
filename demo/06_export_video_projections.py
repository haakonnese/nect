"""
Demo 06: export video of the projections for data analysis, when one does not have a configuration file.
"""

from pathlib import Path

import numpy as np
from .download_demo_data import download_demo_data

from nect import Geometry
from nect.config import get_config
from nect.data import NeCTDataset

geometry = Geometry(
    DSD=1500.0,  # Distance Source Detector
    DSO=1000.0,  # Distance Source Origin
    nDetector=[256, 512],  # Number of detector pixels [rows, columns]/[height, width]
    dDetector=[1.75, 1.75],  # Size of detector pixels [row, columns]/[height, width]
    nVoxel=[256, 512, 256],  # Number of voxels [height, width, depth]/[z, y, x]
    dVoxel=[1.0, 1.0, 1.0],  # Size of voxels [height, width, depth]/[z, y, x]
    angles=np.linspace(0, 360, 49, endpoint=False),  # Projection angles
    mode="cone",  # Geometry mode (cone or parallel)
    radians=True,  # Angle units (radians (True) or degrees (False))
)
download_demo_data(mode="dynamic", folder="cone")
projections = Path(__file__).parent / "NeCT-data" / "dynamic" / "cone" / "projections.npy"

config = get_config(geometry, projections, mode="dynamic")
NeCTDataset(config).export_video(file=Path(__file__).parent / "video_projections.mp4")
