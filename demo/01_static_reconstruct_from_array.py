"""
Demo 01: Reconstruct a static volume from an array"""

from pathlib import Path

import numpy as np
from .download_demo_data import download_demo_data

import nect

geometry = nect.Geometry(
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
demo_dir = Path(__file__).parent
download_demo_data(mode="static", folder="cone")
projections = np.load(demo_dir / "NeCT-data" / "static" / "cone" / "projections.npy")
volume = nect.reconstruct(geometry=geometry, projections=projections, quality="high")
np.save(demo_dir / "volume.npy", volume)
