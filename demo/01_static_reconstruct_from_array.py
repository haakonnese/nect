"""
Demo 01: Reconstruct a static volume from an array"""

from pathlib import Path

import numpy as np
from nect.download_demo_data import download_demo_data, get_demo_data_path

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
    radians=False,  # Angle units (radians (True) or degrees (False))
)
download_demo_data("Carp-cone", force_download=True) # Download the demo data. You can force a re-download by setting force_download=True
demo_dir = get_demo_data_path("Carp-cone")
projections = np.load(demo_dir / "projections.npy")
volume = nect.reconstruct(geometry=geometry, projections=projections, quality="high")
np.save("volume.npy", volume)
