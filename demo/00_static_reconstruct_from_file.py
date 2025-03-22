"""
Demo 00: reconstruct a static volume from a file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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
demo_dir = get_demo_data_path("Carp-cone")
download_demo_data("Carp-cone")
volume = nect.reconstruct(
    geometry=geometry,
    projections=demo_dir / "projections.npy",
    log=True,
    exp_name="carp",
    quality="low",
)
plt.imsave("carp.png", volume[128], cmap="gray", dpi=300)
