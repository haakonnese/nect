"""
Demo 05: reconstruct using parallel beam geometry
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nect.download_demo_data import download_demo_data, get_demo_data_path

import nect

geometry = nect.Geometry(
    nDetector=[256, 512],  # Number of detector pixels [rows, columns]/[height, width]
    dDetector=[1., 1.],  # Size of detector pixels [row, columns]/[height, width]
    nVoxel=[256, 512, 256],  # Number of voxels [height, width, depth]/[z, y, x]
    dVoxel=[1.0, 1.0, 1.0],  # Size of voxels [height, width, depth]/[z, y, x]
    angles=np.linspace(0, 360, 49, endpoint=False),  # Projection angles
    mode="parallel",  # Geometry mode (cone or parallel)
    radians=False,  # Angle units (radians (True) or degrees (False))
)
demo_dir = get_demo_data_path("Carp-parallel")
download_demo_data("Carp-parallel")
volume = nect.reconstruct(
    geometry=geometry,
    projections=demo_dir / "projections.npy",
    log=True,
    exp_name="carp_parallel",
    quality="medium",
)
plt.imsave("carp.png", volume[128], cmap="gray", dpi=300)
