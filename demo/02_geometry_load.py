"""
Demo 02: Load geometry from a YAML file and reconstruct a static volume from an array"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from .download_demo_data import download_demo_data

import nect

demo_dir = Path(__file__).parent
data_dir = demo_dir / "NeCT-data" / "static" / "cone"
download_demo_data(mode="static", folder="cone")
geometry = nect.Geometry.from_yaml(data_dir / "geometry.yaml")
volume = nect.reconstruct(geometry=geometry, projections=data_dir / "projections.npy")
plt.imsave(str(demo_dir / "carp.png"), volume[128], cmap="gray", dpi=300)
