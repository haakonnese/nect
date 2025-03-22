"""
Demo 02: Load geometry from a YAML file and reconstruct a static volume from an array"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nect.download_demo_data import download_demo_data, get_demo_data_path

import nect

demo_dir = get_demo_data_path("Carp-cone")
download_demo_data("Carp-cone")
geometry = nect.Geometry.from_yaml(demo_dir / "geometry.yaml")
volume = nect.reconstruct(geometry=geometry, projections=demo_dir / "projections.npy")
plt.imsave("carp.png", volume[128], cmap="gray", dpi=300)
