# NeCT: Neural Computed Tomography

NeCT leverages deep learning to improve computed tomography (CT) image quality, supporting both static and dynamic CT reconstruction. For dynamic (4D) CT, the temporal resolution approaches the time between subsequent projections.
<table>
  <tr>
    <td>
      <img src="docs/images/showcase1.gif" width="480">
      <p>
        Rendering of spontaneous imbibition in a Bentheimer sandstone reconstructed using NeCT. The brine flowing into the sample is shown in light blue, while the salt grains dissolving are presented in red.
      </p>
    </td>
    <td>
        <img src="docs/images/showcase2.gif" width="480">
      <p>
        Rendering of the dissolution of a salt grain. Three orthogonal slices visualize its temporal evolution. In the xz slice, it is possible to observe the brine coming into contact with the salt before it starts to dissolve.
      </p>
    </td>
  </tr>
</table>

<p align="center">
    <a href="https://rdcu.be/eBgrb" target="_blank">
        <img src="https://img.shields.io/badge/Paper-red?style=for-the-badge" alt="Paper"/>
    </a>
    <a href="https://haakonnese.github.io/nect/" target="_blank">
        <img src="https://img.shields.io/badge/NeCT%20Documentation-blueviolet?style=for-the-badge&logo=readthedocs" alt="NeCT Documentation"/>
    </a>
</p>

- [Installation](#installation)
- [Demo](#demo)
- [Data](#data)
- [GUI](#gui)
- [Licensing and Citation](#licensing-and-citation)

![NeCT Reconstruction Pipeline](docs/images/pipeline.png)

## Installation

NeCT has been tested on **Windows** and **Linux** with the following dependencies:

| Package         | Version           | Notes              |
|-----------------|-------------------|--------------------|
| python          | 3.11 \| 3.12      |                    |
| pytorch         | 2.4 - 2.7         |                    |
| CUDA            | 12.X              |                    |
| CMake (Linux)   | 3.24              | For tiny-cuda-nn   |
| C++17 (Windows) |                   | For tiny-cuda-nn   |

> **Recommended:** Use [conda](https://docs.anaconda.com/free/anaconda/install/) or [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage your Python environment.
>
> - For video export, the `avc1` codec for `ffmpeg` is only available with conda. With uv, video export uses the `mp4v` codec. If video export using `avc1` is vital, use conda.
> - Tested with `python=3.11, 3.12` and `pytorch>=2.4,<2.8`.
> - To install for multiple compute capabilities, see [below](#install-multiple-compute-capabilities).

**Note:** Ensure `PATH` and `LD_LIBRARY_PATH` include the CUDA binaries as described in [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/). If you encounter installation errors related to `tiny-cuda-nn`, check their [issues page](https://github.com/NVlabs/tiny-cuda-nn/issues). Building binaries for both tiny-cuda-nn and NeCT may take several minutes.

### Uv Installation

If you don't have uv installed, follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/). This will install pytorch with CUDA 12.4.

```bash
uv venv --python=3.12
source venv/bin/activate
uv pip install git+https://github.com/haakonnese/nect[torch]
uv pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
```

#### Custom PyTorch Version

To use a specific PyTorch version (2.4-2.7) visit the [PyTorch Installation Page](https://pytorch.org/get-started/locally/) and install the desired PyTorch version into your uv environment. Then install NeCT with:

```bash
uv pip install git+https://github.com/haakonnese/nect --no-build-isolation-package torch
uv pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
```

### Conda Installation

```bash
conda create -n nect python=3.12 -y
conda activate nect
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 lightning==2.1 conda-forge::opencv -c pytorch -c nvidia -c conda-forge -y
pip install -v git+https://github.com/haakonnese/nect
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Install Multiple Compute Capabilities

To build for multiple compute capabilities (e.g., 60=P100, 70=V100, 80=A100, 90=H100), set these environment variables **before** installing NeCT:

```bash
export CUDA_ARCHITECTURES="60;70;80;90"
export CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
export TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
export TORCH_CUDA_ARCH_LIST="6.0 7.0 8.0 9.0"
export FORCE_CUDA="1"
```

## Demo

Demo scripts are in the [demo](./demo/) folder. The first time you reconstruct demo objects, projection data will be downloaded automatically.

## Data

All projections from the dynamic experiments are available at [Zenodo](https://zenodo.org/records/16448474).

## GUI

The GUI for NeCT is based on the GPL licensed PyQt5 library. For that reason, the GUI is in a separate repository: [nect-gui](https://github.com/haakonnese/nect-gui).

## Licensing and Citation

This project is licensed under the MIT license.

A collaboration between the Norwegian University of Science and Technology (NTNU) and the CT lab at Equinor.

If you use NeCT in your research, please cite: 
```bibtex
@article{Friis2025Nect,
author={Friis, Henrik
        and Nese, Håkon
        and Luani, Giacomo
        and Pryme, Colin
        and Rennan, Lars
        and Chattopadhyay, Basab
        and Kristoffersen, Anders
        and Mengshoel, Ole Jakob
        and Breiby, Dag Werner},
title={Implicit neural representation for fast 4D computed tomography of multiphase flow in porous media},
journal={Communications Physics},
year={2025},
volume={8},
number={1},
pages={339},
issn={2399-3650},
doi={10.1038/s42005-025-02249-0},
url={https://doi.org/10.1038/s42005-025-02249-0}
}
```