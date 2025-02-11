# NeCT: Neural Computed Tomography
NeCT is a project that aims to use deep learning to improve the quality of computed tomography (CT) images. NeCT allows for both static and dynamic CT reconstruction, and the project is based on the INR framework `tiny-cuda-nn`.
- [Installation](#installation)
- [Demo](#demo)
- [GUI](#gui)
- [Licensing and Citation](#licensing-and-citation)

![NeCT Reconstruction Pipeline](docs/images/Reconstruction%20pipeline.png)

## Installation
The project has been tested to work on both Windows and Linux with the following dependencies
<!-- make a table with name and version -->
<table>
<tr><th>Package</th><th>Version</th></tr>
<tr><td>python</td><td>3.10 | 3.11</td><td></td></tr>
<tr><td>pytorch</td><td>2.1</td><td></td></tr>
<tr><td> CUDA</td><td>11.7 | 12.0</td><td></td></tr>
<tr><td> CMake (For Linux)</td><td>3.24</td><td>For tiny-cuda-nn</td></tr>
<tr><td> C++17 (For Windows)</td><td></td><td>For tiny-cuda-nn</td></tr>
</table>

We recommend using [conda](https://docs.anaconda.com/free/anaconda/install/) to manage the python environment. 
The project has been tested for `python=3.10,3.11` and `pytorch=2.1`, but we assume it will work with newer versions as well. To install the project for multiple different compute capabilities, follow the instructions [here](#install-multiple-compute-capabilities).


Then create a conda environment and install all the dependencies. Make sure that both `PATH` and `LD_LIBRARY_PATH` includes the paths to the CUDA binaries as described in [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/) before installing. If you experience some error when installing related to `tiny-cuda-nn`, please go to [issues of tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/issues) and search to try to solve the problem. As binaries will be built for both tiny-cuda-nn and NeCT, the installation will usually take at least several minutes. 

```bash
conda create -n nect python=3.11
conda activate nect
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 lightning==2.1 conda-forge::opencv -c pytorch -c nvidia -c conda-forge -y
pip install -v git+https://github.com/haakonnese/nect
```

### Install multiple compute capabilities
To install the project for multiple different compute capabilities, follow the instructions below. The following environment variables needs to be set before installing NeCT if you want to build the binaries for multiple compute capabilities. In the example below, we install for compute capabilities 60 (P100), 70 (V100), 80 (A100) and 90 (H100).  
```bash
export CUDA_ARCHITECTURES="60;70;80;90"
export CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
export TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
export TORCH_CUDA_ARCH_LIST="6.0 7.0 8.0 9.0"
export FORCE_CUDA="1"
```

### Installing TIGRE for projection synthesis (optional)
```bash
pip install Cmake
pip install git+https://github.com/CERN/TIGRE/#subdirectory=Python
```


## Demo
Demo scripts can be found under [demo](./demo/). First time reconstructing the demo-objects, projection-data will be downloaded.

## GUI
The GUI is a tool for visualizing the reconstructed 4D reconstruction (works only for 4D now). The GUI is based on the `PyQt5` library and have only been tested on Windows. The GUI can be started by running the following command:
```bash
python -m nect.gui
```

## Licensing and Citation
The project is licensed under the MIT license. 
The project is a collaboration between the Norwegian Univeristy of Science and Technology (NTNU) and the CT lab at Equinor.
If you use NeCT in your research, please cite the following paper: To be added.
