
NeCT has been tested on **Windows** and **Linux** with the following dependencies:

| Package         | Version           | Notes              |
|-----------------|-------------------|--------------------|
| python          | 3.11 \| 3.12      |                    |
| pytorch         | 2.5               |                    |
| CUDA            | 12.1 \| 12.4      |                    |
| CMake (Linux)   | 3.24              | For tiny-cuda-nn   |
| C++17 (Windows) |                   | For tiny-cuda-nn   |

> **Recommended:** Use [conda](https://docs.anaconda.com/free/anaconda/install/) or [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage your Python environment.
>
> - For video export, the `avc1` codec for `ffmpeg` is only available with conda. With uv, video export uses the `mp4v` codec. If video export using `avc1` is vital, use conda.
> - Tested with `python=3.11, 3.12` and `pytorch=2.5`. Newer (and older) versions may also work.
> - To install for multiple compute capabilities, see [below](#install-multiple-compute-capabilities).

**Note:** Ensure `PATH` and `LD_LIBRARY_PATH` include the CUDA binaries as described in [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/). If you encounter installation errors related to `tiny-cuda-nn`, check their [issues page](https://github.com/NVlabs/tiny-cuda-nn/issues). Building binaries for both tiny-cuda-nn and NeCT may take several minutes.

### Uv Installation

```bash
uv venv --python=3.12
source venv/bin/activate
uv pip install git+https://github.com/haakonnese/nect
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