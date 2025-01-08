import pathlib
import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess
import sys
# Check if CUDA is available
cuda = torch.cuda.is_available()

def install_pybind11():
    try:
        import pybind11
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11>=2.11"])

# Install pybind11 if it's not installed
install_pybind11()
import pybind11

if cuda:
    source_files = [str(pathlib.Path("nect/sampling/ct_sampling.cu"))]
    cxx_args = ["-g", "-D__USE_GPU", "-I%s" % pybind11.get_include()]
    nvcc_args = ["-g", "-D__USE_GPU", "-DCUDA_VERSION=12030", "-O3"]
    if os.name == 'nt':  # if Windows
        cxx_args.extend(['/MD', '/EHsc'])  # MSVC specific flags
        nvcc_args.extend(['-Xcompiler', '/MD', '-Xcompiler', '/EHsc'])  # NVCC flags for MSVC

    ext_mod = CUDAExtension(
        name="nect.sampling.ct_sampling",
        sources=source_files,
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": nvcc_args,
        },
    )
else:
    raise ImportError("CUDA is not available.")

setup(
    name="nect",
    version="0.0.1",
    description="4D-CT reconstruction using Machine Learning and AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Håkon Nese, Henrik Friis",
    author_email="haakon.nese@gmail.com, henrik.friis@outlook.com",
    license="MIT",  # Change if using a different license
    url="https://github.com/haakonnese/nect",
    packages=find_packages(".", include=["nect", "nect.*", "torch_extra", "torch_extra.*"]),
    package_data={"nect.cfg": ["**/*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=[ext_mod],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        'numpy>=1.23, <2',
        'matplotlib>=3.7',
        'Pillow>=10.0',
        'scipy>=1.10',
        'jinja2>=3.1',
        'tqdm>=4.66',
        'loguru>=0.7',
        'pynvml>=11.5',
        'python-dotenv>=1.0',
        'dacite>=1.8.1',
        'torchinfo>=1.8.0',
        'tensorboard>=2.16.2',
        'tifffile>=2024.5.22',
        'napari>=0.5.1',
        'pyqt5>=5.15.11',
        'tinycudann @ git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch'
    ],
    python_requires='>=3.9',
)