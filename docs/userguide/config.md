# Reconstruction Configuration

The config file defines the parameters for the reconstruction when using the command line interface. The config file is a `yaml` file with the default parameters defined below.

In addition to the default parameters, there are seperate config files for the different reconstruction methods. The config files are located in the `nect/cfg` folder.
The following import order of the config files are used, and the parameters are overwritten by the definitions in the later files if they are defined in multiple files.:

1. The default parameters in `nect/cfg/default.yaml` are loaded first.
2. The parameters specific to the reconstruction method used are loaded next. This can e.g. be `nect/cfg/dynamic/quadcubes.yaml` for the 4D-CT Quadcubes method.
3. The user specific `yaml` file is loaded last. This file is defined by the user when running the reconstruction, `python -m nect.reconstruct <path/to/config/file.yaml>`

In the user specific `yaml` file, the user can overwrite any parameter defined in the default or method specific config files.
The following parameters must be defined in the user specific `yaml` file if using the command line interface:

```yaml
geometry: <path/to/geometry/file.yaml>
img_path: <path/to/projections/folder> or <path/to/projections/file.npy>
model: model_type
```

The supported `model_type` are:

- For static:
  - `hash_grid`
  - `kplanes`
- For dynamic:
  - `quadcubes`
  - `double_hash_grid`
  - `kplanes_dynamic`
  - `hypercubes`

```yaml title="nect/cfg/default.yaml"
{% include "../../nect/cfg/default.yaml" %}
```
