# Demo

We provide several demo-files to easier get started. The files are placed under [NeCT/demo](https://github.com/haakonnese/nect/demo).

### Static CT image reconstruction

- [00 - Static: from file](00_static_from_file.md) show how to do a static reconstruction providing the path to the projections.
- [01 - Static: from array](01_static_from_array.md) show how to do a static reconstruction with the projections already loaded into memory.

### Geometry

- [02 - Geometry](02_geometry.md) contains info of how to load geometry from file.
- [05 - Parallel beam geometry](05_parallel_beam.md) shows how to use a parallel beam geometry instead of cone-beam geometry.

### Dynamic CT reconstruction

- [03 - Dynamic: export video](03_dynamic_video.md) show how to do a 4DCT reconstruction and export videos (original and difference video)
- [04 - Dynamic: export volumes](04_dynamic_volumes.md) show how to do a 4DCT reconstruction when the pair of angles and projections are not ordered by the acqusition time. In the end, volumes for multiple timesteps are exported and saved as 3D tiff-files.
- [06 - Dynamic: export video of projections](06_dynamic_export_video_projections.md) show how to do export videos of the projections. This can be used as a quality check of the projections.
- [07 - Reconstruct from config file](07_reconstruct_from_cfg_file.md) show how to reconstruct from a configuration file. This is useful when you have a pre-defined setup and want to run the reconstruction without manually specifying all parameters. An example of this is the Bentheimer experiment. Download the data by clicking [this link](https://files.de-1.osf.io/v1/resources/2w8xc/providers/osfstorage/67d08a82ec5b8741b044b5e4/?zip=) (will take some time, it is over 12 GB), fill in the missing file paths in the config file, and run this script.
