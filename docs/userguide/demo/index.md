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