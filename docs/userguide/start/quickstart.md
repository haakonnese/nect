Start using `NeCT` to reconstruct CT images or 4DCT videos. For more advanced usage, look at the [demos](../demo/index.md), the [geometry configuration](../geometry.md), and how to set up the [configuration file](../config.md).

=== "Static CT"

    A static reconstruction using `NeCT` with `medium` quality, creating a CT image that is saved to a file.

    !!! Example 
        
        ```python
        import numpy as np
        import nect

        geometry = nect.Geometry(
            DSD=1500.0,  # Distance Source Detector
            DSO=1000.0,  # Distance Source Origin
            nDetector=[256, 512],  # Number of detector pixels [rows, columns]/[height, width]
            dDetector=[1.75, 1.75],  # Size of detector pixels [row, columns]/[height, width]
            nVoxel=[256, 512, 256],  # Number of voxels [height, width, depth]/[z, y, x]
            dVoxel=[1.0, 1.0, 1.0],  # Size of voxels [height, width, depth]/[z, y, x]
            angles=np.linspace(0, 360, 49, endpoint=False),  # Projection angles
            mode="cone",  # Geometry mode (cone or parallel)
            angles_type="degrees",  # Angle units (degrees or radians)
        )

        volume = nect.reconstruct(
            geometry=geometry,
            projections=<projections-path>, # a single npy-file of shape [nProjections, height, width] 
                                            # or a directory of images
            quality="medium",
        )
        np.save("volume.npy", volume)
        ```

=== "4DCT"

    4DCT reconstruction with `NeCT` is done with `higher` quality. Afterwards, a video of the middle slice of the object is created through the whole time-series. The video is saved in the same folder as the reconstruction with config and the model weights.

    !!! Example 
        
        ```python
        import numpy as np
        import nect
        
        geometry = nect.Geometry(
            DSD=1500.0,  # Distance Source Detector
            DSO=1000.0,  # Distance Source Origin
            nDetector=[256, 512],  # Number of detector pixels [rows, columns]/[height, width]
            dDetector=[1.75, 1.75],  # Size of detector pixels [row, columns]/[height, width]
            nVoxel=[256, 512, 256],  # Number of voxels [height, width, depth]/[z, y, x]
            dVoxel=[1.0, 1.0, 1.0],  # Size of voxels [height, width, depth]/[z, y, x]
            angles=np.linspace(0, 360, 49, endpoint=False),  # Projection angles
            mode="cone",  # Geometry mode (cone or parallel)
            angles_type="degrees",  # Angle units (degrees or radians)
        )
        projections = np.load(<projections-path>)
        
        reconstruction_path = nect.reconstruct(
            geometry=geometry,
            projections=projections,
            quality="highest",
            mode="dynamic"
        )
        nect.export_video(reconstruction_path, add_scale_bar=True, acquisition_time_minutes=60)
        ```
