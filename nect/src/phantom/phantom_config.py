from enum import IntEnum

import numpy as np
import porespy as ps
import torch

from nect.src.phantom.geometric import (
    Circle,
    Cuboid,
    CustomGeometry,
    Ellipse,
    Phantom,
    PhantomObject,
    Rectangle,
    Sphere,
    Triangle,
)
from nect.src.phantom.utils import (
    create_random_closed_geometry,
    eoi_simple_dynamic_pore,
)


class IntensityConfig(IntEnum):
    ROCK = 3
    AIR = 1
    WATER = 2  #  0.12 /cm at 300keV https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html


def test_phantom_3D():
    cuboid = Cuboid(15, 30, 50)
    sphere = Sphere(30, 30, 30, 10)

    cuboid_obj = PhantomObject(
        eom=lambda tlbr, t: tlbr + t,
        eoi=lambda intensity, t: intensity,
        tl=torch.tensor([40, 40, 40]),
        geometry=cuboid,
        intensity=1,
    )
    sphere_obj = PhantomObject(
        eom=lambda tlbr, t: tlbr + t,
        eoi=lambda intensity, t: intensity,
        tl=torch.tensor([25, 65, 65]),
        geometry=sphere,
        intensity=0.5,
    )

    phantom = Phantom(size=(64, 128, 128))
    phantom.add_phantom_object(obj=[cuboid_obj, sphere_obj])
    return phantom


def moving_objects_constant_intensity_phantom(save_video=False) -> Phantom:
    """Creates a phantom with moving objects.

    Returns:
        Phantom: The phantom with moving objects.
    """

    phantom = Phantom(size=(300, 300), background_method="zeros")
    circle = Circle(300, 300, 150)
    rectangle = Rectangle(40, 80)
    triangle = Triangle(40, 80)
    small_circle = Circle(20, 20, 10)
    ellipse = Ellipse(40, 80, 10, 20)
    random = create_random_closed_geometry(size=(50, 50))

    def eom_horizontal(tlbr, t):
        return tlbr + t * torch.tensor([0, 1, 0, 1])

    def eom_vertical(tlbr, t):
        return tlbr + t * torch.tensor([1, 0, 1, 0])

    def eom_diagonal(tlbr, t):
        return tlbr + t

    phantom_circle = PhantomObject(
        lambda tlbr, t: tlbr,
        lambda intensity, t: intensity,
        torch.tensor([0, 0]),
        circle,
        intensity=0.1,
    )
    phantom_rectangle = PhantomObject(
        eom_horizontal,
        lambda intensity, t: intensity,
        torch.tensor([60, 60]),
        rectangle,
        intensity=1,
    )
    phantom_random = PhantomObject(
        eom_diagonal,
        lambda intensity, t: intensity,
        torch.tensor([100, 100]),
        random,
        intensity=1,
    )
    phantom_ellipse = PhantomObject(
        eom_vertical,
        lambda intensity, t: intensity,
        torch.tensor([200, 200]),
        ellipse,
        intensity=1,
    )
    phantom_small_circle = PhantomObject(
        eom_diagonal,
        lambda intensity, t: intensity,
        torch.tensor([100, 200]),
        small_circle,
        intensity=1,
    )
    phantom_triangle = PhantomObject(
        eom_diagonal,
        lambda intensity, t: intensity,
        torch.tensor([200, 100]),
        triangle,
        intensity=1,
    )

    phantom.add_phantom_object(
        [
            phantom_circle,
            phantom_rectangle,
            phantom_random,
            phantom_ellipse,
            phantom_small_circle,
            phantom_triangle,
        ]
    )

    if save_video:
        # Save the phantom to a video
        time_steps = np.arange(0, 30, 1)
        phantom.save_video(time_steps, "src/phantom/cfg_videos/moving_objects.mp4")
    return phantom


def moving_objects_one_constant_constant_intensity_phantom(save_video=False) -> Phantom:
    """Creates a phantom with moving objects.

    Returns:
        Phantom: The phantom with moving objects.
    """

    phantom = Phantom(size=(300, 300), background_method="zeros")
    circle = Circle(300, 300, 150)
    rectangle = Rectangle(40, 80)
    square = Rectangle(40, 40)
    triangle = Triangle(40, 80)
    small_circle = Circle(20, 20, 10)
    ellipse = Ellipse(40, 80, 10, 20)
    random = create_random_closed_geometry(size=(50, 50))

    def eom_horizontal(tlbr, t):
        return tlbr + t * torch.tensor([0, 1, 0, 1])

    def eom_vertical(tlbr, t):
        return tlbr + t * torch.tensor([1, 0, 1, 0])

    def eom_diagonal(tlbr, t):
        return tlbr + t

    def eom_constant(tlbr, t):
        return tlbr

    phantom_circle = PhantomObject(
        lambda tlbr, t: tlbr,
        lambda intensity, t: intensity,
        torch.tensor([0, 0]),
        circle,
        intensity=0.1,
    )
    phantom_rectangle = PhantomObject(
        eom_horizontal,
        lambda intensity, t: intensity,
        torch.tensor([60, 60]),
        rectangle,
        intensity=1,
    )
    phantom_random = PhantomObject(
        eom_diagonal,
        lambda intensity, t: intensity,
        torch.tensor([100, 100]),
        random,
        intensity=1,
    )
    phantom_ellipse = PhantomObject(
        eom_vertical,
        lambda intensity, t: intensity,
        torch.tensor([200, 200]),
        ellipse,
        intensity=1,
    )
    phantom_small_circle = PhantomObject(
        eom_diagonal,
        lambda intensity, t: intensity,
        torch.tensor([100, 200]),
        small_circle,
        intensity=1,
    )
    phantom_triangle = PhantomObject(
        eom_diagonal,
        lambda intensity, t: intensity,
        torch.tensor([200, 100]),
        triangle,
        intensity=1,
    )
    phantom_static_rectangle = PhantomObject(
        eom_constant,
        lambda intensity, t: intensity,
        torch.tensor([150, 50]),
        square,
        intensity=1,
    )

    phantom.add_phantom_object(
        [
            phantom_circle,
            phantom_rectangle,
            phantom_random,
            phantom_ellipse,
            phantom_small_circle,
            phantom_triangle,
            phantom_static_rectangle,
        ]
    )

    if save_video:
        # Save the phantom to a video
        time_steps = np.arange(0, 30, 1)
        phantom.save_video(time_steps, "src/phantom/cfg_videos/moving_objects_one_static.mp4")
    return phantom


def moving_objects_one_constant_changing_intensity_phantom(save_video=False) -> Phantom:
    """Creates a phantom with moving objects.

    Returns:
        Phantom: The phantom with moving objects.
    """

    phantom = Phantom(size=(300, 300), background_method="zeros")
    circle = Circle(300, 300, 150)
    rectangle = Rectangle(40, 80)
    square = Rectangle(40, 40)
    triangle = Triangle(40, 80)
    small_circle = Circle(20, 20, 10)
    ellipse = Ellipse(40, 80, 10, 20)
    random = create_random_closed_geometry(size=(50, 50))

    def eom_horizontal(tlbr, t):
        return tlbr + t * torch.tensor([0, 1, 0, 1])

    def eom_vertical(tlbr, t):
        return tlbr + t * torch.tensor([1, 0, 1, 0])

    def eom_diagonal(tlbr, t):
        return tlbr + t

    def eom_constant(tlbr, t):
        return tlbr

    def decreaseing_intensity(intensity, t):
        return intensity - t / 30

    def increasing_intensity(intensity, t):
        return intensity + t / 15

    phantom_circle = PhantomObject(
        lambda tlbr, t: tlbr,
        lambda intensity, t: intensity,
        torch.tensor([0, 0]),
        circle,
        intensity=0.1,
    )
    phantom_rectangle = PhantomObject(
        eom_horizontal,
        lambda intensity, t: intensity,
        torch.tensor([60, 60]),
        rectangle,
        intensity=1,
    )
    phantom_random = PhantomObject(
        eom_diagonal,
        lambda intensity, t: intensity,
        torch.tensor([100, 100]),
        random,
        intensity=1,
    )
    phantom_ellipse = PhantomObject(
        eom_vertical,
        lambda intensity, t: intensity,
        torch.tensor([200, 200]),
        ellipse,
        intensity=1,
    )
    phantom_small_circle = PhantomObject(
        eom_diagonal,
        lambda intensity, t: intensity,
        torch.tensor([100, 200]),
        small_circle,
        intensity=1,
    )
    phantom_triangle = PhantomObject(
        eom_diagonal,
        decreaseing_intensity,
        torch.tensor([200, 100]),
        triangle,
        intensity=1,
    )
    phantom_static_rectangle = PhantomObject(
        eom_constant, increasing_intensity, torch.tensor([150, 50]), square, intensity=1
    )

    phantom.add_phantom_object(
        [
            phantom_circle,
            phantom_rectangle,
            phantom_random,
            phantom_ellipse,
            phantom_small_circle,
            phantom_triangle,
            phantom_static_rectangle,
        ]
    )

    if save_video:
        # Save the phantom to a video
        time_steps = np.arange(0, 30, 1)
        phantom.save_video(
            time_steps,
            "src/phantom/cfg_videos/moving_objects_one_static_chaning_intensity.mp4",
        )
    return phantom


def dynamic_2d_pores(save_video=False):
    custom = CustomGeometry(mask=ps.generators.blobs(shape=[100, 100], porosity=0.5, blobiness=2))
    custom_object = PhantomObject(
        eom=lambda tlbr, t: tlbr,
        eoi=eoi_simple_dynamic_pore,
        tl=torch.tensor([0, 0]),
        geometry=custom,
        intensity=custom.mask.astype(np.uint8),
    )
    phantom = Phantom(size=(100, 100))
    phantom.add_phantom_object(custom_object)

    if save_video:
        # Save the phantom to a video
        time_steps = np.arange(0, 30, 1)
        phantom.save_video(time_steps, "src/phantom/cfg_videos/dynamic_2d_pores.mp4")


# if __name__ == "__main__":
#     #phantom = moving_objects_constant_intensity_phantom(save_video=True)
#     #phantom = moving_objects_one_constant_constant_intensity_phantom(save_video=True)
#     #phantom = moving_objects_one_constant_changing_intensity_phantom(save_video=True)
#     phantom = dynamic_2d_pores(save_video=True)
