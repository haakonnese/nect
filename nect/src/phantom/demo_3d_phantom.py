"""
The demo demonstrates how to create a 3D phantom and visualize it using Plotly. The 3D phantom can be used
to verify the correctness of the forward model or the reconstruction algorithm. The demo creates a cuboid and a sphere
that moves in a predefined trajectory.
"""

import numpy as np
import plotly.graph_objects as go
import torch

from nect.src.phantom.geometric import Cuboid, Phantom, PhantomObject, Sphere

if __name__ == "__main__":
    cuboid = Cuboid(5, 10, 25)
    sphere = Sphere(10, 10, 10, 5)

    cuboid_obj = PhantomObject(
        eom=lambda tlbr, t: tlbr + t,
        eoi=lambda intensity, t: intensity,
        tl=torch.tensor([0, 0, 0]),
        geometry=cuboid,
        intensity=1,
    )
    sphere_obj = PhantomObject(
        eom=lambda tlbr, t: tlbr + t,
        eoi=lambda intensity, t: intensity,
        tl=torch.tensor([15, 15, 15]),
        geometry=sphere,
        intensity=0.5,
    )

    phantom = Phantom(size=(30, 30, 30))
    phantom.add_phantom_object(obj=[cuboid_obj, sphere_obj])

    frames = []
    values_list = [phantom.get_phantom(top=t).numpy() for t in np.linspace(0, 5, 5)]
    for i, values in enumerate(values_list):
        X, Y, Z = np.mgrid[0 : values.shape[0], 0 : values.shape[1], 0 : values.shape[2]]
        frame = go.Frame(
            data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=values.flatten(),
                isomin=0.1,
                isomax=0.8,
                opacity=0.1,
                surface_count=17,
            ),
            name=f"{i}",  # Optional, you can name your frames
        )
        frames.append(frame)

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values_list[0].flatten(),
            isomin=0.1,
            isomax=0.8,
            opacity=0.1,
            surface_count=17,
        ),
        frames=frames,  # Add the frames you created
        layout=go.Layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                },
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Frame:",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": 30, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                }
            ],
        ),
    )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(frame=dict(duration=100, redraw=True), fromcurrent=True),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ]
    )

    frame_slider_steps = [
        {
            "args": [
                [f"Frame {i}"],
                {
                    "frame": {"duration": 30, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 30},
                },
            ],
            "label": f"Frame {i}",
            "method": "animate",
        }
        for i in range(len(frames))
    ]

    fig.update_layout(
        sliders=[
            dict(
                steps=frame_slider_steps,
                active=0,
                x=0.1,
                y=0,
                pad={"t": 50, "b": 10},
                currentvalue={"prefix": "Frame: "},
            )
        ]
    )

    fig.show()
