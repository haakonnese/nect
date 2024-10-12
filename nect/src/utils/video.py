import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def save_video(video_images: list[np.ndarray], file_name: str, fps: int = 5):
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()  # Clear the previous frame
        ax.imshow(video_images[frame], cmap="gray")
        ax.axis("off")

    ani = FuncAnimation(fig, update, frames=len(video_images), repeat=False)

    # Set up the writer (FFMpegWriter for MP4)
    Writer = FuncAnimation.save
    writer = Writer(ani, file_name, writer="ffmpeg", fps=fps)

    ani.save(file_name, writer=writer)  # Save the video
    plt.close(fig)
