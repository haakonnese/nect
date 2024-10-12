from skimage.transform import iradon


def fbp(sinogram, theta, filter_name="ramp", circle=True):
    return iradon(radon_image=sinogram, theta=theta, filter_name=filter_name, circle=circle)
