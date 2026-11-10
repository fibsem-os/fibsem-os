import numpy as np

from fibsem.structures.image import FibsemImage


def create_distance_map_px(w: int, h: int) -> np.ndarray:
    x = np.arange(0, w)
    y = np.arange(0, h)

    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(((w / 2) - X) ** 2 + ((h / 2) - Y) ** 2)

    return distance


def rotate_image(image: FibsemImage):
    """Rotate the AdornedImage 180 degrees."""
    data = np.rot90(np.rot90(np.copy(image.data)))
    reference = FibsemImage(data=data, metadata=image.metadata)
    return reference


def normalise_image(img: FibsemImage) -> np.ndarray:
    """Normalise the image"""
    return (img.data - np.mean(img.data)) / np.std(img.data)
