import numpy as np

from fibsem.structures import FibsemImage


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


def percentile_stretch(
    data: np.ndarray, clip_lo: float = 0.5, clip_hi: float = 99.5
) -> np.ndarray:
    """Linearly stretch *data* so the [clip_lo, clip_hi] percentile range fills the full dtype range.

    Args:
        data: Input array (integer dtype).
        clip_lo: Lower clip percentile (default 0.5).
        clip_hi: Upper clip percentile (default 99.5).

    Returns:
        Stretched array with the same dtype as *data*, or an unchanged copy if
        the histogram is degenerate (p_hi <= p_lo).
    """
    dtype = data.dtype
    dtype_max = np.iinfo(dtype).max
    p_lo = np.percentile(data, clip_lo)
    p_hi = np.percentile(data, clip_hi)
    if p_hi <= p_lo:
        return data.copy()
    clipped = np.clip(data.astype(np.float64), p_lo, p_hi)
    return ((clipped - p_lo) / (p_hi - p_lo) * dtype_max).astype(dtype)
