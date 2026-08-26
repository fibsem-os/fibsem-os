from typing import Tuple

import numpy as np
from fibsem.structures import Point, FibsemImage
from PIL import Image
from scipy import ndimage as ndi


def create_distance_map_px(w: int, h: int) -> np.ndarray:
    x = np.arange(0, w)
    y = np.arange(0, h)

    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(((w / 2) - X) ** 2 + ((h / 2) - Y) ** 2)

    return distance


def measure_brightness(img: FibsemImage) -> float:

    return np.mean(img.data)


def rotate_image(image: FibsemImage):
    """Rotate the AdornedImage 180 degrees."""
    data = np.rot90(np.rot90(np.copy(image.data)))
    reference = FibsemImage(data=data, metadata=image.metadata)
    return reference


def normalise_image(img: FibsemImage) -> np.ndarray:
    """Normalise the image"""
    return (img.data - np.mean(img.data)) / np.std(img.data)


def difference_of_gaussians(
    data: np.ndarray, low_sigma: float = 1.5, high_sigma: float = 16.0
) -> np.ndarray:
    """Difference-of-Gaussians band-pass filter.

    Suppresses both the slow background gradient (charging, uneven illumination)
    and the pixel noise, leaving the spatial band that the fiducial occupies.
    Unlike `normalise_image`, the result is offset-free by construction rather
    than by subtracting a global mean, so a gradient that differs between the
    reference and the new image cannot bias the correlation.

    `low_sigma` must stay below the fiducial arm width and `high_sigma` well
    below the alignment ROI, or the fiducial itself is removed. Both are in
    pixels, so at a different resolution or HFW they are a different physical
    scale -- see FIB-711.

    Args:
        data: 2-D image data. Not modified.
        low_sigma: standard deviation of the fine-scale Gaussian, in pixels.
        high_sigma: standard deviation of the coarse-scale Gaussian, in pixels.

    Returns:
        float64 array of the same shape.
    """
    if high_sigma <= low_sigma:
        raise ValueError(
            f"high_sigma ({high_sigma}) must be greater than low_sigma ({low_sigma})"
        )
    f = np.asarray(data, dtype=np.float64)
    return ndi.gaussian_filter(f, low_sigma) - ndi.gaussian_filter(f, high_sigma)


def hann_window(shape: Tuple[int, int]) -> np.ndarray:
    """Separable 2-D Hann window.

    Applied before an FFT-based correlation it tapers the image to zero at the
    border, which removes the wrap-around edge energy that otherwise appears as
    a cross-shaped artefact through the centre of the correlation surface.

    Args:
        shape: (height, width) of the image to window.

    Returns:
        float64 array of the given shape, values in [0, 1].
    """
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w))


def cosine_stretch(img: FibsemImage, tilt_degrees: float):
    """Apply a cosine stretch to an image based on the relative tilt.

    This is required when aligning images with different tilts to ensure features are the same size.

    Args:
        img (AdornedImage): _description_
        tilt_degrees (float): _description_

    Returns:
        _type_: _description_
    """
    # note: do smaller version for negative tilt??

    tilt = np.deg2rad(tilt_degrees)

    shape = int(img.data.shape[0] / np.cos(tilt)), int(img.data.shape[1] / np.cos(tilt))

    # cosine stretch
    # larger
    resized_img = np.asarray(
        Image.fromarray(img.data).resize(size=(shape[1], shape[0]))
    )

    # crop centre out?
    c = Point(resized_img.shape[1] // 2, resized_img.shape[0] // 2)
    dy, dx = img.data.shape[0] // 2, img.data.shape[1] // 2
    scaled_img = resized_img[c.y - dy : c.y + dy, c.x - dx : c.x + dx]

    return FibsemImage(data=scaled_img, metadata=img.metadata)


def apply_image_mask(img: FibsemImage, mask: np.ndarray) -> np.ndarray:

    return normalise_image(img) * mask


def percentile_stretch(data: np.ndarray, clip_lo: float = 0.5, clip_hi: float = 99.5) -> np.ndarray:
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
