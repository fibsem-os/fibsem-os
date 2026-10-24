import numpy as np
import scipy.ndimage as ndi

from fibsem.imaging import utils as image_utils
from fibsem.structures.settings import Point


### MASKING
def create_circle_mask(
    shape: tuple = (128, 128), radius: int = 32, sigma: int = 3
) -> np.ndarray:
    """_summary_

    Args:
        shape (tuple, optional): _description_. Defaults to (128, 128).
        radius (int, optional): _description_. Defaults to 32.
        sigma (int, optional): _description_. Defaults to 3.

    Returns:
        np.ndarray: _description_
    """
    distance = image_utils.create_distance_map_px(w=shape[1], h=shape[0])
    mask = distance <= radius

    if sigma:
        mask = ndi.gaussian_filter(mask, sigma=sigma)

    return mask


def create_bandpass_mask(
    shape: tuple = (256, 256), lp: int = 32, hp: int = 2, sigma: int = 3
) -> np.ndarray:
    """_summary_

    Args:
        shape (tuple, optional): _description_. Defaults to (256, 256).
        lp (int, optional): _description_. Defaults to 32.
        hp (int, optional): _description_. Defaults to 2.
        sigma (int, optional): _description_. Defaults to 3.

    Returns:
        np.ndarray: _description_
    """

    distance = image_utils.create_distance_map_px(w=shape[1], h=shape[0])

    lowpass = distance <= lp
    highpass = distance >= hp

    mask = (lowpass * highpass).astype(np.float32)
    if sigma:
        mask = ndi.gaussian_filter(mask, sigma=sigma)

    return mask


# FROM AUTOLAMELLA
def _mask_rectangular(image_shape, sigma=5.0, *, start=None, extent=None):
    """Make a rectangular mask with soft edges for image normalization.

    Parameters
    ----------
    image_shape : tuple
        Shape of the original image array
    sigma : float, optional
        Sigma value (in pixels) for gaussian blur function, by default 5.
    start : tuple, optional
        Origin point of the rectangle, e.g., ([plane,] row, column).
        Default start is 5% of the total image width and height.
    extent : int, optional
        The extent (size) of the drawn rectangle.
        E.g., ([num_planes,] num_rows, num_cols).
        Default is for the rectangle to cover 95% of the image width & height.

    Returns
    -------
    ndarray
        Rectangular mask with soft edges in array matching input image_shape.
    """
    import skimage

    if extent is None:
        # leave at least a 5% gap on each edge
        start = np.round(np.array(image_shape) * 0.05)
        extent = np.round(np.array(image_shape) * 0.90)
    rr, cc = skimage.draw.rectangle(start, extent=extent, shape=image_shape)
    mask = np.zeros(image_shape)
    mask[rr.astype(int), cc.astype(int)] = 1.0
    mask = ndi.gaussian_filter(mask, sigma=sigma)
    return mask


def create_rect_mask(
    img: np.ndarray, w: int, h: int, sigma: int = 0, pt: Point = None
) -> np.ndarray:
    """Create a rectangular mask at centred at the desired point.

    Args:
        img (np.ndarray): Image to be masked
        w (int): mask width
        h (int): mask height
        sigma (int, optional): gaussian blur to apply to mask (softness). Defaults to 0.
        pt (Point): Mask centre point. Defaults to None

    Returns:
        np.ndarray: mask
    """
    # centre mask point
    if pt is None:
        pt = Point(img.data.shape[1] // 2, img.data.shape[0] // 2)

    mask = np.zeros_like(img)

    y_min, y_max = (
        int(np.clip(pt.y - h / 2, 0, img.shape[1])),
        int(np.clip(pt.y + h / 2, 0, img.shape[1])),
    )
    x_min, x_max = (
        int(np.clip(pt.x - w / 2, 0, img.shape[1])),
        int(np.clip(pt.x + w / 2, 0, img.shape[1])),
    )

    mask[y_min:y_max, x_min:x_max] = 1

    if sigma:
        mask = ndi.gaussian_filter(mask, sigma=sigma)

    return mask


def apply_circular_mask(img: np.ndarray, radius: int, sigma: int = 0) -> np.ndarray:
    circ_mask = create_circle_mask(img.shape, radius=radius, sigma=sigma)

    if img.ndim == 3:
        circ_mask = np.moveaxis(np.array([circ_mask, circ_mask, circ_mask]), 0, 2)

    return img * circ_mask
