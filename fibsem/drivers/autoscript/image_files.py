"""Reading images saved by the instrument's own software (xT, Maps), rather than
by fibsem: the pixel size is in the instrument's own TIFF tag, and a saved image may
carry the databar under it. The registry offers :func:`read_image` to the core image
loaders (``DriverEntry.image_reader``). Needs only ``tifffile``, never the SDK."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tifffile as tff
from PIL import Image

# The instrument's metadata tag: its name, and its code when tifffile has no name for it.
METADATA_TAG_NAME = "FEI_HELIOS"
METADATA_TAG_CODE = 34682


def read_metadata(filename: str) -> Dict[str, Any]:
    """The instrument's metadata in *filename*, or an empty dict without its tag."""
    md: Dict[str, Any] = {}
    with tff.TiffFile(filename) as tif:
        for page in tif.pages:
            for tag in page.tags.values():
                if tag.name == METADATA_TAG_NAME or tag.code == METADATA_TAG_CODE:
                    md = tag.value
    return md


def read_image(filename: str) -> Tuple[np.ndarray, Optional[float]]:
    """The image in *filename* and its pixel size in metres, without the databar.

    Raises ``ValueError`` for a file without the instrument's metadata tag, so a
    loader can try the next format.
    """
    md = read_metadata(filename)
    if not md:
        raise ValueError(f"{filename} has no {METADATA_TAG_NAME} metadata")
    image = tff.imread(filename)

    pixel_size = None
    try:
        pixel_size = md["Scan"]["PixelWidth"]
    except KeyError as e:
        logging.warning(f"Pixel size not found in metadata: {e}")

    # convert to grayscale
    if image.ndim == 3:
        image = np.asarray(Image.fromarray(image).convert("L"))

    trim_metadata: bool = False
    try:
        shape = md["Image"]["ResolutionY"], md["Image"]["ResolutionX"]
        if image.shape != shape:
            logging.info(
                f"Image shape {image.shape} does not match metadata shape {shape}, likely a metadata bar present"
            )
            trim_metadata = True
    except KeyError as e:
        logging.warning(f"Image shape not found in metadata: {e}")

    # trim the image to before the first row with all zeros
    if trim_metadata:
        try:
            # crop the image to the metadata bar
            cropped_img = image[: shape[0], : shape[1]]
            # remove the metadata bar with image processing
            trimmed_img = remove_metadata_bar(image)

            logging.info(
                f"Cropped Shape: {cropped_img.shape}, Trimmed Shape: {trimmed_img.shape}"
            )
            if cropped_img.shape != trimmed_img.shape:
                raise ValueError(
                    "Cropped image shape does not match trimmed image shape"
                )

            if image.shape != trimmed_img.shape:
                logging.info(f"Image trimmed from {image.shape} to {trimmed_img.shape}")
                image = trimmed_img
        except Exception as e:
            logging.error(f"Error trimming image: {e}")

    return image, pixel_size


def remove_metadata_bar(img: np.ndarray) -> np.ndarray:
    """The image above its first all-zero row, where the databar starts."""
    for i, row in enumerate(img):
        if not np.any(row):
            # trim the image when the first row with all zeros is found
            break
    return img[:i]
