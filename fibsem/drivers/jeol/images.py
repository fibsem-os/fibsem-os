"""Reading JEOL's frames into arrays.

The FIB's ``getDispImage`` returns the displayed frame as a PNG inside a longer
payload; the SEM writes a photo to a TIFF file, which is read once it stops growing.
"""

from __future__ import annotations

import io
import os
import time
from typing import Callable

import numpy as np
from PIL import Image

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# How long a SEM photo may take to be written: a slow full-resolution one takes minutes.
PHOTO_TIMEOUT = 240.0
PHOTO_POLL_INTERVAL = 0.5


def decode_display_image(payload: bytes) -> np.ndarray:
    """The 8-bit greyscale frame in a ``getDispImage`` payload: the PNG wherever its
    header starts, converted from RGBA."""
    if payload is None or len(payload) < 10:
        raise ValueError("The JEOL display image payload is empty.")
    start = payload.find(_PNG_SIGNATURE)
    if start == -1:
        raise ValueError("No PNG header in the JEOL display image payload.")
    with Image.open(io.BytesIO(payload[start:])) as image:
        image.load()
        return np.array(image.convert("L"))


def read_photo(
    path: str,
    timeout: float = PHOTO_TIMEOUT,
    wait: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    """The SEM photo JEOL is writing to *path*, once the file exists and has stopped
    growing. Raises `TimeoutError` when it is not written within *timeout* seconds."""
    start = time.monotonic()
    while not os.path.exists(path):
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"The SEM photo was not created within {timeout} s.")
        wait(0.1)
    previous = -1
    while True:
        size = os.path.getsize(path)
        if size > 0 and size == previous:
            break
        if time.monotonic() - start > timeout:
            raise TimeoutError(
                f"The SEM photo was still being written after {timeout} s."
            )
        previous = size
        wait(PHOTO_POLL_INTERVAL)
    with Image.open(path) as image:
        image.load()
        return np.array(image)
