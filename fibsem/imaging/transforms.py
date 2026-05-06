from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import numpy as np
import cv2

if TYPE_CHECKING:
    from fibsem.structures import FibsemImage


def cosine_stretch(image: "FibsemImage", tilt_angle: float) -> "FibsemImage":
    """Stretch a FIB image along Y to correct for the cosine foreshortening caused by the tilt angle.

    The FIB column views the sample at `tilt_angle` degrees from the SEM (vertical) column,
    which compresses the Y-axis of the FIB image by cos(tilt_angle). This function inverts
    that compression so the FIB image is in the same coordinate space as the SEM image.

    Args:
        image: FibsemImage (typically IB) to correct.
        tilt_angle: angle in degrees between the SEM and FIB columns at the sample.
    Returns:
        FibsemImage with Y-axis stretched by 1/cos(tilt_angle), same width as input.
    """
    from fibsem.structures import FibsemImage

    h, w = image.data.shape[:2]
    scale_y = 1.0 / np.cos(np.radians(tilt_angle))
    new_h = int(round(h * scale_y))

    data = image.data.astype(np.float32)
    stretched = cv2.resize(data, (w, new_h), interpolation=cv2.INTER_LINEAR)
    stretched = stretched.astype(image.data.dtype)

    return FibsemImage(data=stretched, metadata=image.metadata)


def perspective_transform(image: "FibsemImage", tilt_angle: float) -> "FibsemImage":
    """Apply a perspective (tilt) transform to a FIB image to align it with a SEM image.

    Builds a homography that compresses Y by cos(tilt_angle) anchored at the image centre,
    which maps each FIB pixel to its corresponding SEM pixel position. The output image
    is the same shape as the input — the transform is equivalent to a Y-scale around the
    image midpoint.

    Use the inverse (tilt_angle negated or H inverted) to warp the FIB image into SEM space.

    Args:
        image: FibsemImage (typically IB) to transform.
        tilt_angle: angle in degrees between the SEM and FIB columns at the sample.
    Returns:
        FibsemImage warped into SEM coordinate space, same shape as input.
    """
    from fibsem.structures import FibsemImage

    h, w = image.data.shape[:2]
    s = np.cos(np.radians(tilt_angle))
    cy = h / 2.0

    # H maps FIB pixel (x, y) -> SEM pixel (x, y*cos + cy*(1-cos))
    # To warp FIB image into SEM space we need the inverse: stretch Y by 1/s
    H_fib_to_sem = np.array(
        [
            [1, 0, 0],
            [0, 1 / s, cy * (1 - 1 / s)],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    data = image.data.astype(np.float32)
    warped = cv2.warpPerspective(data, H_fib_to_sem, (w, h))
    warped = warped.astype(image.data.dtype)

    return FibsemImage(data=warped, metadata=image.metadata)
