"""Odemis images: a frame from the odemis client as a ``FibsemImage``.

The client returns the bare frame, so its metadata is rebuilt from the beam's
settings and the microscope state, as every backend stamps it.

This module imports nothing from odemis; the backend's guarded import is the only one.
"""

from __future__ import annotations

import numpy as np

from fibsem.structures import (
    FibsemImage,
    FibsemImageMetadata,
    ImageSettings,
    MicroscopeState,
    Point,
)


def fibsem_image_from_frame(
    data: np.ndarray, image_settings: ImageSettings, state: MicroscopeState
) -> FibsemImage:
    """A FibsemImage of *data*, with its pixel size from the settings' hfw."""
    # TODO: retrieve the full image metadata from image md, rather than reconstruct
    pixel_size = image_settings.hfw / image_settings.resolution[0]
    md = FibsemImageMetadata(
        image_settings=image_settings,
        pixel_size=Point(pixel_size, pixel_size),
        microscope_state=state,
    )
    return FibsemImage(data, md)
