"""Turning a fluorescence z-stack on disk into one displayable RGB image.

Separate from the widgets that show it, and Qt-free, so it can be tested in CI —
anything reaching through `fibsem.ui` is skipped there for lack of napari/PyQt5.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from fibsem.fm.composite import AVAILABLE_COLORS, FMLayer, composite_fm_layers
from fibsem.fm.structures import FluorescenceImage


def is_fluorescence_image(filepath: str) -> bool:
    """Whether a recorded output is a fluorescence z-stack rather than a .tif."""
    return os.fspath(filepath).lower().endswith((".ome.tiff", ".ome.tif"))


def composite_projection(image: FluorescenceImage) -> np.ndarray:
    """Max-project each channel over z, tint by its colour, blend: an (H, W, 3) RGB.

    The same composite the FM canvas shows, from an image in hand -- a mosaic just
    stitched, say -- so a thumbnail need not re-read a file that was just written.
    """
    data = np.asarray(image.data)
    if data.ndim == 5:  # TCZYX: one time point
        data = data[0]
    projected = data.max(axis=1)  # (C, Z, Y, X) -> (C, Y, X)
    channels = list(image.metadata.channels)

    layers = []
    for index, plane in enumerate(projected):
        # metadata should describe every channel, but never drop a plane when it
        # doesn't — an unnamed channel is better than a silently missing one.
        if index < len(channels):
            name = channels[index].name
            color = (
                getattr(channels[index], "color", None)
                or AVAILABLE_COLORS[index % len(AVAILABLE_COLORS)]
            )
        else:
            name = f"Channel-{index + 1:02d}"
            color = AVAILABLE_COLORS[index % len(AVAILABLE_COLORS)]
        layers.append(FMLayer(name=name, data=plane, color=color))

    rgb = composite_fm_layers(layers)
    if rgb is None:
        raise ValueError("fluorescence image has no displayable channels")
    return rgb


def load_projection(filepath: str) -> Tuple[np.ndarray, float]:
    """Composite a fluorescence z-stack file into one RGB image.

    The volume is released before compositing: a real METEOR stack is ~530 MB in
    memory where its projection is ~25 MB, and callers may load several in a row.

    Returns:
        (H, W, 3) uint8 RGB, and the pixel size in metres.
    """
    image = FluorescenceImage.load(filepath)
    pixel_size_x = image.metadata.pixel_size_x
    try:
        rgb = composite_projection(image)
    except ValueError as e:
        raise ValueError(f"{e}: {filepath}") from e
    finally:
        del image  # drop the volume, keep the projection
    return rgb, pixel_size_x
