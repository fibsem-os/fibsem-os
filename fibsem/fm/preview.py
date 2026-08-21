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


def load_projection(filepath: str) -> Tuple[np.ndarray, float]:
    """Composite a fluorescence z-stack into one RGB image.

    Max-projects each channel over z, tints it by the channel's colour and blends
    additively — the same composite the FM canvas shows, so a stack looks the same
    in the review panel as it did live.

    The volume is released before compositing: a real METEOR stack is ~530 MB in
    memory where its projection is ~25 MB, and callers may load several in a row.

    Returns:
        (H, W, 3) uint8 RGB, and the pixel size in metres.
    """
    image = FluorescenceImage.load(filepath)
    projected = image.data.max(axis=1)  # (C, Z, Y, X) -> (C, Y, X), a new array
    channels = list(image.metadata.channels)
    pixel_size_x = image.metadata.pixel_size_x
    del image  # drop the volume, keep the projection

    layers = []
    for index, plane in enumerate(projected):
        # metadata should describe every channel, but never drop a plane when it
        # doesn't — an unnamed channel is better than a silently missing one.
        if index < len(channels):
            name, color = channels[index].name, channels[index].color
        else:
            name = f"Channel-{index + 1:02d}"
            color = AVAILABLE_COLORS[index % len(AVAILABLE_COLORS)]
        layers.append(FMLayer(name=name, data=plane, color=color))

    rgb = composite_fm_layers(layers)
    if rgb is None:
        raise ValueError(f"fluorescence image has no displayable channels: {filepath}")
    return rgb, pixel_size_x
