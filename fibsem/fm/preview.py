"""Turning a fluorescence z-stack on disk into one displayable RGB image.

Separate from the widgets that show it, and Qt-free, so it can be tested in CI —
anything reaching through `fibsem.ui` is skipped there for lack of napari/PyQt5.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np

from fibsem.fm.composite import AVAILABLE_COLORS, FMLayer, composite_fm_layers
from fibsem.fm.structures import FluorescenceImage


def is_fluorescence_image(filepath: str) -> bool:
    """Whether a recorded output is a fluorescence z-stack rather than a .tif."""
    return os.fspath(filepath).lower().endswith((".ome.tiff", ".ome.tif"))


def projection_layers(image: FluorescenceImage) -> List[FMLayer]:
    """One layer per channel: its max projection over z, named and coloured.

    What :func:`composite_projection` blends, kept apart so a caller that lets the
    user recolour or hide a channel can re-blend without re-reading or re-projecting.
    """
    data = np.asarray(image.data)
    if data.ndim == 5:  # TCZYX: one time point
        data = data[0]
    projected = data.max(axis=1)  # (C, Z, Y, X) -> (C, Y, X)
    channels = list(image.metadata.channels)
    layers = []
    for index, plane in enumerate(projected):
        name, color = _channel_look(channels, index, len(projected))
        layers.append(FMLayer(name=name, data=plane, color=color))
    return layers


def _channel_look(channels: list, index: int, count: int = 1) -> Tuple[str, str]:
    """A channel's name and colour as the projection draws it. Metadata should
    describe every channel, but never drop a plane when it doesn't: an unnamed
    channel is better than a silently missing one. Gray is the settings'
    default, so in a stack of several channels it takes the palette's colour."""
    fallback = AVAILABLE_COLORS[index % len(AVAILABLE_COLORS)]
    if index < len(channels):
        color = getattr(channels[index], "color", None)
        if not color or (color == "gray" and count > 1):
            color = fallback
        return channels[index].name, color
    return f"Channel-{index + 1:02d}", fallback


def stack_facts(image: FluorescenceImage) -> Dict[str, Any]:
    """What a legend says about a stack: each channel as the projection draws
    it (name, colour, excitation in nm, emission filter, exposure in s), the
    number of z planes, their step in metres (None when not recorded), and when
    it was acquired."""
    shape = np.shape(image.data)
    planes = int(shape[-3]) if len(shape) >= 3 else 1
    count = int(shape[-4]) if len(shape) >= 4 else 1
    metadata = image.metadata
    channels = list(metadata.channels)
    facts = []
    for index in range(count):
        name, color = _channel_look(channels, index, count)
        known = channels[index] if index < len(channels) else None
        emission = getattr(known, "emission_filter", None)
        facts.append(
            {
                "name": name,
                "color": color,
                "excitation_wavelength": getattr(known, "excitation_wavelength", None),
                "emission": getattr(emission, "name", None),
                "exposure_time": getattr(known, "exposure_time", None),
            }
        )
    step = metadata.pixel_size_z
    z = metadata.z_positions or []
    if step is None and len(z) >= 2:
        step = abs(z[1] - z[0])
    acquired = metadata.acquisition_datetime
    return {
        "channels": facts,
        "z_planes": planes,
        "z_step": step,
        "acquired_at": acquired.isoformat() if acquired is not None else None,
    }


def composite_projection(image: FluorescenceImage) -> np.ndarray:
    """Max-project each channel over z, tint by its colour, blend: an (H, W, 3) RGB.

    The same composite the FM canvas shows, from an image in hand -- a mosaic just
    stitched, say -- so a thumbnail need not re-read a file that was just written.
    """
    rgb = composite_fm_layers(projection_layers(image))
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
    rgb, pixel_size_x, _ = load_stack_preview(filepath)
    return rgb, pixel_size_x


def load_stack_preview(filepath: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """As :func:`load_projection`, with the stack's :func:`stack_facts`."""
    image = FluorescenceImage.load(filepath)
    pixel_size_x = image.metadata.pixel_size_x
    try:
        facts = stack_facts(image)
        rgb = composite_projection(image)
    except ValueError as e:
        raise ValueError(f"{e}: {filepath}") from e
    finally:
        del image  # drop the volume, keep the projection
    return rgb, pixel_size_x, facts
