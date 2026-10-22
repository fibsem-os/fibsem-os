"""Turning a fluorescence z-stack on disk into one displayable RGB image.

Separate from the widgets that show it, and Qt-free, so it can be tested in CI —
anything reaching through `fibsem.ui` is skipped there for lack of napari/PyQt5.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fibsem.fm.composite import AVAILABLE_COLORS, FMLayer, composite_fm_layers
from fibsem.fm.structures import FluorescenceImage, mean_z_step

# How plane_sharpness scores a plane (`fibsem.autofunctions.metrics`).
SHARPNESS_METHOD = "variance"


def is_fluorescence_image(filepath: str) -> bool:
    """Whether a recorded output is a fluorescence z-stack rather than a .tif."""
    return os.fspath(filepath).lower().endswith((".ome.tiff", ".ome.tif"))


def projection_layers(image: FluorescenceImage) -> List[FMLayer]:
    """One layer per channel: its max projection over z, named and coloured.

    What :func:`composite_projection` blends, kept apart so a caller that lets the
    user recolour or hide a channel can re-blend without re-reading or re-projecting.
    """
    return stack_layers(image)


def stack_layers(
    image: FluorescenceImage,
    z: Optional[int] = None,
    channels: Optional[Sequence[str]] = None,
) -> List[FMLayer]:
    """One layer per channel, named and coloured: plane *z* (0 is the first
    plane), or the max projection over z when *z* is None. *channels* names
    the channels shown; the rest are hidden. ValueError names what is wrong:
    a plane outside the stack, or a channel it doesn't have."""
    data = _czyx(image)
    planes = data.shape[1]
    if z is None:
        shown = data.max(axis=1)  # (C, Z, Y, X) -> (C, Y, X)
    elif 0 <= z < planes:
        shown = data[:, z]
    else:
        raise ValueError(f"plane {z} is outside the stack's 0 to {planes - 1}")
    looks = [
        _channel_look(list(image.metadata.channels), i, len(shown))
        for i in range(len(shown))
    ]
    names = [name for name, _ in looks]
    unknown = [c for c in channels or [] if c not in names]
    if unknown:
        raise ValueError(
            f"no channel {', '.join(unknown)} in the stack: it has {', '.join(names)}"
        )
    return [
        FMLayer(
            name=name,
            data=plane,
            color=color,
            visible=channels is None or name in channels,
        )
        for plane, (name, color) in zip(shown, looks)
    ]


def _czyx(image: FluorescenceImage) -> np.ndarray:
    data = np.asarray(image.data)
    if data.ndim == 5:  # TCZYX: one time point
        data = data[0]
    while data.ndim < 4:  # a single channel, or a single plane
        data = data[None]
    return data


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
    if planes > 1:
        data = _czyx(image)
        for fact, channel in zip(facts, data):
            fact["sharpness"] = plane_sharpness(channel)
            fact["sharpest_plane"] = int(np.argmax(fact["sharpness"]))
    step = metadata.pixel_size_z
    z = metadata.z_positions or []
    if step is None and len(z) >= 2:
        step = mean_z_step(z)
    acquired = metadata.acquisition_datetime
    return {
        "channels": facts,
        "z_planes": planes,
        "z_step": step,
        "z_positions": [float(p) for p in z] if len(z) == planes else None,
        "acquired_at": acquired.isoformat() if acquired is not None else None,
    }


def plane_sharpness(planes: np.ndarray) -> List[float]:
    """How sharp each plane of one channel's (Z, Y, X) stack is, 1 the
    sharpest: the mean local variance of each plane reduced to at most 256 px
    a side, which camera noise barely moves."""
    from fibsem.autofunctions.metrics import calculate_focus_quality
    from fibsem.imaging.reduce import downsample

    scores = [
        calculate_focus_quality(downsample(np.asarray(plane), 256), SHARPNESS_METHOD)
        for plane in planes
    ]
    best = max(scores, default=0.0)
    return [round(s / best, 3) if best > 0 else 0.0 for s in scores]


def composite_projection(image: FluorescenceImage) -> np.ndarray:
    """Max-project each channel over z, tint by its colour, blend: an (H, W, 3) RGB.

    The same composite the FM canvas shows, from an image in hand -- a mosaic just
    stitched, say -- so a thumbnail need not re-read a file that was just written.
    """
    return render_stack(image)


def render_stack(
    image: FluorescenceImage,
    z: Optional[int] = None,
    channels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Plane *z* of *image* (the max projection when None) in *channels* (all
    when None), each tinted by its colour and blended: an (H, W, 3) RGB, as the
    FM canvas shows it."""
    layers = stack_layers(image, z, channels)
    rgb = composite_fm_layers(layers, shape=layers[0].data.shape if layers else None)
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


def load_stack_preview(
    filepath: str,
    z: Optional[int] = None,
    channels: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """As :func:`load_projection`, with the stack's :func:`stack_facts`: plane
    *z* and *channels* as :func:`render_stack` takes them."""
    image = FluorescenceImage.load(filepath)
    pixel_size_x = image.metadata.pixel_size_x
    try:
        facts = stack_facts(image)
        rgb = render_stack(image, z, channels)
    except ValueError as e:
        raise ValueError(f"{e}: {filepath}") from e
    finally:
        del image  # drop the volume, keep the projection
    return rgb, pixel_size_x, facts
