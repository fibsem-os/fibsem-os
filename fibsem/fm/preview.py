"""Turning a fluorescence z-stack on disk into one displayable RGB image.

Separate from the widgets that show it, and Qt-free, so it can be tested in CI —
anything reaching through `fibsem.ui` is skipped there for lack of napari/PyQt5.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fibsem.fm.composite import AVAILABLE_COLORS, FMLayer, composite_fm_layers
from fibsem.fm.structures import FluorescenceImage, mean_z_step

if TYPE_CHECKING:
    from fibsem.fm.stack_file import StackFile

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
    _check_plane(z, data.shape[1])
    shown = data.max(axis=1) if z is None else data[:, z]  # (C, Y, X)
    return _layers(
        list(image.metadata.channels), len(shown), shown.__getitem__, channels
    )


def _check_plane(z: Optional[int], planes: int) -> None:
    if z is not None and not 0 <= z < planes:
        raise ValueError(f"plane {z} is outside the stack's 0 to {planes - 1}")


def _layers(
    metadata_channels: list,
    count: int,
    plane_of: Callable[[int], np.ndarray],
    channels: Optional[Sequence[str]],
    shape_of: Optional[Callable[[int], np.ndarray]] = None,
) -> List[FMLayer]:
    """The layers of *count* channels, each named and coloured, the ones
    *channels* names shown (all when None). *plane_of* gives a channel's
    plane, and is asked only for the channels shown; a hidden first layer,
    there for the shape, takes its plane from *shape_of* when given."""
    looks = [channel_look(metadata_channels, i, count) for i in range(count)]
    names = [name for name, _ in looks]
    unknown = [c for c in channels or [] if c not in names]
    if unknown:
        raise ValueError(
            f"no channel {', '.join(unknown)} in the stack: it has {', '.join(names)}"
        )
    layers = []
    for i, (name, color) in enumerate(looks):
        visible = channels is None or name in channels
        if visible or i == 0:  # a hidden first layer still gives the shape
            plane = plane_of(i) if visible or shape_of is None else shape_of(i)
            layers.append(FMLayer(name=name, data=plane, color=color, visible=visible))
    return layers


def _czyx(image: FluorescenceImage) -> np.ndarray:
    data = np.asarray(image.data)
    if data.ndim == 5:  # TCZYX: one time point
        data = data[0]
    while data.ndim < 4:  # a single channel, or a single plane
        data = data[None]
    return data


def channel_look(channels: list, index: int, count: int = 1) -> Tuple[str, str]:
    """A channel's name and colour as the projection draws it. Metadata should
    describe every channel, but never drop a plane when it doesn't: an unnamed
    channel is better than a silently missing one. Gray is the settings'
    default, so in a stack of several channels a gray channel takes the colour
    of its light (`_light_color`), and the palette's when that isn't known."""
    fallback = AVAILABLE_COLORS[index % len(AVAILABLE_COLORS)]
    if index < len(channels):
        channel = channels[index]
        color = getattr(channel, "color", None)
        if not color or (color == "gray" and count > 1):
            color = _light_color(channel) or fallback
        return channel.name, color
    return f"Channel-{index + 1:02d}", fallback


def _light_color(channel: Any) -> Optional[str]:
    """The colour a channel's light suggests: gray for reflection, else by its
    excitation in nm (405 blue, 488 green, 561 red, 640 violet)."""
    if "reflect" in str(getattr(channel, "name", "")).lower():
        return "gray"
    nm = getattr(channel, "excitation_wavelength", None)
    if not nm:
        return None
    if nm < 450:
        return "blue"
    if nm < 530:
        return "green"
    if nm < 610:
        return "red"
    return "violet"


def stack_facts(image: FluorescenceImage) -> Dict[str, Any]:
    """What a legend says about a stack: each channel as the projection draws
    it (name, colour, excitation in nm, emission filter, exposure in s, and
    each plane's sharpness), the number of z planes, their step in metres (None
    when not recorded), and when it was acquired."""
    shape = np.shape(image.data)
    planes = int(shape[-3]) if len(shape) >= 3 else 1
    count = int(shape[-4]) if len(shape) >= 4 else 1
    sharpness = (
        [plane_sharpness(channel) for channel in _czyx(image)] if planes > 1 else None
    )
    return _facts(image.metadata, count, planes, sharpness)


def _facts(
    metadata: Any,
    count: int,
    planes: int,
    sharpness: Optional[List[List[float]]],
) -> Dict[str, Any]:
    channels = list(metadata.channels)
    facts = []
    for index in range(count):
        name, color = channel_look(channels, index, count)
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
    for fact, scores in zip(facts, sharpness or []):
        fact["sharpness"] = scores
        fact["sharpest_plane"] = int(np.argmax(scores))
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
    return _relative([_plane_score(plane) for plane in planes])


def _plane_score(plane: np.ndarray) -> float:
    from fibsem.autofunctions.metrics import calculate_focus_quality
    from fibsem.imaging.reduce import downsample

    return calculate_focus_quality(downsample(np.asarray(plane), 256), SHARPNESS_METHOD)


def _relative(scores: List[float]) -> List[float]:
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
    rgb, pixel_size_x, _, _ = load_stack_preview(filepath)
    return rgb, pixel_size_x


def load_stack_preview(
    filepath: str,
    z: Optional[int] = None,
    channels: Optional[Sequence[str]] = None,
    max_px: Optional[int] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any], Tuple[int, int]]:
    """As :func:`load_projection`, with the stack's :func:`stack_facts` and its
    full (Y, X) size: plane *z* and *channels* as :func:`render_stack` takes them.

    Reads the file a plane at a time (`StackFile`): a plane reads one page per
    channel shown, and the projection and the sharpness come from one pass
    over the stack, kept for the next read of the same file. With *max_px*, from
    the smallest copy the file holds that is still that size on its long side;
    the pixel size stays the full size's."""
    from fibsem.fm.stack_file import StackFile

    with StackFile(filepath) as stack:
        level = stack.level_for(max_px)
        projection, sharpness = stack_summary(stack, level)
        try:
            _check_plane(z, stack.planes)
            layers = _layers(
                list(stack.metadata.channels),
                stack.channels,
                (lambda c: projection[c])
                if z is None
                else (lambda c: stack.plane(c, z, level)),
                channels,
                shape_of=lambda c: projection[c],
            )
        except ValueError as e:
            raise ValueError(f"{e}: {filepath}") from e
        facts = _facts(stack.metadata, stack.channels, stack.planes, sharpness)
        pixel_size_x = stack.metadata.pixel_size_x
        full = tuple(stack.shape[-2:])
    rgb = composite_fm_layers(layers, shape=layers[0].data.shape) if layers else None
    if rgb is None:
        raise ValueError(f"fluorescence image has no displayable channels: {filepath}")
    return rgb, pixel_size_x, facts, full


# Projections and sharpness of the stacks read lately, by path, size and
# modification time: a viewer steps planes and channels of one stack, and each
# projection is one pass over gigabytes.
_SUMMARIES: "OrderedDict[Tuple[str, int, int, int], Tuple[np.ndarray, Optional[List[List[float]]]]]" = OrderedDict()
_SUMMARIES_KEPT = 8
_SUMMARIES_LOCK = threading.Lock()


def stack_summary(
    stack: "StackFile", level: int = 0
) -> Tuple[np.ndarray, Optional[List[List[float]]]]:
    """Each channel's max projection over z, (C, Y, X), and each plane's
    sharpness per channel (None for a single plane), from one pass over the
    stack a plane at a time at *level*; the last few stacks' are kept."""
    stat = os.stat(stack.path)
    key = (os.path.realpath(stack.path), stat.st_size, stat.st_mtime_ns, level)
    with _SUMMARIES_LOCK:
        if key in _SUMMARIES:
            _SUMMARIES.move_to_end(key)
            return _SUMMARIES[key]
    projection = None
    scores: List[List[float]] = []
    for c in range(stack.channels):
        channel_scores = []
        for z in range(stack.planes):
            plane = stack.plane(c, z, level)
            if projection is None:
                projection = np.empty((stack.channels, *plane.shape), plane.dtype)
            if z == 0:
                projection[c] = plane
            else:
                np.maximum(projection[c], plane, out=projection[c])
            if stack.planes > 1:
                channel_scores.append(_plane_score(plane))
        scores.append(_relative(channel_scores))
    summary = (projection, scores if stack.planes > 1 else None)
    with _SUMMARIES_LOCK:
        _SUMMARIES[key] = summary
        while len(_SUMMARIES) > _SUMMARIES_KEPT:
            _SUMMARIES.popitem(last=False)
    return summary
