"""How a picture of an image is drawn, as the web viewer's layers panel sets it,
for whoever draws by name rather than by index (an agent's drawing tools).

A layer is a fluorescence channel, or a beam image's one greyscale layer, named
by its kind (``SEM``, ``FIB``). Each takes a `ChannelLook`: contrast limits in
the image's own values (None: auto, the 1st and 99th percentiles), gamma,
opacity and colour. `layer_stats` says what each layer holds and how much of it
the limits clip, so a picture that is too dark or saturated can be redrawn.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fibsem.fm.composite import auto_clim
from fibsem.imaging.export import (
    ChannelLook,
    ExportImage,
    ExportOptions,
    saturation_value,
)

# The keys a look is given by.
LOOK_KEYS = ("clim", "gamma", "opacity", "color")


def layer_names(image: ExportImage) -> List[str]:
    """The names *image*'s layers are given by: its channels, else its kind."""
    if image.layers:
        return [layer.name for layer in image.layers]
    return [image.kind] if image.gray is not None else []


def check_look(look: ChannelLook) -> None:
    """ValueError unless *look* can be drawn."""
    if look.clim is not None and not look.clim[0] < look.clim[1]:
        raise ValueError("a look's clim is (low, high), low below high.")
    if look.gamma <= 0 or not 0.0 <= look.opacity <= 1.0:
        raise ValueError("a look's gamma is above 0 and its opacity 0 to 1.")


def looks_for(
    image: ExportImage, looks: Optional[Dict[str, Dict[str, Any]]]
) -> List[ChannelLook]:
    """*looks* (by layer name, each any of `LOOK_KEYS`) as *image*'s looks in
    layer order, a layer not named left as it is. ValueError names a layer the
    image doesn't have or a look that doesn't read."""
    names = layer_names(image)
    if not looks:
        return []
    if not names:
        raise ValueError("looks are for an SEM, FIB or fluorescence image.")
    unknown = [name for name in looks if name not in names]
    if unknown:
        raise ValueError(
            f"no layer {', '.join(unknown)}: the image's layers are {', '.join(names)}"
        )
    out = []
    for name in names:
        given = looks.get(name) or {}
        if not isinstance(given, dict) or set(given) - set(LOOK_KEYS):
            raise ValueError(f"a look is any of {', '.join(LOOK_KEYS)}: {given}")
        clim = given.get("clim")
        look = ChannelLook(
            clim=None if clim is None else (float(clim[0]), float(clim[1])),
            gamma=float(given.get("gamma", 1.0)),
            opacity=float(given.get("opacity", 1.0)),
            color=str(given.get("color", "")),
        )
        check_look(look)
        out.append(look)
    return out


def cropped(image: ExportImage, box: Sequence[float]) -> ExportImage:
    """*image*'s region *box* (x, y, width, height in its pixels), at its own
    scale. ValueError for a box outside the image."""
    height, width = image.rgb.shape[:2]
    x, y, w, h = (int(round(v)) for v in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"box {list(box)} is outside the {width} x {height} image")
    region = (slice(y0, y1), slice(x0, x1))
    out = copy.copy(image)
    out.rgb = image.rgb[region]
    out.gray = None if image.gray is None else image.gray[region]
    out.layers = [
        replace(layer, data=layer.data[region], _clim_cache=None)
        for layer in image.layers
    ]
    return out


def _raw(image: ExportImage) -> List[Tuple[str, np.ndarray, Optional[float]]]:
    """Each layer's data in the image's own values, and where it saturates."""
    if image.layers:
        return [
            (layer.name, np.asarray(layer.data), saturation_value(layer.data))
            for layer in image.layers
            if layer.visible and layer.data is not None
        ]
    if image.gray is None:
        return []
    lo, hi = image.value_range or (0.0, 1.0)
    return [(image.kind, image.gray * (hi - lo) + lo, image.full_value)]


def _drawn_as_acquired(image: ExportImage, options: ExportOptions) -> bool:
    """Whether a beam image is drawn as acquired: no looks, no contrast."""
    return not image.layers and not options.looks and options.contrast_is_default()


def layer_stats(image: ExportImage, options: ExportOptions) -> Dict[str, Dict]:
    """Per layer drawn: the look it was drawn with, its auto limits resolved to
    numbers, and what it holds: min, 1st percentile, median, 99th percentile and
    max, the percent of pixels at or above the high limit (``clipped_high``) and
    at or below the low one (``clipped_low``), and the percent at its full value
    (``saturated``; None for float data)."""
    stats = {}
    for i, (name, data, full) in enumerate(_raw(image)):
        look = options.looks[i] if i < len(options.looks) else ChannelLook()
        if look.clim is not None:
            clim = look.clim
        elif _drawn_as_acquired(image, options):
            clim = (float(data.min()), float(data.max()))
        else:
            clim = auto_clim(np.asarray(data, dtype=np.float32))
        p1, median, p99 = np.percentile(data, (1, 50, 99))
        count = data.size or 1
        stats[name] = {
            "look": {**asdict(look), "clim": [float(clim[0]), float(clim[1])]},
            "auto": look.clim is None,
            "min": float(data.min()),
            "p1": float(p1),
            "median": float(median),
            "p99": float(p99),
            "max": float(data.max()),
            "clipped_low": round(100.0 * np.count_nonzero(data <= clim[0]) / count, 2),
            "clipped_high": round(100.0 * np.count_nonzero(data >= clim[1]) / count, 2),
            "saturated": None
            if full is None
            else round(100.0 * np.count_nonzero(data >= full) / count, 2),
        }
    return stats
