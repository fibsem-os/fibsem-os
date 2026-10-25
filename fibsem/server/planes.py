"""An image's raw pixel planes, for a viewer that draws them itself.

The web viewer composites channels in the browser, so contrast, colour, gamma and
the plane on screen change without asking the server again. It needs each
channel's values as stored, not a rendered picture: plane *z*, every plane, or a
projection over a range of planes, at full size or as a 1/4 or 1/8 copy. Zoomed
into a large image, it asks for the region on screen at full size (*box*), which a
0.60 file serves from the tiles that region covers.

A file saved by 0.60 holds those copies (`fibsem.image_format`). For an older
file, or a METEOR one, they are made on first open, a plane at a time, and kept
in memory within `COPIES_BUDGET`; the file is never rewritten.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from fibsem.server.files import FileRefusal, kind, resolve

# The reductions a viewer asks for: full size, and the copies 0.60 files hold.
FACTORS = (1, 4, 8)
PROJECTIONS = ("max", "mean", "min")
# Small copies made for files that don't hold them, by bytes.
COPIES_BUDGET = 512 * 1024**2
# The most pixels a region asks for: a large screen's worth, twice over.
BOX_PIXELS = 4096 * 4096

Plane = Union[int, str]

_COPIES: "OrderedDict[Tuple[str, int, int, int], np.ndarray]" = OrderedDict()
_COPIES_LOCK = threading.Lock()


def read(
    root: Optional[str],
    path: str,
    factor: int = 1,
    z: Optional[Plane] = None,
    first: Optional[int] = None,
    last: Optional[int] = None,
    channels: Optional[Sequence[str]] = None,
    box: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """The planes of the image at *path* as a (C, Z, Y, X) array, and what they
    are: their shape, dtype, the *factor* they are reduced by, the channels'
    names, and the full image's (Y, X). *box* (x, y, width, height), at full
    size only, reads that region of the planes instead of the whole.

    A fluorescence stack answers plane *z* (0 the first), ``"all"`` its planes,
    or a projection (``"max"``, ``"mean"``, ``"min"``; max when *z* is None)
    over planes *first* to *last*, in *channels* (all when None). Any other
    image is one plane of one channel, its detector or its grey values.
    """
    from fibsem.fm.preview import channel_look, is_fluorescence_image

    if factor not in FACTORS:
        raise FileRefusal(
            422, "invalid", f"factor is one of {', '.join(map(str, FACTORS))}."
        )
    if box is not None and factor != 1:
        raise FileRefusal(422, "invalid", "A box is read at full size, factor 1.")
    target = resolve(root, path)
    if kind(target) != "image":
        raise FileRefusal(422, "invalid", f"{path} is not an image.")
    if not is_fluorescence_image(target):
        if z not in (None, 0, "max", "all") or channels:
            raise FileRefusal(
                422,
                "invalid",
                "Only a fluorescence stack (.ome.tiff) has planes and channels to pick.",
            )
        data, full = _single(target, factor, box)
        return _header(data, factor, ["Image"], full, box), data

    from fibsem.fm.stack_file import StackFile

    with StackFile(target) as stack:
        looks = [
            channel_look(list(stack.metadata.channels), i, stack.channels)
            for i in range(stack.channels)
        ]
        names = [name for name, _ in looks]
        unknown = [c for c in channels or [] if c not in names]
        if unknown:
            raise FileRefusal(
                422,
                "invalid",
                f"No channel {', '.join(unknown)} in the stack: it has {', '.join(names)}.",
            )
        picked = [i for i, n in enumerate(names) if not channels or n in channels]
        full = tuple(int(n) for n in stack.shape[-2:])
        region = _checked_box(box, full)
        data = _stack_planes(stack, factor, z, first, last, picked, region)
    return _header(data, factor, [names[i] for i in picked], full, box), data


def _header(
    data: np.ndarray, factor: int, names: List[str], full, box=None
) -> Dict[str, Any]:
    header = {
        "shape": [int(n) for n in data.shape],
        "dtype": data.dtype.name,
        "factor": factor,
        "channels": names,
        "full_shape": [int(n) for n in full],
    }
    if box is not None:
        header["box"] = [int(n) for n in box]
    return header


Region = Tuple[int, int, int, int]  # (top, left, height, width)


def _checked_box(
    box: Optional[Sequence[int]], full: Tuple[int, int]
) -> Optional[Region]:
    """*box* (x, y, width, height) as a region of a (Y, X) *full* image, or the
    422 saying why it isn't one."""
    if box is None:
        return None
    if len(box) != 4:
        raise FileRefusal(422, "invalid", "box is x,y,width,height in pixels.")
    x, y, w, h = (int(n) for n in box)
    if w < 1 or h < 1 or x < 0 or y < 0 or x + w > full[1] or y + h > full[0]:
        raise FileRefusal(
            422,
            "invalid",
            f"box {x},{y},{w},{h} is not inside the {full[1]} x {full[0]} image.",
        )
    if w * h > BOX_PIXELS:
        raise FileRefusal(
            422, "invalid", f"box is at most {BOX_PIXELS} pixels; ask for less."
        )
    return y, x, h, w


def _plain_dtype(data: np.ndarray) -> np.ndarray:
    """*data* in a type the browser reads: uint8, uint16 or float32, little-endian."""
    if data.dtype in (np.uint8, np.uint16, np.float32):
        return data.astype(data.dtype.newbyteorder("<"), copy=False)
    if data.dtype == np.bool_:
        return data.astype(np.uint8) * 255
    if np.issubdtype(data.dtype, np.integer) and data.min() >= 0 and data.max() < 2**16:
        return data.astype("<u2")
    return data.astype("<f4")


def _single(
    path: str, factor: int, box: Optional[Sequence[int]] = None
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """A beam image's (or a plain picture's) grey values as (1, 1, Y, X) at
    *factor*, from the file's own copy when it has one, and its full (Y, X);
    only the region *box* (x, y, width, height) of it when given."""
    data = None
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile

        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            full = tuple(int(n) for n in series.shape[-2:])
            region = _checked_box(box, full)
            if region is not None and len(series.shape) == 2:
                from fibsem.image_format import read_region

                data = read_region(series.levels[0].pages[0], *region)
                return _plain_dtype(data)[None, None], full
            for level in series.levels[1:] if region is None else ():
                if full[-1] // level.shape[-1] == factor:
                    data = level.asarray()
            if data is None:
                data = series.levels[0].asarray()
    else:
        from PIL import Image

        with Image.open(path) as im:
            data = np.asarray(im.convert("L"))
        full = tuple(data.shape[-2:])
    while data.ndim > 2:  # a stack saved as a beam image: its projection
        data = data.max(axis=0)
    region = _checked_box(box, full)
    if region is not None:
        top, left, height, width = region
        data = data[top : top + height, left : left + width]
    elif data.shape[-1] == full[-1] and factor > 1:
        data = _reduced(data, factor)
    return _plain_dtype(data)[None, None], full


def _reduced(data: np.ndarray, factor: int) -> np.ndarray:
    from fibsem.imaging.reduce import reduce_by

    return reduce_by(data, factor)


def _stack_planes(
    stack,
    factor: int,
    z: Optional[Plane],
    first: Optional[int],
    last: Optional[int],
    picked: List[int],
    region: Optional[Region] = None,
) -> np.ndarray:
    planes = stack.planes
    if isinstance(z, int):
        if not 0 <= z < planes:
            raise FileRefusal(
                422, "invalid", f"Plane {z} is outside the stack's 0 to {planes - 1}."
            )
        if region is not None:
            return _plain_dtype(
                np.stack([stack.region(c, z, *region) for c in picked])[:, None]
            )
        return _plain_dtype(
            np.stack([_plane(stack, c, z, factor) for c in picked])[:, None]
        )
    if z == "all" and region is not None:
        raise FileRefusal(
            422, "invalid", "A box is read of a plane or a projection, not all planes."
        )
    if z == "all":
        if factor == 1 and stack.channels * planes * stack.plane(0, 0).nbytes > (
            COPIES_BUDGET
        ):
            raise FileRefusal(
                422,
                "invalid",
                "The stack is too large to send whole at full size: ask for a "
                "plane, or its planes at factor 4 or 8.",
            )
        return _plain_dtype(_copy(stack, factor)[picked])
    mode = z or "max"
    if mode not in PROJECTIONS:
        raise FileRefusal(
            422,
            "invalid",
            "z is a plane number, all, or a projection: max, mean or min.",
        )
    lo = 0 if first is None else first
    hi = planes - 1 if last is None else last
    if not 0 <= lo <= hi < planes:
        raise FileRefusal(
            422,
            "invalid",
            f"Planes {lo} to {hi} are not in the stack's 0 to {planes - 1}.",
        )
    out = [_project(stack, c, factor, mode, lo, hi, region) for c in picked]
    return _plain_dtype(np.stack(out)[:, None])


def _project(
    stack,
    channel: int,
    factor: int,
    mode: str,
    lo: int,
    hi: int,
    region: Optional[Region] = None,
):
    """One channel's projection over planes *lo* to *hi*, at *factor*, of
    *region* when given."""
    level = 0 if factor == 1 else _file_level(stack, factor)
    whole_max = mode == "max" and (lo, hi) == (0, stack.planes - 1)
    if whole_max and level is not None and region is None:
        from fibsem.fm.preview import stack_summary

        # the whole stack's max: the pass the previews make, kept for both
        return stack_summary(stack, level)[0][channel]
    if factor > 1:
        planes = _copy(stack, factor)[channel, lo : hi + 1]
        if mode == "max":
            return planes.max(axis=0)
        if mode == "min":
            return planes.min(axis=0)
        return planes.mean(axis=0, dtype=np.float32)
    # full size: a plane at a time, never the whole stack in memory
    out = None
    for z in range(lo, hi + 1):
        if region is None:
            plane = stack.plane(channel, z)
        else:
            plane = stack.region(channel, z, *region)
        if out is None:
            out = plane.astype(np.float32) if mode == "mean" else plane.copy()
        elif mode == "max":
            np.maximum(out, plane, out=out)
        elif mode == "min":
            np.minimum(out, plane, out=out)
        else:
            out += plane
    if mode == "mean":
        out /= hi - lo + 1
    return out


def _file_level(stack, factor: int) -> Optional[int]:
    """The level of *stack* that holds its *factor* copies, if the file has them."""
    full = stack.shape[-1]
    for level in range(1, stack.levels):
        if full // stack.level_shape(level)[-1] == factor:
            return level
    return None


def _plane(stack, channel: int, z: int, factor: int) -> np.ndarray:
    if factor == 1:
        return stack.plane(channel, z)
    level = _file_level(stack, factor)
    if level is not None:
        return stack.plane(channel, z, level)
    return _copy(stack, factor)[channel, z]


def _copy(stack, factor: int) -> np.ndarray:
    """Every plane of *stack* at *factor*, (C, Z, y, x): from the file's own copies
    when it has them, else reduced from the full planes on first open."""
    if factor == 1:
        return np.stack(
            [
                np.stack([stack.plane(c, z) for z in range(stack.planes)])
                for c in range(stack.channels)
            ]
        )
    level = _file_level(stack, factor)

    def make() -> np.ndarray:
        planes = [
            [
                stack.plane(c, z, level)
                if level is not None
                else _reduced(stack.plane(c, z), factor)
                for z in range(stack.planes)
            ]
            for c in range(stack.channels)
        ]
        return np.asarray(planes)

    return _cached(stack.path, factor, make)


def _cached(path: str, factor: int, make) -> np.ndarray:
    """*make()*, kept by file and factor until the copies kept pass the budget."""
    stat = os.stat(path)
    key = (os.path.realpath(path), stat.st_size, stat.st_mtime_ns, factor)
    with _COPIES_LOCK:
        if key in _COPIES:
            _COPIES.move_to_end(key)
            return _COPIES[key]
    data = make()
    with _COPIES_LOCK:
        _COPIES[key] = data
        while len(_COPIES) > 1 and sum(a.nbytes for a in _COPIES.values()) > (
            COPIES_BUDGET
        ):
            _COPIES.popitem(last=False)
    return data
