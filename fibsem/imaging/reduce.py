"""Reducing an image to a display size without deleting what is in it.

Anything that shows a large image in a small space has to throw pixels away, and there
are two ways to do it. Sampling — ``arr[::n, ::n]`` — is free, and wrong: it does not
blur what it leaves out, it deletes it. A feature a pixel or two across survives only
when it happens to land on a sampled row, so it is absent at one zoom and present at the
next, and blinks when the view moves by a single pixel, with nothing on screen saying
the picture is incomplete. For fluorescence that is the worst available failure mode,
the small bright thing being what someone is looking for.

Averaging over each block instead attenuates such a feature rather than removing it, and
holds it steady while panning. It costs an allocation where sampling returned a view.
That is a real cost and not a hidden saving: a strided view does keep its parent's whole
buffer alive, but matplotlib copies at ``imshow`` anyway, so on the canvas path the
source was already released either way. Measured with weakrefs over twenty placed tiles,
neither reduction leaves a single source alive.

Kept out of the UI package so it is usable, and testable, without Qt.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

__all__ = ["downsample"]


# Element types `cv2.resize` accepts. Anything else takes the numpy path, which is a few
# hundred times slower but answers for every dtype. Pinned by a test that feeds each one
# to cv2 for real, so the list cannot quietly rot against a new OpenCV.
_CV2_DTYPES = frozenset(
    np.dtype(name) for name in ("uint8", "uint16", "int16", "float32", "float64")
)


def _box_mean_numpy(arr: np.ndarray, factor: int, out_h: int, out_w: int) -> np.ndarray:
    """Block mean in numpy, for the arrays cv2 will not take."""
    pad = [(0, out_h * factor - arr.shape[0]), (0, out_w * factor - arr.shape[1])]
    pad += [(0, 0)] * (arr.ndim - 2)
    # float32 for anything narrower, so an 8-bit mosaic is not accumulated at 8 bytes a
    # pixel; float64 input stays float64. A block holds at most `factor**2` values, well
    # short of what either accumulator can represent exactly.
    acc = np.promote_types(arr.dtype, np.float32)
    padded = np.pad(arr.astype(acc, copy=False), pad, mode="edge")
    out = padded.reshape(out_h, factor, out_w, factor, *arr.shape[2:]).mean(axis=(1, 3))
    if np.issubdtype(arr.dtype, np.integer):
        out = np.rint(out)  # truncating would bias every reduced image darker
    return out.astype(arr.dtype, copy=False)


def downsample(arr: np.ndarray, max_px: int) -> np.ndarray:
    """*arr* reduced until neither side exceeds *max_px*, by averaging over each block.

    Averaged, not sampled — see the module docstring for why that distinction is the
    whole point of this function. Returns *arr* itself when it is already small enough.

    Shape is ``ceil(n / factor)`` per axis, which is what ``arr[::factor, ::factor]``
    gave, so this drops into a sampling call site unchanged. Note that the reduction is
    uniform across both axes, so a non-square image keeps its aspect ratio but may exceed
    *max_px* on neither side and fall well under it on one.

    dtype is preserved, which matters more than it looks: matplotlib reads an RGB array's
    value range from its dtype, ``uint8`` meaning 0-255 and float meaning 0-1, so
    returning a float for a ``uint8`` image renders it white.

    The ragged final block is edge-padded rather than left short, and that is what keeps
    this cheap: ``cv2.resize`` has a fast path for an exactly-integer scale and a generic
    one about fifty times slower, so padding to an exact multiple is what buys the
    former. A 1024x1024 RGBA image reduces to 512 in 0.14 ms; the same in numpy is 72 ms.

    Padding rather than shortening biases one output row and one output column toward
    the edge pixel, since the block is completed with copies of it. On a 1024x4712 plane
    at factor 10 that block is 60% replicated, but it is one row of 103, and the error
    tracks how much the last few rows resemble each other: 0.1/255 on smooth data,
    1.8 on lightly textured, 32 on pure noise, 76 on an alternating hard edge. Accepted
    because an overview's tiles overlap, so that row is covered by the neighbour drawn
    over it -- 10% by default for FM. Note the beam tiler defaults to no overlap at all,
    where nothing covers it; still negligible at these magnitudes, but not for the same
    reason. Switch to :func:`_box_mean_numpy` for the exact short-block answer at ~5x.
    """
    h, w = arr.shape[:2]
    if h <= max_px and w <= max_px:
        return arr
    factor = max(1, math.ceil(max(h, w) / max_px))
    out_h, out_w = math.ceil(h / factor), math.ceil(w / factor)
    too_many_channels = arr.ndim > 3 or (arr.ndim == 3 and arr.shape[2] > 4)
    if arr.dtype not in _CV2_DTYPES or too_many_channels:
        return _box_mean_numpy(arr, factor, out_h, out_w)
    pad_h, pad_w = out_h * factor - h, out_w * factor - w
    if pad_h or pad_w:
        arr = cv2.copyMakeBorder(arr, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    return cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_AREA)
