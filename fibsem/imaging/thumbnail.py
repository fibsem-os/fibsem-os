"""Display-sized copies of acquired frames, written so a reader never sees a partial one.

One writer for every thumbnail the app keeps beside a frame: a lamella's, a grid
overview's. Written at display size rather than at the acquired resolution -- this
is a thumbnail, not a copy of the frame, which is saved separately by whatever
acquired it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union

import numpy as np

THUMBNAIL_MAX_EDGE = 512


def write_thumbnail(
    data: np.ndarray,
    destination: Union[str, Path],
    max_edge: int = THUMBNAIL_MAX_EDGE,
) -> str:
    """Write `data` (2-D grey or (H, W, 3) RGB) as a PNG no larger than `max_edge`.

    Staged in the destination's directory and moved into place: `os.replace` is
    atomic on POSIX and on Windows provided both are on one filesystem, so a
    reader on another thread sees either the previous thumbnail or the new one,
    never a partial. Never upscales -- a frame smaller than the bound keeps its
    size. Returns the path written.
    """
    from PIL import Image

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    if data.ndim == 2:
        data = np.stack([data, data, data], axis=2)
    handle, staged = tempfile.mkstemp(
        dir=str(destination.parent), prefix=".thumbnail-", suffix=".png"
    )
    os.close(handle)
    try:
        thumbnail = Image.fromarray(data.astype(np.uint8))
        thumbnail.thumbnail((max_edge, max_edge), Image.LANCZOS)
        thumbnail.save(staged)
        os.replace(staged, str(destination))
    except BaseException:
        # Including cancellation: a staged file left behind would accumulate, and
        # it is hidden, so nobody would notice it doing so.
        try:
            os.remove(staged)
        except OSError:
            pass
        raise
    return str(destination)
