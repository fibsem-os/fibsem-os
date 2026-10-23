"""A fluorescence stack on disk, read one plane at a time.

`FluorescenceImage.load` reads the whole stack, which on a real acquisition is
gigabytes. Showing one plane, or the projection, needs a plane at a time: this
opens the file once and reads only the pages asked for. A page is one (Y, X)
plane of one channel; which page holds channel c, plane z comes from the axes
tifffile reads out of the file, arranged by `to_czyx` as `load` arranges the data.
A file whose pages are not single planes (an RGB file) is read whole, once. A file
saved by 0.60 also holds each plane's small copies, read by level.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from fibsem.fm.structures import (
    FluorescenceImage,
    FluorescenceImageMetadata,
    open_tiff,
    to_czyx,
)


class StackFile:
    """The stack at *path*: its (C, Z, Y, X) shape, its metadata, and its
    planes on demand. A context manager; `close` releases the file."""

    def __init__(self, path: str):
        self.path = str(path)
        self._tif = open_tiff(self.path)
        series = self._tif.series[0]
        axes = series.axes.upper()
        self._whole: Optional[np.ndarray] = None
        self._pages: Optional[np.ndarray] = None
        self._levels = series.levels
        if axes.endswith("YX") and len(series.shape) > 2:
            lead = tuple(series.shape[:-2])
            index = np.arange(int(np.prod(lead))).reshape(lead + (1, 1))
            pages = to_czyx(index, axes)
            if pages.ndim == 4:  # one time point
                self._pages = pages[:, :, 0, 0]
                self.shape: Tuple[int, int, int, int] = (
                    *self._pages.shape,
                    *series.shape[-2:],
                )
        if self._pages is None:
            self._whole = to_czyx(series.asarray(), axes)
            if self._whole.ndim == 5:
                self._whole = self._whole[0]
            self.shape = tuple(self._whole.shape)
        self.metadata: FluorescenceImageMetadata = FluorescenceImage.read_metadata(
            self.path, self.shape
        )

    @property
    def channels(self) -> int:
        return self.shape[0]

    @property
    def planes(self) -> int:
        return self.shape[1]

    @property
    def levels(self) -> int:
        """How many sizes each plane is stored at: 1, or with the small copies 0.60
        writes (`fibsem.image_format`), 1 + their number, level 0 the full size."""
        return 1 if self._whole is not None else len(self._levels)

    def level_for(self, max_px: Optional[int]) -> int:
        """The smallest level whose long side is still at least *max_px* (0, the
        full size, when None)."""
        level = 0
        if max_px is not None:
            for i in range(1, self.levels):
                if max(self._levels[i].shape[-2:]) < max_px:
                    break
                level = i
        return level

    def plane(self, channel: int, z: int, level: int = 0) -> np.ndarray:
        """Plane *z* of *channel*, (Y, X), at *level*: one page read, not the stack."""
        if self._whole is not None:
            return self._whole[channel, z]
        page = int(self._pages[channel, z])
        if level:
            return self._levels[level].pages[page].asarray()
        return self._tif.asarray(key=page, series=0)

    def close(self) -> None:
        self._tif.close()

    def __enter__(self) -> "StackFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
