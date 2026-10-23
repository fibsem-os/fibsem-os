"""A fluorescence stack on disk, read one plane at a time.

`FluorescenceImage.load` reads the whole stack, which on a real acquisition is
gigabytes. Showing one plane, or the projection, needs a plane at a time: this
opens the file once and reads only the pages asked for. A page is one (Y, X)
plane of one channel; which page holds channel c, plane z comes from the axes
tifffile reads out of the file, arranged by `to_czyx` as `load` arranges the data.
A file whose pages are not single planes (an RGB file) is read whole, once.
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

    def plane(self, channel: int, z: int) -> np.ndarray:
        """Plane *z* of *channel*, (Y, X): one page read, not the stack."""
        if self._whole is not None:
            return self._whole[channel, z]
        return self._tif.asarray(key=int(self._pages[channel, z]), series=0)

    def close(self) -> None:
        self._tif.close()

    def __enter__(self) -> "StackFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
