"""The Scanner: what drives a beam's scan and returns the frame it scanned.

A beam images through the vendor's own scan unless its ``scanner`` role is filled
(``Beam.scanner``), as it is when the configuration binds it to an external scan
generator::

    hardware:
      devices:
        - name: scan_generator
          type: scan_generator
          driver: demo
        - name: electron
          roles: {scanner: scan_generator}

The interface is deliberately minimal until a real unit's SDK says more: acquire one
frame with a resolution and a dwell time, and stop. The beam keeps the imaging
settings (``resolution``, ``dwell_time``, ``hfw``) and builds the ``FibsemImage``;
the scanner returns the raw frame.

A driver implements ``_acquire`` and ``_stop``. ``acquire`` claims the scanner's
resource for the frame; ``stop`` does not, since it must reach a frame in flight.

A driver whose unit has limits says so with ``_dwell_time_limits`` and
``_resolution_limits``: ``acquire`` refuses a frame outside them, and a beam the
scanner is bound to offers only the dwell times and resolutions both can do. The
limits are also its ``facts``, which is how a remote copy (``RemoteScanner``) knows them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from fibsem.devices.core import Device, command
from fibsem.structures import RangeLimit

Resolution = Tuple[int, int]

SCANNER_RESOURCE = "scanner"


class Scanner(Device):
    @command
    def acquire(self, resolution: Tuple[int, int], dwell_time: float) -> np.ndarray:
        """Scan one frame of *resolution* (width, height) pixels with *dwell_time*
        seconds per pixel, and return it as a (height, width) array."""
        width, height = (int(n) for n in resolution)
        if width <= 0 or height <= 0:
            raise ValueError(
                f"{self.name}: resolution must be positive, not {resolution}"
            )
        if dwell_time <= 0:
            raise ValueError(
                f"{self.name}: dwell time must be positive, not {dwell_time}"
            )
        dwell_limits = self.dwell_time_limits
        if dwell_limits is not None and not (
            dwell_limits.min <= dwell_time <= dwell_limits.max
        ):
            raise ValueError(
                f"{self.name}: dwell time {dwell_time} s is outside "
                f"{dwell_limits.min}-{dwell_limits.max} s"
            )
        resolution_limits = self.resolution_limits
        if resolution_limits is not None and not resolution_within(
            (width, height), resolution_limits
        ):
            low, high = resolution_limits
            raise ValueError(
                f"{self.name}: resolution {width}x{height} is outside "
                f"{low[0]}x{low[1]}-{high[0]}x{high[1]}"
            )
        with self.resources.claim(SCANNER_RESOURCE):
            return self._acquire((width, height), float(dwell_time))

    @command
    def stop(self) -> None:
        """Stop the scan: a frame in flight ends early. Safe when not scanning."""
        self._stop()

    @property
    def dwell_time_limits(self) -> Optional[RangeLimit]:
        """The dwell times (s) the unit can scan with, or None for no limit of its own."""
        return self._dwell_time_limits()

    @property
    def resolution_limits(self) -> Optional[Tuple[Resolution, Resolution]]:
        """The smallest and largest (width, height) frames the unit can scan, or None
        for no limit of its own."""
        return self._resolution_limits()

    def facts(self) -> Dict[str, Any]:
        """Its limits, so a remote copy refuses and narrows a beam the same way. A
        driver adding facts of its own keeps these: ``{**super().facts(), ...}``."""
        dwell, frames = self.dwell_time_limits, self.resolution_limits
        return {
            "dwell_time_limits": None if dwell is None else [dwell.min, dwell.max],
            "resolution_limits": None if frames is None else [list(f) for f in frames],
        }

    def _dwell_time_limits(self) -> Optional[RangeLimit]:
        return None

    def _resolution_limits(self) -> Optional[Tuple[Resolution, Resolution]]:
        return None

    def _acquire(self, resolution: Tuple[int, int], dwell_time: float) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} can't acquire")

    def _stop(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} can't stop")


def resolution_within(
    resolution: Resolution, limits: Tuple[Resolution, Resolution]
) -> bool:
    """Whether *resolution* is no smaller than ``limits[0]`` and no larger than
    ``limits[1]`` on each side."""
    (low_w, low_h), (high_w, high_h) = limits
    width, height = resolution
    return low_w <= width <= high_w and low_h <= height <= high_h
