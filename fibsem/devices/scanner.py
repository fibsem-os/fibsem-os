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
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from fibsem.devices.core import Device, command

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
        with self.resources.claim(SCANNER_RESOURCE):
            return self._acquire((width, height), float(dwell_time))

    @command
    def stop(self) -> None:
        """Stop the scan: a frame in flight ends early. Safe when not scanning."""
        self._stop()

    def _acquire(self, resolution: Tuple[int, int], dwell_time: float) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} can't acquire")

    def _stop(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} can't stop")
