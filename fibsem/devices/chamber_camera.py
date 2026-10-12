"""The ChamberCamera device: the camera that looks into the chamber.

It has no parameters; its one command, ``acquire``, returns a frame as the camera gives
it. ``FibsemMicroscope.acquire_chamber_image`` wraps that frame in a ``FibsemImage``.

A backend implements ``_acquire``. On Thermo the chamber camera is a third device on
the channel the beams and the FM share, so the backend holds that channel for the
frame and puts it back afterwards; the base class claims nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fibsem.devices.core import Device, command


class ChamberCamera(Device):
    def __init__(self, parent: Any = None, **kwargs: Any):
        super().__init__(name="chamber_camera", parent=parent, **kwargs)

    @command
    def acquire(self) -> np.ndarray:
        """One frame of the chamber, as the camera gives it."""
        return self._acquire()

    # -- what a backend implements -------------------------------------------------

    def _acquire(self) -> np.ndarray:
        raise NotImplementedError
