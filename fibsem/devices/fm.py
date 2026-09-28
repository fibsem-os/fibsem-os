"""The fluorescence microscope's parts as devices, and the FM that groups them.

Each part is its own device with its own parameters: the camera, the light source,
the filter set and the objective. ``FM`` holds no parameters of its own; it is where
sequences that use several parts live, such as acquiring one channel (filter, light,
exposure, camera). A driver runs those sequences next to the hardware, which matters
when the FM is on another computer: one call over the network, not one per step, and
a dropped connection can't leave the light on.

Commands call a hook the driver implements (``_acquire``, ``_insert``, ...), the way
the stage's moves do, so the command's name, arguments and checks are the same on
every driver.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from fibsem.devices.core import Device, Parameter, command

OBJECTIVE_STATES = ("Inserted", "Retracted", "Busy", "Error", "Other")


class Camera(Device):
    exposure_time = Parameter(float, unit="s")
    binning = Parameter(int)
    gain = Parameter(float)
    offset = Parameter(float)

    @command
    def acquire(self) -> np.ndarray:
        """One frame with the current settings, as the camera gives it."""
        return self._acquire()

    def _acquire(self) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} can't acquire")


class LightSource(Device):
    power = Parameter(float, doc="A fraction of the source's maximum, 0 to 1.")


class FilterSet(Device):
    excitation_wavelength = Parameter(float, unit="nm")
    emission_wavelength = Parameter(
        object,
        doc="A wavelength in nm, a filter's name, or None for reflection. Mixed, as "
        "the FM classes have it; one type would need a decision.",
    )


class Objective(Device):
    """Where the objective is and what it's doing. It moves only through commands."""

    position = Parameter(float, unit="m")
    state = Parameter(str, choices=OBJECTIVE_STATES)
    magnification = Parameter(float)
    numerical_aperture = Parameter(float)
    limit_position = Parameter(float, unit="m", doc="The furthest a move may go in.")

    @command
    def insert(self) -> None:
        """Move the objective in, to its imaging position."""
        self._insert()

    @command
    def retract(self) -> None:
        """Move the objective out, clear of the sample."""
        self._retract()

    @command
    def move_absolute(self, position: float) -> None:
        """Move the objective to a position, in metres."""
        self._move_absolute(position)

    @command
    def move_relative(self, delta: float) -> None:
        """Move the objective by a distance, in metres (positive is towards the sample)."""
        self._move_relative(delta)

    def _insert(self) -> None:
        raise NotImplementedError

    def _retract(self) -> None:
        raise NotImplementedError

    def _move_absolute(self, position: float) -> None:
        raise NotImplementedError

    def _move_relative(self, delta: float) -> None:
        raise NotImplementedError


class FM(Device):
    """The group: sequences over several parts, run by the driver next to them."""

    @command
    def acquire_channel(self, channel: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Set up one channel (``ChannelSettings.to_dict()``), then acquire a frame.

        Without a channel, acquires with the current settings.
        """
        return self._acquire_channel(channel)

    def _acquire_channel(self, channel: Optional[Dict[str, Any]]) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} can't acquire")
