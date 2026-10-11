"""The Manipulator device: the needle.

Like the stage, it is the raw hardware contract only. Two read-only parameters
describe it, ``position`` and ``state``, and the moves are commands:
``insert``, ``retract``, ``move_absolute``, ``move_relative`` and ``stop``. The corrected
move (``move_manipulator_corrected``) depends on the beam as much as the needle,
so it is not here; it belongs to the views, as the stage's corrected moves do.

Named positions (``PARK``, ``EUCENTRIC``) are the instrument's, so the driver
answers ``named_positions()`` and ``saved_position(name)``, and
``move_to_offset`` moves relative to one. ``axes()`` says which axes the arm has, so
whether it rotates or tilts is the device's to say, not the backend's.

A backend implements the parameters with ``read_position``/``read_state`` and
the commands with four hooks: ``_insert``, ``_retract``, ``_move_absolute`` and
``_move_relative``, plus ``saved_position``. The base class claims the
``manipulator`` resource and reads the position and state back, which updates
their caches and emits their change signals. A backend that can stop a move
implements ``_stop``; ``axes()`` defaults to x, y and z.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from fibsem.devices.core import Device, Parameter, command
from fibsem.structures import FibsemManipulatorPosition, InsertableDeviceState

MANIPULATOR_RESOURCE = "manipulator"

# Today's manipulator keys, for a backend that routes them to its manipulator
# device. Both are reads; the old API moves the needle with methods, not keys.
MANIPULATOR_ROUTES: Dict[str, str] = {
    "manipulator_position": "position",
    "manipulator_state": "state",
}


class Manipulator(Device):
    position = Parameter(FibsemManipulatorPosition, doc="Raw needle coordinates.")
    state = Parameter(InsertableDeviceState, doc="Where the needle is.")

    def __init__(self, parent: Any = None, **kwargs: Any):
        super().__init__(name="manipulator", parent=parent, **kwargs)

    @command
    def insert(self, name: str = "PARK") -> FibsemManipulatorPosition:
        """Insert the needle to a named position. Returns where it is afterwards."""
        with self.resources.claim(MANIPULATOR_RESOURCE):
            self._insert(name)
            return self._read_back()

    @command
    def retract(self) -> FibsemManipulatorPosition:
        """Retract the needle. Returns where it is afterwards."""
        with self.resources.claim(MANIPULATOR_RESOURCE):
            self._retract()
            return self._read_back()

    @command
    def move_absolute(
        self, position: FibsemManipulatorPosition
    ) -> FibsemManipulatorPosition:
        """Move to a position. Returns where it is afterwards."""
        with self.resources.claim(MANIPULATOR_RESOURCE):
            self._move_absolute(position)
            return self.position.get_value()

    @command
    def move_relative(
        self, delta: FibsemManipulatorPosition
    ) -> FibsemManipulatorPosition:
        """Move by an offset. Returns where it is afterwards."""
        with self.resources.claim(MANIPULATOR_RESOURCE):
            self._move_relative(delta)
            return self.position.get_value()

    @command
    def move_to_offset(
        self, offset: FibsemManipulatorPosition, name: str
    ) -> FibsemManipulatorPosition:
        """Move to a named position plus an offset. Returns where it is afterwards."""
        return self.move_absolute(self.saved_position(name) + offset)

    @command
    def stop(self) -> FibsemManipulatorPosition:
        """Stop the needle where it is. Returns where it stopped.

        It does not claim the manipulator: the move it stops holds that claim.
        """
        self._stop()
        return self._read_back()

    def _read_back(self) -> FibsemManipulatorPosition:
        self.state.get_value()
        return self.position.get_value()

    def facts(self) -> Dict[str, Any]:
        """The arm's axes, and its named positions with where each one is now."""
        saved = {}
        for name in self.named_positions():
            try:
                saved[name] = self.saved_position(name).to_dict()
            except Exception as error:  # one unreadable name hides only itself
                logging.warning(f"{self.name}: saved position {name}: {error}")
        return {
            "axes": list(self.axes()),
            "named_positions": self.named_positions(),
            "saved_positions": saved,
        }

    # -- what a backend implements -------------------------------------------------

    def named_positions(self) -> List[str]:
        """The names `saved_position` answers, if the instrument has any."""
        return []

    def axes(self) -> Tuple[str, ...]:
        """The axes the arm moves on, of ``x``, ``y``, ``z``, ``r`` (rotation) and
        ``t`` (tilt)."""
        return ("x", "y", "z")

    def saved_position(self, name: str) -> FibsemManipulatorPosition:
        """The instrument's named position. Raises ValueError for an unknown name."""
        raise NotImplementedError

    def _insert(self, name: str) -> None:
        raise NotImplementedError

    def _retract(self) -> None:
        raise NotImplementedError

    def _move_absolute(self, position: FibsemManipulatorPosition) -> None:
        raise NotImplementedError

    def _move_relative(self, delta: FibsemManipulatorPosition) -> None:
        raise NotImplementedError

    def _stop(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} cannot stop the needle.")
