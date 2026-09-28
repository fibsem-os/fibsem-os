"""The Manipulator device: the needle.

Like the stage, it is the raw hardware contract only. Two read-only parameters
describe it, ``position`` and ``inserted``, and the moves are commands:
``insert``, ``retract``, ``move_absolute`` and ``move_relative``. The corrected
move (``move_manipulator_corrected``) depends on the beam as much as the needle,
so it is not here; it belongs to the views, as the stage's corrected moves do.

Named positions (``PARK``, ``EUCENTRIC``) are the instrument's, so the driver
answers ``saved_position(name)``, and ``move_to_offset`` moves relative to one.

A backend implements the parameters with ``read_position``/``read_inserted`` and
the commands with four hooks: ``_insert``, ``_retract``, ``_move_absolute`` and
``_move_relative``, plus ``saved_position``. The base class claims the
``manipulator`` resource and reads the position and state back, which updates
their caches and emits their change signals.
"""

from __future__ import annotations

from typing import Any

from fibsem.devices.core import Device, Parameter, command
from fibsem.structures import FibsemManipulatorPosition

MANIPULATOR_RESOURCE = "manipulator"


class Manipulator(Device):
    position = Parameter(FibsemManipulatorPosition, doc="Raw needle coordinates.")
    inserted = Parameter(bool, doc="The needle is inserted, not retracted.")

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

    def _read_back(self) -> FibsemManipulatorPosition:
        self.inserted.get_value()
        return self.position.get_value()

    # -- what a backend implements -------------------------------------------------

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
