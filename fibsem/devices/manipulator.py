"""The Manipulator device: the needle.

Like the stage, it is the raw hardware contract only. Two read-only parameters
describe it, ``position`` and ``state``, and the moves are commands:
``insert``, ``retract``, ``move_absolute``, ``move_relative`` and ``stop``.
``move_corrected`` moves the needle in a beam's image coordinates; where the needle's
axes follow the stage tilt (Thermo), the driver reads the tilt from the device in the
``stage`` role, which the microscope fills with its stage.

Named positions (``PARK``, ``EUCENTRIC``) are the instrument's, so the driver
answers ``named_positions()`` and ``saved_position(name)``, and
``move_to_offset`` moves relative to one. ``axes()`` says which axes the arm has, so
whether it rotates or tilts is the device's to say, not the backend's.

A backend implements the parameters with ``read_position``/``read_state`` and
the commands with four hooks: ``_insert``, ``_retract``, ``_move_absolute`` and
``_move_relative``, plus ``saved_position``. The base class claims the
``manipulator`` resource and reads the position and state back, which updates
their caches and emits their change signals. A backend that can stop a move
implements ``_stop``, and one that can make a corrected move implements
``_move_corrected``; ``axes()`` defaults to x, y and z.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from fibsem.devices.core import Device, Parameter, Role, command
from fibsem.devices.stage import Stage
from fibsem.structures import (
    BeamType,
    FibsemManipulatorPosition,
    InsertableDeviceState,
)

MANIPULATOR_RESOURCE = "manipulator"


class Manipulator(Device):
    position = Parameter(FibsemManipulatorPosition, doc="Raw needle coordinates.")
    state = Parameter(InsertableDeviceState, doc="Where the needle is.")
    stage = Role(
        Stage,
        required=False,
        doc="The stage whose tilt a corrected move follows, where the driver needs it.",
    )

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

    @command(available=lambda needle: _implements(needle, "_move_corrected"))
    def move_corrected(
        self, dx: float, dy: float, beam_type: BeamType
    ) -> FibsemManipulatorPosition:
        """Move by (dx, dy) in the image coordinates of *beam_type*'s view. Returns
        where it is afterwards."""
        with self.resources.claim(MANIPULATOR_RESOURCE):
            self._move_corrected(dx, dy, beam_type)
            return self.position.get_value()

    @command
    def move_to_offset(
        self, offset: FibsemManipulatorPosition, name: str
    ) -> FibsemManipulatorPosition:
        """Move to a named position plus an offset. Returns where it is afterwards."""
        return self.move_absolute(self._offset_target(offset, name))

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
            "is_installed": self._installed_or_none(),
        }

    def _installed_or_none(self) -> Optional[bool]:
        # A raising answer is "cannot say": a missing arm and a sick connection
        # look the same.
        try:
            return self.is_installed()
        except Exception as error:
            logging.debug(f"{self.name}: is_installed: {error}")
            return None

    # -- what a backend implements -------------------------------------------------

    def is_installed(self) -> Optional[bool]:
        """Whether the instrument reports this arm fitted, or None if it cannot say.

        A backend builds its own manipulator and then asks this; one the instrument
        says is not fitted is taken out of the device map
        (`FibsemMicroscope._keep_if_fitted`).
        """
        return None

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

    def _offset_target(
        self, offset: FibsemManipulatorPosition, name: str
    ) -> FibsemManipulatorPosition:
        """Where ``move_to_offset`` goes: the named position plus the offset. A driver
        whose axes follow the stage tilt corrects the offset first."""
        return self.saved_position(name) + offset

    def _move_corrected(self, dx: float, dy: float, beam_type: BeamType) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} cannot make a corrected move."
        )


def _implements(needle: Manipulator, hook: str) -> bool:
    """Whether the needle's driver overrides *hook*."""
    return getattr(type(needle), hook) is not getattr(Manipulator, hook)
