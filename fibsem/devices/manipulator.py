"""The Manipulator device: the needle.

Like the stage, it is the raw hardware contract only. Two read-only parameters
describe it, ``position`` and ``state``, and the moves are commands:
``insert``, ``retract``, ``move_absolute``, ``move_relative`` and ``stop``.
``move_corrected`` moves the needle in a beam's image coordinates; where the needle's
axes follow the stage tilt (Thermo), the driver reads the tilt from the device in the
``stage`` role, which the microscope fills with its stage.

Named positions are the instrument's, so the driver answers ``named_positions()``
and ``saved_position(name)``, and ``move_to_offset`` moves relative to one.

Which axes the arm has is the driver's to say, the same way as for the stage: they
are the keys of the per-axis limits in ``position``'s metadata, and
``needle.axes.r`` is a view of one of them. A move outside the limits is refused
with ``ManipulatorLimitError``, never clipped. A driver that gives no limits has x,
y and z, unbounded.

A backend implements the parameters with ``read_position``/``read_state`` and
the commands with four hooks: ``_insert``, ``_retract``, ``_move_absolute`` and
``_move_relative``, plus ``saved_position``. The base class claims the
``manipulator`` resource and reads the position and state back, which updates
their caches and emits their change signals. A backend that can stop a move
implements ``_stop``, and one that can make a corrected move implements
``_move_corrected``, and one whose arm has other axes or bounds implements
``metadata_position``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fibsem.devices.core import (
    UNLIMITED,
    Axes,
    AxisLimitError,
    Device,
    DeviceParameter,
    ParameterMetadata,
    Role,
    axes_outside_limits,
    build_axes,
    command,
)
from fibsem.devices.stage import Stage
from fibsem.structures import (
    BeamType,
    FibsemManipulatorPosition,
    InsertableDeviceState,
)

MANIPULATOR_RESOURCE = "manipulator"


class ManipulatorLimitError(AxisLimitError):
    """A needle move whose target is outside an axis's limits. Nothing moved."""


class Manipulator(Device):
    position = DeviceParameter(FibsemManipulatorPosition, doc="Raw needle coordinates.")
    state = DeviceParameter(InsertableDeviceState, doc="Where the needle is.")
    stage = Role(
        Stage,
        required=False,
        doc="The stage whose tilt a corrected move follows, where the driver needs it.",
    )

    def __init__(self, parent: Any = None, **kwargs: Any):
        super().__init__(name="manipulator", parent=parent, **kwargs)
        self.axes = Axes({})

    def connect(self) -> Manipulator:
        """Bind the parameters, then build an axis for each one the driver reports."""
        super().connect()
        self._build_axes()
        return self

    def _build_axes(self) -> None:
        self.axes = build_axes(self) if "position" in self.parameters else Axes({})

    def check_limits(self, position: FibsemManipulatorPosition) -> None:
        """Raise ``ManipulatorLimitError`` naming every axis of *position* out of
        limits."""
        outside = axes_outside_limits(self.axes, position)
        if outside:
            raise ManipulatorLimitError(
                f"{self.name}: target outside the needle limits: {', '.join(outside)}"
            )

    @command
    def insert(self, name: Optional[str] = None) -> FibsemManipulatorPosition:
        """Insert the needle to a named position, or to the instrument's own insert
        position when no name is given. Returns where it is afterwards."""
        if name is None:
            name = self.default_insert_position()
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
        self.check_limits(position)
        with self.resources.claim(MANIPULATOR_RESOURCE):
            self._move_absolute(position)
            return self.position.get_value()

    @command
    def move_relative(
        self, delta: FibsemManipulatorPosition
    ) -> FibsemManipulatorPosition:
        """Move by an offset. Returns where it is afterwards."""
        with self.resources.claim(MANIPULATOR_RESOURCE):
            self.check_limits(self.position.get_value() + delta)
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
        """The arm's named positions with where each one is now. Its axes travel
        as the limits of ``position``."""
        saved = {}
        for name in self.named_positions():
            try:
                saved[name] = self.saved_position(name).to_dict()
            except Exception as error:  # one unreadable name hides only itself
                logging.warning(f"{self.name}: saved position {name}: {error}")
        return {
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

    def default_insert_position(self) -> Optional[str]:
        """The named position ``insert`` goes to when given none: the first of
        `named_positions`, or None for an arm with no named positions."""
        names = self.named_positions()
        return names[0] if names else None

    def metadata_position(self) -> ParameterMetadata:
        """The arm's axes and their limits: x, y and z, unbounded, unless the driver
        says otherwise. Add ``r`` (rotation) or ``t`` (tilt) for an arm that has
        them."""
        return ParameterMetadata(limits={axis: UNLIMITED for axis in ("x", "y", "z")})

    def saved_position(self, name: str) -> FibsemManipulatorPosition:
        """The instrument's named position. Raises ValueError for an unknown name."""
        raise NotImplementedError

    def _insert(self, name: Optional[str]) -> None:
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
