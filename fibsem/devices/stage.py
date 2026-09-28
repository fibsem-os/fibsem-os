"""The Stage device, the reverse of today's ``Stage``.

Today ``fibsem.microscopes._stage.Stage`` owns nothing: every property and move asks
its parent microscope, which asks its backend's ``get``/``set`` chain. Here the device
owns the hardware contract and the microscope would ask the device.

The stage is the raw hardware contract only: where it is, what it can reach, and
"move to these coordinates" in the fibsem-os frame (FIB-1083). View-corrected moves
(``stable_move``, ``vertical_move``, ``project_stable_move``) depend on the beams as
much as the stage, so they are not here; they belong to the views, built on top.

Parameters describe state and are read-only here: ``position``, one parameter per
axis (``x``, ``y``, ``z``, ``r``, ``t``, carrying that axis's limits as metadata),
``homed`` and ``linked``. Changing where the stage is takes time and can fail, so it
is a command, not an assignment: ``move_absolute``, ``move_relative``, ``home`` and
``link``. A move outside the axis limits is refused, never clipped, for the same
reason ``raise_if_outside_stage_limits`` refuses a tile grid: a stage that stopped
somewhere else than asked is a position everything downstream takes for the truth.

A backend implements the parameters with ``read_<name>``/``metadata_<name>`` methods,
as for a beam, and the commands with four hooks: ``_move_absolute``,
``_move_relative``, ``_home`` and ``_link``. The base class does the rest once: the
limit check, the ``stage`` resource, the read-back, and the change signals.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from fibsem.devices.core import BoundParameter, Device, Parameter, command
from fibsem.structures import FibsemStagePosition

STAGE_RESOURCE = "stage"
AXES: Tuple[str, ...] = ("x", "y", "z", "r", "t")


class StageLimitError(ValueError):
    """A move whose target is outside an axis's limits. Nothing moved."""


class Stage(Device):
    position = Parameter(
        FibsemStagePosition, doc="Raw stage coordinates, in the fibsem-os frame."
    )
    x = Parameter(float, unit="m")
    y = Parameter(float, unit="m")
    z = Parameter(float, unit="m")
    r = Parameter(float, unit="rad")
    t = Parameter(float, unit="rad")
    homed = Parameter(bool)
    linked = Parameter(bool, doc="z linked to the working distance.")

    def __init__(self, parent: Any = None, **kwargs: Any):
        super().__init__(name="stage", parent=parent, **kwargs)
        # An axis follows every position read, so a UI watching one axis stays
        # current without another instrument call.
        self.changed.connect(self._position_read)

    # -- description --------------------------------------------------------------

    @property
    def axes(self) -> Tuple[str, ...]:
        """The axes this stage has. A compustage has no rotation axis."""
        return tuple(axis for axis in AXES if axis in self.parameters)

    @property
    def limits(self) -> Dict[str, Tuple[float, float]]:
        """Each axis's limits, in SI units, from the metadata cached at connect."""
        return {
            axis: self.parameters[axis].limits
            for axis in self.axes
            if self.parameters[axis].limits is not None
        }

    # -- commands -----------------------------------------------------------------

    @command
    def move_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """Move to a position. Axes that are None stay where they are."""
        self.check_limits(position)
        return self.move_through(position)

    @command
    def move_relative(self, delta: FibsemStagePosition) -> FibsemStagePosition:
        """Move by an offset. Axes that are None don't move."""
        with self.resources.claim(STAGE_RESOURCE):
            self.check_limits(self.position.get_value() + delta)
            return self.move_through(delta, relative=True)

    @command(available=lambda stage: "homed" in stage.parameters)
    def home(self) -> bool:
        """Home the stage. Returns whether it is homed afterwards."""
        with self.resources.claim(STAGE_RESOURCE):
            self._home()
            return self.homed.get_value()

    @command(available=lambda stage: "linked" in stage.parameters)
    def link(self) -> bool:
        """Link z to the working distance. Returns whether it is linked afterwards."""
        with self.resources.claim(STAGE_RESOURCE):
            self._link()
            return self.linked.get_value()

    def move_through(
        self, position: FibsemStagePosition, relative: bool = False
    ) -> FibsemStagePosition:
        """The old API's move: the same path with no limit check, so nothing changes.

        Today no backend refuses an out-of-limit move before the vendor does; the
        old ``move_stage_absolute``/``move_stage_relative`` keep that by calling this.
        Claims the stage, moves, and reads back the position, which updates the cache
        and emits ``position.changed``.
        """
        with self.resources.claim(STAGE_RESOURCE):
            if relative:
                self._move_relative(position)
            else:
                self._move_absolute(position)
            return self.position.get_value()

    def check_limits(self, position: FibsemStagePosition) -> None:
        """Raise ``StageLimitError`` naming every axis of *position* out of limits."""
        outside = []
        for axis, (low, high) in self.limits.items():
            value = getattr(position, axis, None)
            if value is not None and not low <= value <= high:
                outside.append(f"{axis}={value:g} not in [{low:g}, {high:g}]")
        if outside:
            raise StageLimitError(
                f"{self.name}: target outside the stage limits: {', '.join(outside)}"
            )

    # -- what a backend implements -------------------------------------------------

    def _move_absolute(self, position: FibsemStagePosition) -> None:
        raise NotImplementedError

    def _move_relative(self, delta: FibsemStagePosition) -> None:
        raise NotImplementedError

    def _home(self) -> None:
        raise NotImplementedError

    def _link(self) -> None:
        raise NotImplementedError

    # -- internals -----------------------------------------------------------------

    def _position_read(self, name: str, value: Any) -> None:
        if name != "position":
            return
        for axis in self.axes:
            axis_value: Optional[float] = getattr(value, axis, None)
            if axis_value is not None:
                param: BoundParameter = self.parameters[axis]
                param.report(axis_value)
