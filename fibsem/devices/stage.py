"""The Stage device, the reverse of today's ``Stage``.

Today ``fibsem.microscopes._stage.Stage`` owns nothing: every property and move asks
its parent microscope, which asks its backend's ``get``/``set`` chain. Here the device
owns the hardware contract and the microscope would ask the device.

The stage is the raw hardware contract only: where it is, what it can reach, and
"move to these coordinates" in the fibsem-os frame (FIB-1083). View-corrected moves
(``stable_move``, ``vertical_move``, ``project_stable_move``) depend on the beams as
much as the stage, so they are not here; they belong to the views, built on top.

Parameters describe state and are read-only here: ``position``, ``homed`` and
``linked``. The stage has one position, so it is one parameter with one read and one
signal. ``stage.axes.t`` is a view of one axis of it, with its unit and limits and a
``changed`` signal of its own, but no read of its own. Which axes exist is the
driver's to say: they are the keys of the per-axis limits in ``position``'s metadata.

Changing where the stage is takes time and can fail, so it is a command, not an
assignment: ``move_absolute``, ``move_relative``, ``home`` and ``link``. A move
outside the axis limits is refused, never clipped, for the same reason
``raise_if_outside_stage_limits`` refuses a tile grid: a stage that stopped
somewhere else than asked is a position everything downstream takes for the truth.

A backend implements the parameters with ``read_<name>``/``metadata_<name>`` methods,
as for a beam, and the commands with four hooks: ``_move_absolute``,
``_move_relative``, ``_home`` and ``_link``. The base class does the rest once: the
limit check, the ``stage`` resource, the read-back, and the change signals.

The stage also declares its poses (FIB-1101): the rotation and tilt that each
orientation name (SEM, FIB, and FM where the stage reaches it by re-posing) means on
this stage. ``poses`` is a pure function of the geometry the microscope passes in,
so the stage never asks its parent. MILLING is not a stage pose: it depends on the
milling angle setting, so the microscope builds it from the SEM rotation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, Mapping

from psygnal import Signal

from fibsem.devices.core import Device, Parameter, command
from fibsem.structures import STAGE_FRAME_FIBSEM, FibsemStagePosition, RangeLimit

STAGE_RESOURCE = "stage"
AXIS_UNITS: Dict[str, str] = {"x": "m", "y": "m", "z": "m", "r": "rad", "t": "rad"}
"""Every axis a ``FibsemStagePosition`` can carry, and its unit."""

UNLIMITED = RangeLimit(min=-math.inf, max=math.inf)
"""The limits a driver gives an axis it has but cannot bound."""


def rotating_stage_poses(
    rotation_reference: float,
    shuttle_pre_tilt: float,
    fib_column_tilt: float,
    rotates: bool = True,
) -> Dict[str, FibsemStagePosition]:
    """SEM and FIB for a stage that turns round to face the ion beam.

    Angles in degrees in, radians out. SEM is the reference rotation, with the
    pre-tilt cancelled so the sample faces the electron beam. FIB is half a turn from
    it, tilted to the ion column less the pre-tilt. A stage without a rotation axis
    (``rotates`` False) stays at the reference for FIB, as ``rotation_180`` does.
    """
    fib_rotation = (rotation_reference + 180) % 360 if rotates else rotation_reference
    return {
        "SEM": FibsemStagePosition(
            r=math.radians(rotation_reference), t=math.radians(shuttle_pre_tilt)
        ),
        "FIB": FibsemStagePosition(
            r=math.radians(fib_rotation),
            t=math.radians(fib_column_tilt - shuttle_pre_tilt),
        ),
    }


def compustage_poses(
    rotation_reference: float, shuttle_pre_tilt: float, fib_column_tilt: float
) -> Dict[str, FibsemStagePosition]:
    """SEM, FIB and FM for a compustage, which tilts over instead of turning round.

    It has no rotation axis, so every pose is at the reference rotation. FIB is the
    rotating stage's tilt turned over by 180 degrees, because the grid is imaged from
    its back. FM is the grid turned fully over to face the objective underneath.

    FM is a pose only here. On an offset mount the FM is a place, not a pose: the
    stage travels there holding whatever orientation it was in, so no FM pose exists.
    """
    poses = rotating_stage_poses(
        rotation_reference, shuttle_pre_tilt, fib_column_tilt, rotates=False
    )
    poses["FIB"].t -= math.radians(180)
    poses["FM"] = FibsemStagePosition(
        r=math.radians(rotation_reference), t=math.radians(-180)
    )
    return poses


def axis_limits_from_degrees(limits: Mapping[str, RangeLimit]) -> Dict[str, RangeLimit]:
    """Per-axis limits in the axes' units, from limits that give rotations in degrees
    (as today's ``_get_axis_limits`` does)."""
    converted: Dict[str, RangeLimit] = {}
    for axis, limit in limits.items():
        low, high = limit.min, limit.max
        if AXIS_UNITS.get(axis) == "rad":
            low, high = math.radians(low), math.radians(high)
        converted[axis] = RangeLimit(min=low, max=high)
    return converted


class StageLimitError(ValueError):
    """A move whose target is outside an axis's limits. Nothing moved."""


class Axis:
    """One axis of the stage position: a view, with no read of its own."""

    changed = Signal(float)
    """The axis's new value, when a position read found it moved."""

    def __init__(self, stage: Stage, name: str):
        self.stage = stage
        self.name = name
        self.unit = AXIS_UNITS[name]

    @property
    def limits(self) -> RangeLimit:
        """In SI units, from the position metadata cached at connect."""
        return self.stage.position.limits[self.name]

    @property
    def value(self) -> float:
        """A live read: reads the whole position and returns this axis."""
        return getattr(self.stage.position.get_value(), self.name)

    @property
    def cached(self) -> float:
        """The last known value, with no instrument call."""
        return getattr(self.stage.position.cached, self.name)

    def _position_changed(self, position: FibsemStagePosition) -> None:
        value = getattr(position, self.name, None)
        previous = getattr(self.stage.position.previous, self.name, None)
        if value is not None and value != previous:
            self.changed.emit(value)

    def __repr__(self) -> str:
        return f"<Axis {self.name} [{self.unit}] {self.limits}>"


class Axes:
    """The stage's axes by name: ``stage.axes.t``, ``stage.axes["t"]``, ``"r" in``."""

    def __init__(self, axes: Dict[str, Axis]):
        self._axes = axes

    def __getattr__(self, name: str) -> Axis:
        try:
            return self.__dict__["_axes"][name]
        except KeyError:
            raise AttributeError(f"the stage has no '{name}' axis") from None

    def __getitem__(self, name: str) -> Axis:
        return self._axes[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._axes)

    def __contains__(self, name: object) -> bool:
        return name in self._axes

    def __len__(self) -> int:
        return len(self._axes)

    def __repr__(self) -> str:
        return f"<Axes {list(self._axes)}>"


class Stage(Device):
    position = Parameter(
        FibsemStagePosition,
        doc="Raw stage coordinates, in the fibsem-os frame. Its limits are per axis.",
    )
    homed = Parameter(bool)
    linked = Parameter(bool, doc="z linked to the working distance.")

    frame: str = STAGE_FRAME_FIBSEM
    """The frame ``position`` is in, stamped on every image (FIB-1114)."""

    def __init__(self, parent: Any = None, **kwargs: Any):
        super().__init__(name="stage", parent=parent, **kwargs)
        self.axes = Axes({})

    def connect(self) -> Stage:
        """Bind the parameters, then build an axis for each one the driver reports."""
        super().connect()
        limits = self.position.limits or {}
        axes = {name: Axis(self, name) for name in AXIS_UNITS if name in limits}
        for axis in axes.values():
            self.position.changed.connect(axis._position_changed)
        self.axes = Axes(axes)
        return self

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
        and emits ``position.changed`` and each moved axis's ``changed``.
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
        for name in self.axes:
            limit = self.axes[name].limits
            low, high = limit.min, limit.max
            value = getattr(position, name, None)
            if value is not None and not low <= value <= high:
                outside.append(f"{name}={value:g} not in [{low:g}, {high:g}]")
        if outside:
            raise StageLimitError(
                f"{self.name}: target outside the stage limits: {', '.join(outside)}"
            )

    # -- poses ------------------------------------------------------------------------

    def poses(
        self, rotation_reference: float, shuttle_pre_tilt: float, fib_column_tilt: float
    ) -> Dict[str, FibsemStagePosition]:
        """The pose for each orientation name on this stage, from the geometry in degrees.

        A stage that turns round to face the ion beam by default, or stays at the
        reference if it has no ``r`` axis. A stage that reaches the beams some other
        way (a compustage) overrides this.
        """
        return rotating_stage_poses(
            rotation_reference,
            shuttle_pre_tilt,
            fib_column_tilt,
            rotates="r" in self.axes,
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
