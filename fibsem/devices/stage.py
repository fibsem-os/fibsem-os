"""The Stage device: it owns the stage's hardware contract, and the microscope asks
the device.

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

import logging
import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional

from fibsem.constants import DEGREE_SYMBOL
from fibsem.devices.core import (
    AXIS_UNITS,
    UNLIMITED,  # noqa: F401  (re-exported: drivers import it from here)
    Axes,
    AxisLimitError,
    Device,
    DeviceParameter,
    Session,
    axes_outside_limits,
    build_axes,
    command,
    plain,
    query,
)
from fibsem.structures import (
    BEAMS_STAGE_DEVICE,
    KNOWN_ORIENTATIONS,
    STAGE_FRAME_FIBSEM,
    Display,
    FibsemStagePosition,
    RangeLimit,
)

STAGE_RESOURCE = "stage"


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


def flip_stage_poses(
    rotation_reference: float, shuttle_pre_tilt: float, fib_column_tilt: float
) -> Dict[str, FibsemStagePosition]:
    """SEM, FIB and FM for a flip stage, which tilts over instead of turning round.

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


def flip_stage_device_at_pose(orientation: str) -> str:
    """Which device a flip stage is at in a pose: the FM in the FM pose, else the beams.

    The objective is under the grid, so the stage reaches it by turning the grid over,
    not by travelling. The pose says where it is, wherever x and y are.
    """
    return "FM" if orientation == "FM" else BEAMS_STAGE_DEVICE


def tilted_past_vertical(tilt: float) -> bool:
    """Whether a stage tilt in radians has the sample turned over: past -90 degrees.

    The default for `Stage.turned_over`, and for a backend without a stage device.
    """
    return tilt < math.radians(-90)


def axis_limits_from_degrees(limits: Mapping[str, RangeLimit]) -> Dict[str, RangeLimit]:
    """Per-axis limits in the axes' units, from limits that give rotations in degrees
    (as the fixed stage tables do)."""
    converted: Dict[str, RangeLimit] = {}
    for axis, limit in limits.items():
        low, high = limit.min, limit.max
        if AXIS_UNITS.get(axis) == "rad":
            low, high = math.radians(low), math.radians(high)
        converted[axis] = RangeLimit(min=low, max=high)
    return converted


def axis_limits_to_degrees(limits: Mapping[str, RangeLimit]) -> Dict[str, RangeLimit]:
    """The inverse of ``axis_limits_from_degrees``: rotations back in degrees, for
    readers that take limits in the sample stage's units (metres, and degrees for r
    and t)."""
    converted: Dict[str, RangeLimit] = {}
    for axis, limit in limits.items():
        low, high = limit.min, limit.max
        if AXIS_UNITS.get(axis) == "rad":
            low, high = math.degrees(low), math.degrees(high)
        converted[axis] = RangeLimit(min=low, max=high)
    return converted


class StageLimitError(AxisLimitError):
    """A stage move whose target is outside an axis's limits. Nothing moved."""


class Stage(Device):
    position = DeviceParameter(
        FibsemStagePosition,
        doc="Raw stage coordinates, in the fibsem-os frame. Its limits are per axis.",
        display={
            **{
                axis: Display(axis.upper(), scale=1e3, step=0.001, decimals=5)
                for axis in ("x", "y", "z")
            },
            "r": Display("Rotation", scale=180 / math.pi, unit=DEGREE_SYMBOL),
            "t": Display("Tilt", scale=180 / math.pi, unit=DEGREE_SYMBOL),
        },
    )
    homed = DeviceParameter(bool)
    linked = DeviceParameter(bool, doc="z linked to the working distance.")

    frame: str = STAGE_FRAME_FIBSEM
    """The frame ``position`` is in, stamped on every image (FIB-1114)."""

    def __init__(self, parent: Any = None, **kwargs: Any):
        super().__init__(name="stage", parent=parent, **kwargs)
        self.axes = Axes({})

    def connect(self) -> Stage:
        """Bind the parameters, then build an axis for each one the driver reports."""
        super().connect()
        self._build_axes()
        return self

    def _build_axes(self) -> None:
        """An axis for each one the driver gives ``position`` limits for."""
        self.axes = build_axes(self)

    def facts(self) -> Dict[str, Any]:
        """The frame, the shuttle, and the device each orientation puts the stage at:
        the answers a remote stage gives without asking again."""
        return {
            "frame": self.frame,
            "has_builtin_shuttle": self.has_builtin_shuttle(),
            "mirrors_back_view": self.mirrors_back_view(),
            "device_at_pose": {
                orientation: self.device_at_pose(orientation)
                for orientation in KNOWN_ORIENTATIONS
            },
        }

    @property
    def is_homed(self) -> Optional[bool]:
        """Whether the stage is homed; None where its driver can't say."""
        param = self.parameters.get("homed")
        return None if param is None else param.get_value()

    # -- commands -----------------------------------------------------------------

    @command
    def move_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """Move to a position. Axes that are None stay where they are.

        Claims the stage, moves, and reads back the position, which updates the cache
        and emits ``position.changed`` and each moved axis's ``changed``.
        """
        with recording_move(self.session, self, "move_absolute", position=position):
            self.check_limits(position)
            with self.resources.claim(STAGE_RESOURCE):
                self._move_absolute(position)
                return self.position.get_value()

    @command
    def move_relative(self, delta: FibsemStagePosition) -> FibsemStagePosition:
        """Move by an offset. Axes that are None don't move."""
        with recording_move(self.session, self, "move_relative", position=delta):
            with self.resources.claim(STAGE_RESOURCE):
                self.check_limits(self.position.get_value() + delta)
                self._move_relative(delta)
                return self.position.get_value()

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

    @query
    def pose_table(
        self, rotation_reference: float, shuttle_pre_tilt: float, fib_column_tilt: float
    ) -> Dict[str, FibsemStagePosition]:
        """``poses()`` as a query, so a stage on another computer can be asked for
        its poses. It moves nothing."""
        return self.poses(rotation_reference, shuttle_pre_tilt, fib_column_tilt)

    def check_limits(self, position: FibsemStagePosition) -> None:
        """Raise ``StageLimitError`` naming every axis of *position* out of limits."""
        outside = axes_outside_limits(self.axes, position)
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

    def device_at_pose(self, orientation: str) -> Optional[str]:
        """The device this stage is at whenever it holds a pose, or None.

        None by default: the stage travels between devices, so where it is decides
        which one it is at (the nearest origin, `FibsemMicroscope.get_current_device`).
        A stage that reaches a device by re-posing instead of travelling (a compustage
        and its FM) overrides this, so the pose decides wherever x and y are.
        """
        return None

    def has_builtin_shuttle(self) -> bool:
        """Whether the stage carries its own one-grid shuttle, so the sample holder is
        built in rather than read from the configuration.

        False by default. A compustage, which is the autoloader's stage, overrides it.
        """
        return False

    def mirrors_back_view(self) -> bool:
        """Whether the instrument mirrors an image a beam takes of the back of the
        grid, so it reads as if seen from the front. Stamped on every image
        (`FibsemHardwareGeometry.mirrors_back_view`) for its reprojection.

        False by default: the image is as the beam saw it.
        """
        return False

    def turned_over(self, tilt: float) -> bool:
        """Whether the sample is turned over at this stage tilt, in radians.

        Past -90 degrees by default, which only a compustage reaches. A stage that
        turns the sample over somewhere else overrides this.
        """
        return tilted_past_vertical(tilt)

    # -- what a backend implements -------------------------------------------------

    def _move_absolute(self, position: FibsemStagePosition) -> None:
        raise NotImplementedError

    def _move_relative(self, delta: FibsemStagePosition) -> None:
        raise NotImplementedError

    def _home(self) -> None:
        raise NotImplementedError

    def _link(self) -> None:
        raise NotImplementedError


@contextmanager
def recording_move(
    session: Session, stage: Any, move: str, **request: Any
) -> Iterator[None]:
    """Record the stage moves inside as one ``stage_moved`` event named ``move``, with
    ``request`` (the arguments it was given), and where ``stage`` was before and after.
    A stage move inside another records nothing of its own (`Session.recording`), so
    a stage movement service's move is recorded under the service's name.

    ``start`` and ``end`` are the position last read, never a new read, which would be
    a hardware call the move did not make: a move ends by reading where it got to, and
    one that failed first ends where it began.
    """
    with session.recording("stage_moved", move=move, request=None, start=None) as entry:
        try:
            entry["request"] = {name: plain(value) for name, value in request.items()}
            entry["start"] = plain(_last_position(stage))
        except Exception:  # noqa: BLE001 - recording must not matter
            logging.debug(f"could not describe a {move} stage move", exc_info=True)
        try:
            yield
        finally:
            try:
                entry["end"] = plain(_last_position(stage))
            except Exception:  # noqa: BLE001 - recording must not matter
                entry["end"] = None


def _last_position(stage: Any) -> Optional[FibsemStagePosition]:
    position = getattr(stage, "parameters", {}).get("position")
    return None if position is None else position.last_known
