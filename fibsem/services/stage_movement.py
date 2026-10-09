"""Stage movement as a service: moves measured in a view of the sample, turned into
stage moves.

`StageMovement` holds the moves that coordinate the stage with the beams: a stable
move by a displacement seen in a beam image, the vertical move that restores
coincidence, the safe absolute move that tilts flat before a large rotation, and the
moves to a named orientation and a milling angle. It commands the stage and writes the
beams' working distance through its roles. What it asks of the instrument's geometry
(the poses, the column tilt, the eucentric heights, the rotation centre, the milling
angle) it reads from its parent microscope, which still owns that model.

A driver subclasses it where the instrument moves differently: TESCAN keeps its own
coincidence move from the SEM view, verified on hardware.

`FibsemMicroscope`'s moves (``stable_move``, ``vertical_move`` and the rest) go to the
microscope's service, ``microscope.stage_movement``, so they keep their signatures on
every backend.
"""

from __future__ import annotations

import functools
import logging
from copy import deepcopy
from typing import Any, Optional, Tuple, Type, TypeVar

import numpy as np

from fibsem.devices.beam import Beam
from fibsem.devices.core import Role, command
from fibsem.devices.stage import Stage
from fibsem.geometry.movement import (
    apply_delta,
    fib_offset_after_sem_move,
    image_to_stage_delta,
    undo_scan_rotation,
    vertical_move_delta,
)
from fibsem.services.core import Service
from fibsem.structures import BeamType, FibsemStagePosition

_S = TypeVar("_S", bound="StageMovement")

# A vertical move whose stage-z travel is larger than this resets the ion working
# distance to its eucentric height, the best estimate at the new coincidence plane.
EUCENTRIC_RESET_THRESHOLD = 100e-6  # m

_BEAM_ROLES = {BeamType.ELECTRON: "electron", BeamType.ION: "ion"}


def _records_stage_move(method):
    """Record the move as one ``stage_moved`` event on the parent microscope's
    record; see `fibsem.microscope._records_stage_move`. Only the outermost move on a
    thread is recorded, so a microscope method that routes here records under its own
    name, as before."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        from fibsem.microscope import _records_stage_move as records

        recorded = records(method, microscope_of=lambda service: service.parent)
        return recorded(self, *args, **kwargs)

    return wrapper


class StageMovement(Service):
    """The stage, moved by what a view of the sample shows.

    Every command returns the stage position afterwards, as the old methods did,
    except ``safe_absolute_stage_movement``, which returns nothing, and
    ``move_to_milling_angle``, which says whether the stage reached the angle.
    """

    stage = Role(Stage, doc="The stage that moves.")
    electron = Role(
        Beam,
        required=False,
        doc="The electron beam: its view, and its working distance kept on a move.",
    )
    ion = Role(Beam, required=False, doc="The ion beam: its view.")

    @property
    def vertical_move_views(self) -> Tuple[BeamType, ...]:
        """The views a coincidence correction can be measured in: the beam_type
        values vertical_move accepts. The backend declares them
        (`FibsemMicroscope.vertical_move_views`)."""
        return tuple(self.parent.vertical_move_views)

    def __init__(self, name: str = "stage_movement", **kwargs: Any):
        super().__init__(name, **kwargs)

    # -- reading and writing through the roles -------------------------------------------

    def _beam(self, beam_type: BeamType) -> Optional[Beam]:
        role = _BEAM_ROLES.get(beam_type)
        return None if role is None else self._roles.get(role)

    def _beam_value(self, beam_type: BeamType, name: str) -> Any:
        """A beam parameter's value, or None where there is no such beam or
        parameter (absent = unsupported), as the microscope's beam reads give it."""
        beam = self._beam(beam_type)
        param = None if beam is None else beam.parameters.get(name)
        if param is None:
            return None
        return param.get_value()

    def _set_working_distance(self, beam_type: BeamType, wd: float) -> None:
        """Write a working distance as the old API does (no new checks), or nothing
        where the beam has none (the TESCAN ion column)."""
        beam = self._beam(beam_type)
        param = None if beam is None else beam.parameters.get("working_distance")
        if param is None:
            logging.debug(f"No {beam_type.name} working distance to set.")
            return
        param.write_through(wd)

    def _position(self) -> FibsemStagePosition:
        """The stage position, read the way the microscope reads it, so its
        ``stage_position_changed`` still says where the stage went."""
        return self.parent.get_stage_position()

    def _move_stage(
        self, position: FibsemStagePosition, relative: bool = False
    ) -> FibsemStagePosition:
        """One stage move, without the limit check the old API never had.

        Through the microscope's raw move, which goes to the stage device: it is the
        one place a backend's raw stage move is recorded and its position published,
        and what every existing pin of these moves records. It goes straight to the
        stage device once the old raw moves are retired.
        """
        if relative:
            return self.parent.move_stage_relative(position)
        return self.parent.move_stage_absolute(position)

    def _linked(self) -> Optional[bool]:
        param = self.stage.parameters.get("linked")
        return None if param is None else param.get_value()

    def _view_tilt(self, beam_type: BeamType) -> float:
        """Tilt of a beam column's viewing axis from the electron column, in radians."""
        if beam_type is BeamType.ELECTRON:
            return 0.0
        if beam_type is BeamType.ION:
            return np.deg2rad(self.parent.system.ion.column_tilt)
        raise ValueError(f"Unsupported beam type: {beam_type}")

    def _view_stage_delta(
        self, dx: float, dy: float, view_tilt: float
    ) -> FibsemStagePosition:
        """`fibsem.geometry.movement.image_to_stage_delta` at the current pose."""
        position = self._position()
        return image_to_stage_delta(
            dx,
            dy,
            view_tilt=view_tilt,
            geometry=self.parent.hardware_geometry(),
            stage_rotation=position.r,
            stage_tilt=position.t,
        )

    # -- moves the stage device has itself --------------------------------------------------

    @command
    @_records_stage_move
    def move_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """Move the stage to a position, within its limits. Axes that are None stay."""
        self.stage.move_absolute(position)
        return self._position()

    @command
    @_records_stage_move
    def move_relative(self, delta: FibsemStagePosition) -> FibsemStagePosition:
        """Move the stage by an offset, within its limits. Axes that are None stay."""
        self.stage.move_relative(delta)
        return self._position()

    # -- moves by what a view shows ------------------------------------------------------

    @command
    @_records_stage_move
    def stable_move(
        self, dx: float, dy: float, beam_type: BeamType, static_wd: bool = False
    ) -> FibsemStagePosition:
        """Move the stage by a displacement seen in a beam image, in the sample plane.

        Args:
            dx: distance along the image x-axis, in metres.
            dy: distance along the image y-axis, in metres.
            beam_type: the beam whose image the displacement was measured in.
            static_wd: put the electron working distance at its eucentric height,
                rather than where it was.
        """
        wd = self._beam_value(BeamType.ELECTRON, "working_distance")

        scan_rotation = self._beam_value(beam_type, "scan_rotation")
        dx, dy = undo_scan_rotation(dx, dy, scan_rotation)

        stage_position = self._view_stage_delta(
            dx, dy, view_tilt=self._view_tilt(beam_type)
        )
        self._move_stage(stage_position, relative=True)

        if static_wd:
            wd = self.parent.system.electron.eucentric_height

        # A linked stage's z moves the working distance with it, so put it back. An
        # unlinked stage (a compustage never links) leaves it where it was.
        if self._linked():
            self._set_working_distance(BeamType.ELECTRON, wd)

        logging.debug(
            {
                "msg": "stable_move",
                "dx": dx,
                "dy": dy,
                "beam_type": beam_type.name,
                "static_wd": static_wd,
                "working_distance": wd,
                "scan_rotation": scan_rotation,
                "position": stage_position.to_dict(),
            }
        )
        return self._position()

    def project_stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        base_position: FibsemStagePosition,
    ) -> FibsemStagePosition:
        """Where the stage would end up after a stable move from ``base_position``.

        Nothing moves. The projection is taken at the current stage pose, not at
        ``base_position``, as it always has been.
        """
        scan_rotation = self._beam_value(beam_type, "scan_rotation")
        dx, dy = undo_scan_rotation(dx, dy, scan_rotation)
        delta = self._view_stage_delta(dx, dy, view_tilt=self._view_tilt(beam_type))
        return apply_delta(base_position, delta)

    def supports_vertical_move(self, beam_type: BeamType = BeamType.ION) -> bool:
        """Whether coincidence can be restored from the given view."""
        return beam_type in self.vertical_move_views

    @command
    @_records_stage_move
    def vertical_move(
        self,
        dy: float,
        dx: float = 0.0,
        beam_type: BeamType = BeamType.ION,
        relaxation: float = 1.0,
    ) -> FibsemStagePosition:
        """Restore coincidence from an offset measured in one of the beam views.

        See `FibsemMicroscope.vertical_move`.

        Raises:
            NotImplementedError: if coincidence can't be restored from that view.
        """
        if not self.supports_vertical_move(beam_type):
            raise NotImplementedError(
                f"{type(self.parent).__name__} cannot restore coincidence from the "
                f"{beam_type.name} view."
            )
        if beam_type is BeamType.ELECTRON:
            return self._vertical_move_from_sem(dx=dx, dy=dy, relaxation=relaxation)
        return self._vertical_move_from_fib(dx=dx, dy=dy, relaxation=relaxation)

    def _vertical_move_from_fib(
        self, dy: float, dx: float = 0.0, relaxation: float = 1.0
    ) -> FibsemStagePosition:
        """Move the stage vertically to correct the coincidence point, from an offset
        measured in the FIB view: the feature is already centred in the SEM, and a
        chamber-vertical move is invisible to the electron beam."""
        wd = self._beam_value(BeamType.ELECTRON, "working_distance")

        scan_rotation = self._beam_value(BeamType.ION, "scan_rotation")
        stage_tilt = self._position().t
        stage_position = vertical_move_delta(
            dx=dx,
            dy=dy,
            scan_rotation=scan_rotation,
            fib_column_tilt=self.parent.system.ion.column_tilt,
            stage_tilt=stage_tilt,
            turned_over=self.stage.turned_over(stage_tilt),
            relaxation=relaxation,
        )
        logging.info(f"Vertical movement: {stage_position}")
        self._move_stage(stage_position, relative=True)

        # Always restore the pre-move SEM working distance so fine corrections keep
        # their focus. For a large correction, snap the FIB working distance to
        # eucentric; small corrections keep the current FIB focus.
        self._set_working_distance(BeamType.ELECTRON, wd)
        if abs(stage_position.z) > EUCENTRIC_RESET_THRESHOLD:
            self._set_working_distance(
                BeamType.ION, self.parent.system.ion.eucentric_height
            )

        logging.debug(
            {
                "msg": "vertical_move",
                "dy": stage_position.y,
                "dx": stage_position.x,
                "wd": wd,
                "scan_rotation": scan_rotation,
                "position": stage_position.to_dict(),
            }
        )
        return self._position()

    def _vertical_move_from_sem(
        self, dx: float, dy: float, relaxation: float = 1.0
    ) -> FibsemStagePosition:
        """Correct the coincidence point from an offset measured in the SEM view: a
        stable move brings the feature to the centre of the SEM, and the height
        correction that follows puts the FIB back. A driver may replace it."""
        base_position = self._position()
        self.stable_move(dx=dx, dy=dy, beam_type=BeamType.ELECTRON)
        position_after_sem_move = self._position()

        milling_angle = None
        if self.parent.get_stage_orientation() in ["SEM", "MILLING"]:
            milling_angle = self.parent.get_current_milling_angle()  # deg

        dy = fib_offset_after_sem_move(
            stage_dy=position_after_sem_move.y - base_position.y,
            ion_scan_rotation=self._beam_value(BeamType.ION, "scan_rotation"),
            milling_angle=milling_angle,
        )
        self._vertical_move_from_fib(dx=0, dy=dy, relaxation=relaxation)
        return self._position()

    # -- moves to a place ------------------------------------------------------------------

    @command
    @_records_stage_move
    def safe_absolute_stage_movement(self, stage_position: FibsemStagePosition) -> None:
        """Move the stage to a position safely: tilted flat for a large rotation, and
        rotated compucentrically before the rest of the move."""
        # Before anything moves; see `FibsemMicroscope._refuse_rotation_at_the_
        # fluorescence_microscope` (FIB-841).
        self.parent._refuse_rotation_at_the_fluorescence_microscope(stage_position)
        self._safe_rotation(stage_position)

        logging.debug(f"safe moving to {stage_position}")
        self._move_stage(stage_position)
        logging.debug("safe movement complete.")

    def _safe_rotation(self, stage_position: FibsemStagePosition) -> None:
        """What comes before the move itself: on a stage with a rotation axis, tilt
        flat for a large rotation, then rotate compucentrically. A driver whose
        instrument does this itself replaces it."""
        # The safe sequence is about rotating, so a stage with no rotation axis (a
        # compustage) skips it. `rotation` is `"r" in` the stage's axes.
        if not self.parent.system.stage.rotation:
            return

        from fibsem import movement

        current_position = self._position()
        if movement.rotation_angle_is_larger(stage_position.r, current_position.r):
            self._move_stage(FibsemStagePosition(t=0))
            logging.info("tilting to flat for large rotation.")

        self._move_stage(
            FibsemStagePosition(r=stage_position.r, coordinate_system="RAW")
        )  # TODO: support compucentric rotation directly

    @command
    @_records_stage_move
    def move_to_orientation(self, orientation: str) -> FibsemStagePosition:
        """Move the stage to a named orientation ('SEM', 'FIB', 'MILLING', ...)."""
        stage_position = self._orientation_move_target(orientation)
        self.safe_absolute_stage_movement(stage_position)
        return self._position()

    def _orientation_move_target(self, orientation: str) -> FibsemStagePosition:
        """Where a move to a named orientation ends: its pose (r, t), and on a stage
        that turns a half turn, x and y carried round the rotation centre.

        Only where the driver reports its ``rotation_centre`` (FIB-655). Without one,
        or from a position at no named orientation, the pose alone, as before.
        """
        microscope = self.parent
        pose = deepcopy(microscope.get_orientation(orientation))
        if not microscope.system.stage.rotation or microscope.rotation_centre is None:
            return pose
        current = self._position()
        try:
            target = microscope.get_target_position(deepcopy(current), orientation)
        except ValueError as e:
            logging.warning(
                f"Moving to {orientation} by its pose alone, so the rotation centre "
                f"correction is not applied: {e}"
            )
            return pose
        if target.r == current.r and target.t == current.t:
            # Already at this orientation: nothing turns, so x and y stay put.
            return pose
        pose.x, pose.y = target.x, target.y
        return pose

    @command
    @_records_stage_move
    def move_to_milling_angle(
        self, milling_angle: float, rotation: Optional[float] = None
    ) -> bool:
        """Move the stage to a milling angle, in radians, from the pretilt and the
        column tilt. Returns whether the stage is close to it afterwards.

        Args:
            milling_angle: the milling angle, in radians.
            rotation: the stage rotation, in radians; the rotation reference if None.
        """
        from fibsem.transformations import get_stage_tilt_from_milling_angle

        microscope = self.parent
        if rotation is None:
            rotation = np.radians(microscope.system.stage.rotation_reference)

        stage_tilt = get_stage_tilt_from_milling_angle(microscope, milling_angle)
        self.safe_absolute_stage_movement(FibsemStagePosition(t=stage_tilt, r=rotation))

        # milling_angle is radians here; is_close_to_milling_angle compares degrees
        # (FIB-853)
        return microscope.is_close_to_milling_angle(np.degrees(milling_angle))


def bind_stage_movement(service: Type[_S], microscope: Any) -> Optional[_S]:
    """Build a microscope's stage movement service of class *service* over its stage
    and beams, or None when it has no stage device, which leaves the microscope's own
    move code in charge."""
    if microscope.stage is None:
        return None
    movement = service(parent=microscope)
    movement.fill_roles(stage=microscope.stage)
    for beam_type, role in _BEAM_ROLES.items():
        beam = microscope.beams.get(beam_type)
        if beam is not None:
            movement.fill_roles(**{role: beam})
    movement.connect()
    return movement
