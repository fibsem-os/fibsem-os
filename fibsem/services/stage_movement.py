"""Stage movement as a service: moves measured in a view of the sample, turned into
stage moves.

`StageMovement` holds the moves that coordinate the stage with the beams and the
fluorescence microscope: a stable move by a displacement seen in a beam image or the
FM image, the vertical move that restores coincidence, the safe absolute move that
tilts flat before a large rotation, the moves to a named orientation and a milling
angle, and the move to a device (the FM, or back to the beams) that retracts and
inserts the objective around the traverse. It commands the stage, writes the beams'
working distance, and reads the FM camera and objective through its roles. What it asks of the instrument's geometry
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
from fibsem.devices.fm import FM, mount_transform_from_name
from fibsem.devices.stage import Stage
from fibsem.fm.structures import objective_state_name
from fibsem.geometry.movement import (
    apply_delta,
    fib_offset_after_sem_move,
    image_to_stage_delta,
    undo_scan_rotation,
    vertical_move_delta,
)
from fibsem.services.core import Service
from fibsem.structures import BeamType, CameraImageTransform, FibsemStagePosition

_S = TypeVar("_S", bound="StageMovement")

# A vertical move whose stage-z travel is larger than this resets the ion working
# distance to its eucentric height, the best estimate at the new coincidence plane.
EUCENTRIC_RESET_THRESHOLD = 100e-6  # m

_BEAM_ROLES = {BeamType.ELECTRON: "electron", BeamType.ION: "ion"}


def _moves(translation: FibsemStagePosition) -> bool:
    """Whether a relative move moves anything: an axis set and non-zero."""
    from fibsem.microscope import _moves as moves

    return moves(translation)


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
    fm = Role(
        FM,
        required=False,
        doc="The fluorescence microscope: its camera's view, and the objective that "
        "is retracted for a traverse and inserted at the FM.",
    )

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

    # -- the fluorescence view -----------------------------------------------------------

    def _fm_device(self) -> Optional[FM]:
        """The FM, or None. Some backends build their FM after the stage, so it is
        found in the microscope's devices the first time it is asked for."""
        fm = self._roles.get("fm")
        if fm is None:
            found = self.parent.devices.get("fm")
            if isinstance(found, FM):
                self.fill_roles(fm=found)
                fm = found
        return fm

    def _require_fm(self, message: str) -> Any:
        """The FM the moves look through: the microscope's FM wrapper where it has
        one (it may have no FM device, as on the Odemis stream FM), else the device."""
        wrapper = getattr(self.parent, "fm", None)
        if wrapper is not None:
            return wrapper
        fm = self._fm_device()
        if fm is None:
            raise ValueError(message)
        return fm

    @property
    def camera_tilt(self) -> float:
        """Tilt of the FM camera's optical axis from the SEM column, in degrees.

        The FM entry's ``camera_tilt`` where the configuration states one (FIB-335).
        Otherwise from the mount: an FM the stage reaches by turning the grid over (a
        compustage) looks up at it from the opposite side to the SEM, 180 degrees; an
        offset FM sits parallel to the FIB column and shares its tilt.
        """
        return fm_camera_tilt(self.parent)

    def _display_transform(self, fm: Any) -> CameraImageTransform:
        """The transform the user displays FM images under, from the camera (through
        the wrapper, which keeps it on the camera device when there is one)."""
        if not isinstance(fm, FM):
            return fm._transform
        camera = fm._roles.get("camera")
        param = None if camera is None else camera.parameters.get("display_transform")
        if param is None:
            return CameraImageTransform.NONE
        return mount_transform_from_name(param.get_value())

    def _objective_state(self, fm: Any) -> Optional[str]:
        if not isinstance(fm, FM):
            return fm.objective.state
        objective = fm._roles.get("objective")
        if objective is None:
            return None
        return objective_state_name(objective.state.get_value())

    def _fm_stage_delta(self, fm: Any, dx: float, dy: float) -> FibsemStagePosition:
        """Relative stage movement for a displacement seen in the displayed FM image.

        Shared by `fm_stable_move` and `project_fm_stable_move`, so the two cannot
        disagree. The input is in the frame the user is looking at, so the display
        transform is undone first, the counterpart of a beam's scan rotation. The
        driver hands out stage-aligned images (the camera's ``mount_transform`` is
        applied before the display transform), and every transform is its own
        inverse, so applying it maps the displacement back.
        """
        dx, dy = self._display_transform(fm).apply_to_delta(dx, dy)
        return self._view_stage_delta(dx, dy, view_tilt=np.deg2rad(self.camera_tilt))

    def project_fm_stable_move(
        self, dx: float, dy: float, base_position: FibsemStagePosition
    ) -> FibsemStagePosition:
        """Where the stage would end up after an FM displacement, without moving.

        The fluorescence counterpart of `project_stable_move`, in displayed image
        coordinates.

        Raises:
            ValueError: if there is no fluorescence microscope.
        """
        fm = self._require_fm("Fluorescence microscope is not available.")
        return apply_delta(base_position, self._fm_stage_delta(fm, dx, dy))

    @command
    def fm_stable_move(self, dx: float, dy: float) -> FibsemStagePosition:
        """Move the stage by a displacement seen in the fluorescence image, holding
        the focal plane: the same projection the beams use, with the camera's tilt in
        place of a column tilt, so the move stays in the sample plane.

        Raises:
            ValueError: if there is no fluorescence microscope.
        """
        fm = self._require_fm("Fluorescence microscope is not available. Cannot move.")
        state = self._objective_state(fm)
        if state is not None:
            if state != "Inserted":
                logging.warning(
                    "Moving via the fluorescence image while the objective is not "
                    f"inserted (state: {state}); the view may not match the sample."
                )

        stage_position = self._fm_stage_delta(fm, dx, dy)

        # No working-distance restore: that is beam bookkeeping; the objective keeps
        # focus because the move stays in the sample plane.
        self._move_stage(stage_position, relative=True)

        logging.debug(
            {
                "msg": "fm_stable_move",
                "dx": dx,
                "dy": dy,
                "camera_tilt": self.camera_tilt,
                "position": stage_position.to_dict(),
            }
        )
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

    def _objective(self) -> Any:
        """The objective to insert and retract: through the FM API where the
        microscope has one, which tells the displays the objective moved
        (`ObjectiveLens.position_changed`), else the FM's objective device."""
        if self.parent.fm is not None:
            return self.parent.fm.objective
        return self._require_fm("No fluorescence microscope.").objective

    @command
    @_records_stage_move
    def move_to_device(self, device: str, orientation: Optional[str] = None) -> None:
        """Travel to `device`, re-posing on the way when the pose has to change.

        One call that owns the safe order -- retract the objective, re-pose at the
        beams, travel out -- so a rotation never happens with the stage parked under
        an objective. The rotation guard (FIB-841) stays underneath as the last-line
        assert; the route this composes never trips it.

        `orientation` names the pose to arrive in. Omitted, the pose is carried
        across untouched whenever the target device can image from it -- that is the
        point of a traverse, and the reason this must not pass the current
        orientation's *name* through `move_to_orientation`: doing so would snap r and
        t to nominal and quietly discard a milling angle somebody dialled in. When
        the pose does have to change (an offset FM images in the FIB pose; asking
        for it from SEM used to be a refusal), the device's first declared
        acquisition orientation is used.
        """
        target_device = self.parent._get_device(device)  # refuses by name

        if device == "FM" and self._fm_device() is None:
            raise ValueError("FM module is not available. Cannot move to FM position.")

        stage_position = self._position()
        source = self.parent.get_current_device(stage_position)
        if source is None:
            raise ValueError(
                f"The stage is not at any configured device "
                f"({sorted(self.parent.system.stage.devices)}), so there is nothing to "
                f"travel from. Position: {stage_position}."
            )

        # The pose to arrive in. An explicit ask is honoured as asked; otherwise the
        # pose is carried across, unless the target device cannot image from it --
        # then its first declared acquisition orientation stands in.
        desired = self.parent._arrival_orientation(device, stage_position, orientation)
        if desired is not None and orientation is None:
            logging.info(
                f"The {device} device images from "
                f"{target_device.available_orientations}; re-posing to {desired} "
                f"at the beams before travelling."
            )

        if desired is None and source == device:
            logging.info(f"Already at {device} position, no need to move.")
        else:
            # Both ends checked before anything moves: the source by finding it, the
            # target by converting to where the stage will arrive.
            arrival = self._planned_arrival(device, stage_position, desired)
            if arrival is not None:
                self._check_arrival(device, arrival)

            if desired is not None:
                # The bracketing order: every re-pose happens at the beams, where the
                # rotation is about the sample rather than a 48.8 mm arm.
                #
                # Driven to the *converted* position, not to the orientation by name.
                # `move_to_orientation` rewrites r and t where the stage stands; a half
                # turn there is compucentric about a centre that is not the sample, so the
                # point that was under the beam is swung away and the traverse carries the
                # wrong piece of sample out. The transform is what every pose derivation
                # and overview marker uses, so arriving where it says is what puts the
                # stage on the marked point. Falls back to the bare re-pose only from a
                # pose the classifier cannot name: there is no point to keep there, and
                # the fallback is how a stage in an unsupported pose gets back to a
                # supported one.
                try:
                    at_the_beams = self.parent.get_target_position(
                        stage_position, desired, target_device="FIBSEM"
                    )
                except ValueError as e:
                    logging.warning(
                        f"Re-posing to {desired} without keeping the sample point: {e}"
                    )
                    at_the_beams = None

                self._retract_objective_to_move(device)
                self._travel(source, "FIBSEM")
                if at_the_beams is not None:
                    self.safe_absolute_stage_movement(at_the_beams)
                else:
                    self.move_to_orientation(desired)
                self._travel("FIBSEM", device)
            elif _moves(self.parent._device_translation(source, device)):
                self._retract_objective_to_move(device)
                self._travel(source, device)

        # Unconditional, so that the postcondition is the device *and* the objective
        # state together: asking again for a device the stage is already at cannot
        # leave the FM blind.
        if device == "FM":
            self._objective().insert()

    def _travel(self, source: str, target: str) -> None:
        """Move the stage by the translation from one device to another, if any.

        Nothing is commanded between two devices at one place (a compustage's beams
        and an FM without an origin of its own)."""
        translation = self.parent._device_translation(source, target)
        if _moves(translation):
            self._move_stage(translation, relative=True)

    def _retract_objective_to_move(self, device: str) -> None:
        """Retract the objective immediately before the stage moves, and only then.

        The objective must not be out over the sample while the stage moves, but every
        reason to retract it is the motion itself -- so a call that refuses, or finds
        it has nowhere to go, leaves the objective exactly as it found it rather than
        pulling it out of the sample for nothing.
        """
        logging.info(f"Moving to {device} position...")
        if self._fm_device() is not None:
            self._objective().retract()

    def _planned_arrival(
        self,
        device: str,
        stage_position: FibsemStagePosition,
        desired: Optional[str],
    ) -> Optional[FibsemStagePosition]:
        """Where `move_to_device` will put the stage: the `get_target_position`
        conversion it drives to, or the bare translation from a pose the conversion
        cannot name. `None` when neither can say -- a re-pose from such a pose, which
        has no point to keep and is checked by nothing but the move itself."""
        try:
            return self.parent.get_target_position(
                deepcopy(stage_position), desired, target_device=device
            )
        except ValueError as e:
            if desired is not None:
                logging.warning(f"Not checking where {device} will be reached: {e}")
                return None
            source = self.parent.get_current_device(stage_position)
            return stage_position + self.parent._device_translation(source, device)

    def _check_arrival(self, device: str, arrival: FibsemStagePosition) -> None:
        """Refuse a traverse that would arrive outside *device*'s range.

        The start is checked by finding the source (`get_current_device`); this is the
        other end. Each device has its own range, and the stage keeps its offset from
        the source's origin, so a position well inside the beams' range can land
        outside an FM's -- where the stage would report it had not arrived, and could
        not go back. Checked before the stage moves, and before the objective is
        retracted for it.
        """
        if not self.parent.is_at_device(device, arrival):
            raise ValueError(
                f"Travelling to {device} from here would arrive at {arrival}, outside "
                f"its range {self.parent._get_device(device).range} of its origin "
                f"{self.parent._get_device(device).origin}. Move the stage nearer the "
                "source's origin first."
            )


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
    fm = microscope.devices.get("fm")
    if isinstance(fm, FM):
        movement.fill_roles(fm=fm)
    movement.connect()
    return movement


def fm_camera_tilt(microscope: Any) -> float:
    """Tilt of the FM camera's optical axis from the SEM column, in degrees: the FM
    entry's ``camera_tilt``, else 180 where the FM is a stage pose, else the ion
    column's tilt. See `StageMovement.camera_tilt`."""
    configured = microscope.system.fm.camera_tilt
    if configured is not None:
        return float(configured)
    if microscope._fm_is_a_pose():
        return 180.0
    return microscope.system.ion.column_tilt
