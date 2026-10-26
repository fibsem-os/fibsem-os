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

import logging
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, TypeVar

import numpy as np

from fibsem.devices.beam import Beam
from fibsem.devices.core import DeviceParameter, Role, command, query
from fibsem.devices.fm import FM, mount_transform_from_name
from fibsem.devices.stage import Stage, recording_move
from fibsem.geometry.angles import (
    rotation_angle_is_larger,
    rotation_angle_is_smaller,
    wrap_angle,
)
from fibsem.geometry.frames import StageModel
from fibsem.geometry.frames import fm_camera_tilt as derived_camera_tilt
from fibsem.geometry.movement import (
    apply_delta,
    beam_view_tilt,
    fib_offset_after_sem_move,
    image_to_stage_delta,
    inverse_view_corrected_dy,
    undo_scan_rotation,
    vertical_move_delta,
)
from fibsem.geometry.orientations import (
    classify_orientation,
    milling_angle_at,
    milling_angle_from_tilt,
    repose,
)
from fibsem.services.core import Service
from fibsem.structures.settings import BeamType, FibsemStagePosition
from fibsem.structures.system import (
    BEAMS_STAGE_DEVICE,
    DEVICE_AXES,
    CameraImageTransform,
    DeviceImagingState,
    InsertableDeviceState,
    StageDeviceSettings,
)
from fibsem.transformations import get_stage_tilt_from_milling_angle

_S = TypeVar("_S", bound="StageMovement")

# A vertical move whose stage-z travel is larger than this resets the ion working
# distance to its eucentric height, the best estimate at the new coincidence plane.
EUCENTRIC_RESET_THRESHOLD = 100e-6  # m

_BEAM_ROLES = {BeamType.ELECTRON: "electron", BeamType.ION: "ion"}


def _moves(translation: FibsemStagePosition) -> bool:
    """Whether a relative move moves anything: an axis set and non-zero."""
    from fibsem.microscope import _moves as moves

    return moves(translation)


# The device the orientation transform is defined at. A re-pose is a half turn about
# a chamber-fixed centre, and the beams are the only place it has ever been applied on
# any instrument -- so `target_position` carries a position into this device's frame
# before re-posing it, and back out afterwards.
ROTATION_FRAME_DEVICE = "FIBSEM"


# How near the stage's tilt must be to the tilt an insert asks for to stay put.
SAFE_INSERT_TILT_TOLERANCE_DEG = 0.5


@dataclass
class RouteStep:
    """One step of `StageMovement.move_to_device`'s route, with where it ends.

    ``step`` is retract, turn, travel or insert. A retract or insert carries the
    objective's z there (None where the driver can't say); a turn its orientation,
    r and t; a travel the device it goes to and its translation. ``position`` is
    the stage position the step ends at, where it is known.
    """

    step: str
    objective_z: Optional[float] = None
    moves_objective: bool = True
    orientation: Optional[str] = None
    r: Optional[float] = None
    t: Optional[float] = None
    device: Optional[str] = None
    translation: Optional[FibsemStagePosition] = None
    position: Optional[FibsemStagePosition] = None

    @property
    def text(self) -> str:
        """The step as an operator reads it in a confirmation."""
        if self.step in ("retract", "insert"):
            verb = "Retract" if self.step == "retract" else "Insert"
            z = "" if self.objective_z is None else f" to z = {_mm(self.objective_z)}"
            return f"{verb} the objective{z}"
        if self.step == "turn":
            pose = ", ".join(
                f"{name} = {np.degrees(value):.1f}°"
                for name, value in (("r", self.r), ("t", self.t))
                if value is not None
            )
            return f"Turn to the {self.orientation} orientation at the beams ({pose})"
        place = (
            "back to the beams"
            if self.device == BEAMS_STAGE_DEVICE
            else f"to the {self.device}"
        )
        legs = [
            getattr(self.translation, axis)
            for axis in DEVICE_AXES
            if getattr(self.translation, axis) is not None
        ]
        text = f"Travel {float(np.hypot.reduce(legs)) * 1e3:.1f} mm {place}"
        if self.position is not None:
            text += ", to " + ", ".join(
                f"{axis} = {_mm(getattr(self.position, axis))}"
                for axis in DEVICE_AXES
                if getattr(self.position, axis) is not None
            )
        return text

    def to_dict(self) -> Dict[str, Any]:
        values: Dict[str, Any] = {"step": self.step, "text": self.text}
        if self.step in ("retract", "insert"):
            values["objective_z"] = self.objective_z
        if self.step == "turn":
            values.update(orientation=self.orientation, r=self.r, t=self.t)
        if self.step == "travel":
            values.update(device=self.device, translation=self.translation.to_dict())
        if self.position is not None:
            values["position"] = self.position.to_dict()
        return values


def _mm(metres: float) -> str:
    return f"{metres * 1e3:.3f} mm"


class StageMovement(Service):
    """The stage, moved by what a view of the sample shows.

    Every command returns the stage position afterwards,
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

    milling_angle = DeviceParameter(
        float,
        unit="deg",
        doc="The stored milling angle the MILLING orientation is built from, in "
        "degrees. The configuration gives its value at connect.",
    )

    def __init__(self, name: str = "stage_movement", **kwargs: Any):
        super().__init__(name, **kwargs)
        # The orientation table, built from the stage's poses and the milling angle
        # on first use, and again whenever either changes.
        self._orientations: Optional[Dict[str, FibsemStagePosition]] = None
        # Set while the SEM to FM calibration travels by the nominal origin.
        self._calibrating = False

    # -- orientations --------------------------------------------------------------------

    def read_milling_angle(self) -> float:
        return float(self.parent.system.stage.milling_angle)

    def write_milling_angle(self, value: float) -> None:
        self.parent.system.stage.milling_angle = value
        self.update_orientations()

    @property
    def orientations(self) -> Dict[str, FibsemStagePosition]:
        """Each orientation's pose (r and t), by name: SEM, FIB, MILLING, and FM on a
        compustage."""
        if self._orientations is None:
            self.update_orientations()
        return self._orientations

    @orientations.setter
    def orientations(self, table: Dict[str, FibsemStagePosition]) -> None:
        self._orientations = table

    def turned_over(self, tilt: float) -> bool:
        """Whether the stage has the sample turned over at this tilt, in radians.

        The stage device says; with no stage, nothing is turned over.
        """
        stage = self.parent.stage_device
        return False if stage is None else stage.turned_over(tilt)

    @query
    def stage_poses(self) -> Dict[str, FibsemStagePosition]:
        """The stage's pose for each orientation name, from the configured geometry.

        The stage device declares them. With no stage there are none.
        """
        stage = self.parent.stage_device
        if stage is None:
            return {}
        system = self.parent.system
        return stage.poses(
            rotation_reference=system.stage.rotation_reference,
            shuttle_pre_tilt=system.stage.shuttle_pre_tilt,
            fib_column_tilt=system.ion.column_tilt,
        )

    @query
    def fm_is_a_pose(self) -> bool:
        """Does the stage reach the FM by re-posing rather than by travelling?

        True where the stage declares an FM pose: the objective is under
        the grid and the stage turns the grid over to face it, so the beams and the
        FM are one place. False on an offset mount, where the FM is a place the stage
        travels to in whatever pose it holds. Today it is true exactly on a compustage.
        """
        return "FM" in self.stage_poses()

    def update_orientations(self) -> None:
        """Rebuild the orientation table from the stage's poses and the milling angle."""
        poses = self.stage_poses()
        if not poses:  # with no stage there is nothing to orient
            self._orientations = {}
            return
        # MILLING sits between FIB and the rest: it is the SEM rotation at the tilt
        # the milling angle needs, so it is not the stage's to declare.
        milling_tilt = get_stage_tilt_from_milling_angle(
            self.parent, np.radians(self.parent.system.stage.milling_angle)
        )
        sem = poses.pop("SEM")
        fib = poses.pop("FIB")
        self._orientations = {
            "SEM": sem,
            "FIB": fib,
            "MILLING": FibsemStagePosition(r=sem.r, t=milling_tilt),
            **poses,
        }

    @query
    def orientation(self, name: str) -> FibsemStagePosition:
        """The pose (r and t) of the orientation called ``name``."""
        if name not in self.orientations:
            raise ValueError(f"Orientation {name} not supported.")
        return self.orientations[name]

    @query
    def stage_orientation(self, position: Optional[FibsemStagePosition] = None) -> str:
        """The orientation a stage position is in ("SEM", "FIB", "MILLING", "FM" or
        "NONE"), the stage's own when ``position`` is None."""
        if position is None:
            position = self._position()
        if position.r is None or position.t is None:
            raise ValueError(
                "Stage position must have both rotation (r) and tilt (t) defined."
            )
        return classify_orientation(position.r, position.t, self.orientations)

    @query
    def current_milling_angle(
        self, position: Optional[FibsemStagePosition] = None
    ) -> float:
        """The milling angle a stage position gives, in degrees, the stage's own when
        ``position`` is None: the angle between the ion beam and the sample surface,
        at whatever rotation and tilt the stage holds (90 at the FIB pose)."""
        if position is None:
            position = self._position()
        if position.t is None:
            raise ValueError(
                "Stage tilt is not available. Cannot calculate milling angle."
            )
        geometry = self.parent.hardware_geometry()
        at_reference = position.r is None or np.isclose(
            wrap_angle(position.r - np.radians(geometry.rotation_reference)), 0.0
        )
        if at_reference:  # the SEM rotation (or none read): the tilt-plane formula
            return milling_angle_from_tilt(
                position.t,
                pretilt=self.parent.system.stage.shuttle_pre_tilt,
                column_tilt=self.parent.system.ion.column_tilt,
                turned_over=self.turned_over(position.t),
            )
        return milling_angle_at(
            position.r,
            position.t,
            model=StageModel.from_geometry(geometry),
            column_tilt=self.parent.system.ion.column_tilt,
            turned_over=self.turned_over(position.t),
        )

    @query
    def is_close_to_milling_angle(
        self, milling_angle: float, atol: float = 2.0
    ) -> bool:
        """Whether the stage's milling angle is within ``atol`` of ``milling_angle``,
        both in degrees."""
        return bool(np.isclose(self.current_milling_angle(), milling_angle, atol=atol))

    # -- the geometry, in one read -------------------------------------------------------

    @query
    def geometry(self) -> Dict[str, Any]:
        """The instrument's geometry and what it means at the stage's pose now: one read
        for an agent meeting a new instrument, before its first move.

        Angles in degrees (named ``_deg``), lengths in metres. ``views`` gives, for each
        beam, the stage move a stable move of +1 µm along each image axis makes at the
        current pose and scan rotation, from the same projection ``stable_move`` uses;
        ``stage_plus_x_in_image`` is the image x direction a stage +x move slides the
        sample. These are the model's answers: confirm them once with a small move.
        """
        hardware = self.parent.hardware_geometry()
        position = self._position()
        model = StageModel.from_geometry(hardware)

        def pose(p: FibsemStagePosition) -> Dict[str, Optional[float]]:
            return {
                "r_deg": None if p.r is None else float(np.degrees(p.r)),
                "t_deg": None if p.t is None else float(np.degrees(p.t)),
            }

        answer: Dict[str, Any] = {
            "column_tilt_deg": {
                "electron": float(hardware.column_tilt),
                "ion": float(hardware.fib_column_tilt),
            },
            "shuttle_pre_tilt_deg": float(hardware.shuttle_pre_tilt),
            "rotation_reference_deg": float(hardware.rotation_reference),
            "rotation_centre": (
                None
                if hardware.rotation_centre is None
                else [float(c) for c in hardware.rotation_centre]
            ),
            "stage_frame": hardware.stage_frame,
            "fm_is_pose": hardware.fm_is_pose,
            "mirrors_back_view": hardware.mirrors_back_view,
            "orientations": {name: pose(p) for name, p in self.orientations.items()},
            "stage_position": {
                "x": position.x,
                "y": position.y,
                "z": position.z,
                **pose(position),
            },
        }
        if position.r is not None and position.t is not None:
            answer["orientation"] = self.stage_orientation(position)
            answer["turned_round"] = bool(model.turned_round(position.r))
            answer["turned_over"] = bool(self.turned_over(position.t))
            # Chamber frame: x the tilt axis, y horizontal, z up towards the electron
            # column. Where a raw stage +z move goes, at this tilt.
            plus_z = model.stage_axes(position.t) @ np.array([0.0, 0.0, 1.0])
            answer["stage_plus_z_in_chamber"] = [float(v) for v in plus_z]
            answer["milling_angle"] = {
                "current_deg": self.current_milling_angle(position),
                "stored_deg": self.read_milling_angle(),
                # MILLING is built at the SEM rotation: from here, that is where
                # move_to_milling_angle goes.
                "pose_for_stored": pose(self.orientations["MILLING"])
                if "MILLING" in self.orientations
                else None,
            }
            answer["views"] = {
                beam_type.name: self._view_geometry(beam_type)
                for beam_type in (BeamType.ELECTRON, BeamType.ION)
                if self._beam(beam_type) is not None
            }
        devices = self.parent.system.stage.devices or {}
        answer["devices"] = {
            name: None
            if device.origin is None
            else {"x": device.origin.x, "y": device.origin.y, "z": device.origin.z}
            for name, device in devices.items()
        }
        return answer

    def _view_geometry(self, beam_type: BeamType) -> Dict[str, Any]:
        """What one beam's view means for a move at the current pose."""
        step = 1e-6
        scan_rotation = self._beam_value(beam_type, "scan_rotation") or 0.0
        view_tilt = self._view_tilt(beam_type)

        def stage_move(dx: float, dy: float) -> Dict[str, float]:
            ux, uy = undo_scan_rotation(dx, dy, scan_rotation)
            delta = self._view_stage_delta(ux, uy, view_tilt=view_tilt)
            return {"x": float(delta.x), "y": float(delta.y), "z": float(delta.z)}

        handled = bool(
            np.isclose(scan_rotation, 0.0) or np.isclose(scan_rotation, np.pi)
        )
        return {
            "view_tilt_deg": float(np.degrees(view_tilt)),
            "scan_rotation_deg": float(np.degrees(scan_rotation)),
            # stable_move undoes a half turn of scan rotation and nothing else.
            "scan_rotation_handled": handled,
            "stable_move_plus_dx_1um": stage_move(step, 0.0),
            "stable_move_plus_dy_1um": stage_move(0.0, step),
            "stage_plus_x_in_image": "+x" if stage_move(step, 0.0)["x"] > 0 else "-x",
        }

    # -- devices: where the stage is, and a position carried between them -------------
    #
    # A device is a place the stage travels to for an instrument to see the sample
    # (the beams, an offset FM), configured under system.stage.devices; an
    # orientation is the pose it is held in once there.

    def _device(self, device: str) -> StageDeviceSettings:
        """The configuration for `device`, or a refusal naming the ones there are.
        An offset FM's origin is the SEM to FM calibration's once it is run
        (`StageSystemSettings.calibrated_device`)."""
        try:
            return self.parent.system.stage.calibrated_device(device)
        except KeyError:
            raise ValueError(
                f"Microscope {device} not supported. "
                f"Configured devices: {sorted(self.parent.system.stage.devices)}."
            ) from None

    @query
    def device_origin(self, device: str) -> FibsemStagePosition:
        """Where the stage travels for `device` to see the sample.

        The device's *origin*, not a position expressed at it -- for that, ask
        `get_target_position(position, target_device=...)`.

        Partial: an offset fluorescence microscope is an x location and leaves y, z, r
        and t free, so the axes it does not constrain come back `None`.
        """
        return deepcopy(self._device(device).origin)

    @query
    def is_at_device(
        self, device: str, stage_position: Optional[FibsemStagePosition] = None
    ) -> bool:
        """Is the stage at `device`?

        The question nothing could ask before. `get_stage_orientation` cannot answer
        it on an offset mount -- the FM orientation there is byte-identical to the FIB
        one -- because it is not a question about orientation at all.

        Only meaningful for a device the stage goes to. A device whose entry gives no
        origin has no stage position to be at, and is refused: `False` would read as
        "not there yet".

        Where every device shares one origin -- the objective under the grid, which
        is also what a configuration that declares no devices describes -- there is
        nowhere else to travel to, and the answer is `True` wherever the stage is.
        A device's `range` exists to tell places apart; with one place it tells
        nothing, and reading it literally refused a lamella 25 mm along an Arctis grid
        as "needs travel".

        Each device has its own range, and the beams' is unbounded, so the stage can be
        within more than one; it is at the one `current_device` picks, the nearest
        origin. Within the beams' range but nearer the FM's origin is at the FM. Where
        the stage device says a pose puts it at a device (`Stage.device_at_pose`), the
        pose decides and no range is checked.
        """
        target = self._device(device)
        if all(getattr(target.origin, axis) is None for axis in DEVICE_AXES):
            raise ValueError(
                f"The {device} device has no stage position (its entry gives no "
                "origin), so the stage cannot be at it."
            )
        if self._is_the_only_place(target):
            return True
        if stage_position is None:
            stage_position = self._position()
        # Where the pose says, the range does not: a compustage's FM reaches the whole
        # grid by moving the stage, so its stage limits bound it, not a window.
        by_pose = self._device_at_position_pose(stage_position)
        if by_pose is not None:
            return by_pose == device
        if not target.contains(stage_position):
            return False
        return self.current_device(stage_position) == device

    def _is_the_only_place(self, target: StageDeviceSettings) -> bool:
        """Does every device sit at *target*'s origin, so there is nowhere to travel?"""
        if all(getattr(target.origin, axis) is None for axis in DEVICE_AXES):
            return False
        origin = self._resolved_origin(target)
        return all(
            getattr(self._resolved_origin(self._device(name)), axis)
            == getattr(origin, axis)
            for name in self.parent.system.stage.devices
            for axis in DEVICE_AXES
        )

    def _resolved_origin(self, device: StageDeviceSettings) -> FibsemStagePosition:
        """*device*'s origin, with each axis it leaves unset taken from the beams'.

        An FM declared at `{x: 48.8 mm}` is at the beams' y, so a y offset on another
        device is travelled along from it rather than skipped.
        """
        beams = self.parent.system.stage.devices.get(BEAMS_STAGE_DEVICE)
        origin = deepcopy(device.origin)
        if beams is None:
            return origin
        for axis in DEVICE_AXES:
            if getattr(origin, axis) is None:
                setattr(origin, axis, getattr(beams.origin, axis))
        return origin

    @query
    def current_device(
        self, stage_position: Optional[FibsemStagePosition] = None
    ) -> Optional[str]:
        """Which device the stage is at, or `None` if it is at no configured device.

        Every device has its own range and the beams' is unbounded, so more than one
        can contain a position: an offset FM's window lies inside the beams'. Then the
        device with the **nearest origin** wins -- the stage at the FM is nearer the
        FM's origin than the beams' -- and a tie goes to the first configured. `None`
        means no device contains the position, which with the beams unbounded only a
        position missing an axis an origin sets can be.

        A stage that reaches a device by re-posing rather than travelling says so
        (`Stage.device_at_pose`), and then the pose decides: a compustage is at the FM
        in the FM pose and at the beams in any other, wherever x and y are.
        """
        if stage_position is None:
            stage_position = self._position()

        by_pose = self._device_at_position_pose(stage_position)
        if by_pose is not None:
            return by_pose

        devices = {
            name: self._device(name) for name in self.parent.system.stage.devices
        }
        containing = [
            name
            for name, device in devices.items()
            if self._is_the_only_place(device) or device.contains(stage_position)
        ]
        if not containing:
            return None
        return min(
            containing,
            key=lambda name: self._resolved_origin(devices[name]).euclidean_distance(
                stage_position
            ),
        )

    def _device_at_position_pose(
        self, stage_position: FibsemStagePosition
    ) -> Optional[str]:
        """`_device_at_pose` for the pose a position is in. A partial position (no r
        or t) has no pose to ask about."""
        if stage_position.r is None or stage_position.t is None:
            return None
        return self._device_at_pose(self.stage_orientation(stage_position))

    def _device_at_pose(self, orientation: str) -> Optional[str]:
        """The configured device the stage device says a pose puts it at, or None
        when its position decides (`Stage.device_at_pose`)."""
        if self.parent.stage_device is None:
            return None
        device = self.parent.stage_device.device_at_pose(orientation)
        return device if device in self.parent.system.stage.devices else None

    @query
    def device_imaging_state(
        self, device: str, stage_position: Optional[FibsemStagePosition] = None
    ) -> DeviceImagingState:
        """Can `device` see the sample from where the stage is -- and if not, why not.

        One question, answered the same way on both mountings, with the mounting
        expressed entirely in configuration. Two terms:

        * **place** -- `is_at_device`, against the device's declared origin
        * **pose** -- `get_stage_orientation`, against the device's declared
          `available_orientations`; an empty list constrains nothing and the term
          is vacuously true

        Each mounting makes a *different* term trivially true. A compustage FM shares
        the beams' origin, so the place carries nothing and the pose carries it all;
        an offset FM images from the pose the sample was carried out in, so the pose
        carries nothing and the place carries it all. Which is why the same
        conjunction discriminates on both -- and why the failing term names the
        remedy: a wrong place means travel, a wrong pose means re-pose.

        Callers act on the value by policy, not uniformly -- see
        `DeviceImagingState`. Pass `stage_position` to ask about a stored pose rather
        than the current one; both workflow tasks do.
        """
        # The fluorescence microscope is the one device whose instrument can be absent
        # -- the beams always exist, and there is no third device yet. `fm` is an object
        # or None (a present-but-faulted state is not modelled), so this is the whole of
        # the "no device" test.
        if device == "FM" and self.parent.fm is None:
            return DeviceImagingState.NO_DEVICE

        at_device = self.is_at_device(device, stage_position)
        orientations = self._device(device).available_orientations
        in_orientation = (
            not orientations or self.stage_orientation(stage_position) in orientations
        )

        if at_device and in_orientation:
            return DeviceImagingState.READY
        if in_orientation:
            return DeviceImagingState.NEEDS_TRAVEL
        if at_device:
            return DeviceImagingState.NEEDS_REPOSE
        return DeviceImagingState.NEEDS_REPOSE_THEN_TRAVEL

    @query
    def explain_imaging_state(
        self,
        device: str,
        state: Optional[DeviceImagingState] = None,
        stage_position: Optional[FibsemStagePosition] = None,
    ) -> str:
        """One sentence a person can act on, for each imaging state.

        The vocabulary is the two axes': the stage travels between
        *devices* and is re-posed between *orientations*, and the failing term names
        the verb -- so a refusal built from this says which of the two to do, in
        which order, rather than announcing a false generality like "not in a valid
        orientation" for a stage that is 48.8 mm from the instrument.

        Pass `state` when it has already been asked, so a message and the decision it
        explains cannot be about two different moments.
        """
        if state is None:
            state = self.device_imaging_state(device, stage_position)

        instrument = (
            "the fluorescence microscope" if device == "FM" else f"the {device} device"
        )

        if state is DeviceImagingState.NO_DEVICE:
            return "This system has no fluorescence microscope."
        if state is DeviceImagingState.READY:
            return f"{instrument.capitalize()} can image the sample from here."

        orientation = self.stage_orientation(stage_position)
        held = (
            f"held in the {orientation} orientation"
            if orientation != "NONE"
            else "held in an unrecognised orientation"
        )
        allowed = self._device(device).available_orientations
        images_from = (
            f"images from the {' or '.join(allowed)} orientation"
            if allowed
            else "images from any orientation"
        )

        if state is DeviceImagingState.NEEDS_TRAVEL:
            return (
                f"The stage is {held}, which {instrument} images from, but it is "
                f"away from the {device} device: travel there "
                f"(move_to_device('{device}'))."
            )
        if state is DeviceImagingState.NEEDS_REPOSE:
            return (
                f"The stage is at the {device} device but {held}; {instrument} "
                f"{images_from}. Re-pose via the beams "
                f"(move_to_device('{device}', orientation='{allowed[0]}'))."
            )
        return (
            f"The stage is {held}, away from the {device} device; {instrument} "
            f"{images_from}. Re-pose at the beams and travel out "
            f"(move_to_device('{device}'))."
        )

    def _device_translation(self, source: str, target: str) -> FibsemStagePosition:
        """The relative stage move from one device to another.

        A difference of two configured places rather than a constant, so the traverse
        and the "am I already there" windows cannot drift apart. Relative
        rather than absolute on purpose: the devices constrain x only, and a relative
        move carries y, z, r and t across unchanged.

        The same where the FM is a pose (a compustage): the stage turns the grid over
        to face the objective, and the FM's origin is the offset between the beams'
        coincidence point and the objective's centre, in stage coordinates at the FM
        pose. Zero unless one is configured.
        """
        source_origin = self._resolved_origin(self._device(source))
        target_origin = self._resolved_origin(self._device(target))

        translation = FibsemStagePosition()
        for axis in DEVICE_AXES:
            start, end = getattr(source_origin, axis), getattr(target_origin, axis)
            if start is not None and end is not None:
                setattr(translation, axis, end - start)
        return translation

    def _apply_device_translation(
        self, stage_position: FibsemStagePosition, source: str, target: str
    ) -> FibsemStagePosition:
        """`stage_position`, moved from one device to another. Mutates and returns it."""
        translation = self._device_translation(source, target)
        for axis in DEVICE_AXES:
            delta = getattr(translation, axis)
            if not delta:
                continue
            value = getattr(stage_position, axis)
            if value is None:
                raise ValueError(
                    f"Cannot convert between devices {source} and {target}: the "
                    f"position has no {axis}, and the two differ along it."
                )
            setattr(stage_position, axis, value + delta)
        return stage_position

    def _arrival_orientation(
        self,
        device: str,
        stage_position: FibsemStagePosition,
        orientation: Optional[str] = None,
    ) -> Optional[str]:
        """The orientation a position has to be re-posed into for `device`, or None.

        One rule, shared by the move (`move_to_device`) and the conversion
        (`to_device`) so the stage arrives where the conversion said it would. An
        explicit ask is honoured as asked. Otherwise the pose is kept whenever the
        device can image from it -- a traverse must not discard a tilt somebody
        dialled in -- and the device's first declared acquisition orientation stands
        in when it cannot. None means "keep the pose".
        """
        if orientation is not None:
            return orientation
        allowed = self._device(device).available_orientations
        if allowed and self.stage_orientation(stage_position) not in allowed:
            return allowed[0]
        return None

    @query
    def target_position(
        self,
        stage_position: FibsemStagePosition,
        target_orientation: Optional[str] = None,
        target_device: Optional[str] = None,
    ) -> FibsemStagePosition:
        """Convert a stage position across the orientation axis, the device axis, or both.

        Where the objective is offset the two are independent -- the device is a place
        the stage travels to, the orientation is the pose it is held in once there --
        so either may be asked for alone:

            (p, "FIB")                        re-pose, stay put
            (p, target_device="FM")           relocate, keep the pose
            (p, "MILLING", target_device="FM") both -- this is FM-MILLING, which is
                                              a pair rather than a fifth orientation

        **Asking for an orientation snaps the pose.** An orientation *is* a canonical
        r and t, so a position a degree or two off nominal is written to nominal
        rather than carried across. That is right for a conversion and wrong for a
        relocation, which is why the second form exists: the traverse must not discard
        a milling angle somebody dialled in on the way to the FM.

        Note `target_orientation="FM"` means something on a compustage, where the FM
        really is an orientation -- a flip to `t = -180`. On an offset mount it is
        refused: the FM is a device there, and `orientations["FM"]` is a copy of the
        FIB entry carrying no positional term.
        """

        currrent_orientation = self.stage_orientation(stage_position)
        logging.debug(
            f"Getting target position for {target_orientation} from {currrent_orientation}"
        )

        if target_orientation is None and target_device is None:
            return stage_position

        if currrent_orientation == target_orientation and target_device is None:
            return stage_position

        if currrent_orientation == "NONE":
            raise ValueError("Unknown orientation. Cannot convert stage position.")

        # The FM is an orientation only where the objective is under the grid. On an
        # offset mount it is a *place*, reached by translating rather than re-posing,
        # and `orientations["FM"]` there is a copy of the FIB entry carrying no
        # positional term -- so converting into it would return a position under the
        # beam wearing the FM's rotation and tilt. Ask for it as a device instead:
        # `target_device="FM"`, with whichever orientation the sample should be in.
        if "FM" in (currrent_orientation, target_orientation) and (
            not self.fm_is_a_pose()
        ):
            raise ValueError(
                "Cannot move to the FM position: this stage reaches the FM by "
                "travelling, not by re-posing."
            )

        # Which device the stage is at, read *before* either leg runs. The orientation
        # leg below can move x and y most of the way across the grid, so asking
        # afterwards would sometimes name a different device, or none.
        source_device = self.current_device(stage_position)

        # Where a pose puts the stage at a device, asking for the pose is asking for
        # the device: on a compustage, MILLING from the FM is back at the beams, and
        # FM from the beams is at the FM, offset by its origin.
        if target_device is None and target_orientation is not None:
            target_device = self._device_at_pose(target_orientation)

        if target_device is not None and source_device is None:
            raise ValueError(
                f"The stage is not at any configured device "
                f"({sorted(self.parent.system.stage.devices)}), so there is nothing to "
                f"convert from. Position: {stage_position}."
            )

        stage_position = deepcopy(stage_position)
        orientation = (
            self.orientation(target_orientation)
            if target_orientation is not None
            else None
        )

        # The device legs **bracket** the orientation leg rather than following it,
        # so that the re-pose always happens at the beams. Written out, that is the
        # order an operator would use:
        #
        #   leaving the beams:  re-pose first, then traverse to the device
        #   returning:          traverse back to the beams first, then re-pose
        #
        # Both come out of one expression, `f(p) = rotate(p - source) + target`.
        # Leaving, `source` is zero and the rotation happens before the traverse;
        # returning, `target` is zero and the traverse happens before the rotation.
        #
        # **The stage must not be re-posed while it is parked at the FM.** The
        # objective is inserted over the sample there, and a half turn swings the
        # sample about a centre ~48.8 mm away. Bracketing is what keeps the rotation
        # out of that pose; the arrangement that re-poses wherever the stage happens
        # to be would compute a target that commands exactly that move.
        #
        # It is also the only arrangement that round-trips, which is how the wrong one
        # was caught rather than reasoned about: re-posing at the FM and translating
        # afterwards sends `(SEM, beams) -> (FIB, FM) -> (SEM, beams)` back to
        # -97.6 mm instead of 0. `rotate` is an involution, so `f` inverts by swapping
        # source and target -- which is exactly what the reverse call does.
        #
        # This computes a target; it does not order the moves. Nothing yet stops a
        # caller re-posing while at the FM --, a prerequisite for opening
        # the connection gate.
        #
        # A compustage brackets with a zero translation, so it takes this path and
        # gets its old answer untouched.
        if source_device is not None:
            stage_position = self._apply_device_translation(
                stage_position, source_device, ROTATION_FRAME_DEVICE
            )

        # The re-pose, at the beams: r and t become the target's, and x and y are
        # carried round the compucentric centre only when the rotation between the two
        # *poses* is a half turn (fibsem.geometry.orientations.repose says why).
        # Relocating alone changes no rotation, so with no pose asked for the bracket
        # collapses to a plain translation from one device to the other.
        if orientation is not None:
            stage_position = repose(
                stage_position,
                current=self.orientation(currrent_orientation),
                target=orientation,
                rotation_centre=self.parent._compucentric_centre(),
                has_rotation=bool(self.parent.system.stage.rotation),
            )

        # ... and back out, to whichever device was asked for. `target_device` of None
        # means the one it started at, so an orientation-only conversion at the FM is
        # still re-posed about the right centre rather than about wherever it is
        # parked.
        if source_device is not None:
            stage_position = self._apply_device_translation(
                stage_position, ROTATION_FRAME_DEVICE, target_device or source_device
            )

        return stage_position

    @query
    def to_device(
        self,
        stage_position: FibsemStagePosition,
        device: str,
        orientation: Optional[str] = None,
    ) -> FibsemStagePosition:
        """*stage_position* as `device` sees it: where `move_to_device` would arrive.

        The one spelling of "this piece of sample, at that instrument" -- a lamella's
        fluorescence pose from its milling pose, a grid slot on the FM canvas, a
        milling pose from a target found in fluorescence. The same on both mountings:
        a compustage flips and travels by its FM origin (zero unless configured), an
        offset mount gets the traverse.

        `orientation` names the pose to arrive in; omitted, `_arrival_orientation`
        decides -- kept if the device images from it, else the first it declares.

        Raises:
            ValueError: from `get_target_position` -- a position in no supported
                orientation (`"NONE"`), or at no configured device, has no conversion.
        """
        return self.target_position(
            deepcopy(stage_position),
            target_orientation=self._arrival_orientation(
                device, stage_position, orientation
            ),
            target_device=device,
        )

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
        """Write a working distance with no new checks, or nothing
        where the beam has none (the TESCAN ion column)."""
        beam = self._beam(beam_type)
        param = None if beam is None else beam.parameters.get("working_distance")
        if param is None:
            logging.debug(f"No {beam_type.name} working distance to set.")
            return
        param.write_through(wd)

    def _position(self) -> FibsemStagePosition:
        """The stage position, read the way the microscope reads it, so
        ``stage.position.changed`` still says where the stage went."""
        return self.parent.get_stage_position()

    def _move_stage(
        self, position: FibsemStagePosition, relative: bool = False
    ) -> FibsemStagePosition:
        """One stage move, refused outside the stage limits.

        Through the microscope's raw move, which goes to the stage device's checked
        move: it is the one place a backend's raw stage move is recorded and its
        position published, and what every existing pin of these moves records.
        """
        self._refuse_moving_with_the_objective_in(position, relative)
        if relative:
            return self.parent.move_stage_relative(position)
        return self.parent.move_stage_absolute(position)

    def _linked(self) -> Optional[bool]:
        param = self.stage.parameters.get("linked")
        return None if param is None else param.get_value()

    def _view_tilt(self, beam_type: BeamType) -> float:
        """Tilt of a beam column's viewing axis from the electron column, in radians."""
        return beam_view_tilt(beam_type, self.parent.hardware_geometry())

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

    # -- a displacement seen in a view, as a stage movement, and back ------------------
    #
    # The sample sits on the shuttle pre-tilt, so a y-displacement in an image moves it
    # along the holder, split across the stage y- and z-axes. Every view shares one
    # projection and differs only in how far its axis is tilted from the electron
    # column: 0 for the electron beam, the column tilt for the ion beam, the camera tilt
    # for the FM. Taken at the current pose; nothing moves.

    @query
    def view_corrected_movement(
        self, expected_y: float, view_tilt: float = 0.0
    ) -> FibsemStagePosition:
        """The relative y/z stage movement for a y-displacement (metres) in a view
        tilted ``view_tilt`` radians from the electron column."""
        delta = self._view_stage_delta(0, expected_y, view_tilt=view_tilt)
        return FibsemStagePosition(x=0, y=delta.y, z=delta.z)

    @query
    def y_corrected_movement(
        self, expected_y: float, beam_type: BeamType = BeamType.ELECTRON
    ) -> FibsemStagePosition:
        """`view_corrected_movement` in a beam's view."""
        return self.view_corrected_movement(
            expected_y, view_tilt=self._view_tilt(beam_type)
        )

    @query
    def inverse_view_corrected_movement(
        self, dy: float, dz: float, view_tilt: float = 0.0
    ) -> float:
        """The y-displacement in the view that a y/z stage movement corresponds to.

        The inverse of `view_corrected_movement`, through the
        `fibsem.geometry.movement.inverse_view_corrected_dy` the saved-image
        reprojection shares.
        """
        position = self._position()
        return inverse_view_corrected_dy(
            dy=dy,
            dz=dz,
            view_tilt=view_tilt,
            geometry=self.parent.hardware_geometry(),
            stage_rotation=position.r if position.r is not None else 0.0,
            stage_tilt=position.t if position.t is not None else 0.0,
        )

    @query
    def inverse_y_corrected_movement(
        self, dy: float, dz: float, beam_type: BeamType = BeamType.ELECTRON
    ) -> float:
        """`inverse_view_corrected_movement` in a beam's view."""
        return self.inverse_view_corrected_movement(
            dy, dz, view_tilt=self._view_tilt(beam_type)
        )

    # -- moves the stage device has itself --------------------------------------------------

    @command
    def move_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """Move the stage to a position, within its limits. Axes that are None stay."""
        with recording_move(
            self.session, self.stage, "move_absolute", position=position
        ):
            self._refuse_moving_with_the_objective_in(position)
            self.stage.move_absolute(position)
            return self._position()

    @command
    def move_relative(self, delta: FibsemStagePosition) -> FibsemStagePosition:
        """Move the stage by an offset, within its limits. Axes that are None stay."""
        with recording_move(self.session, self.stage, "move_relative", position=delta):
            self._refuse_moving_with_the_objective_in(delta, relative=True)
            self.stage.move_relative(delta)
            return self._position()

    # -- moves by what a view shows ------------------------------------------------------

    @command
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
        with recording_move(
            self.session,
            self.stage,
            "stable_move",
            dx=dx,
            dy=dy,
            beam_type=beam_type,
            static_wd=static_wd,
        ):
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

    @query
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
        with recording_move(
            self.session,
            self.stage,
            "vertical_move",
            dy=dy,
            dx=dx,
            beam_type=beam_type,
            relaxation=relaxation,
        ):
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
            column_tilt=np.deg2rad(self.parent.system.ion.column_tilt),
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
        """The FM device, or None: the microscope's FM's own, since every FM is built
        over its devices. Found the first time it is asked for, because some backends
        build their FM after the stage."""
        fm = self._roles.get("fm")
        if fm is None and getattr(self.parent, "fm", None) is not None:
            fm = self.parent.fm.devices["fm"]
            self.fill_roles(fm=fm)
        return fm

    def _require_fm(self, message: str) -> FM:
        fm = self._fm_device()
        if fm is None:
            raise ValueError(message)
        return fm

    @property
    def camera_tilt(self) -> float:
        """Tilt of the FM camera's optical axis from the SEM column, in degrees.

        The FM entry's ``camera_tilt`` where the configuration states one,
        else derived from the mount (`fibsem.geometry.frames.fm_camera_tilt`).
        """
        return fm_camera_tilt(self.parent)

    def _display_transform(self, fm: FM) -> CameraImageTransform:
        """The transform the user displays FM images under, held by the camera."""
        camera = fm._roles["camera"]
        return mount_transform_from_name(
            camera.parameters["display_transform"].get_value()
        )

    def _objective_state(self, fm: FM) -> Optional[InsertableDeviceState]:
        objective = fm._roles.get("objective")
        if objective is None:
            return None
        return objective.state.get_value()

    def _fm_stage_delta(self, fm: FM, dx: float, dy: float) -> FibsemStagePosition:
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

    @query
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
            if state is not InsertableDeviceState.INSERTED:
                logging.warning(
                    "Moving via the fluorescence image while the objective is not "
                    f"inserted (state: {state.value}); the view may not match the sample."
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

    # -- the axes a move leaves alone -------------------------------------------------------

    @query
    def blocked_axes(
        self, position: Optional[FibsemStagePosition] = None
    ) -> Tuple[str, ...]:
        """The stage axes an absolute move to *position* leaves alone, because the
        microscope would refuse them.

        Two halves, and they are not the same rule.

        The **orientation** half asks where the move is *going*, not where the stage is
        standing, and blocks z (and r, which a stage with an FM pose does not have).
        Asking the current pose would drop z from the very move that leaves the
        fluorescence pose. Measured on the Arctis: with the objective retracted, z and t
        are both available at t = -180, so a move out of that pose has nothing to lose.
        A move *into* it is still restricted here -- more conservative than the
        measurement requires, and free, because the one path that enters the pose sends
        r and t only.

        `position` is the destination. Without one, or with a partial pose that has no
        orientation to read (the safe move's rotation step sends a bare tilt), the stage's
        own pose stands in -- what such a move got before.

        `stage_orientation` can never return "FM" on an offset mount -- the FM is a
        device there and `orientations["FM"]` is a copy of the FIB entry -- so that half
        is naturally confined to the stages that declare an FM pose.

        The **objective** half: while the objective is inserted, the axes it names
        (`ObjectiveLens.blocked_axes`, z and t as measured) are left alone. Only a
        driver whose absolute move asks (`without_blocked_axes`, AutoScript's) drops
        them; an odemis stage moves z with the objective in, as an FM stable move
        must. The objective state is read live, every move: a stale
        "Retracted" here moves the stage with the objective in the chamber.
        """
        blocked = []
        destination = (
            position
            if position is not None
            and position.r is not None
            and position.t is not None
            else None
        )
        if self.stage_orientation(destination) == "FM":
            blocked += ["z", "r"]

        fm = self.parent.fm
        if fm is not None and fm.objective.state is InsertableDeviceState.INSERTED:
            blocked += [a for a in fm.objective.blocked_axes if a not in blocked]
        return tuple(blocked)

    def without_blocked_axes(
        self, position: FibsemStagePosition
    ) -> FibsemStagePosition:
        """*position* with the axes `blocked_axes` names left unset, so an absolute
        move does not send them. A driver whose instrument refuses those axes calls this
        in its stage's absolute move."""
        blocked = self.blocked_axes(position)
        if not blocked:
            return position
        position = deepcopy(position)
        for axis in blocked:
            setattr(position, axis, None)
        return position

    def _refuse_rotation_at_the_fm(self, stage_position: FibsemStagePosition) -> None:
        """Refuse a stage rotation while the stage is parked at the FM.

        The objective is inserted over the sample there, and the rotation is
        compucentric about a centre back at the beams -- some 48.8 mm away -- so a half
        turn swings the sample most of the width of the chamber, under the objective.

        The route to another pose at the FM is the one `target_position` computes:
        traverse back to the beams, re-pose there, traverse out again. This refuses the
        shortcut. It is a refusal rather than a silent correction because a caller
        asking for the shortcut has a wrong idea of where the stage is going, and
        quietly sending it somewhere else would leave that idea intact.

        **Rotation only.** A tilt pivots about an axis through the sample instead of
        swinging it, and where the objective does restrict tilt the microscope refuses
        it itself (z and t, measured). This does not duplicate that.

        Not a restriction on FM-MILLING. That pose is a half turn from the FM's own
        orientation (measured: FM sits at r=180, MILLING at r=0), so it was never
        reachable by rotating in place -- it is reached the way everything else at the
        FM is, via the beams.
        """
        # A stage that reaches the FM by re-posing (a compustage, flipping) has its
        # devices at one place, so "parked at the FM" is not a state it can be in --
        # and it has no rotation axis to be compucentric about either.
        if self.fm_is_a_pose() or self.parent.fm is None:
            return

        if stage_position.r is None:
            return

        current_position = self._position()
        if self.current_device(current_position) != "FM":
            return

        # The same 5 degrees `stage_orientation` classifies within, so a caller asking
        # for the pose the stage is already in does not trip this on the slop a real
        # stage always carries.
        if rotation_angle_is_smaller(stage_position.r, current_position.r, atol=5):
            return

        raise ValueError(
            "Cannot rotate the stage while it is at the fluorescence microscope: the "
            "rotation is compucentric about a centre at the beams, so it would swing "
            "the sample across the chamber under the objective. Move to the beams "
            "first (move_to_device('FIBSEM')), re-pose there, and travel back -- "
            "or ask move_to_device for the pose and let it order the legs."
        )

    def _refuse_moving_with_the_objective_in(
        self, position: FibsemStagePosition, relative: bool = False
    ) -> None:
        """Refuse one stage move that would leave the current device, or rotate at the
        FM, while the FM objective is inserted.

        Every move in this service reaches the stage through here, so a relative move
        or a saved position cannot carry the inserted objective across the chamber.
        `move_to_device` and the safe move retract the objective before they travel,
        so their own legs pass. The objective state is read live, every move. Moves
        within the FM's range (tiles, focus, stable moves) are left alone, and a stage
        that reaches the FM by re-posing has `blocked_axes` instead.
        """
        fm = self.parent.fm
        if fm is None or self.fm_is_a_pose():
            return
        if fm.objective.state is not InsertableDeviceState.INSERTED:
            return

        current = self._position()
        if relative:
            target = current + position
        else:
            target = deepcopy(position)
            for axis in DEVICE_AXES + ("r", "t"):
                if getattr(target, axis) is None:
                    setattr(target, axis, getattr(current, axis))

        source = self.current_device(current)
        rotates = (
            source == "FM"
            and target.r is not None
            and current.r is not None
            and not rotation_angle_is_smaller(target.r, current.r, atol=5)
        )
        leaves = self.current_device(target) != source
        if not (rotates or leaves):
            return
        raise ValueError(
            "The fluorescence objective is inserted, and this move would "
            + (
                "rotate the stage under it"
                if rotates
                else f"take the stage away from the {source}"
            )
            + f" (to {target}). Retract the objective, or use move_to_device, which "
            "retracts it before travelling."
        )

    # -- saved positions -------------------------------------------------------------------
    # Named stage positions, kept with the microscope configuration's session state
    # (`fibsem.saved_positions`): what the Saved Positions panel lists.

    def _saved_state(self, writable: bool = False):
        from fibsem.session_state import session_state_for

        return session_state_for(self.parent, writable=writable)

    @staticmethod
    def _named(positions: List[FibsemStagePosition], name: str) -> FibsemStagePosition:
        for position in positions:
            if position.name == name:
                return position
        raise ValueError(
            f"No saved position named {name!r}; saved: {[p.name for p in positions]}."
        )

    def _write_saved(self, positions: List[FibsemStagePosition]) -> None:
        from fibsem.saved_positions import save_saved_positions

        save_saved_positions(positions, self._saved_state(writable=True))

    @query
    def saved_positions(self) -> List[FibsemStagePosition]:
        """The saved stage positions, in the order they were saved."""
        from fibsem.saved_positions import load_saved_positions

        return load_saved_positions(self._saved_state())

    @command
    def save_position(
        self, name: str, stage_position: Optional[FibsemStagePosition] = None
    ) -> FibsemStagePosition:
        """Save the stage position now, or *stage_position*, as *name*: one
        already saved under that name is replaced where it stands, otherwise
        it is added at the end."""
        name = str(name or "").strip()
        if not name:
            raise ValueError("A saved position needs a name.")
        position = deepcopy(stage_position or self._position())
        position.name = name
        positions = self.saved_positions()
        index = next((i for i, p in enumerate(positions) if p.name == name), None)
        if index is None:
            positions.append(position)
        else:
            positions[index] = position
        self._write_saved(positions)
        return position

    @command
    def rename_saved_position(self, name: str, new_name: str) -> FibsemStagePosition:
        """Rename a saved position; refused onto a name already saved."""
        new_name = str(new_name or "").strip()
        if not new_name:
            raise ValueError("A saved position needs a name.")
        positions = self.saved_positions()
        position = self._named(positions, name)
        if new_name != name and any(p.name == new_name for p in positions):
            raise ValueError(f"A position named {new_name!r} is already saved.")
        position.name = new_name
        self._write_saved(positions)
        return position

    @command
    def remove_saved_position(self, name: str) -> None:
        """Forget a saved position."""
        positions = self.saved_positions()
        self._named(positions, name)
        self._write_saved([p for p in positions if p.name != name])

    @command
    def move_to_saved_position(self, name: str) -> FibsemStagePosition:
        """Move the stage to a saved position, through the safe absolute move."""
        return self.safe_absolute_stage_movement(
            self._named(self.saved_positions(), name)
        )

    # -- moves to a place ------------------------------------------------------------------

    @command
    def safe_absolute_stage_movement(
        self, stage_position: FibsemStagePosition
    ) -> FibsemStagePosition:
        """Move the stage to a position safely: tilted flat for a large rotation, and
        rotated compucentrically before the rest of the move. A position at another
        device on an offset mount goes by `move_to_device`'s route. Returns where it
        is, as every move does."""
        with recording_move(
            self.session,
            self.stage,
            "safe_absolute_stage_movement",
            stage_position=stage_position,
        ):
            crossing = self._device_crossing(stage_position)
            if crossing is not None:
                self._move_across_devices(stage_position, *crossing)
                return self._position()

            # Before anything moves.
            self._refuse_rotation_at_the_fm(stage_position)
            self._safe_rotation(stage_position)

            logging.debug(f"safe moving to {stage_position}")
            self._move_stage(stage_position)
            logging.debug("safe movement complete.")
            return self._position()

    def _device_crossing(
        self, stage_position: FibsemStagePosition
    ) -> Optional[Tuple[str, str]]:
        """(source, target) when *stage_position* is at another device the stage
        travels to, else None. Never on a compustage, whose devices are poses."""
        if self.fm_is_a_pose() or self.parent.fm is None:
            return None
        source = self.current_device(self._position())
        target = self.current_device(stage_position)
        if source is None or target is None or source == target:
            return None
        if not _moves(self._device_translation(source, target)):
            return None
        return source, target

    def _move_across_devices(
        self, stage_position: FibsemStagePosition, source: str, target: str
    ) -> None:
        """Position at the beams with the objective out, then travel.

        One absolute move would cross the chamber with the objective wherever it was,
        and re-pose at the far end. Instead every re-pose happens at the beams and the
        legs between devices are `move_to_device`'s pure traverses.
        """
        logging.info(f"{stage_position} is at {target}; moving there from {source}.")
        if source != ROTATION_FRAME_DEVICE:
            self.move_to_device(ROTATION_FRAME_DEVICE)
        if target == ROTATION_FRAME_DEVICE:
            self.safe_absolute_stage_movement(stage_position)
            return
        self._retract_objective_to_move(target)
        self.safe_absolute_stage_movement(
            self.target_position(stage_position, target_device=ROTATION_FRAME_DEVICE)
        )
        self.move_to_device(target)
        # The traverse arrives where the conversion says; this takes up what is left.
        self.safe_absolute_stage_movement(stage_position)

    def _safe_rotation(self, stage_position: FibsemStagePosition) -> None:
        """What comes before the move itself: on a stage with a rotation axis, tilt
        flat for a large rotation, then rotate compucentrically. A driver whose
        instrument does this itself replaces it."""
        # The safe sequence is about rotating, so a stage with no rotation axis (a
        # compustage) skips it, and so does a target with no rotation (an x/y-only
        # move), which leaves r where it is. `rotation` is `"r" in` the stage's axes.
        if not self.parent.system.stage.rotation or stage_position.r is None:
            return

        current_position = self._position()
        if rotation_angle_is_larger(stage_position.r, current_position.r):
            self._move_stage(FibsemStagePosition(t=0))
            logging.info("tilting to flat for large rotation.")

        self._move_stage(FibsemStagePosition(r=stage_position.r))

    @command
    def move_to_orientation(self, orientation: str) -> FibsemStagePosition:
        """Move the stage to a named orientation ('SEM', 'FIB', 'MILLING', ...)."""
        with recording_move(
            self.session, self.stage, "move_to_orientation", orientation=orientation
        ):
            stage_position = self._orientation_move_target(orientation)
            self.safe_absolute_stage_movement(stage_position)
            return self._position()

    def _orientation_move_target(self, orientation: str) -> FibsemStagePosition:
        """Where a move to a named orientation ends: its pose (r, t), and on a stage
        that turns a half turn, x and y carried round the rotation centre.

        Only where the driver reports its ``rotation_centre``. Without one,
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
    def move_to_milling_angle(
        self,
        milling_angle: float,
        rotation: Optional[float] = None,
        keep_rotation: bool = False,
    ) -> bool:
        """Move the stage to a milling angle, in degrees like the service's other
        milling-angle reads, from the pretilt and the column tilt. Returns whether
        the stage is close to it afterwards.

        Args:
            milling_angle: the milling angle, in degrees.
            rotation: the stage rotation, in degrees. If None, the rotation
                reference, or the stage's current rotation with ``keep_rotation``
                (as the agent's tool asks: a tilt it asked for should not turn
                the stage).
            keep_rotation: with no ``rotation``, stay at the current rotation.
        """
        microscope = self.parent
        if rotation is None and keep_rotation:
            rotation = float(np.degrees(self._position().r))
        elif rotation is None:
            rotation = microscope.system.stage.rotation_reference
        # The move event keeps radians, the stage's own unit, as it always has.
        with recording_move(
            self.session,
            self.stage,
            "move_to_milling_angle",
            milling_angle=float(np.radians(milling_angle)),
            rotation=float(np.radians(rotation)),
        ):
            from fibsem.transformations import get_stage_tilt_from_milling_angle

            stage_tilt = get_stage_tilt_from_milling_angle(
                microscope, np.radians(milling_angle)
            )
            stage_tilt = self._tilt_for_milling_angle(
                milling_angle, np.radians(rotation), stage_tilt
            )
            self.safe_absolute_stage_movement(
                FibsemStagePosition(t=stage_tilt, r=float(np.radians(rotation)))
            )
            return microscope.is_close_to_milling_angle(milling_angle)

    def _tilt_for_milling_angle(
        self, milling_angle: float, rotation: float, sem_tilt: float
    ) -> float:
        """The stage tilt, in radians, that gives ``milling_angle`` (degrees) at
        ``rotation`` (radians). At the SEM rotation that is ``sem_tilt``, the
        tilt-plane formula's answer; elsewhere the pre-tilt leans another way, so the
        tilt is searched for with the same model ``current_milling_angle`` reads,
        between -90 and 90 degrees, nearest ``sem_tilt`` on a tie."""
        reference = np.radians(self.parent.system.stage.rotation_reference)
        if np.isclose(wrap_angle(rotation - reference), 0.0):
            return sem_tilt
        tilts = np.radians(np.arange(-90.0, 90.0001, 0.05))
        errors = np.array(
            [
                abs(
                    self.current_milling_angle(FibsemStagePosition(r=rotation, t=t))
                    - milling_angle
                )
                for t in tilts
            ]
        )
        best = np.flatnonzero(np.isclose(errors, errors.min(), atol=1e-6))
        return float(tilts[best[np.argmin(np.abs(tilts[best] - sem_tilt))]])

    def _objective(self) -> Any:
        """The objective to insert and retract, through the FM API, which tells the
        displays the objective moved (`ObjectiveLens.position_changed`)."""
        self._require_fm("No fluorescence microscope.")
        return self.parent.fm.objective

    @command
    def insert_objective(self, tilt: Optional[float] = None) -> None:
        """Insert the FM objective where the stage stands, or tilted to *tilt*
        (radians) first, as the FM tab's Insert does: the objective is retracted
        for the tilt, and a stage already within half a degree of it stays put.
        A tilt is refused away from the FM device."""
        with recording_move(self.session, self.stage, "insert_objective", tilt=tilt):
            objective = self._objective()
            if tilt is not None:
                stage_position = self._position()
                if not self._at_the_fm(stage_position):
                    raise ValueError(
                        "The stage is not at the FM, so it is not tilted there: "
                        "travel with move_to_device('FM'), which inserts the "
                        "objective."
                    )
                if not np.isclose(
                    np.degrees(stage_position.t or 0.0),
                    np.degrees(tilt),
                    atol=SAFE_INSERT_TILT_TOLERANCE_DEG,
                ):
                    self._retract_objective_to_move("FM")
                    stage_position.t = float(tilt)
                    self._move_stage(stage_position)
            objective.insert()

    @command
    def retract_objective(self) -> None:
        """Retract the FM objective, as the FM tab's Retract does."""
        with recording_move(self.session, self.stage, "retract_objective"):
            self._objective().retract()

    def _at_the_fm(self, stage_position: FibsemStagePosition) -> bool:
        """Whether the stage is at the FM; True where the FM is no device the
        stage travels to."""
        if "FM" not in self.parent.system.stage.devices:
            return True
        try:
            return self.is_at_device("FM", stage_position)
        except ValueError:
            return True

    @query
    def travels_to_fm(self) -> bool:
        """Is the FM a place the stage travels to (an offset mount)?"""
        return (
            self.parent.fm is not None
            and "FM" in self.parent.system.stage.devices
            and not self.fm_is_a_pose()
        )

    def fm_calibration_needed(self) -> bool:
        """Is travel to the FM refused until SEM to FM is calibrated? True on an
        offset mount until a calibration holds for the fitted holder; never on a
        compustage or a system with no FM."""
        return self.travels_to_fm() and not self.parent.system.stage.fm_calibrated()

    def _refuse_uncalibrated_fm_travel(self) -> None:
        if self._calibrating or not self.fm_calibration_needed():
            return
        stage = self.parent.system.stage
        calibration = stage.device_calibration
        if calibration is not None:
            raise ValueError(
                f"Recheck SEM to FM first: it was calibrated on the holder "
                f"'{calibration.holder}', and '{stage.active_holder}' is fitted. "
                "Travel to the fluorescence microscope stays off until then."
            )
        raise ValueError(
            "Calibrate SEM to FM first: travel to the fluorescence microscope stays "
            "off until the SEM to FM calibration is run for this holder."
        )

    @contextmanager
    def calibrating(self) -> Iterator[None]:
        """Travel to the FM by its nominal origin while the SEM to FM calibration
        measures it: the one travel the gate lets through."""
        self._calibrating = True
        try:
            yield
        finally:
            self._calibrating = False

    @query
    def plan_move_to_device(
        self, device: str, orientation: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """The steps `move_to_device` would take, in order, with where each one ends,
        without moving: what a confirmation lists before a travel. Each step has
        ``step`` (retract, turn, travel or insert), ``text`` (one line for an
        operator) and its values. Refuses what `move_to_device` refuses."""
        return [step.to_dict() for step in self._route(device, orientation)]

    def _route(self, device: str, orientation: Optional[str]) -> List[RouteStep]:
        """`move_to_device`'s steps: every check it makes before anything moves, then
        the moves. One list for the move and for `plan_move_to_device`, so a
        confirmation can't list a route the stage doesn't take."""
        target_device = self._device(device)  # refuses by name

        if device == "FM" and self._fm_device() is None:
            raise ValueError("FM module is not available. Cannot move to FM position.")

        stage_position = self._position()
        source = self.current_device(stage_position)
        if source is None:
            raise ValueError(
                f"The stage is not at any configured device "
                f"({sorted(self.parent.system.stage.devices)}), so there is nothing to "
                f"travel from. Position: {stage_position}."
            )

        # The pose to arrive in. An explicit ask is honoured as asked; otherwise the
        # pose is carried across, unless the target device cannot image from it --
        # then its first declared acquisition orientation stands in.
        desired = self._arrival_orientation(device, stage_position, orientation)
        if desired is not None and orientation is None:
            logging.info(
                f"The {device} device images from "
                f"{target_device.available_orientations}; re-posing to {desired} "
                f"at the beams before travelling."
            )

        if device == "FM" and source != "FM":
            self._refuse_uncalibrated_fm_travel()

        steps: List[RouteStep] = []
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
                    at_the_beams = self.target_position(
                        stage_position, desired, target_device=ROTATION_FRAME_DEVICE
                    )
                except ValueError as e:
                    logging.warning(
                        f"Re-posing to {desired} without keeping the sample point: {e}"
                    )
                    at_the_beams = None

                steps.append(self._retract_step())
                steps += self._travel_steps(
                    source, ROTATION_FRAME_DEVICE, stage_position
                )
                pose = self.orientation(desired)
                steps.append(
                    RouteStep(
                        "turn",
                        orientation=desired,
                        position=at_the_beams,
                        r=pose.r,
                        t=pose.t,
                    )
                )
                steps += self._travel_steps(ROTATION_FRAME_DEVICE, device, at_the_beams)
            elif _moves(self._device_translation(source, device)):
                steps.append(self._retract_step())
                steps += self._travel_steps(source, device, stage_position)

        # Unconditional, so that the postcondition is the device *and* the objective
        # state together: asking again for a device the stage is already at cannot
        # leave the FM blind.
        if device == "FM":
            steps.append(
                RouteStep("insert", objective_z=self._objective().insert_position)
            )
        return steps

    def _retract_step(self) -> RouteStep:
        fm = self._fm_device()
        return RouteStep(
            "retract",
            objective_z=None if fm is None else self._objective().retract_position,
            moves_objective=fm is not None,
        )

    def _travel_steps(
        self, source: str, target: str, start: Optional[FibsemStagePosition]
    ) -> List[RouteStep]:
        """The traverse from one device to another, or none where they share a place.
        Ends at *start* moved by it, where *start* is known."""
        translation = self._device_translation(source, target)
        if not _moves(translation):
            return []
        return [
            RouteStep(
                "travel",
                device=target,
                translation=translation,
                position=None if start is None else start + translation,
            )
        ]

    @command
    def move_to_device(self, device: str, orientation: Optional[str] = None) -> None:
        """Travel to `device`, re-posing on the way when the pose has to change.

        One call that owns the safe order -- retract the objective, re-pose at the
        beams, travel out -- so a rotation never happens with the stage parked under
        an objective. The rotation guard stays underneath as the last-line
        assert; the route this composes never trips it. `plan_move_to_device` lists
        the same route without moving.

        `orientation` names the pose to arrive in. Omitted, the pose is carried
        across untouched whenever the target device can image from it -- that is the
        point of a traverse, and the reason this must not pass the current
        orientation's *name* through `move_to_orientation`: doing so would snap r and
        t to nominal and quietly discard a milling angle somebody dialled in. When
        the pose does have to change (an offset FM images in the FIB pose; asking
        for it from SEM), the device's first declared
        acquisition orientation is used.
        """
        with recording_move(
            self.session,
            self.stage,
            "move_to_device",
            device=device,
            orientation=orientation,
        ):
            for step in self._route(device, orientation):
                if step.step == "retract":
                    self._retract_objective_to_move(device)
                elif step.step == "travel":
                    self._move_stage(step.translation, relative=True)
                elif step.step == "turn":
                    if step.position is not None:
                        self.safe_absolute_stage_movement(step.position)
                    else:
                        self.move_to_orientation(step.orientation)
                elif step.step == "insert":
                    self._objective().insert()

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
            return self.target_position(
                deepcopy(stage_position), desired, target_device=device
            )
        except ValueError as e:
            if desired is not None:
                logging.warning(f"Not checking where {device} will be reached: {e}")
                return None
            source = self.current_device(stage_position)
            return stage_position + self._device_translation(source, device)

    def _check_arrival(self, device: str, arrival: FibsemStagePosition) -> None:
        """Refuse a traverse that would arrive outside *device*'s range.

        The start is checked by finding the source (`current_device`); this is the
        other end. Each device has its own range, and the stage keeps its offset from
        the source's origin, so a position well inside the beams' range can land
        outside an FM's -- where the stage would report it had not arrived, and could
        not go back. Checked before the stage moves, and before the objective is
        retracted for it.
        """
        if not self.is_at_device(device, arrival):
            raise ValueError(
                f"Travelling to {device} from here would arrive at {arrival}, outside "
                f"its range {self._device(device).range} of its origin "
                f"{self._device(device).origin}. Move the stage nearer the "
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
    if getattr(microscope, "fm", None) is not None:
        movement.fill_roles(fm=microscope.fm.devices["fm"])
    movement.connect()
    return movement


def fm_camera_tilt(microscope: Any) -> float:
    """Tilt of the FM camera's optical axis from the SEM column, in degrees: the FM
    entry's ``camera_tilt``, else 180 where the FM is a stage pose, else the ion
    column's tilt. See `StageMovement.camera_tilt`."""
    configured = microscope.system.fm.camera_tilt
    if configured is not None:
        return float(configured)
    return derived_camera_tilt(
        microscope._fm_is_a_pose(), microscope.system.ion.column_tilt
    )
