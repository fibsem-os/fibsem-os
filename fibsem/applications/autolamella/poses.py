"""Deriving a lamella's poses from the position it was marked at.

A lamella carries two poses: where it is milled, and where it is looked at under
fluorescence. One of them is the position a person marked; the other is worked out from
it with `FibsemMicroscope.to_device` -- the same conversion on a compustage, where the
FM is a flip, and on an offset mount, where it is a place 48 mm away.

Which one was marked is read off the geometry, not declared:

* A position **only the beams** can use is the milling pose, verbatim, and the
  fluorescence pose is derived from it.
* A position **only the objective** sees the sample from -- the FM device on an offset
  mount, the t = -180 flip on a compustage -- is the fluorescence pose, and the milling
  pose is derived from it.
* A position **both** can use is both poses, the fluorescence one a copy.
* A position in **no supported orientation** is refused on a system with a fluorescence
  microscope, where both poses are owed and only one could be built.

`_is_beam_side` is the one place that asks about the mounting: on a compustage the pose
decides the side, on an offset mount the place does.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from fibsem.structures import (
    DeviceImagingState,
    FibsemStagePosition,
    InsertableDeviceState,
    MicroscopeState,
)
from fibsem.workflows.proposals import ItemPoses

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.microscope import FibsemMicroscope

# The orientation a lamella is milled at, when one has to be chosen rather than
# recovered. See `_to_milling` for when that happens.
MILLING_ORIENTATION = "MILLING"

# The compustage's FM orientation; on an offset mount the FM is a place, not a pose.
FLUORESCENCE_ORIENTATION = "FM"

# What `get_stage_orientation` calls a pose it has no name for.
UNSUPPORTED_ORIENTATION = "NONE"

# The two devices a lamella's poses are at, as the stage configuration names them.
BEAMS_DEVICE = "FIBSEM"
FM_DEVICE = "FM"


def build_lamella_poses(
    microscope: "FibsemMicroscope",
    position: Optional[FibsemStagePosition] = None,
    objective_position: Optional[float] = None,
    state: Optional[MicroscopeState] = None,
    marked_at: Optional[str] = None,
) -> ItemPoses:
    """The milling and fluorescence poses for a lamella marked at *position*.

    See the module docstring for how the side is decided.

    Args:
        microscope: the instrument, for the conversion and the objective.
        position: where the lamella was marked. Defaults to where the stage is.
        objective_position: the focus for the fluorescence pose. Defaults to the
            objective's configured focus position, which is what a beam-side caller
            wants — it has no way to know a better one.
        state: the microscope state to base both poses on. Defaults to the live one.
        marked_at: `FLUORESCENCE_ORIENTATION` from a caller marking through the FM.
            A check, never the decision: a position the objective cannot see the
            sample from is refused rather than taken as a fluorescence pose.

    Raises:
        ValueError: if the position is in no supported orientation, is on the FM
            side somewhere the objective cannot see the sample from, or was marked
            through the FM somewhere the FM cannot see.
    """
    state = deepcopy(state if state is not None else microscope.get_microscope_state())
    if position is not None:
        state.stage_position = deepcopy(position)
    marked = state.stage_position

    if microscope.fm is not None:
        _refuse_unsupported(microscope, marked)
        if marked_at == FLUORESCENCE_ORIENTATION:
            _refuse_unless_fm_sees(microscope, marked)

    if _is_beam_side(microscope, marked):
        # The milling pose verbatim, whatever orientation it happens to be in. That is
        # not always the MILLING orientation -- a lamella marked while looking at the
        # SEM keeps the SEM tilt -- and `update_milling_angle` derives the angle from
        # whatever is there.
        milling_position = deepcopy(marked)
        fluorescence_position = _to_fluorescence(microscope, marked)
    else:
        milling_position = _to_milling(microscope, marked)
        fluorescence_position = deepcopy(marked)

    milling = deepcopy(state)
    milling.stage_position = milling_position

    if microscope.fm is None or fluorescence_position is None:
        return ItemPoses(beam=milling, fluorescence=None)

    fluorescence = deepcopy(state)
    fluorescence.stage_position = fluorescence_position
    if objective_position is None:
        objective_position = microscope.fm.objective.focus_position
    fluorescence.objective_position = objective_position

    return ItemPoses(beam=milling, fluorescence=fluorescence)


def _is_beam_side(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> bool:
    """Is *position* at the beams, rather than somewhere only the objective sees?

    On an offset mount that is a place: the FM is 48 mm away. On a compustage the beams
    and the FM are one place and the pose decides: t = -180 is the FM's alone. With no
    FM every position is the beams'.
    """
    if microscope.fm is None:
        return True
    if microscope._fm_is_a_pose():
        return microscope.get_stage_orientation(position) != FLUORESCENCE_ORIENTATION
    return microscope.is_at_device(BEAMS_DEVICE, position)


def _refuse_unsupported(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> None:
    """Raise for a position in no supported orientation, naming the pose: a lamella
    marked there would get a milling pose and silently no fluorescence pose."""
    if microscope.get_stage_orientation(position) != UNSUPPORTED_ORIENTATION:
        return
    raise ValueError(
        f"Cannot mark a lamella here: the stage pose (rotation "
        f"{np.degrees(position.r):.1f} deg, tilt {np.degrees(position.t):.1f} deg) is "
        f"not a supported orientation, so its fluorescence pose cannot be worked out. "
        f"Move to a supported orientation and mark it again."
    )


def _refuse_unless_fm_sees(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> None:
    """Raise unless the objective sees the sample from *position*."""
    imaging = microscope.get_device_imaging_state(FM_DEVICE, position)
    if imaging is not DeviceImagingState.READY:
        raise ValueError(
            "Cannot take this as a fluorescence position: "
            + microscope.explain_imaging_state(FM_DEVICE, imaging, position)
        )


def sync_fluorescence_pose(microscope: "FibsemMicroscope", lamella) -> bool:
    """Bring a lamella's fluorescence pose back in line with its milling pose.

    For the callers that move a lamella on the beam side. They set the milling pose;
    stopping there would leave the fluorescence pose describing where the lamella *was*
    -- a stale pose being worse than a missing one, because nothing about it looks wrong.

    Only the stage position is rewritten. Everything else the fluorescence pose carries
    is kept, the objective position most of all: someone focused on this lamella by hand,
    and moving it sideways is not a reason to throw that away.

    Deliberately does **not** invent a pose for a lamella that has none. A lamella with
    no fluorescence pose has never been marked under fluorescence, and `fluorescence_
    selected` asks only whether an objective position exists -- so conjuring one here
    would make a lamella nobody has ever looked at report itself as focused.

    Returns:
        True if the pose was updated; False if there was none to update, or the
        instrument cannot work one out.
    """
    pose = getattr(lamella, "fluorescence_pose", None)
    if pose is None:
        return False

    milling = getattr(lamella, "milling_pose", None)
    if milling is None or milling.stage_position is None:
        logging.debug(
            f"Cannot sync the fluorescence pose of {getattr(lamella, 'name', '?')}: "
            f"it has no milling pose to derive one from."
        )
        return False

    # Into the orientation the pose is already in, where the FM images from it: a pose
    # somebody chose at the SEM tilt is not flipped because its lamella moved.
    position = _to_fluorescence(
        microscope,
        milling.stage_position,
        orientation=_kept_orientation(microscope, pose.stage_position),
    )
    if position is None:
        # Left as it stands rather than cleared. It cannot be derived on this system, so
        # whatever is there was put there deliberately and is the better of two bad
        # answers -- but it is now stale, and saying so is the only thing left to do.
        logging.warning(
            f"Could not update the fluorescence pose of "
            f"{getattr(lamella, 'name', '?')} to follow its milling pose; it still "
            f"describes the previous position."
        )
        return False

    pose.stage_position = position
    return True


def _kept_orientation(
    microscope: "FibsemMicroscope", position: Optional[FibsemStagePosition]
) -> Optional[str]:
    """The orientation *position* is in, if the FM images from it; else None."""
    if position is None or position.r is None or position.t is None:
        return None
    current = microscope.get_stage_orientation(position)
    allowed = microscope._get_device(FM_DEVICE).available_orientations
    return current if current in allowed else None


def _to_milling(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> FibsemStagePosition:
    """A fluorescence position re-posed for milling, under the beams.

    The canonical milling orientation, not a recovered one: a fluorescence pose does not
    record which beam orientation it came from, and for a target found *in*
    fluorescence there is no earlier pose to recover anyway.

    Checked before converting, because the conversion converts whatever it is given: a
    position the objective cannot see the sample from is not a fluorescence position,
    and re-posing it would write a milling pose somewhere the lamella is not.
    """
    if microscope.fm is not None:
        _refuse_unless_fm_sees(microscope, position)
    try:
        return microscope.to_device(position, BEAMS_DEVICE, MILLING_ORIENTATION)
    except ValueError as e:
        raise ValueError(
            f"Could not derive a milling pose from the fluorescence position: {e}"
        ) from e


def _to_fluorescence(
    microscope: "FibsemMicroscope",
    position: FibsemStagePosition,
    orientation: Optional[str] = None,
) -> Optional[FibsemStagePosition]:
    """A beam position as the FM sees it, or None if it cannot be worked out.

    Unavailability is not an error here, unlike the other direction: the fluorescence
    pose is a convenience to a beam-side caller, and refusing the whole lamella for it
    would break marking lamellae outright.

    `to_device` keeps the pose where the objective images from it. Otherwise, on a
    compustage, the pose is the FM's Default Orientation, as it always was; on an
    offset mount, where that control is disabled, the first orientation the FM
    declares. *orientation* overrides both, for a caller re-deriving a pose already
    in one.
    """
    if microscope.fm is None:
        return None
    if (
        orientation is None
        and microscope._fm_is_a_pose()
        and _kept_orientation(microscope, position) is None
    ):
        orientation = microscope.fm.default_orientation
    try:
        return microscope.to_device(position, FM_DEVICE, orientation)
    except ValueError as e:
        logging.warning(f"Could not derive a fluorescence pose for {position}: {e}")
        return None


# -- adding a lamella and setting its poses, without the window ----------------------
#
# What the Lamella tab's buttons do, minus the dialogs and the redraw, so the App
# Server runs the very same steps for an agent. The window calls
# these and then refreshes itself; the server calls them on the GUI thread too.


def grid_id_at(experiment, microscope: "FibsemMicroscope", position) -> Optional[str]:
    """The record id of the grid a lamella at *position* is on, or None.

    Resolved the way the rest of the system finds a grid, by name: with a loader the
    one grid on the stage, on a fixed holder the calibrated slot the position falls
    in. Then the experiment's record of that name. None whenever any step has no
    answer: nothing loaded, an uncalibrated slot, a grid on the hardware with no
    record. Never a guess, and never a record created as a side effect of marking a
    lamella.
    """
    stage = getattr(microscope, "sample_handling", None)
    if experiment is None or stage is None:
        return None
    try:
        if stage.magazine is not None:
            loaded = stage.loaded_grids
            grid = loaded[0] if len(loaded) == 1 else None
        else:
            grid = stage.grid_at_position(position)
    except Exception as e:  # noqa: BLE001 - a lamella is never refused for this
        logging.debug(f"Could not resolve the grid for a new lamella: {e}")
        return None
    if grid is None:
        return None
    record = experiment.get_grid_by_name(grid.name)
    return record.id if record is not None else None


def add_lamella(
    experiment,
    microscope: "FibsemMicroscope",
    stage_position: Optional[FibsemStagePosition] = None,
    name: Optional[str] = None,
    objective_position: Optional[float] = None,
    marked_at: Optional[str] = None,
    grid_id: Optional[str] = None,
):
    """Add a lamella to *experiment* and save it; returns the new lamella.

    `ExperimentTab.add_new_lamella` without the redraw: see it for the arguments.
    The current stage position when *stage_position* is None.
    """
    poses = build_lamella_poses(
        microscope=microscope,
        position=stage_position,
        objective_position=objective_position,
        marked_at=marked_at,
    )
    # Both poses go on at creation -- see `Experiment.add_new_lamella`. Assigning the
    # fluorescence pose after the append left every listener that redraws on
    # `inserted` deciding the new lamella had none.
    experiment.add_new_lamella(
        microscope_state=poses.beam,
        task_config=experiment.task_protocol.task_config,
        name=name,
        fluorescence_pose=poses.fluorescence,
        grid_id=(
            grid_id
            if grid_id is not None
            else grid_id_at(experiment, microscope, poses.beam.stage_position)
        ),
    )
    lamella = experiment.positions[-1]
    lamella.update_milling_angle(microscope)  # from the milling-pose stage tilt
    experiment.save()
    return lamella


def set_lamella_pose(
    experiment,
    microscope: "FibsemMicroscope",
    lamella,
    pose_name: str,
    state: MicroscopeState,
) -> bool:
    """Set *state* as the lamella's pose *pose_name*, and save *experiment* when
    there is one. Returns whether the fluorescence pose was moved to follow it
    (only ever for MILLING). The Experiment tab's and the coincidence viewer's
    Set pose, and the App Server's.

    The objective position of an existing pose is kept: the microscope state does
    not capture it, so replacing the pose outright would wipe its focus. Replacing
    the milling pose moves the lamella, so what derives from it follows: the
    milling angle, and the fluorescence pose of the same piece of sample. The
    experiment's positions emit changed: the overview canvases draw the poses
    themselves, and a pose written in place emits nothing on its own.
    """
    existing = lamella.poses.get(pose_name)
    if existing is not None and existing.objective_position is not None:
        state.objective_position = existing.objective_position
    lamella.poses[pose_name] = state
    followed = False
    if pose_name == "MILLING":
        lamella.update_milling_angle(microscope)
        followed = sync_fluorescence_pose(microscope, lamella)
    if experiment is not None:
        experiment.save()
        experiment.positions.events.changed.emit()
    return followed


def fluorescence_move(
    microscope: "FibsemMicroscope",
    lamella,
    position: FibsemStagePosition,
    objective_position: Optional[float] = None,
) -> ItemPoses:
    """The poses moving *lamella* to *position*, a point marked on the
    fluorescence side, gives it, for :func:`move_lamella`; built on what the
    lamella recorded. Ask before moving: it moves the milling pose too.

    Raises:
        ValueError: the lamella has no milling pose, or the point cannot be
            converted to a beam orientation.
    """
    if lamella.milling_pose is None:
        raise ValueError(f"{lamella.name} has no milling pose to move.")
    return build_lamella_poses(
        microscope=microscope,
        position=position,
        objective_position=objective_position,
        state=lamella.milling_pose,
        marked_at=FLUORESCENCE_ORIENTATION,
    )


def move_lamella(
    experiment,
    microscope: "FibsemMicroscope",
    lamella,
    position: Optional[FibsemStagePosition] = None,
    poses: Optional[ItemPoses] = None,
) -> None:
    """Move *lamella* to a point marked on an overview, and save.

    On the beam side, *position* is the new milling pose and the fluorescence
    pose follows it; from the fluorescence side, *poses* (:func:`fluorescence_move`)
    gives both. Only the stage positions are replaced, so what else the poses
    carry -- the objective position of a lamella focused by hand -- is kept.
    The experiment's positions emit changed: a pose written in place emits
    nothing on its own.
    """
    if poses is None:
        lamella.stage_position = position
        lamella.update_milling_angle(microscope)
        sync_fluorescence_pose(microscope, lamella)
    else:
        lamella.stage_position = poses.beam.stage_position
        lamella.update_milling_angle(microscope)
        if lamella.fluorescence_pose is None:
            lamella.fluorescence_pose = poses.fluorescence
        else:
            lamella.fluorescence_pose.stage_position = poses.fluorescence.stage_position
    experiment.save()
    experiment.positions.events.changed.emit()


def remove_lamella(experiment, lamella) -> bool:
    """Remove *lamella* from *experiment* and save; its files stay on disk,
    and its directory goes only when it is empty. False when it was
    not there."""
    try:
        experiment.positions.remove(lamella)
    except ValueError:
        return False
    experiment.save()
    try:
        os.rmdir(lamella.path)  # refuses unless the directory is empty
    except (OSError, TypeError):
        pass
    return True


def set_objective_position(experiment, lamella, value: float) -> None:
    """Store *value* as the objective position on *lamella*'s fluorescence pose,
    and save *experiment* when there is one. The selected lamella's spinbox and
    Use current, and the App Server's."""
    lamella.fluorescence_pose.objective_position = value
    if experiment is not None:
        experiment.save()


def set_objective_position_on_all(
    experiment, value: float, skip: Sequence[str] = ()
) -> int:
    """Store *value* on every lamella with a fluorescence pose but those named
    in *skip*, saved: Apply to All. How many it was stored on."""
    count = 0
    for lamella in experiment.positions:
        if lamella.fluorescence_pose is not None and lamella.name not in skip:
            lamella.fluorescence_pose.objective_position = value
            count += 1
    if count:
        experiment.save()
    return count


def current_objective_position(fm) -> Optional[float]:
    """The FM objective position to store on a fluorescence pose: where it is
    when inserted, else its focus position. None when it can't be read."""
    objective = fm.objective
    if objective.state is InsertableDeviceState.INSERTED:
        return objective.position
    return objective.focus_position


def stored_objective_position(fm, lamella) -> float:
    """The objective position stored on *lamella*'s fluorescence pose, to move
    the objective back to. Raises ValueError saying why not: no pose, nothing
    stored, or the objective not inserted (a stored position is an inserted
    one)."""
    pose = lamella.fluorescence_pose
    if pose is None:
        raise ValueError(f"{lamella.name} has no fluorescence pose.")
    if pose.objective_position is None:
        raise ValueError(f"{lamella.name} has no stored objective position.")
    if fm.objective.state is not InsertableDeviceState.INSERTED:
        raise ValueError("Insert the objective before moving to a stored position.")
    return pose.objective_position
