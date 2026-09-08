"""Deriving a lamella's poses from the position it was marked at.

A lamella carries two poses: where it is milled, and where it is looked at under
fluorescence. Only one of them is ever picked by a user — the other is derived by
re-posing the stage position into the other orientation.

Which one was picked depends on where the marking happened, and *that* is the thing
worth being careful about. Every caller until now marked positions on the beam side, so
`AutoLamellaUI.add_new_lamella` could take its argument as the milling pose and derive
the fluorescence pose from it. The FM overview tab marks positions on the fluorescence
side, and handing one of those to the same function sets a milling pose at t = -180 --
an orientation nothing mills at. Nothing rejects it; it fails later, somewhere else.

So which side a position was marked from is read off the position rather than assumed.
On a compustage it is in the rotation and tilt; on an offset mount it is in *where the
stage is parked*, ~48 mm out along x with the pose it was carried out in. One question
covers both -- can the objective see the sample from here -- and the microscope answers
it (`get_device_imaging_state`, FIB-839), so nothing here branches on the mounting.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from fibsem.structures import DeviceImagingState, FibsemStagePosition, MicroscopeState

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.microscope import FibsemMicroscope

# The orientation a lamella is milled at, when one has to be chosen rather than
# recovered. See `build_lamella_poses` for when that happens.
MILLING_ORIENTATION = "MILLING"

# The side a lamella is looked at from. On a compustage it is also the orientation the
# stage flips to; on an offset mount it is a device the stage travels to, and the pose
# held there is whatever the FM declares (`stage.devices.FM.acquisition_orientations`).
# Callers use it to *declare a side* to `build_lamella_poses`, never to re-pose with.
FLUORESCENCE_ORIENTATION = "FM"

# The two devices a lamella's poses are at, by the names the stage configuration uses.
BEAMS_DEVICE = "FIBSEM"
FM_DEVICE = "FM"


@dataclass(frozen=True)
class LamellaPoses:
    """Both poses for a lamella, whichever side it was marked from.

    `fluorescence` is None when the microscope has no fluorescence detector — there is
    nothing to derive it from, and a pose invented for an instrument that does not exist
    would be worse than its absence.
    """

    milling: MicroscopeState
    fluorescence: Optional[MicroscopeState] = None


def build_lamella_poses(
    microscope: "FibsemMicroscope",
    position: Optional[FibsemStagePosition] = None,
    objective_position: Optional[float] = None,
    state: Optional[MicroscopeState] = None,
    marked_at: Optional[str] = None,
) -> LamellaPoses:
    """The milling and fluorescence poses for a lamella marked at *position*.

    *position* may be in any orientation. A position in the fluorescence orientation is
    taken as the fluorescence pose and the milling pose derived from it; anything else
    is taken as the milling pose, which is what every beam-side caller has always meant.

    Args:
        microscope: the instrument, for the orientation transform and the objective.
        position: where the lamella was marked. Defaults to where the stage is.
        objective_position: the focus for the fluorescence pose. Defaults to the
            objective's configured focus position, which is what a beam-side caller
            wants — it has no way to know a better one.
        state: the microscope state to base both poses on. Defaults to the live one.
        marked_at: the orientation *position* is in, for a caller that knows. Defaults
            to reading it off the position, which is right on a compustage and **not
            good enough on an offset mount** — see below.

    Raises:
        ValueError: if a fluorescence position is given on a system that cannot convert
            it to a beam orientation.

    Note:
        Orientation is normally derived rather than declared, because it is already in
        the rotation and tilt and a caller that has to declare it is a caller that can
        get it wrong — which is the bug this function exists to remove.

        That fails on an offset mount, where the fluorescence position is distinguished
        by travelling ~48 mm in x and *not* by its tilt. Measured on the simulator, the
        FM and FIB orientations there share a tilt of 17 degrees and both classify as
        MILLING, so a fluorescence position is indistinguishable from somewhere to mill.
        A caller that knows which side it is marking from says so, and gets a refusal
        rather than a plausible wrong answer.
    """
    state = deepcopy(state if state is not None else microscope.get_microscope_state())
    if position is not None:
        state.stage_position = deepcopy(position)

    if marked_at is None:
        marked_from_fluorescence = _is_fluorescence_position(
            microscope, state.stage_position
        )
    else:
        marked_from_fluorescence = marked_at == FLUORESCENCE_ORIENTATION

    if marked_from_fluorescence:
        milling_position = _to_milling(microscope, state.stage_position)
        fluorescence_position = deepcopy(state.stage_position)
    else:
        # Unchanged from what every beam-side caller has always got: the position is
        # the milling pose verbatim, whatever orientation it happens to be in. That is
        # not always the MILLING orientation -- a lamella marked while looking at the
        # SEM keeps the SEM tilt -- and `update_milling_angle` derives the angle from
        # whatever is there, so re-posing it here would change existing behaviour.
        milling_position = deepcopy(state.stage_position)
        fluorescence_position = _to_fluorescence(microscope, state.stage_position)

    milling = deepcopy(state)
    milling.stage_position = milling_position

    if microscope.fm is None or fluorescence_position is None:
        return LamellaPoses(milling=milling, fluorescence=None)

    fluorescence = deepcopy(state)
    fluorescence.stage_position = fluorescence_position
    if objective_position is None:
        objective_position = microscope.fm.objective.focus_position
    fluorescence.objective_position = objective_position

    return LamellaPoses(milling=milling, fluorescence=fluorescence)


def sync_fluorescence_pose(microscope: "FibsemMicroscope", lamella) -> bool:
    """Bring a lamella's fluorescence pose back in line with its milling pose.

    For the callers that move a lamella on the beam side. They set the milling pose and
    have historically stopped there, which leaves the fluorescence pose describing where
    the lamella *used to be* -- a stale pose being worse than a missing one, because
    nothing about it looks wrong.

    Only the stage position is rewritten. Everything else the fluorescence pose carries
    is kept, the objective position most of all: someone focused on this lamella by hand,
    and moving it sideways is not a reason to throw that away.

    Deliberately does **not** invent a pose for a lamella that has none. A lamella with
    no fluorescence pose has never been marked under fluorescence, and `fluorescence_
    selected` asks only whether an objective position exists -- so conjuring one here
    would make a lamella nobody has ever looked at report itself as focused.

    Returns:
        True if the pose was updated; False if there was none to update, or the
        instrument could not work one out.
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

    position = _to_fluorescence(microscope, milling.stage_position)
    if position is None:
        # Left as it stands rather than cleared. Whatever is there was put there
        # deliberately and is the better of two bad answers -- but it is now stale, and
        # saying so is the only thing left to do.
        logging.warning(
            f"Could not update the fluorescence pose of "
            f"{getattr(lamella, 'name', '?')} to follow its milling pose; it still "
            f"describes the previous position."
        )
        return False

    pose.stage_position = position
    return True


def _is_fluorescence_position(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> bool:
    """Can the objective see the sample from *position*?

    READY strictly, not `allows_acquisition`: this decides which of a lamella's poses
    gets *written* from the position, and a position the objective could image after a
    re-pose is still not the fluorescence pose.

    Without an FM the device question has no answer, so the orientation stands in for
    it -- which is all a compustage ever needed.
    """
    if microscope.fm is None:
        return microscope.get_stage_orientation(position) == FLUORESCENCE_ORIENTATION
    state = microscope.get_device_imaging_state("FM", position)
    return state is DeviceImagingState.READY


def _to_milling(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> FibsemStagePosition:
    """A fluorescence position re-posed for milling, under the beams.

    The canonical milling orientation, not a recovered one: a fluorescence pose does not
    record which beam orientation it came from -- the transform only rewrites rotation
    and tilt, so a lamella marked at the SEM and one marked at the milling angle both
    arrive at the same fluorescence pose and are indistinguishable afterwards. For a
    target found *in* fluorescence there is no earlier pose to recover anyway, and the
    orientation milling actually happens at is the only defensible answer.

    Checked before converting, because `get_target_position` will not: it converts
    whatever it is given. A position the objective cannot see the sample from is not a
    fluorescence position, and re-posing it "for milling" would write a milling pose
    somewhere the lamella is not. Refused instead, naming what is wrong with it.
    """
    if microscope.fm is not None:
        state = microscope.get_device_imaging_state("FM", position)
        if state is not DeviceImagingState.READY:
            raise ValueError(
                "Cannot take this as a fluorescence position: "
                + microscope.describe_device_imaging_state("FM", state, position)
            )
    try:
        return microscope.get_target_position(
            stage_position=deepcopy(position),
            target_orientation=MILLING_ORIENTATION,
            target_device=BEAMS_DEVICE,
        )
    except ValueError as e:
        raise ValueError(
            f"Could not derive a milling pose from the fluorescence position: {e}"
        ) from e


def _to_fluorescence(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> Optional[FibsemStagePosition]:
    """A beam position re-posed and relocated for fluorescence, or None if it cannot be.

    Unavailability is not an error here, unlike the other direction. A beam-side caller
    is marking somewhere to mill and the fluorescence pose is a convenience; refusing
    the whole lamella because the system cannot work one out would break marking
    lamellae outright.

    Asked for as the pair -- the FM's acquisition orientation *at* the FM device --
    which is what a fluorescence pose is on either mounting. A compustage takes the
    device leg with a zero translation and gets the flip it always had; an offset mount
    gets the traverse. An FM declared with no acquisition orientations constrains the
    pose not at all (`pose_orientation` is None), so the position is relocated with
    its pose kept.
    """
    fm = microscope.fm
    if fm is None:
        return None
    try:
        return microscope.get_target_position(
            stage_position=deepcopy(position),
            target_orientation=fm.pose_orientation,
            target_device=FM_DEVICE,
        )
    except ValueError as e:
        logging.debug(f"Could not derive a fluorescence pose for {position}: {e}")
        return None
