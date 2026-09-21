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
  pose is derived from it. Checked first, because a milling pose nothing can mill at is
  the dangerous outcome.
* A position **both** can use (a compustage whose FM also images from the beam pose it
  is in) is both poses, the fluorescence one a copy rather than a flip. Geometry cannot
  say which instrument the person was looking through, so the caller may: `observed`.
  It breaks that tie and nothing else -- it cannot make a position 48 mm from the beams
  into somewhere to mill, which is what declaring the side outright used to risk.
* A position in **no supported orientation** is refused on a system with a fluorescence
  microscope, where both poses are owed and only one could be built.

`_is_beam_side` is the one place that asks about the mounting, and it has to: on a
compustage the pose decides the side, on an offset mount the place does.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from fibsem.structures import DeviceImagingState, FibsemStagePosition, MicroscopeState

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.microscope import FibsemMicroscope

# The orientation a lamella is milled at, when one has to be chosen rather than
# recovered. See `_to_milling` for when that happens.
MILLING_ORIENTATION = "MILLING"

# The compustage's FM orientation: the one pose there that only the objective sees the
# sample from. On an offset mount there is no such orientation -- the FM is a place.
FLUORESCENCE_ORIENTATION = "FM"

# What `get_stage_orientation` calls a pose it has no name for. Unsupported.
UNSUPPORTED_ORIENTATION = "NONE"

# The two devices a lamella's poses are at, by the names the stage configuration uses.
BEAMS_DEVICE = "FIBSEM"
FM_DEVICE = "FM"

# The two poses, by the names `Lamella.poses` keys them under.
MILLING_POSE = "MILLING"
FLUORESCENCE_POSE = "FLUORESCENCE"


@dataclass(frozen=True)
class LamellaPoses:
    """Both poses for a lamella, whichever side it was marked from.

    `fluorescence` is None when the microscope has no fluorescence detector -- there is
    nothing to derive it from, and a pose invented for an instrument that does not exist
    would be worse than its absence.

    `observed` names the pose that *is* the marked position -- `MILLING_POSE` or
    `FLUORESCENCE_POSE`. The other one was worked out from it.
    """

    milling: MicroscopeState
    fluorescence: Optional[MicroscopeState] = None
    observed: str = MILLING_POSE


def build_lamella_poses(
    microscope: "FibsemMicroscope",
    position: Optional[FibsemStagePosition] = None,
    objective_position: Optional[float] = None,
    state: Optional[MicroscopeState] = None,
    observed: Optional[str] = None,
) -> LamellaPoses:
    """The milling and fluorescence poses for a lamella marked at *position*.

    See the module docstring for how the side is decided.

    Args:
        microscope: the instrument, for the conversion and the objective.
        position: where the lamella was marked. Defaults to where the stage is.
        objective_position: the focus for the fluorescence pose. Defaults to the
            objective's configured focus position, which is what a beam-side caller
            wants -- it has no way to know a better one.
        state: the microscope state to base both poses on. Defaults to the live one.
        observed: `MILLING_POSE` or `FLUORESCENCE_POSE`, from a caller that knows
            which instrument the position was marked through. Read only when the
            position is one both can use; the geometry decides everything else.

    Raises:
        ValueError: if the position is in no supported orientation, or is neither
            somewhere to mill nor somewhere the objective sees the sample from --
            mid-traverse on an offset mount, say.
    """
    state = deepcopy(state if state is not None else microscope.get_microscope_state())
    if position is not None:
        state.stage_position = deepcopy(position)
    marked = state.stage_position

    if microscope.fm is not None:
        _refuse_unsupported(microscope, marked)

    if _is_beam_side(microscope, marked):
        # The position is the milling pose verbatim, whatever orientation it happens to
        # be in. That is not always the MILLING orientation -- a lamella marked while
        # looking at the SEM keeps the SEM tilt -- and `update_milling_angle` derives
        # the angle from whatever is there, so re-posing it here would change existing
        # behaviour.
        milling_position = deepcopy(marked)
        fluorescence_position = _to_fluorescence(microscope, marked)
        both_can_use_it = (
            microscope.fm is not None
            and microscope.get_device_imaging_state(FM_DEVICE, marked)
            is DeviceImagingState.READY
        )
        which = (
            FLUORESCENCE_POSE
            if both_can_use_it and observed == FLUORESCENCE_POSE
            else MILLING_POSE
        )
    else:
        milling_position = _to_milling(microscope, marked)
        fluorescence_position = deepcopy(marked)
        which = FLUORESCENCE_POSE

    milling = deepcopy(state)
    milling.stage_position = milling_position

    if microscope.fm is None or fluorescence_position is None:
        return LamellaPoses(milling=milling, fluorescence=None, observed=MILLING_POSE)

    fluorescence = deepcopy(state)
    fluorescence.stage_position = fluorescence_position
    if objective_position is None:
        objective_position = microscope.fm.objective.focus_position
    fluorescence.objective_position = objective_position

    return LamellaPoses(milling=milling, fluorescence=fluorescence, observed=which)


def _is_beam_side(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> bool:
    """Is *position* at the beams, rather than somewhere only the objective sees?

    Not a question about whether milling is feasible there -- tilt limits and reach are
    somebody else's -- only about which side of the instrument the position is on.

    On an offset mount that is a place: the FM is 48 mm away, and no pose at the beams
    is the FM's. On a compustage the beams and the FM are one place and the pose
    decides: t = -180 is the FM's alone, and every other supported pose -- SEM, FIB,
    MILLING -- is somewhere a beam looks at the sample, whether or not the objective
    can image from it too.
    """
    if microscope.stage_is_compustage:
        return microscope.get_stage_orientation(position) != FLUORESCENCE_ORIENTATION
    return microscope.is_at_device(BEAMS_DEVICE, position)


def _refuse_unsupported(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> None:
    """Raise for a position in no supported orientation, naming the pose.

    There is no conversion from one, so a lamella marked there would get a milling
    pose and silently no fluorescence pose. Said at the point of marking instead,
    where there is a person to tell.
    """
    if microscope.get_stage_orientation(position) != UNSUPPORTED_ORIENTATION:
        return
    raise ValueError(
        f"Cannot mark a lamella here: the stage pose (rotation "
        f"{np.degrees(position.r):.1f} deg, tilt {np.degrees(position.t):.1f} deg) is "
        f"not a supported orientation, so its fluorescence pose cannot be worked out. "
        f"Move to a supported orientation and mark it again."
    )


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

    # Into the orientation the pose is already in, where the FM images from it: a
    # pose somebody chose at the SEM tilt is not flipped because its lamella moved.
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
    device = microscope.system.stage.devices.get(FM_DEVICE)
    allowed = device.acquisition_orientations if device is not None else []
    return current if current in allowed else None


def _to_milling(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> FibsemStagePosition:
    """A fluorescence position re-posed for milling, under the beams.

    The canonical milling orientation, not a recovered one: a fluorescence pose does not
    record which beam orientation it came from -- the conversion only rewrites rotation
    and tilt, so a lamella marked at the SEM and one marked at the milling angle both
    arrive at the same fluorescence pose and are indistinguishable afterwards. For a
    target found *in* fluorescence there is no earlier pose to recover anyway, and the
    orientation milling actually happens at is the only defensible answer.

    Checked before converting, because the conversion will not: it converts whatever it
    is given. A position the objective cannot see the sample from is not a fluorescence
    position, and re-posing it "for milling" would write a milling pose somewhere the
    lamella is not. Refused instead, in the words that say what is wrong with it.
    """
    if microscope.fm is not None:
        imaging = microscope.get_device_imaging_state(FM_DEVICE, position)
        if imaging is not DeviceImagingState.READY:
            raise ValueError(
                "Cannot take this as a fluorescence position: "
                + microscope.describe_device_imaging_state(FM_DEVICE, imaging, position)
            )
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

    Unavailability is not an error here, unlike the other direction. A beam-side caller
    is marking somewhere to mill and the fluorescence pose is a convenience; refusing
    the whole lamella because the system cannot work one out would break marking
    lamellae outright.

    `to_device` keeps the pose where the objective images from it -- the fluorescence
    pose is then a copy, not a flip -- and otherwise uses the first orientation the FM
    declares. *orientation* overrides that, for a caller re-deriving a pose that is
    already in one.
    """
    if microscope.fm is None:
        return None
    try:
        return microscope.to_device(position, FM_DEVICE, orientation)
    except ValueError as e:
        logging.warning(f"Could not derive a fluorescence pose for {position}: {e}")
        return None
