"""Deriving a lamella's poses from the position it was marked at.

A lamella carries two poses: where it is milled, and where it is looked at under
fluorescence. They are two *observations* of one piece of sample from two instruments,
and the transform between them is a first guess for the one nobody has looked through
yet -- not a coupling. Only a person, or an alignment, can say where the sample really
is under either. So:

* Creation asks one question: **can the beams mill from this position?** A beam pose at
  the beams is the milling pose verbatim, whatever tilt it happens to be at -- what every
  beam-side caller has always had. Anything else -- the FM device on an offset mount,
  the t = -180 flip on a compustage -- gets the milling pose derived first, and the
  fluorescence pose derived from *that*. Nothing branches on the mounting: the
  microscope answers the question (`get_device_imaging_state`, FIB-839).
* Nothing here syncs the two afterwards. `derive_fluorescence_pose` and
  `derive_milling_pose` exist for a caller that has decided to overwrite one from the
  other -- the overview tabs when their Link preference is on, the lamella details on
  request -- and they say so on the lamella (`PoseProvenance.DERIVED`).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from fibsem.structures import DeviceImagingState, FibsemStagePosition, MicroscopeState

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.applications.autolamella.structures import Lamella
    from fibsem.microscope import FibsemMicroscope

# The orientation a lamella is milled at, when one has to be chosen rather than
# recovered. See `build_lamella_poses` for when that happens.
MILLING_ORIENTATION = "MILLING"

# The compustage's FM orientation: the one pose there that only the objective sees the
# sample from. On an offset mount there is no such orientation -- the FM is a place --
# and this name is never read off a position.
FLUORESCENCE_ORIENTATION = "FM"

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
    `FLUORESCENCE_POSE` -- so the caller can record which one a person chose and which
    one was worked out.
    """

    milling: MicroscopeState
    fluorescence: Optional[MicroscopeState] = None
    observed: str = MILLING_POSE


def build_lamella_poses(
    microscope: "FibsemMicroscope",
    position: Optional[FibsemStagePosition] = None,
    objective_position: Optional[float] = None,
    state: Optional[MicroscopeState] = None,
    orientation: Optional[str] = None,
) -> LamellaPoses:
    """The milling and fluorescence poses for a lamella marked at *position*.

    *position* may be anywhere. If the beams can mill from it, it is the milling pose
    verbatim and the fluorescence pose is derived. Otherwise it is somewhere only the
    objective sees the sample from, so it is the fluorescence pose, and the milling
    pose is derived -- checked first, because a milling pose nothing can mill at is the
    dangerous outcome.

    Args:
        microscope: the instrument, for the transform and the objective.
        position: where the lamella was marked. Defaults to where the stage is.
        objective_position: the focus for the fluorescence pose. Defaults to the
            objective's configured focus position, which is what a beam-side caller
            wants -- it has no way to know a better one.
        state: the microscope state to base both poses on. Defaults to the live one.
        orientation: the orientation a *derived* fluorescence pose is put in. Defaults
            to the FM's `pose_orientation`, the first it declares it images from.

    Raises:
        ValueError: if the position is neither somewhere to mill nor somewhere the
            objective can see the sample from -- mid-traverse on an offset mount, say.
    """
    state = deepcopy(state if state is not None else microscope.get_microscope_state())
    if position is not None:
        state.stage_position = deepcopy(position)

    if is_beam_position(microscope, state.stage_position):
        # The position is the milling pose verbatim, whatever orientation it happens to
        # be in. That is not always the MILLING orientation -- a lamella marked while
        # looking at the SEM keeps the SEM tilt -- and `update_milling_angle` derives
        # the angle from whatever is there, so re-posing it here would change existing
        # behaviour.
        observed = MILLING_POSE
        milling_position = deepcopy(state.stage_position)
        fluorescence_position = _to_fluorescence(
            microscope, state.stage_position, orientation
        )
    else:
        observed = FLUORESCENCE_POSE
        milling_position = _to_milling(microscope, state.stage_position)
        fluorescence_position = deepcopy(state.stage_position)

    milling = deepcopy(state)
    milling.stage_position = milling_position

    if microscope.fm is None or fluorescence_position is None:
        return LamellaPoses(milling=milling, fluorescence=None, observed=MILLING_POSE)

    fluorescence = deepcopy(state)
    fluorescence.stage_position = fluorescence_position
    if objective_position is None:
        objective_position = microscope.fm.objective.focus_position
    fluorescence.objective_position = objective_position

    return LamellaPoses(milling=milling, fluorescence=fluorescence, observed=observed)


def is_beam_position(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> bool:
    """Can the beams mill from *position*?

    Under the beams, in any pose but the one only the objective sees the sample from.
    On an offset mount that is a place: the FM is 48 mm away and no pose at the beams
    is the FM's. On a compustage the beams and the FM are one place and the pose
    decides: t = -180 is the FM's alone, everything else -- SEM, FIB, MILLING, an
    unnamed tilt -- is somewhere a beam looks at the sample. SEM being *also* an FM
    orientation there (the objective can image from it) does not change that: a
    position both instruments can use is a milling pose, and its fluorescence pose is
    a copy or a flip of it, as the lamella's orientation says.
    """
    if microscope.stage_is_compustage:
        return microscope.get_stage_orientation(position) != FLUORESCENCE_ORIENTATION
    return microscope.is_at_device(BEAMS_DEVICE, position)


def derive_fluorescence_pose(
    microscope: "FibsemMicroscope",
    lamella: "Lamella",
    orientation: Optional[str] = None,
) -> bool:
    """Overwrite a lamella's fluorescence pose with one derived from its milling pose.

    For a caller that has decided the fluorescence pose should follow the milling
    pose -- the beam overview with its Link preference on, the lamella details on
    request. Nothing calls this on its own behalf: the transform is a guess, and a pose
    somebody centred by hand is not to be replaced without being asked.

    Only the stage position is derived. The objective position is kept -- someone
    focused on this lamella, and moving it sideways is not a reason to throw that
    away -- and set from the objective's configured focus when there was no pose.

    Returns:
        True if a pose was written; False if the instrument could not derive one, in
        which case the existing pose is left as it stands.
    """
    milling = lamella.milling_pose
    if milling is None or milling.stage_position is None:
        logging.debug(
            f"Cannot derive the fluorescence pose of {lamella.name}: "
            f"it has no milling pose to derive one from."
        )
        return False
    position = _to_fluorescence(microscope, milling.stage_position, orientation)
    if position is None:
        logging.warning(
            f"Could not derive the fluorescence pose of {lamella.name} from its "
            f"milling pose; the existing one is left as it stands."
        )
        return False

    existing = lamella.fluorescence_pose
    pose = deepcopy(existing if existing is not None else milling)
    pose.stage_position = position
    if existing is None or existing.objective_position is None:
        pose.objective_position = microscope.fm.objective.focus_position
    from fibsem.applications.autolamella.structures import PoseProvenance

    lamella.set_pose(FLUORESCENCE_POSE, pose, PoseProvenance.DERIVED)
    return True


def derive_milling_pose(microscope: "FibsemMicroscope", lamella: "Lamella") -> bool:
    """Overwrite a lamella's milling pose with one derived from its fluorescence pose.

    The other direction of `derive_fluorescence_pose`, with the other direction's
    caution: a wrong milling pose is dangerous, so this refuses -- returns False and
    writes nothing -- unless the fluorescence pose is somewhere the objective can see
    the sample from. The milling angle is left to the caller, which knows whether it
    wants it re-read from the new pose.
    """
    fluorescence = lamella.fluorescence_pose
    if fluorescence is None or fluorescence.stage_position is None:
        logging.debug(
            f"Cannot derive the milling pose of {lamella.name}: "
            f"it has no fluorescence pose to derive one from."
        )
        return False
    try:
        position = _to_milling(microscope, fluorescence.stage_position)
    except ValueError as e:
        logging.warning(f"Could not derive the milling pose of {lamella.name}: {e}")
        return False

    existing = lamella.milling_pose
    pose = deepcopy(existing if existing is not None else fluorescence)
    pose.stage_position = position
    from fibsem.applications.autolamella.structures import PoseProvenance

    lamella.set_pose(MILLING_POSE, pose, PoseProvenance.DERIVED)
    return True


def follow_milling_pose(
    microscope: "FibsemMicroscope", lamella: "Lamella", link: bool
) -> bool:
    """What happens to the fluorescence pose after a person moved the milling one.

    *link* is the beam overview's Link preference: derive the fluorescence pose from
    the new milling pose, or leave it and say it may be stale. Either way the person
    decided; nothing here decides for them. Returns True if a pose was derived.
    """
    if link:
        return derive_fluorescence_pose(microscope, lamella)
    lamella.mark_pose_stale(FLUORESCENCE_POSE)
    return False


def follow_fluorescence_pose(
    microscope: "FibsemMicroscope", lamella: "Lamella", link: bool
) -> bool:
    """The other direction: after a person moved the fluorescence pose."""
    if link:
        return derive_milling_pose(microscope, lamella)
    lamella.mark_pose_stale(MILLING_POSE)
    return False


POSE_NOUNS = {MILLING_POSE: "milling pose", FLUORESCENCE_POSE: "fluorescence pose"}


def derivation_question(
    lamella: "Lamella", pose_name: str, orientation: Optional[str] = None
) -> str:
    """The confirmation for *Derive*, in the words a person needs before saying yes.

    Every derivation overwrites, so every one confirms -- and the one that overwrites
    a pose somebody set by hand says so, since that is the case the answer changes.
    """
    from fibsem.applications.autolamella.structures import PoseProvenance

    other = MILLING_POSE if pose_name == FLUORESCENCE_POSE else FLUORESCENCE_POSE
    into = f" into the {orientation} orientation" if orientation else ""
    text = (
        f"Derive the {POSE_NOUNS.get(pose_name, pose_name)} of {lamella.name} from "
        f"its {POSE_NOUNS.get(other, other)}{into}?"
    )
    if lamella.provenance_of(pose_name) is PoseProvenance.OBSERVED:
        text += f"\n\nThis overwrites a {POSE_NOUNS.get(pose_name, pose_name)} that was set by hand."
    if pose_name == FLUORESCENCE_POSE:
        text += (
            "\n\nThe objective position is kept; refocus before acquiring if the "
            "orientation changed."
        )
    return text


def derive_pose(
    microscope: "FibsemMicroscope",
    lamella: "Lamella",
    pose_name: str,
    orientation: Optional[str] = None,
) -> bool:
    """Overwrite the named pose with one derived from the other -- by name.

    The lamella details' *Derive* action, which knows only which row it is on.
    *orientation* is honoured for the fluorescence pose and ignored for the milling
    one, which is always the milling orientation under the beams.
    """
    if pose_name == FLUORESCENCE_POSE:
        return derive_fluorescence_pose(microscope, lamella, orientation)
    if pose_name == MILLING_POSE:
        return derive_milling_pose(microscope, lamella)
    raise ValueError(f"No derivation for a pose named {pose_name!r}.")


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
        state = microscope.get_device_imaging_state(FM_DEVICE, position)
        if state is not DeviceImagingState.READY:
            raise ValueError(
                "Cannot take this as a fluorescence position: "
                + microscope.describe_device_imaging_state(FM_DEVICE, state, position)
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
    microscope: "FibsemMicroscope",
    position: FibsemStagePosition,
    orientation: Optional[str] = None,
) -> Optional[FibsemStagePosition]:
    """A beam position re-posed and relocated for fluorescence, or None if it cannot be.

    Unavailability is not an error here, unlike the other direction. A beam-side caller
    is marking somewhere to mill and the fluorescence pose is a convenience; refusing
    the whole lamella because the system cannot work one out would break marking
    lamellae outright.

    Asked for as the pair -- an orientation the FM images in, *at* the FM device --
    which is what a fluorescence pose is on either mounting. A compustage takes the
    device leg with a zero translation and gets the flip it always had; an offset mount
    gets the traverse. *orientation* defaults to the FM's `pose_orientation`, the first
    it declares; None there means the FM constrains the pose not at all, and the
    position is relocated with its pose kept.
    """
    fm = microscope.fm
    if fm is None:
        return None
    try:
        return microscope.get_target_position(
            stage_position=deepcopy(position),
            target_orientation=orientation
            if orientation is not None
            else fm.pose_orientation,
            target_device=FM_DEVICE,
        )
    except ValueError as e:
        logging.debug(f"Could not derive a fluorescence pose for {position}: {e}")
        return None
