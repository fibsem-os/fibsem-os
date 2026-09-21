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
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from fibsem.structures import DeviceImagingState, FibsemStagePosition, MicroscopeState

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.applications.autolamella.structures import Lamella
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


class PoseProvenance(str, Enum):
    """Where a lamella pose came from.

    A lamella's milling and fluorescence poses are two observations of one piece of
    sample from two instruments, and the conversion between them is a first guess for
    the side nobody has looked through yet. So each pose says which it is, and that
    decides what may happen to it: a derived pose follows when the other one moves --
    a guess replacing a guess -- and an observed pose is never rewritten unless
    somebody asks (`poses.move_pose`, `poses.derive_pose`).
    """

    OBSERVED = "observed"  # marked, centred or recorded at that instrument
    DERIVED = "derived"  # worked out from the other pose


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

    @property
    def provenance(self) -> Dict[str, PoseProvenance]:
        """Which pose was marked and which worked out, as `Lamella` records it."""
        if self.fluorescence is None:
            return {MILLING_POSE: PoseProvenance.OBSERVED}
        derived = FLUORESCENCE_POSE if self.observed == MILLING_POSE else MILLING_POSE
        return {
            self.observed: PoseProvenance.OBSERVED,
            derived: PoseProvenance.DERIVED,
        }

    def write_to(self, lamella: "Lamella") -> None:
        """Put both poses on *lamella*, each saying where it came from."""
        provenance = self.provenance
        lamella.set_pose(MILLING_POSE, self.milling, provenance[MILLING_POSE])
        if self.fluorescence is not None:
            lamella.set_pose(
                FLUORESCENCE_POSE, self.fluorescence, provenance[FLUORESCENCE_POSE]
            )


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


class Followed(str, Enum):
    """What `move_pose` did with the *other* pose."""

    DERIVED = "derived"  # it was a guess (or asked for), so it was worked out again
    KEPT = "kept"  # somebody observed it; it stays until they ask
    MISSING = "missing"  # there is none, and a move does not invent one
    FAILED = "failed"  # it should have followed, and could not be worked out


def other_pose(name: str) -> str:
    return MILLING_POSE if name == FLUORESCENCE_POSE else FLUORESCENCE_POSE


def move_pose(
    microscope: "FibsemMicroscope",
    lamella: "Lamella",
    name: str,
    position: Optional[FibsemStagePosition] = None,
    state: Optional[MicroscopeState] = None,
    objective_position: Optional[float] = None,
) -> Followed:
    """A person moved one of a lamella's poses. The one write path for that.

    Writes the pose as observed -- *state* replaces it whole, *position* moves it and
    keeps the rest, the objective position most of all -- and then applies the one
    rule about the other pose:

    * it is still **derived**: worked out again from the new one. Nobody has looked
      through that side, and a guess pointing at where the lamella used to be looks
      exactly like a right one.
    * it is **observed**: left alone. Somebody centred it, and the conversion is a
      guess; only `derive_pose`, asked for by name, overwrites it.
    * there is **none**: left that way. A lamella with no fluorescence pose has never
      been looked at under fluorescence, and `fluorescence_selected` asks only whether
      an objective position exists -- conjuring a pose here would make it report
      itself as focused.

    The milling angle follows every write to the milling pose, here rather than at
    each caller. Saving and announcing the change stay with the caller.

    A fluorescence pose moved on a lamella that has none is built on its milling
    pose's state, with *objective_position* if given.
    """
    if name not in (MILLING_POSE, FLUORESCENCE_POSE):
        raise ValueError(f"No pose named {name!r} to move.")
    if state is not None:
        # A recorded microscope state does not capture the objective, so replacing a
        # pose outright would wipe the focus somebody set on it.
        existing = lamella.poses.get(name)
        if state.objective_position is None and existing is not None:
            state.objective_position = existing.objective_position
        lamella.set_pose(name, state)
    elif position is None:
        raise ValueError("move_pose needs a position or a state.")
    elif lamella.poses.get(name) is not None:
        lamella.set_pose_position(name, deepcopy(position))
    else:
        base = lamella.poses.get(other_pose(name))
        if base is None:
            raise ValueError(f"{lamella.name} has no pose to build a {name} pose on.")
        pose = deepcopy(base)
        pose.stage_position = deepcopy(position)
        if name == FLUORESCENCE_POSE:
            pose.objective_position = objective_position
        lamella.set_pose(name, pose)
    if name == MILLING_POSE:
        lamella.update_milling_angle(microscope)

    other = other_pose(name)
    if lamella.poses.get(other) is None:
        return Followed.MISSING
    if lamella.provenance_of(other) is not PoseProvenance.DERIVED:
        return Followed.KEPT
    if derive_pose(microscope, lamella, other):
        return Followed.DERIVED
    return Followed.FAILED


def record_pose(
    microscope: "FibsemMicroscope",
    lamella: "Lamella",
    name: str,
    state: MicroscopeState,
) -> Followed:
    """Record *state* as the named pose: "set the current position as this pose".

    The milling and fluorescence poses go through `move_pose`. Any other named pose
    is a record with no counterpart, and is just written.
    """
    if name in (MILLING_POSE, FLUORESCENCE_POSE):
        return move_pose(microscope, lamella, name, state=state)
    lamella.set_pose(name, state)
    return Followed.MISSING


POSE_NOUNS = {MILLING_POSE: "milling pose", FLUORESCENCE_POSE: "fluorescence pose"}


def followed_note(name: str, followed: Followed) -> str:
    """What to tell a person about the *other* pose after `move_pose`; "" if nothing.

    Said the same way wherever a pose can be moved, so the rule reads as one rule.
    """
    if name not in POSE_NOUNS:
        return ""
    other = POSE_NOUNS[other_pose(name)]
    if followed is Followed.DERIVED:
        return f"Its {other} was worked out again from the new position."
    if followed is Followed.KEPT:
        return f"Its {other} was set by hand and stays where it is."
    if followed is Followed.FAILED:
        return f"Its {other} could not be worked out from here and stays where it was."
    return ""


def move_consequence(lamella: "Lamella", name: str) -> str:
    """What moving the named pose will do to the other one, said *before* the move.

    For a confirmation on a canvas that shows only one side: the person cannot see
    the other pose from there, so they are told what is about to happen to it.
    """
    other = other_pose(name)
    noun = POSE_NOUNS[other]
    if lamella.poses.get(other) is None:
        return ""
    if lamella.provenance_of(other) is PoseProvenance.DERIVED:
        return (
            f"Its {noun} is worked out again from the new position and moves with it."
        )
    return f"Its {noun} was set by hand and stays where it is."


def derive_pose(
    microscope: "FibsemMicroscope",
    lamella: "Lamella",
    name: str,
    orientation: Optional[str] = None,
) -> bool:
    """Overwrite the named pose with one worked out from the other, and say so.

    For a caller that has decided to: `move_pose` when the pose was a guess already,
    a person asking for it by name, a task configured to. Only the stage position is
    derived. A fluorescence pose keeps its objective position -- someone focused on
    this lamella, and moving it sideways is not a reason to throw that away -- or
    takes the objective's configured focus if it had none.

    *orientation* is for the fluorescence pose: the orientation to derive it into.
    Left out, the one it is already in is kept where the FM images from it, so a pose
    chosen at the SEM tilt is not flipped because its lamella moved. The milling pose
    is always derived into the milling orientation, under the beams.

    Returns:
        True if the pose was written. False, with the existing pose untouched, if
        there is nothing to derive it from or the instrument cannot work it out -- a
        wrong milling pose is the dangerous outcome, so that direction refuses unless
        the fluorescence pose is somewhere the objective sees the sample from.
    """
    if name not in (MILLING_POSE, FLUORESCENCE_POSE):
        raise ValueError(f"No derivation for a pose named {name!r}.")
    source = lamella.poses.get(other_pose(name))
    if source is None or source.stage_position is None:
        logging.debug(f"Cannot derive the {name} pose of {lamella.name}: no source.")
        return False
    existing = lamella.poses.get(name)

    if name == FLUORESCENCE_POSE:
        if orientation is None and existing is not None:
            orientation = _kept_orientation(microscope, existing.stage_position)
        position = _to_fluorescence(microscope, source.stage_position, orientation)
        if position is None:
            return False
    else:
        try:
            position = _to_milling(microscope, source.stage_position)
        except ValueError as e:
            logging.warning(f"Could not derive the milling pose of {lamella.name}: {e}")
            return False

    pose = deepcopy(existing if existing is not None else source)
    pose.stage_position = position
    if name == FLUORESCENCE_POSE and (
        existing is None or existing.objective_position is None
    ):
        pose.objective_position = microscope.fm.objective.focus_position
    lamella.set_pose(name, pose, PoseProvenance.DERIVED)
    if name == MILLING_POSE:
        lamella.update_milling_angle(microscope)
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
