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

So the orientation is read off the position rather than assumed. It is already there,
in the rotation and tilt, which is how `get_target_position` works at all.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from fibsem.structures import FibsemStagePosition, MicroscopeState

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.microscope import FibsemMicroscope

# The orientation a lamella is milled at, when one has to be chosen rather than
# recovered. See `build_lamella_poses` for when that happens.
MILLING_ORIENTATION = "MILLING"

FLUORESCENCE_ORIENTATION = "FM"


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
        marked_at = microscope.get_stage_orientation(state.stage_position)

    if marked_at == FLUORESCENCE_ORIENTATION:
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


def _to_milling(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> FibsemStagePosition:
    """A fluorescence position re-posed for milling.

    The canonical milling orientation, not a recovered one: a fluorescence pose does not
    record which beam orientation it came from -- the transform only rewrites rotation
    and tilt, so a lamella marked at the SEM and one marked at the milling angle both
    arrive at t = -180 and are indistinguishable afterwards. For a target found *in*
    fluorescence there is no earlier pose to recover anyway, and the orientation milling
    actually happens at is the only defensible answer.
    """
    # Checked here rather than left to `get_target_position` to complain, because it
    # will not. That function re-derives the orientation from the position, and on an
    # offset mount a fluorescence position already classifies as MILLING -- so the
    # conversion returns early, unchanged, and the caller gets its own position back as
    # somewhere to mill. The precondition is about the *system*, so ask the system.
    if not microscope.stage_is_compustage:
        raise ValueError(
            "Cannot mark a lamella from the fluorescence view on this system: there is "
            "no transform between the fluorescence and beam positions on an offset "
            "mount, so the milling pose cannot be derived (FIB-93). Mark it from the "
            "beam side instead."
        )
    try:
        return microscope.get_target_position(
            stage_position=deepcopy(position),
            target_orientation=MILLING_ORIENTATION,
        )
    except ValueError as e:
        raise ValueError(
            f"Could not derive a milling pose from the fluorescence position: {e}"
        ) from e


def _to_fluorescence(
    microscope: "FibsemMicroscope", position: FibsemStagePosition
) -> Optional[FibsemStagePosition]:
    """A beam position re-posed for fluorescence, or None if it cannot be.

    Unavailability is not an error here, unlike the other direction. A beam-side caller
    is marking somewhere to mill and the fluorescence pose is a convenience; refusing
    the whole lamella because the system cannot work one out would break marking
    lamellae on every offset system.
    """
    if microscope.fm is None:
        return None
    try:
        return microscope.get_target_position(
            stage_position=deepcopy(position),
            target_orientation=microscope.fm.default_orientation,
        )
    except ValueError as e:
        logging.debug(f"Could not derive a fluorescence pose for {position}: {e}")
        return None
