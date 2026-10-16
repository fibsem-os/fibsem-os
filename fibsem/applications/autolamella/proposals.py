"""AutoLamella's proposal kinds, and how the values they carry are stored and
written to a lamella or the experiment when a decision confirms them.

The records and registries are in ``fibsem.workflows.proposals``; importing
this module registers what is here.
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Any, List

from fibsem.applications.autolamella.poses import LamellaPoses
from fibsem.structures import (
    FibsemRectangle,
    FibsemStagePosition,
    MicroscopeState,
    Point,
)
from fibsem.workflows.proposals import (
    PreparedWrite,
    ProposalKind,
    ValueRefused,
    _quietly,
    register_proposal_kind,
    register_value_codec,
    register_value_writer,
)

__all__ = [
    "ALIGNMENT_AREA",
    "DETECTION",
    "OVERVIEW_POSITIONS",
    "POINT_OF_INTEREST",
    "STATE",
    "features_to_dict",
]

# A point on an image: the point of interest the tasks that follow it sync to.
POINT_OF_INTEREST = "point_of_interest"
# Where the lamellae go on a grid: the positions a decision creates them at.
# The only kind whose confirmed values make items rather than edit the one the
# proposal is on.
OVERVIEW_POSITIONS = "overview_positions"
# Where a model put the features it was asked to find, for someone to correct:
# the one kind so far that is asked *during* a task rather than after it
# (FIB-1025). Its value is the feature set, keyed by name, so a delta is per
# feature and not one number for the lot.
DETECTION = "detection"
# The instrument's state as a task arrived at it, for the operator to confirm
# before the task goes on ("Acquire reference image. Press Continue when
# ready."). The proposal is the stage position then; the decision is a
# confirmation, and the task fills in the position as it stands afterwards,
# so a move the operator made first is the delta. Asked mid-task, so its
# value is written nowhere: the task is what uses it.
STATE = "state"
# The reduced area the alignment reference is taken in: a rectangle in
# fractions of the FIB frame, dragged on the last FIB image. Asked mid-task,
# right before that reference is acquired, so it is written nowhere: the
# task acquires with the area it is handed (FIB-1053).
ALIGNMENT_AREA = "alignment_area"

register_proposal_kind(
    ProposalKind(name=POINT_OF_INTEREST, values=("poi",), label="Point of interest")
)
register_proposal_kind(
    ProposalKind(
        name=OVERVIEW_POSITIONS, values=("positions",), label="Lamella positions"
    )
)
register_proposal_kind(
    ProposalKind(name=DETECTION, values=("features",), label="Detection")
)
register_proposal_kind(
    ProposalKind(name=STATE, values=("stage_position",), label="Position")
)
register_proposal_kind(
    ProposalKind(
        name=ALIGNMENT_AREA, values=("alignment_area",), label="Alignment area"
    )
)

# ---------------------------------------------------------------------------
# Values: how each named value is stored, and how confirming writes it through
# ---------------------------------------------------------------------------
#
# A value exists in a proposal because a later task consumes it. ``poi`` is
# read by the milling tasks that sync their patterns to it. The name is the
# contract, so codecs and writers are keyed by it rather than by kind.


# A position a decision creates a lamella at is carried as the pose pair the
# instrument's geometry produced for it (``build_lamella_poses``), which is
# what ``Experiment.add_new_lamella`` takes.
#
# Not a point on an image and not a bare stage position: either would have
# made a decision need a microscope to turn into poses, and there is no
# instrument where a decision is made -- the Review tab days later, or an
# agent over the server. The geometry is read where the position is placed
# instead, in the renderer or in a proposer running at the beam.


def _positions_to_dict(value: Any) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    return [
        {
            "milling": p.milling.to_dict(),
            "fluorescence": (
                p.fluorescence.to_dict() if p.fluorescence is not None else None
            ),
        }
        if isinstance(p, LamellaPoses)
        else p
        for p in value
    ]


def _positions_from_dict(value: Any) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    out: List[Any] = []
    for p in value:
        if not isinstance(p, dict) or "milling" not in p:
            out.append(p)
            continue
        fm = p.get("fluorescence")
        out.append(
            LamellaPoses(
                milling=MicroscopeState.from_dict(p["milling"]),
                fluorescence=MicroscopeState.from_dict(fm) if fm else None,
            )
        )
    return out


def features_to_dict(value: Any) -> Any:
    """A detected feature set as ``[{name, px}]``.

    Only the name and the pixel it sits on: the mask, the rgb and the image
    are megabytes and already on disk beside the task's outputs, and a delta
    is computed from the points alone.
    """
    if not isinstance(value, (list, tuple)):
        return value
    return [
        {"name": f["name"], "px": _point_to_dict(f["px"])} if isinstance(f, dict) else f
        for f in value
    ]


def _features_from_dict(value: Any) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    return [
        {"name": f.get("name", ""), "px": _point_from_dict(f.get("px"))}
        if isinstance(f, dict)
        else f
        for f in value
    ]


def _point_to_dict(p: Any) -> Any:
    return p.to_dict() if isinstance(p, Point) else p


def _point_from_dict(d: Any) -> Any:
    return Point.from_dict(d) if isinstance(d, dict) and "x" in d else d


register_value_codec("poi", _point_to_dict, _point_from_dict)
register_value_codec("positions", _positions_to_dict, _positions_from_dict)
register_value_codec("features", features_to_dict, _features_from_dict)
register_value_codec(
    "stage_position",
    lambda p: p.to_dict() if isinstance(p, FibsemStagePosition) else p,
    lambda d: FibsemStagePosition.from_dict(d) if isinstance(d, dict) else d,
)
register_value_codec(
    "alignment_area",
    lambda r: r.to_dict() if isinstance(r, FibsemRectangle) else r,
    lambda d: FibsemRectangle.from_dict(d) if isinstance(d, dict) else d,
)


def _prepare_poi(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """The GUI's move path, planned in full before any of it happens: the new
    point, then every pattern that follows it. Same domain plan, same order --
    a write that bypassed the sync left the rough and polishing patterns
    detached from the new point."""
    if not isinstance(value, Point):
        raise ValueRefused(f"poi must be a Point, not {type(value).__name__}.")
    for axis in (value.x, value.y):
        if isinstance(axis, bool) or not isinstance(axis, (int, float)):
            raise ValueRefused(f"poi must have numeric x and y, not {value!r}.")
        if not math.isfinite(axis):
            raise ValueRefused(f"poi must be finite, not {value!r}.")
    plan = getattr(item, "poi_sync_plan", None)
    if plan is None:
        raise ValueRefused(f"{getattr(item, 'name', 'this item')} has no poi.")
    synced, moves = plan(value)
    old_poi = item.poi
    old_points = [(pattern, pattern.point) for pattern, _ in moves]

    def apply() -> List[str]:
        item.poi = value
        for pattern, moved in moves:
            pattern.point = moved
        if synced:
            logging.info(f"Synced tasks to POI: {synced}")
        return list(synced)

    def undo() -> None:
        with _quietly(item):
            item.poi = old_poi
        for pattern, point in reversed(old_points):
            with _quietly(pattern):
                pattern.point = point

    return PreparedWrite(apply=apply, undo=undo)


def _prepare_positions(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """The only write that makes items: one lamella per position, on the grid
    the proposal is on.

    Planned in full before any of it happens, like every other write here, and
    undone together: a half-created set is worse than none, and the decision is
    only appended once the whole set is on the experiment. The poses come from
    the value rather than the instrument (see the codec above), so nothing here
    needs a microscope.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueRefused(f"positions must be a list, not {type(value).__name__}.")
    placed: List[LamellaPoses] = []
    for entry in value:
        if not isinstance(entry, LamellaPoses):
            raise ValueRefused(
                f"every position must be a LamellaPoses, not {type(entry).__name__}."
            )
        if entry.milling is None:
            raise ValueRefused("every position needs a milling pose.")
        placed.append(entry)
    add = getattr(experiment, "add_new_lamella", None)
    if add is None:
        raise ValueRefused("positions can only be written to an experiment.")
    grid_id = getattr(item, "id", None)
    if grid_id is None:
        raise ValueRefused(
            f"{getattr(item, 'name', 'this item')} has no id to stamp on a lamella."
        )
    protocol = getattr(experiment, "task_protocol", None)
    if placed and getattr(protocol, "lamella_defaults", None) is None:
        # Checked here, not discovered half way through creating them: a
        # lamella is built from the protocol's defaults, and an experiment
        # with no protocol loaded cannot make one.
        raise ValueRefused(
            "No protocol is loaded, so a lamella cannot be created. Load one "
            "for this experiment and decide again."
        )
    made: List[Any] = []

    def apply() -> List[str]:
        for entry in placed:
            # add_new_lamella returns nothing and appends, so the lamella it
            # made is the one on the end. Taken here rather than assumed later:
            # undo has to remove exactly what this write added.
            add(
                microscope_state=entry.milling,
                task_config=deepcopy(getattr(protocol, "task_config", None)) or {},
                fluorescence_pose=entry.fluorescence,
                grid_id=grid_id,
            )
            made.append(experiment.positions[-1])
        if made:
            logging.info(
                f"Created {len(made)} lamella(e) on {getattr(item, 'name', '?')}: "
                + ", ".join(getattr(m, "name", "?") or "?" for m in made)
            )
        return []

    def undo() -> None:
        # Removing what this write made, newest first. Nothing else can have
        # taken them: the decision has not been appended and the experiment
        # lock is still held.
        for lamella in reversed(made):
            try:
                experiment.positions.remove(lamella)
            except ValueError:
                logging.warning(
                    f"Could not undo the creation of {getattr(lamella, 'name', '?')}"
                )
        made.clear()

    return PreparedWrite(apply=apply, undo=undo)


def _prepare_features(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """Checked, and written nowhere.

    Every other writer here edits the item, because its proposal is decided
    after the task that made it has ended and there is nobody left to act on
    the answer. A detection is asked *during* a task (FIB-1025): the consumer
    is the task itself, parked on the answer, which applies it with the
    instrument in the state it asked in. Writing it to the item here would be
    a second, later application of the same correction.

    So the value is still checked -- a malformed answer must be refused before
    the run is released on it -- and then left for its waiter.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueRefused(f"features must be a list, not {type(value).__name__}.")
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueRefused(
                f"every feature must be a name and a point, not {type(entry).__name__}."
            )
        if not entry.get("name"):
            raise ValueRefused("every feature needs a name to be matched back by.")
        if not isinstance(entry.get("px"), Point):
            raise ValueRefused(f"{entry['name']} needs a point in image pixels.")
    return PreparedWrite.nothing()


# name -> prepare(experiment, item, value): checks the value and plans every
# effect without touching anything, returning how to apply it and how to undo
# it. The experiment is there for the one write that makes items rather than
# editing the one the proposal is on.
def _prepare_stage_position(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """Checked, and written nowhere, for the reason ``_prepare_features`` gives:
    a state is confirmed mid-task, and the instrument already holds it."""
    if not isinstance(value, FibsemStagePosition):
        raise ValueRefused(
            f"stage_position must be a stage position, not {type(value).__name__}."
        )
    return PreparedWrite.nothing()


def _prepare_alignment_area(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """Checked, and written nowhere, for the reason ``_prepare_features``
    gives: the task acquires its alignment reference with the area it is
    handed. A rectangle that is not a valid reduced area is refused here,
    before the run is released on it."""
    if not isinstance(value, FibsemRectangle):
        raise ValueRefused(
            f"alignment_area must be a rectangle, not {type(value).__name__}."
        )
    if not value.is_valid_reduced_area:
        raise ValueRefused(
            f"alignment_area {value.pretty_string} is not inside the frame."
        )
    return PreparedWrite.nothing()


register_value_writer("poi", _prepare_poi)
register_value_writer("positions", _prepare_positions)
register_value_writer("features", _prepare_features)
register_value_writer("stage_position", _prepare_stage_position)
register_value_writer("alignment_area", _prepare_alignment_area)
