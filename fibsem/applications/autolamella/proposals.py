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
    MicroscopeState,
    Point,
)
from fibsem.workflows.proposals import (
    PreparedWrite,
    ProposalKind,
    ValueRefused,
    _point_from_dict,
    _point_to_dict,
    _quietly,
    register_proposal_kind,
    register_value_codec,
    register_value_writer,
)

__all__ = [
    "OVERVIEW_POSITIONS",
    "POINT_OF_INTEREST",
]

# A point on an image: the point of interest the tasks that follow it sync to.
POINT_OF_INTEREST = "point_of_interest"
# Where the lamellae go on a grid: the positions a decision creates them at.
# The only kind whose confirmed values make items rather than edit the one the
# proposal is on.
OVERVIEW_POSITIONS = "overview_positions"

register_proposal_kind(
    ProposalKind(name=POINT_OF_INTEREST, values=("poi",), label="Point of interest")
)
register_proposal_kind(
    ProposalKind(
        name=OVERVIEW_POSITIONS, values=("positions",), label="Lamella positions"
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


register_value_codec("poi", _point_to_dict, _point_from_dict)
register_value_codec("positions", _positions_to_dict, _positions_from_dict)


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


# name -> prepare(experiment, item, value): checks the value and plans every
# effect without touching anything, returning how to apply it and how to undo
# it. The experiment is there for the one write that makes items rather than
# editing the one the proposal is on.
register_value_writer("poi", _prepare_poi)
register_value_writer("positions", _prepare_positions)
