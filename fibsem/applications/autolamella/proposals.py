"""AutoLamella's proposal kinds, and how the values they carry are stored and
written to a lamella when a decision confirms them.

The records and registries are in ``fibsem.workflows.proposals``; importing
this module registers what is here.
"""

from __future__ import annotations

import logging
import math
from typing import Any, List

from fibsem.structures import Point
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
    "POINT_OF_INTEREST",
]

# A point on an image: the point of interest the tasks that follow it sync to.
POINT_OF_INTEREST = "point_of_interest"

register_proposal_kind(
    ProposalKind(name=POINT_OF_INTEREST, values=("poi",), label="Point of interest")
)

# ---------------------------------------------------------------------------
# Values: how each named value is stored, and how confirming writes it through
# ---------------------------------------------------------------------------
#
# A value exists in a proposal because a later task consumes it. ``poi`` is
# read by the milling tasks that sync their patterns to it. The name is the
# contract, so codecs and writers are keyed by it rather than by kind.


register_value_codec("poi", _point_to_dict, _point_from_dict)


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


# name -> prepare(experiment, item, value): checks the value and plans every
# effect without touching anything, returning how to apply it and how to undo
# it. Positions on an overview are a built-in value (fibsem.workflows.proposals);
# the lamellae they make are planned by Experiment.prepare_items_at.
register_value_writer("poi", _prepare_poi)
