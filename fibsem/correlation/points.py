"""The rules for correlation points, without Qt.

Which canvas a point type is drawn on, which types hold one point and clear each
other, how a FIB fiducial pairs with an FM one, what a point is called, and how
an edit changes a point's status. The point store
(:mod:`fibsem.ui.widgets.point_store`) holds the points and announces changes;
it asks this module what each change means, and so do the correlation tab and
its lists, so the rules are written once.

Points are held by identity, never by equality: ``Coordinate`` is a dataclass,
and two coordinates can hold equal values and still be different points. The
status functions change the caller's coordinate in place, because the verdict
notes and the fit code key on ``id(coord)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from fibsem.correlation.config import DELETE_FIDUCIAL_ONE, DELETE_FIDUCIAL_PAIR
from fibsem.correlation.structures import (
    Coordinate,
    PointProvenance,
    PointStatus,
    PointType,
)

__all__ = [
    "PointRule",
    "POINT_RULES",
    "POINT_FIELDS",
    "identity_index",
    "side_of",
    "types_on",
    "cleared_by",
    "check_additions",
    "partner_type",
    "fiducial_partner",
    "removal_for",
    "point_name",
    "point_names",
    "place_by_hand",
    "set_field",
    "apply_refinement",
    "accept",
    "toggle_rejected",
    "reset_to_predicted",
]


@dataclass(frozen=True)
class PointRule:
    """What holds for one point type."""

    side: str  # "fib" | "fm": the canvas the points are drawn on
    max_one: bool = False  # adding replaces the existing point (surfaces)
    exclusive_group: Optional[str] = None  # adding clears the rest of the group


POINT_RULES: Dict[PointType, PointRule] = {
    PointType.FIB: PointRule("fib"),
    PointType.SURFACE: PointRule("fib", max_one=True, exclusive_group="surface"),
    PointType.FM: PointRule("fm"),
    PointType.POI: PointRule("fm"),
    PointType.SURFACE_FM: PointRule("fm", max_one=True, exclusive_group="surface"),
}

# The values a typed edit may set.
POINT_FIELDS = ("x", "y", "z")

# Fiducials pair by their place in the lists: FIB i with FM i.
_PARTNER_TYPES: Dict[PointType, PointType] = {
    PointType.FIB: PointType.FM,
    PointType.FM: PointType.FIB,
}


def identity_index(coords: Sequence[Coordinate], coord: Coordinate) -> Optional[int]:
    """Where ``coord`` is in ``coords``, found by identity, or None."""
    return next((i for i, c in enumerate(coords) if c is coord), None)


# ---------------------------------------------------------------------------
# Point types
# ---------------------------------------------------------------------------


def side_of(point_type: PointType) -> str:
    """The canvas a point type is drawn on, "fib" or "fm"."""
    return POINT_RULES[point_type].side


def types_on(side: str) -> List[PointType]:
    """The point types drawn on one canvas, in ``POINT_RULES`` order."""
    return [pt for pt, rule in POINT_RULES.items() if rule.side == side]


def cleared_by(point_type: PointType) -> List[PointType]:
    """The types emptied when a point of ``point_type`` is added.

    A ``max_one`` type is replaced rather than appended to, so it is in its own
    list; the other members of its exclusive group are cleared too. Adding a
    FIB surface point clears the FM surface point, and the reverse.
    """
    rule = POINT_RULES[point_type]
    out = [point_type] if rule.max_one else []
    if rule.exclusive_group is not None:
        out += [
            pt
            for pt, other in POINT_RULES.items()
            if pt is not point_type and other.exclusive_group == rule.exclusive_group
        ]
    return out


def check_additions(
    held: Dict[PointType, Sequence[Coordinate]], coords: Sequence[Coordinate]
) -> None:
    """Raise ValueError when appending ``coords`` would put a second point in a
    ``max_one`` type. Nothing is replaced: a bulk addition (predictions, a seed)
    with a second surface point is a caller's mistake, not a gesture."""
    for pt, rule in POINT_RULES.items():
        incoming = sum(1 for c in coords if c.point_type is pt)
        if rule.max_one and incoming and len(held.get(pt, ())) + incoming > 1:
            raise ValueError(f"{pt} holds at most one point")


# ---------------------------------------------------------------------------
# Fiducial pairs
# ---------------------------------------------------------------------------


def partner_type(point_type: PointType) -> Optional[PointType]:
    """The type a fiducial pairs with (FIB <-> FM), or None for a type that
    does not pair."""
    return _PARTNER_TYPES.get(point_type)


def fiducial_partner(
    coord: Coordinate, fib: Sequence[Coordinate], fm: Sequence[Coordinate]
) -> Optional[Coordinate]:
    """The fiducial at the same place in the other side's list, if any.

    ``coord`` is found in its own list by identity; a point in neither list,
    a point that does not pair, and one past the end of the other list have no
    partner.
    """
    other = partner_type(coord.point_type)
    if other is None:
        return None
    own, partners = (fib, fm) if coord.point_type is PointType.FIB else (fm, fib)
    index = identity_index(own, coord)
    if index is None or index >= len(partners):
        return None
    return partners[index]


def removal_for(
    coord: Coordinate, partner: Coordinate, choice: str
) -> Optional[List[Coordinate]]:
    """What deleting a paired fiducial removes, for a Delete fiducial choice.

    Removing FIB 3 alone re-pairs FM 4 with FIB 3 and so on down the lists
    (FIB-1243), so the choice is the pair, that point only, or nothing (None,
    for a cancelled question or any other value). Asking is the caller's.
    """
    if choice == DELETE_FIDUCIAL_PAIR:
        return [coord, partner]
    if choice == DELETE_FIDUCIAL_ONE:
        return [coord]
    return None


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def point_name(point_type: PointType, index: int) -> str:
    """The name of the point at ``index`` (0-based) in its type's list:
    ``FIB 1``, ``FM-SURFACE 1``."""
    return f"{point_type.value} {index + 1}"


def point_names(coordinates: Sequence[Coordinate]) -> List[str]:
    """Per-type 1-based names: ``FIB 1``, ``FIB 2``, ``POI 1`` ...

    Numbered within a type rather than across the list, so a point's label does
    not change when an unrelated type gains or loses one.
    """
    counts: Dict[PointType, int] = {}
    names = []
    for c in coordinates:
        index = counts.get(c.point_type, 0)
        counts[c.point_type] = index + 1
        names.append(point_name(c.point_type, index))
    return names


# ---------------------------------------------------------------------------
# Status. Provenance is never written: it is where the point came from, set
# once by whoever made it.
# ---------------------------------------------------------------------------


def place_by_hand(coord: Coordinate) -> None:
    """A drag or a typed value makes the point ``placed``.

    A drop on a prediction is the user's answer, and the projection never
    moves it again. A fitted or accepted point that is moved is no longer
    what the fitter or the projection said. A rejected point stays rejected.
    """
    coord.fitted = False
    if coord.status in PointStatus.TENTATIVE or coord.status in (
        PointStatus.FITTED,
        PointStatus.ACCEPTED,
    ):
        coord.status = PointStatus.PLACED


def set_field(coord: Coordinate, field: str, value: float) -> None:
    """A typed value: the point is the user's now."""
    if field not in POINT_FIELDS:
        raise ValueError(f"unknown field {field!r}")
    setattr(coord.point, field, value)
    place_by_hand(coord)


def apply_refinement(coord: Coordinate, x: float, y: float, z: float) -> None:
    """Commit an accepted refinement of one point: move it to the refined
    position and mark it fitted."""
    coord.point.x, coord.point.y, coord.point.z = x, y, z
    coord.fitted = True
    coord.status = PointStatus.FITTED


def accept(coords: Sequence[Coordinate]) -> List[Coordinate]:
    """Take predictions as they are, unmoved. Returns the points accepted."""
    accepted = [c for c in coords if c.status in PointStatus.TENTATIVE]
    for coord in accepted:
        coord.status = PointStatus.ACCEPTED
    return accepted


def toggle_rejected(coord: Coordinate) -> bool:
    """Leave a point out of the fit, or bring it back. A prediction is not in
    the fit to begin with, so it cannot be rejected. Returns whether the
    status changed."""
    if coord.status == PointStatus.REJECTED:
        coord.status = PointStatus.FITTED if coord.fitted else PointStatus.PLACED
    elif coord.status not in PointStatus.TENTATIVE:
        coord.status = PointStatus.REJECTED
    else:
        return False
    return True


def reset_to_predicted(coord: Coordinate) -> bool:
    """Make a projected point a guess again; the caller re-projects it.
    Returns False, changing nothing, for a point the projection did not make."""
    if coord.provenance != PointProvenance.PROJECTED:
        return False
    coord.status = PointStatus.PREDICTED
    coord.fitted = False
    return True
