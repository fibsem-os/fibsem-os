"""The correlation points and their rules, without Qt.

Which canvas a point type is drawn on, which types hold one point and clear each
other, how a FIB fiducial pairs with an FM one, what a point is called, and how
an edit changes a point's status. :class:`CorrelationPoints` holds the points
and tells its subscribers of each change; the Qt point store
(:mod:`fibsem.ui.widgets.point_store`) turns those into signals. The correlation
tab and its lists ask this module what a change means too, so the rules are
written once.

Points are held by identity, never by equality: ``Coordinate`` is a dataclass,
and two coordinates can hold equal values and still be different points. The
status functions change the caller's coordinate in place, because the verdict
notes and the fit code key on ``id(coord)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

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
    "CorrelationPoints",
    "STRUCTURE",
    "VALUES",
    "SELECTION",
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

    Removing FIB 3 alone re-pairs FM 4 with FIB 3 and so on down the lists,
    so the choice is the pair, that point only, or nothing (None,
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


# ---------------------------------------------------------------------------
# The points
# ---------------------------------------------------------------------------

# One selected point per canvas, not one overall: a fiducial named in the
# verdict selects its pair, the FM point and its FIB partner, so that both
# canvases show it. Multi-select within a canvas is not built. Selection is a
# tuple and a selection change carries no points so that raising this changes
# the views, not the subscribers.
MAX_SELECTION_PER_SIDE = 1

# What a subscriber is told changed (see CorrelationPoints.subscribe).
STRUCTURE = "structure"  # points added, removed, reordered or replaced
VALUES = "values"  # the given points changed value or status
SELECTION = "selection"  # read ``selection`` / ``current``

_Coords = Union[Coordinate, Sequence[Coordinate], None]
Subscriber = Callable[[str, Tuple[Coordinate, ...]], None]


def _as_list(coords: _Coords) -> List[Coordinate]:
    if coords is None:
        return []
    if isinstance(coords, Coordinate):
        return [coords]
    return list(coords)


def _unique(coords: Iterable[Coordinate]) -> List[Coordinate]:
    out: List[Coordinate] = []
    for coord in coords:
        if identity_index(out, coord) is None:
            out.append(coord)
    return out


class CorrelationPoints:
    """The correlation points, their order, and the selection.

    The correlation widget held the same points in three places: the list widget, the
    canvas overlay, and the tab widget relaying between them. Three bugs in a week were
    the copies disagreeing. This is the single owner; the Qt point store
    (:mod:`fibsem.ui.widgets.point_store`) announces its changes as signals to the lists
    and canvases.

    Selection is one value across every list and both canvases, so it lives
    here and not per point type. A canvas shows several point types, so this
    answers both ``of_type`` (what a list renders) and ``on_side`` (what a
    canvas renders). What a change means is the rules above; this holds the
    points, the order and the selection, and tells its subscribers.

    Every operation finishes its mutation before it tells, so a subscriber
    always reads the state it is told about. A bulk operation tells once.
    The store holds the caller's objects and never replaces them, because the
    verdict notes and the fit code key on ``id(coord)`` and mutate them in place.
    """

    def __init__(self) -> None:
        self._points: Dict[PointType, List[Coordinate]] = {pt: [] for pt in POINT_RULES}
        self._selection: Tuple[Coordinate, ...] = ()
        self._removal_policy: Optional[
            Callable[[List[Coordinate]], Optional[List[Coordinate]]]
        ] = None
        self._subscribers: List[Subscriber] = []

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """Call ``callback(change, coords)`` on every change from now on;
        returns a disposer. ``change`` is :data:`STRUCTURE`, :data:`VALUES`
        (``coords`` are the points that changed) or :data:`SELECTION`."""
        self._subscribers.append(callback)

        def dispose() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return dispose

    def _tell(self, change: str, coords: Tuple[Coordinate, ...] = ()) -> None:
        for subscriber in list(self._subscribers):
            subscriber(change, coords)

    def set_removal_policy(
        self,
        policy: Optional[Callable[[List[Coordinate]], Optional[List[Coordinate]]]],
    ) -> Callable[[], None]:
        """Decide, before a removal, what it removes: given the points asked
        for, return the points to remove (more, for a fiducial's partner), or
        None to remove nothing. Every view removes through :meth:`remove`, so a
        canvas, a list row and the Delete key all go through it.

        Returns an uninstaller that restores the default (remove what is asked
        for) while ``policy`` is still the one installed: a view that asks the
        user calls it when it goes, so a later removal does not ask a window
        that no longer exists."""
        self._removal_policy = policy

        def uninstall() -> None:
            if self._removal_policy is policy:
                self._removal_policy = None

        return uninstall

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def of_type(self, point_type: PointType) -> List[Coordinate]:
        return list(self._points[point_type])

    def on_side(self, side: str) -> List[Coordinate]:
        out: List[Coordinate] = []
        for pt in types_on(side):
            out += self._points[pt]
        return out

    @property
    def coordinates(self) -> List[Coordinate]:
        out: List[Coordinate] = []
        for coords in self._points.values():
            out += coords
        return out

    @property
    def selection(self) -> Tuple[Coordinate, ...]:
        return self._selection

    @property
    def current(self) -> Optional[Coordinate]:
        """The last point selected: what refit, reset and the row highlight use."""
        return self._selection[-1] if self._selection else None

    def selected_on(self, side: str) -> Optional[Coordinate]:
        """The selected point drawn on one canvas, or None."""
        for coord in reversed(self._selection):
            if side_of(coord.point_type) == side:
                return coord
        return None

    def selected_of_type(self, point_type: PointType) -> Optional[Coordinate]:
        """The selected point in one list, or None."""
        for coord in reversed(self._selection):
            if coord.point_type is point_type:
                return coord
        return None

    def index_of(self, coord: Coordinate) -> Optional[int]:
        """The coordinate's row within its own point type, or None."""
        return identity_index(self._points[coord.point_type], coord)

    def __contains__(self, coord: object) -> bool:
        return isinstance(coord, Coordinate) and self.index_of(coord) is not None

    def __len__(self) -> int:
        return sum(len(coords) for coords in self._points.values())

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def add(self, coord: Coordinate) -> None:
        """Add a point under its type's rules, and select it.

        A ``max_one`` type is replaced rather than appended to, and the other
        members of its exclusive group are cleared (:func:`cleared_by`).
        """
        self._require_absent([coord])
        for pt in cleared_by(coord.point_type):
            self._points[pt] = []
        self._points[coord.point_type].append(coord)
        selection_changed = self._set_selection([coord])
        self._tell(STRUCTURE)
        if selection_changed:
            self._tell(SELECTION)

    def add_many(self, coords: Sequence[Coordinate]) -> None:
        """Append points without touching the selection (predictions, seeds).

        Applies no replacement: a second point of a ``max_one`` type is a
        caller's mistake here, not a gesture to interpret.
        """
        coords = _unique(coords)
        if not coords:
            return
        self._require_absent(coords)
        check_additions(self._points, coords)
        for coord in coords:
            self._points[coord.point_type].append(coord)
        self._tell(STRUCTURE)

    def remove(self, coord: Coordinate) -> bool:
        """Whether ``coord`` was removed -- possibly with its partner, which the
        removal policy may add."""
        return coord in self and self.remove_many([coord]) >= 1 and coord not in self

    def remove_many(self, coords: Sequence[Coordinate]) -> int:
        """Remove the points that are here and select the neighbour.

        The neighbour is the point now at the lowest removed row, clamped to the
        end of that type's list; for one point that is the row that followed
        it, or the new last row. The type is the current selection's if it was
        removed, otherwise the first removed point's.

        The removal is told before the new selection: a view answers the
        removal by rebuilding, which would wipe a selection told first.
        Returns how many points were removed.
        """
        present = [c for c in _unique(coords) if c in self]
        if not present:
            return 0
        if self._removal_policy is not None:
            decided = self._removal_policy(present)
            if not decided:
                return 0
            present = [c for c in _unique(decided) if c in self]
        anchor = present[0]
        if (
            self.current is not None
            and identity_index(present, self.current) is not None
        ):
            anchor = self.current
        anchor_type = anchor.point_type
        lowest = min(self.index_of(c) for c in present if c.point_type is anchor_type)

        for coord in present:
            del self._points[coord.point_type][self.index_of(coord)]

        remaining = self._points[anchor_type]
        neighbour = remaining[min(lowest, len(remaining) - 1)] if remaining else None
        selection_changed = self._set_selection(_as_list(neighbour))
        self._tell(STRUCTURE)
        if selection_changed:
            self._tell(SELECTION)
        return len(present)

    def reorder(self, point_type: PointType, coords: Sequence[Coordinate]) -> None:
        """Put one type's points in the given order; the same points, by identity."""
        coords = list(coords)
        held = self._points[point_type]
        if len(coords) != len(held) or any(
            identity_index(coords, c) is None for c in held
        ):
            raise ValueError(f"reorder of {point_type} must keep the same points")
        if all(a is b for a, b in zip(coords, held)):
            return
        self._points[point_type] = coords
        self._tell(STRUCTURE)

    def replace_type(self, point_type: PointType, coords: Sequence[Coordinate]) -> None:
        """Replace one type's points. Selected points that are gone are
        deselected; nothing is selected in their place.

        What assigning ``CoordinateListWidget.coordinates`` does while the tab
        widget still writes whole lists.
        """
        coords = _unique(coords)
        for coord in coords:
            if coord.point_type is not point_type:
                raise ValueError(f"{coord} is not a {point_type} point")
        self._points[point_type] = coords
        kept = [c for c in self._selection if c in self]
        selection_changed = self._set_selection(kept)
        self._tell(STRUCTURE)
        if selection_changed:
            self._tell(SELECTION)

    def replace_all(self, coords: Iterable[Coordinate]) -> None:
        """Replace every point and clear the selection (a load or a seed).

        Selection is an explicit operation: nothing is selected as a side effect
        of replacing the points.
        """
        coords = _unique(coords)
        self._points = {pt: [] for pt in POINT_RULES}
        for coord in coords:
            self._points[coord.point_type].append(coord)
        selection_changed = self._set_selection([])
        self._tell(STRUCTURE)
        if selection_changed:
            self._tell(SELECTION)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, coords: _Coords, extend: bool = False) -> None:
        """Select the given points, or none. ``extend`` adds to the selection,
        replacing what was selected on the same canvas."""
        coords = _as_list(coords)
        self._require_present(coords)
        if extend:
            coords = list(self._selection) + coords
        if self._set_selection(coords):
            self._tell(SELECTION)

    def deselect(self, coords: _Coords) -> None:
        """Take points out of the selection and leave the rest of it."""
        gone = _as_list(coords)
        kept = [c for c in self._selection if identity_index(gone, c) is None]
        if self._set_selection(kept):
            self._tell(SELECTION)

    def _set_selection(self, coords: Sequence[Coordinate]) -> bool:
        # last occurrence wins, so re-selecting a point makes it current
        ordered = _unique(reversed(list(coords)))[::-1]
        kept: List[Coordinate] = []
        per_side: Dict[str, int] = {}
        for coord in reversed(ordered):
            side = side_of(coord.point_type)
            if per_side.get(side, 0) < MAX_SELECTION_PER_SIDE:
                per_side[side] = per_side.get(side, 0) + 1
                kept.append(coord)
        new = tuple(reversed(kept))
        if len(new) == len(self._selection) and all(
            a is b for a, b in zip(new, self._selection)
        ):
            return False
        self._selection = new
        return True

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    def move(self, coord: Coordinate, x: float, y: float) -> None:
        """A finished drag: the point is the user's now."""
        self._require_present([coord])
        coord.point.x = x
        coord.point.y = y
        place_by_hand(coord)
        self._tell(VALUES, (coord,))

    def move_many(self, coords: Sequence[Coordinate], dx: float, dy: float) -> None:
        coords = _unique(coords)
        if not coords:
            return
        self._require_present(coords)
        for coord in coords:
            coord.point.x += dx
            coord.point.y += dy
            place_by_hand(coord)
        self._tell(VALUES, tuple(coords))

    def set_field(self, coord: Coordinate, field: str, value: float) -> None:
        """A typed value: the point is the user's now."""
        self._require_present([coord])
        set_field(coord, field, value)
        self._tell(VALUES, (coord,))

    def notify_changed(self, coords: _Coords) -> None:
        """Tell of points that were changed in place by code outside this
        (``place_predictions`` moves coordinates itself). No status change."""
        coords = _unique(_as_list(coords))
        if not coords:
            return
        self._require_present(coords)
        self._tell(VALUES, tuple(coords))

    # ------------------------------------------------------------------
    # Status transitions: the rule is the function of the same name above;
    # this checks the point is here and tells of the change.
    # ------------------------------------------------------------------

    def apply_refinement(self, coord: Coordinate, x: float, y: float, z: float) -> None:
        """Commit an accepted refinement of one point: move it and mark it
        fitted."""
        self._require_present([coord])
        apply_refinement(coord, x, y, z)
        self._tell(VALUES, (coord,))

    def accept(self, coords: _Coords) -> List[Coordinate]:
        """Take predictions as they are, unmoved. Returns the points accepted."""
        coords = _unique(_as_list(coords))
        self._require_present(coords)
        accepted = accept(coords)
        if accepted:
            self._tell(VALUES, tuple(accepted))
        return accepted

    def toggle_rejected(self, coord: Coordinate) -> bool:
        """Leave a point out of the fit, or bring it back. Returns True when
        the status changed, False too for a point that is not here: this is
        called from a row's menu, and an exception in a Qt slot aborts the app."""
        if coord not in self or not toggle_rejected(coord):
            return False
        self._tell(VALUES, (coord,))
        return True

    def reset_to_predicted(self, coord: Coordinate) -> bool:
        """Make a projected point a guess again. The caller re-projects it and
        then calls ``notify_changed``: this has no transform. Returns False
        for a point the projection did not make, or one that is not here."""
        if coord not in self or not reset_to_predicted(coord):
            return False
        self._tell(VALUES, (coord,))
        return True

    # ------------------------------------------------------------------

    def _require_present(self, coords: Sequence[Coordinate]) -> None:
        for coord in coords:
            if coord not in self:
                raise ValueError(f"{coord} is not in the store")

    def _require_absent(self, coords: Sequence[Coordinate]) -> None:
        for coord in coords:
            if coord in self:
                raise ValueError(f"{coord} is already in the store")
