"""One store for the correlation points (FIB-973).

The correlation widget has held the same points in three places: the list
widget, the canvas overlay, and the tab widget relaying between them. Three
bugs in a week were the copies disagreeing (FIB-958, FIB-965, FIB-972). This is
the single owner the lists and the canvases draw from; the overview's
image-from-points dialog uses it too, for its two canvases.

Two facts shape it. Selection is one value across every list and both canvases,
so it lives here and not per point type. A canvas shows several point types, so
the store answers both ``of_type`` (what a list renders) and ``on_side`` (what a
canvas renders).

What a change means -- which types replace or clear each other, how an edit
changes a point's status -- is :mod:`fibsem.correlation.points`. This holds the
points, the order and the selection, and announces each change.

Every operation finishes its mutation before it emits, so a slot always reads
the state the signal announces. A bulk operation emits once.

Points are held by identity, never by equality: ``Coordinate`` is a dataclass,
and two coordinates can hold equal values and still be different points. The
store holds the caller's objects and never replaces them, because the verdict
notes and the fit code key on ``id(coord)`` and mutate the objects in place.
"""

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.correlation import points
from fibsem.correlation.points import POINT_RULES, identity_index
from fibsem.correlation.structures import Coordinate, PointType

# One selected point per canvas, not one overall: a fiducial named in the
# verdict selects its pair, the FM point and its FIB partner, so that both
# canvases show it. Multi-select within a canvas is not built. Selection is a
# tuple and ``selection_changed`` carries no payload so that raising this
# changes the views, not the store's subscribers.
MAX_SELECTION_PER_SIDE = 1

_Coords = Union[Coordinate, Sequence[Coordinate], None]


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


class PointStore(QObject):
    """The correlation points, their order, and the selection.

    Signals
    -------
    structure_changed()
        Points were added, removed, reordered or replaced. Views rebuild.
    points_changed(tuple)
        The given coordinates changed value or status. Views refresh those rows
        and artists only: rebuilding a list on a typed value would destroy the
        spinbox that has focus.
    selection_changed()
        Read ``selection`` / ``current``.
    """

    structure_changed = pyqtSignal()
    points_changed = pyqtSignal(tuple)  # Tuple[Coordinate, ...]
    selection_changed = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._points: Dict[PointType, List[Coordinate]] = {pt: [] for pt in POINT_RULES}
        self._selection: Tuple[Coordinate, ...] = ()
        self._removal_policy: Optional[
            Callable[[List[Coordinate]], Optional[List[Coordinate]]]
        ] = None

    def set_removal_policy(
        self,
        policy: Optional[Callable[[List[Coordinate]], Optional[List[Coordinate]]]],
    ) -> None:
        """Decide, before a removal, what it removes: given the points asked
        for, return the points to remove (more, for a fiducial's partner), or
        None to remove nothing. Every view removes through :meth:`remove`, so a
        canvas, a list row and the Delete key all go through it (FIB-1243)."""
        self._removal_policy = policy

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def of_type(self, point_type: PointType) -> List[Coordinate]:
        return list(self._points[point_type])

    def on_side(self, side: str) -> List[Coordinate]:
        out: List[Coordinate] = []
        for pt in points.types_on(side):
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
            if points.side_of(coord.point_type) == side:
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
        members of its exclusive group are cleared (``points.cleared_by``).
        """
        self._require_absent([coord])
        for pt in points.cleared_by(coord.point_type):
            self._points[pt] = []
        self._points[coord.point_type].append(coord)
        selection_changed = self._set_selection([coord])
        self.structure_changed.emit()
        if selection_changed:
            self.selection_changed.emit()

    def add_many(self, coords: Sequence[Coordinate]) -> None:
        """Append points without touching the selection (predictions, seeds).

        Applies no replacement: a second point of a ``max_one`` type is a
        caller's mistake here, not a gesture to interpret.
        """
        coords = _unique(coords)
        if not coords:
            return
        self._require_absent(coords)
        points.check_additions(self._points, coords)
        for coord in coords:
            self._points[coord.point_type].append(coord)
        self.structure_changed.emit()

    def remove(self, coord: Coordinate) -> bool:
        """Whether ``coord`` was removed -- possibly with its partner, which the
        removal policy may add."""
        return coord in self and self.remove_many([coord]) >= 1 and coord not in self

    def remove_many(self, coords: Sequence[Coordinate]) -> int:
        """Remove the points that are here and select the neighbour.

        The neighbour is the point now at the lowest removed row, clamped to the
        end of that type's list; for one point that is the row that followed
        it, or the new last row. The type is the current selection's if it was
        removed, otherwise the first removed point's. Selecting row 1 instead
        was FIB-965.

        The removal is announced before the new selection: a view answers the
        removal by rebuilding, which would wipe a selection announced first.
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
        self.structure_changed.emit()
        if selection_changed:
            self.selection_changed.emit()
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
        self.structure_changed.emit()

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
        self.structure_changed.emit()
        if selection_changed:
            self.selection_changed.emit()

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
        self.structure_changed.emit()
        if selection_changed:
            self.selection_changed.emit()

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
            self.selection_changed.emit()

    def deselect(self, coords: _Coords) -> None:
        """Take points out of the selection and leave the rest of it."""
        gone = _as_list(coords)
        kept = [c for c in self._selection if identity_index(gone, c) is None]
        if self._set_selection(kept):
            self.selection_changed.emit()

    def _set_selection(self, coords: Sequence[Coordinate]) -> bool:
        # last occurrence wins, so re-selecting a point makes it current
        ordered = _unique(reversed(list(coords)))[::-1]
        kept: List[Coordinate] = []
        per_side: Dict[str, int] = {}
        for coord in reversed(ordered):
            side = points.side_of(coord.point_type)
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
        points.place_by_hand(coord)
        self.points_changed.emit((coord,))

    def move_many(self, coords: Sequence[Coordinate], dx: float, dy: float) -> None:
        coords = _unique(coords)
        if not coords:
            return
        self._require_present(coords)
        for coord in coords:
            coord.point.x += dx
            coord.point.y += dy
            points.place_by_hand(coord)
        self.points_changed.emit(tuple(coords))

    def set_field(self, coord: Coordinate, field: str, value: float) -> None:
        """A typed value: the point is the user's now."""
        self._require_present([coord])
        points.set_field(coord, field, value)
        self.points_changed.emit((coord,))

    def notify_changed(self, coords: _Coords) -> None:
        """Announce points that were changed in place by code outside the store
        (``place_predictions`` moves coordinates itself). No status change."""
        coords = _unique(_as_list(coords))
        if not coords:
            return
        self._require_present(coords)
        self.points_changed.emit(tuple(coords))

    # ------------------------------------------------------------------
    # Status transitions: the rule is ``points``; the store checks the point
    # is here and announces the change.
    # ------------------------------------------------------------------

    def apply_refinement(self, coord: Coordinate, x: float, y: float, z: float) -> None:
        """Commit an accepted refinement of one point: move it and mark it
        fitted."""
        self._require_present([coord])
        points.apply_refinement(coord, x, y, z)
        self.points_changed.emit((coord,))

    def accept(self, coords: _Coords) -> List[Coordinate]:
        """Take predictions as they are, unmoved. Returns the points accepted."""
        coords = _unique(_as_list(coords))
        self._require_present(coords)
        accepted = points.accept(coords)
        if accepted:
            self.points_changed.emit(tuple(accepted))
        return accepted

    def toggle_rejected(self, coord: Coordinate) -> bool:
        """Leave a point out of the fit, or bring it back. Returns True when
        the status changed, False too for a point that is not here: this is
        called from a row's menu, and an exception in a Qt slot aborts the app."""
        if coord not in self or not points.toggle_rejected(coord):
            return False
        self.points_changed.emit((coord,))
        return True

    def reset_to_predicted(self, coord: Coordinate) -> bool:
        """Make a projected point a guess again. The caller re-projects it and
        then calls ``notify_changed``: the store has no transform. Returns False
        for a point the projection did not make, or one that is not here."""
        if coord not in self or not points.reset_to_predicted(coord):
            return False
        self.points_changed.emit((coord,))
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
