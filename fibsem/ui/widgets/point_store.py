"""The correlation points, as Qt signals.

The points, their order and the selection are
:class:`fibsem.correlation.points.CorrelationPoints`, which holds no Qt: the
correlation session owns one, and a window without a session makes its own.
This is the layer the lists and the canvases bind to. It hands their gestures
to the points and turns each change the points tell of into a signal. The
overview's image-from-points dialog uses it too, for its two canvases.

Every operation finishes its mutation before it emits, so a slot always reads
the state the signal announces. A bulk operation emits once.
"""

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.correlation import points as point_rules
from fibsem.correlation.points import CorrelationPoints
from fibsem.correlation.structures import Coordinate, PointType


class PointStore(QObject):
    """The correlation points, their order, and the selection.

    Each method is the one of the same name on
    :class:`~fibsem.correlation.points.CorrelationPoints`, which says what it
    does. A change made on ``points`` directly is signalled too.

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

    def __init__(
        self,
        parent: Optional[QObject] = None,
        points: Optional[CorrelationPoints] = None,
    ) -> None:
        super().__init__(parent)
        self.points = points if points is not None else CorrelationPoints()
        dispose = self.points.subscribe(self._emit)
        # The points can outlive this (a session's do), so stop being told
        # before the signals are gone.
        self.destroyed.connect(lambda *_: dispose())

    def _emit(self, change: str, coords: Tuple[Coordinate, ...]) -> None:
        if change == point_rules.STRUCTURE:
            self.structure_changed.emit()
        elif change == point_rules.VALUES:
            self.points_changed.emit(coords)
        else:
            self.selection_changed.emit()

    def set_removal_policy(
        self,
        policy: Optional[Callable[[List[Coordinate]], Optional[List[Coordinate]]]],
    ) -> Callable[[], None]:
        return self.points.set_removal_policy(policy)

    # Queries

    def of_type(self, point_type: PointType) -> List[Coordinate]:
        return self.points.of_type(point_type)

    def on_side(self, side: str) -> List[Coordinate]:
        return self.points.on_side(side)

    @property
    def coordinates(self) -> List[Coordinate]:
        return self.points.coordinates

    @property
    def selection(self) -> Tuple[Coordinate, ...]:
        return self.points.selection

    @property
    def current(self) -> Optional[Coordinate]:
        return self.points.current

    def selected_on(self, side: str) -> Optional[Coordinate]:
        return self.points.selected_on(side)

    def selected_of_type(self, point_type: PointType) -> Optional[Coordinate]:
        return self.points.selected_of_type(point_type)

    def index_of(self, coord: Coordinate) -> Optional[int]:
        return self.points.index_of(coord)

    def __contains__(self, coord: object) -> bool:
        return coord in self.points

    def __len__(self) -> int:
        return len(self.points)

    # Structure

    def add(self, coord: Coordinate) -> None:
        self.points.add(coord)

    def add_many(self, coords: Sequence[Coordinate]) -> None:
        self.points.add_many(coords)

    def remove(self, coord: Coordinate) -> bool:
        return self.points.remove(coord)

    def remove_many(self, coords: Sequence[Coordinate]) -> int:
        return self.points.remove_many(coords)

    def reorder(self, point_type: PointType, coords: Sequence[Coordinate]) -> None:
        self.points.reorder(point_type, coords)

    def replace_type(self, point_type: PointType, coords: Sequence[Coordinate]) -> None:
        self.points.replace_type(point_type, coords)

    def replace_all(self, coords: Iterable[Coordinate]) -> None:
        self.points.replace_all(coords)

    # Selection

    def select(self, coords, extend: bool = False) -> None:
        self.points.select(coords, extend=extend)

    def deselect(self, coords) -> None:
        self.points.deselect(coords)

    # Values and status

    def move(self, coord: Coordinate, x: float, y: float) -> None:
        self.points.move(coord, x, y)

    def move_many(self, coords: Sequence[Coordinate], dx: float, dy: float) -> None:
        self.points.move_many(coords, dx, dy)

    def set_field(self, coord: Coordinate, field: str, value: float) -> None:
        self.points.set_field(coord, field, value)

    def notify_changed(self, coords) -> None:
        self.points.notify_changed(coords)

    def apply_refinement(self, coord: Coordinate, x: float, y: float, z: float) -> None:
        self.points.apply_refinement(coord, x, y, z)

    def accept(self, coords) -> List[Coordinate]:
        return self.points.accept(coords)

    def toggle_rejected(self, coord: Coordinate) -> bool:
        return self.points.toggle_rejected(coord)

    def reset_to_predicted(self, coord: Coordinate) -> bool:
        return self.points.reset_to_predicted(coord)
