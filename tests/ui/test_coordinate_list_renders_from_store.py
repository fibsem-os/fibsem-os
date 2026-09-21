"""The coordinate list is a view of the point store (FIB-973).

The list used to own its points and its selection. It now renders what a
``CorrelationPointStore`` holds, so a change made to the store by anything else
shows up in the rows without the list being told. Each list still makes its own
store by default; these also cover several lists on one store, which is how the
tab widget will use them.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_coordinate_list_renders_from_store.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.ui.correlation.point_store import CorrelationPointStore
from fibsem.ui.correlation.widgets.coordinate_list_widget import (
    CoordinateListWidget,
    CoordinateRowWidget,
)

_app = QApplication.instance() or QApplication(sys.argv)


def _coords(pt: PointType, n: int):
    return [Coordinate(PointXYZ(10.0 * i, 10.0 * i, 0), pt) for i in range(n)]


def _rows(lw: CoordinateListWidget):
    return [lw._list.itemWidget(lw._list.item(i)) for i in range(lw._list.count())]


def _row_coords(lw: CoordinateListWidget):
    return [
        lw._list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(lw._list.count())
    ]


def _same(a, b) -> bool:
    return len(a) == len(b) and all(x is y for x, y in zip(a, b))


def _selected_rows(lw: CoordinateListWidget):
    return [w.coord for w in _rows(lw) if w._selected]


def _shared():
    store = CorrelationPointStore()
    fm = CoordinateListWidget(point_type=PointType.FM, store=store)
    poi = CoordinateListWidget(point_type=PointType.POI, store=store)
    return store, fm, poi


def test_a_list_needs_a_point_type():
    with pytest.raises(ValueError):
        CoordinateListWidget()


def test_a_list_makes_its_own_store_by_default():
    a = CoordinateListWidget(point_type=PointType.FIB)
    b = CoordinateListWidget(point_type=PointType.FIB)
    assert a.store is not b.store


def test_the_rows_follow_a_change_made_to_the_store():
    store, fm, _ = _shared()
    coords = _coords(PointType.FM, 3)
    store.add_many(coords)
    assert _same(_row_coords(fm), coords)
    assert fm._empty_label.isHidden()

    store.remove(coords[1])
    assert _same(_row_coords(fm), [coords[0], coords[2]])


def test_the_list_reads_its_points_from_the_store():
    store, fm, poi = _shared()
    fm_coords, poi_coords = _coords(PointType.FM, 2), _coords(PointType.POI, 1)
    fm.coordinates = fm_coords
    poi.coordinates = poi_coords
    assert _same(store.of_type(PointType.FM), fm_coords)
    assert _same(poi.coordinates, poi_coords)
    assert len(store) == 3


def test_a_list_refuses_a_point_of_another_type():
    _, fm, _ = _shared()
    with pytest.raises(ValueError):
        fm.coordinates = _coords(PointType.POI, 1)
    with pytest.raises(ValueError):
        fm.add_coordinate(_coords(PointType.POI, 1)[0])


def test_another_lists_change_does_not_rebuild_these_rows():
    # a rebuild destroys the row widgets, and with them a spinbox being typed in
    store, fm, _ = _shared()
    store.add_many(_coords(PointType.FM, 2))
    before = _rows(fm)

    store.add(_coords(PointType.POI, 1)[0])
    assert _same(_rows(fm), before)


def test_a_value_change_refreshes_the_row_without_rebuilding_it():
    store, fm, _ = _shared()
    coords = _coords(PointType.FM, 2)
    store.add_many(coords)
    before = _rows(fm)

    coords[1].point.x = 77.0
    store.notify_changed([coords[1]])
    assert _same(_rows(fm), before)
    assert before[1].x_spin.value() == 77.0


def test_selecting_in_one_list_clears_the_highlight_in_the_other():
    store, fm, poi = _shared()
    fm_coords, poi_coords = _coords(PointType.FM, 2), _coords(PointType.POI, 2)
    store.add_many(fm_coords + poi_coords)

    store.select(fm_coords[1])
    assert _same(_selected_rows(fm), [fm_coords[1]])
    assert fm.selected_coordinate is fm_coords[1]

    store.select(poi_coords[0])
    assert _selected_rows(fm) == []
    assert fm.selected_coordinate is None
    assert _same(_selected_rows(poi), [poi_coords[0]])


def test_a_store_selection_is_shown_without_being_announced():
    store, fm, _ = _shared()
    coords = _coords(PointType.FM, 2)
    store.add_many(coords)
    emitted = []
    fm.coordinate_selected.connect(emitted.append)

    store.select(coords[0])
    assert _same(_selected_rows(fm), [coords[0]])
    assert emitted == []


def test_a_row_click_selects_in_the_store_and_is_announced():
    store, fm, _ = _shared()
    coords = _coords(PointType.FM, 2)
    store.add_many(coords)
    emitted = []
    fm.coordinate_selected.connect(emitted.append)

    fm._on_row_clicked(coords[1])
    assert store.current is coords[1]
    assert _same(emitted, [coords[1]])


def test_clearing_a_list_silently_leaves_another_lists_selection():
    store, fm, poi = _shared()
    fm_coords, poi_coords = _coords(PointType.FM, 1), _coords(PointType.POI, 1)
    store.add_many(fm_coords + poi_coords)
    store.select(poi_coords[0])

    fm.select_coordinate_silent(None)
    assert store.current is poi_coords[0]


def test_the_selection_survives_the_rebuild_after_a_reorder():
    store, fm, _ = _shared()
    coords = _coords(PointType.FM, 3)
    store.add_many(coords)
    store.select(coords[0])
    order = []
    fm.order_changed.connect(order.append)

    fm._on_reordered(coords[::-1])  # what a drop reports
    assert _same(store.of_type(PointType.FM), coords[::-1])
    assert _same(_selected_rows(fm), [coords[0]])
    assert _same(order[0], coords[::-1])


def test_a_drop_that_changes_nothing_still_gets_its_row_widgets_back():
    # Qt clears the row widgets on an internal move, whatever the final order
    store, fm, _ = _shared()
    coords = _coords(PointType.FM, 2)
    store.add_many(coords)
    for i in range(fm._list.count()):
        fm._list.removeItemWidget(fm._list.item(i))

    fm._on_reordered(list(coords))
    assert all(isinstance(w, CoordinateRowWidget) for w in _rows(fm))


def test_assigning_the_list_still_selects_row_1_and_announces_it():
    # the setter's side effect, kept until the last caller that assigns a list
    # is gone; the store's own replace selects nothing
    store, fm, _ = _shared()
    emitted = []
    fm.coordinate_selected.connect(emitted.append)
    coords = _coords(PointType.FM, 3)

    fm.coordinates = coords
    assert store.current is coords[0]
    assert _same(emitted, [coords[0]])
