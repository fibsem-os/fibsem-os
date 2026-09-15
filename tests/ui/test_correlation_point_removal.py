"""Deleting one point on the correlation canvas removes exactly one point (FIB-958).

`PointOverlay.remove_point` used to emit `point_removed(index)` before popping its own
lists. The tab widget answers that signal by filtering its list model and rebuilding
the overlay from it (`set_coordinates`), so by the time the base method resumed and
popped `index`, the lists had already been rebuilt without the point and the pop took
whichever point now sat there. One Delete removed two markers from the canvas while
the list widget -- the source of truth for the run button -- still held the second.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_correlation_point_removal.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.ui.correlation.widgets.coordinate_list_widget import CoordinateListWidget
from fibsem.ui.correlation.widgets.correlation_canvas_widget import (
    CorrelationCanvasWidget,
)

_app = QApplication.instance() or QApplication(sys.argv)


def _make_canvas(n: int = 5):
    w = CorrelationCanvasWidget()
    w.set_image(np.zeros((200, 200), np.uint8))
    ov = w.picking.points
    model = [
        Coordinate(PointXYZ(20 + i * 30, 20 + i * 30, 0), PointType.FIB)
        for i in range(n)
    ]
    ov.set_coordinates(model)

    removed = []

    def on_removed(c):  # mirrors CorrelationTabWidget._on_canvas_removed
        removed.append(c)
        model[:] = [x for x in model if x is not c]
        ov.set_coordinates(model)  # == _refresh_canvas

    w.point_removed.connect(on_removed)
    return w, ov, model, removed


def _assert_in_sync(ov, model):
    assert len(model) == len(ov.coordinates()) == len(ov._points) == len(ov._artists)
    assert [c.point.x for c in ov.coordinates()] == [c.point.x for c in model]
    assert [p[0] for p in ov._points] == [c.point.x for c in model]


@pytest.mark.parametrize("index", [0, 1, 3, 4])
def test_remove_via_canvas_removes_exactly_one_point(index):
    w, ov, model, removed = _make_canvas(5)
    target = model[index]

    ov.remove_point(index)

    assert removed == [target]
    assert target not in model
    _assert_in_sync(ov, model)
    assert len(model) == 4
    w.close()


def test_remove_every_point_one_at_a_time_stays_in_sync():
    w, ov, model, removed = _make_canvas(5)
    while model:
        ov.remove_point(0)
        _assert_in_sync(ov, model)
    assert len(removed) == 5
    w.close()


def test_remove_coordinate_by_identity_emits_that_coordinate():
    w, ov, model, removed = _make_canvas(3)
    ov.remove_coordinate(model[2])
    assert len(removed) == 1 and removed[0].point.x == 80
    _assert_in_sync(ov, model)
    w.close()


def test_list_widget_removes_by_identity_not_equality():
    lw = CoordinateListWidget(point_type=PointType.FIB)
    twin_a = Coordinate(PointXYZ(10, 10, 0), PointType.FIB)
    twin_b = Coordinate(PointXYZ(10, 10, 0), PointType.FIB)
    assert twin_a == twin_b and twin_a is not twin_b
    lw.coordinates = [twin_a, twin_b]

    lw._on_remove(twin_b)

    assert lw.coordinates == [twin_a]
    assert lw.coordinates[0] is twin_a
    lw.close()


# ── FIB-965: which row is selected after a removal ───────────────────────


def _list_with(n: int) -> CoordinateListWidget:
    lw = CoordinateListWidget(point_type=PointType.FIB)
    lw.coordinates = [
        Coordinate(PointXYZ(10 * i, 10 * i, 0), PointType.FIB) for i in range(n)
    ]
    return lw


def test_remove_coordinate_selects_the_neighbour_and_emits_nothing():
    lw = _list_with(5)
    coords = lw.coordinates
    lw.select_coordinate_silent(coords[3])
    emitted = []
    lw.coordinate_selected.connect(lambda c: emitted.append(("selected", c)))
    lw.coordinate_removed.connect(lambda c: emitted.append(("removed", c)))

    assert lw.remove_coordinate(coords[3]) is True

    assert lw.coordinates == [coords[0], coords[1], coords[2], coords[4]]
    assert lw.selected_coordinate is coords[4]
    assert emitted == []
    lw.close()


def test_remove_coordinate_of_the_last_row_selects_the_new_last():
    lw = _list_with(3)
    coords = lw.coordinates
    lw.remove_coordinate(coords[2])
    assert lw.selected_coordinate is coords[1]
    assert lw.remove_coordinate(coords[2]) is False  # already gone
    lw.close()


def test_trash_button_emits_the_removal_before_the_new_selection():
    lw = _list_with(3)
    coords = lw.coordinates
    order = []
    lw.coordinate_removed.connect(lambda c: order.append(("removed", c)))
    lw.coordinate_selected.connect(lambda c: order.append(("selected", c)))

    lw._on_remove(coords[0])

    assert order == [("removed", coords[0]), ("selected", coords[1])]
    lw.close()


def test_canvas_delete_in_the_tab_widget_selects_the_neighbour_not_row_1():
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    tab = CorrelationTabWidget()
    spec = tab._point_specs[PointType.FIB]
    spec.list_widget.coordinates = [
        Coordinate(PointXYZ(10 * i, 10 * i, 0), PointType.FIB) for i in range(5)
    ]
    coords = spec.list_widget.coordinates
    tab._refresh_canvas(spec.adapter)
    spec.list_widget.select_coordinate_silent(coords[3])

    tab._on_canvas_removed(coords[3])  # what the canvas's Delete key reaches

    assert spec.list_widget.coordinates == [coords[0], coords[1], coords[2], coords[4]]
    assert spec.list_widget.selected_coordinate is coords[4]
    assert spec.adapter._surface.picking.points.selected_coordinate() is coords[4]
    tab.close()
