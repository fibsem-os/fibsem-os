"""The correlation point overlay is a view of the point store (FIB-973).

The overlay used to own the points it drew and the index of the selected one.
It now draws what a ``CorrelationPointStore`` holds, and a gesture on the canvas
is a call on the store. Each overlay still makes its own store by default; these
also cover two overlays on one store, one per canvas, which is how the tab
widget will use them.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_point_overlay_renders_from_store.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from matplotlib.backend_bases import MouseEvent
from PyQt5.QtWidgets import QApplication

from fibsem.correlation.structures import Coordinate, PointStatus, PointType, PointXYZ
from fibsem.ui.correlation.point_store import CorrelationPointStore
from fibsem.ui.correlation.widgets.correlation_point_overlay import (
    CorrelationPointOverlay,
)
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_app = QApplication.instance() or QApplication(sys.argv)


def _coord(x, y, pt=PointType.FIB, **kwargs):
    return Coordinate(PointXYZ(float(x), float(y), 0.0), pt, **kwargs)


def _on_canvas(overlay):
    canvas = FibsemImageCanvas()
    canvas.resize(300, 300)
    canvas.add_overlay(overlay)
    canvas.set_array(np.zeros((100, 100), np.uint8))
    canvas.show()
    _app.processEvents()
    canvas.draw()
    return canvas


def _shared():
    store = CorrelationPointStore()
    fib = CorrelationPointOverlay(store=store, side="fib")
    fm = CorrelationPointOverlay(store=store, side="fm")
    return store, fib, _on_canvas(fib), fm, _on_canvas(fm)


def _mouse(overlay, canvas, name, x, y, button=1):
    sx, sy = overlay._ax.transData.transform((x, y))
    return MouseEvent(name, canvas, sx, sy, button=button)


def _same(a, b) -> bool:
    return len(a) == len(b) and all(p is q for p, q in zip(a, b))


def test_an_overlay_makes_its_own_store_by_default():
    a, b = CorrelationPointOverlay(), CorrelationPointOverlay()
    assert a.store is not b.store


def test_each_canvas_draws_its_own_side_of_a_shared_store():
    store, fib, _, fm, _ = _shared()
    points = [
        _coord(10, 10, PointType.FIB),
        _coord(20, 20, PointType.FM),
        _coord(30, 30, PointType.POI),
        _coord(40, 40, PointType.SURFACE),
    ]
    store.replace_all(points)

    assert _same(fib.coordinates(), [points[0], points[3]])
    assert _same(fm.coordinates(), [points[1], points[2]])
    assert fm.get_points() == [(20.0, 20.0), (30.0, 30.0)]


def test_the_other_canvass_change_does_not_redraw_this_one():
    store, fib, _, fm, _ = _shared()
    store.replace_all([_coord(10, 10, PointType.FIB), _coord(20, 20, PointType.FM)])
    artists = list(fib._artists)

    store.add(_coord(30, 30, PointType.POI))
    assert _same(fib._artists, artists)
    assert len(fm._artists) == 2


def test_set_coordinates_on_one_canvas_leaves_the_other_sides_points():
    store, fib, _, fm, _ = _shared()
    fm_point = _coord(20, 20, PointType.FM)
    store.replace_all([fm_point])

    fib.set_coordinates([_coord(10, 10, PointType.FIB)])
    assert _same(store.of_type(PointType.FM), [fm_point])
    assert len(store) == 2


def test_set_coordinates_redraws_a_changed_status_for_the_same_points():
    # what the tab widget's _refresh_canvas relies on after a reject
    ov = CorrelationPointOverlay()
    _on_canvas(ov)
    point = _coord(10, 10)
    ov.set_coordinates([point])
    assert ov._artists[0].get_alpha() is None

    point.status = PointStatus.REJECTED
    ov.set_coordinates([point])
    assert ov._artists[0].get_alpha() == 0.35


def test_a_placed_prediction_is_redrawn_filled():
    store, _, _, fm, _ = _shared()
    point = _coord(20, 20, PointType.FM, status=PointStatus.PREDICTED)
    store.replace_all([point])
    assert fm._artists[0].get_markerfacecolor() == "none"

    store.move(point, 25.0, 25.0)  # the store makes it `placed`
    assert fm._artists[0].get_markerfacecolor() != "none"
    assert fm.get_points() == [(25.0, 25.0)]


def test_a_value_change_moves_the_artist_without_redrawing_the_rest():
    store, _, _, fm, _ = _shared()
    points = [_coord(20, 20, PointType.FM), _coord(30, 30, PointType.FM)]
    store.replace_all(points)
    artists = list(fm._artists)

    points[1].point.x = 44.0
    store.notify_changed([points[1]])
    assert _same(fm._artists, artists)
    assert fm.get_points()[1] == (44.0, 30.0)


def test_a_store_selection_is_highlighted_without_being_announced():
    store, _, _, fm, _ = _shared()
    points = [_coord(20, 20, PointType.FM), _coord(30, 30, PointType.FM)]
    store.replace_all(points)
    seen = []
    fm.coordinate_selected.connect(seen.append)

    store.select(points[1])
    assert fm._selected == 1
    assert fm.selected_coordinate() is points[1]
    assert seen == []


def test_the_highlight_survives_a_redraw():
    store, _, _, fm, _ = _shared()
    points = [_coord(20, 20, PointType.FM), _coord(30, 30, PointType.FM)]
    store.replace_all(points)
    store.select(points[1])

    store.add_many([_coord(40, 40, PointType.FM)])
    assert fm._selected == 1


def test_a_pair_is_highlighted_on_both_canvases():
    store, fib, _, fm, _ = _shared()
    a, b = _coord(10, 10, PointType.FIB), _coord(20, 20, PointType.FM)
    store.replace_all([a, b])

    fm.set_selected_coordinate(b)
    fib.set_selected_coordinate(a)
    assert fm.selected_coordinate() is b
    assert fib.selected_coordinate() is a

    fib.set_selected_coordinate(None)
    assert fm.selected_coordinate() is b


def test_a_click_selects_in_the_store_and_clears_the_other_canvas():
    store, fib, _, fm, fm_canvas = _shared()
    a, b = _coord(10, 10, PointType.FIB), _coord(20, 20, PointType.FM)
    store.replace_all([a, b])
    store.select(a)
    seen = []
    fm.coordinate_selected.connect(seen.append)

    fm._on_press(_mouse(fm, fm_canvas, "button_press_event", 20, 20))
    fm._on_release(_mouse(fm, fm_canvas, "button_release_event", 20, 20))
    assert _same(store.selection, [b])
    assert fib._selected is None
    assert _same(seen, [b])


def test_a_click_on_nothing_deselects_in_the_store():
    store, _, _, fm, fm_canvas = _shared()
    b = _coord(20, 20, PointType.FM)
    store.replace_all([b])
    store.select(b)

    fm._on_press(_mouse(fm, fm_canvas, "button_press_event", 80, 80))
    assert store.selection == ()


def test_the_store_hears_of_a_drag_once_on_release():
    store, _, _, fm, fm_canvas = _shared()
    b = _coord(20, 20, PointType.FM, status=PointStatus.PREDICTED)
    store.replace_all([b])
    changes, moved = [], []
    store.points_changed.connect(changes.append)
    fm.coordinate_moved.connect(moved.append)

    fm._on_press(_mouse(fm, fm_canvas, "button_press_event", 20, 20))
    fm._on_motion(_mouse(fm, fm_canvas, "motion_notify_event", 30, 30))
    fm._on_motion(_mouse(fm, fm_canvas, "motion_notify_event", 40, 40))
    # mid-drag the position is the overlay's alone
    assert (b.point.x, b.point.y) == (20.0, 20.0)
    assert changes == []

    fm._on_release(_mouse(fm, fm_canvas, "button_release_event", 40, 40))
    assert (round(b.point.x), round(b.point.y)) == (40, 40)
    assert b.status == PointStatus.PLACED
    assert len(changes) == 1
    assert _same(moved, [b])


def test_a_removal_is_made_in_the_store_before_it_is_announced():
    store, _, _, fm, _ = _shared()
    points = [_coord(20, 20, PointType.FM), _coord(30, 30, PointType.FM)]
    store.replace_all(points)
    seen = []
    fm.coordinate_removed.connect(
        lambda c: seen.append((c, store.of_type(PointType.FM), fm.coordinates()))
    )

    fm.remove_coordinate(points[0])
    removed, in_store, drawn = seen[0]
    assert removed is points[0]
    assert _same(in_store, [points[1]])
    assert _same(drawn, [points[1]])
    assert fm.selected_coordinate() is points[1]  # the neighbour
