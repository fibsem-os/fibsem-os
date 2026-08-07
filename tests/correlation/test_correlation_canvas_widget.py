"""CorrelationCanvasWidget: the shared-canvas replacement for ImagePointCanvas.

Two things are pinned here. First, that points survive the trip through the
widget with their identity intact. Second -- and this is the one that earns its
keep -- that the widget's signals and methods still match ImagePointCanvas's, so
the eventual swap in correlation_tab_widget stays a construction-site change.
That test fails the moment the two drift, which is the failure mode of building
a replacement alongside the thing it replaces.
"""
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] without the UI extra

from PyQt5.QtWidgets import QApplication

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.ui.correlation.widgets.correlation_canvas_widget import (
    CorrelationCanvasWidget,
)
from fibsem.ui.correlation.widgets.image_point_canvas import ImagePointCanvas

_app = QApplication.instance() or QApplication(sys.argv)


def _coord(x, y, pt=PointType.FIB):
    return Coordinate(PointXYZ(x, y, 0.0), pt)


def _widget(allowed=None):
    w = CorrelationCanvasWidget(allowed_point_types=allowed)
    w.canvas.set_array(np.zeros((64, 64), dtype=np.uint8))
    return w


# ── the contract with the tab widget ──────────────────────────────────────


def test_signals_match_the_canvas_it_replaces():
    """The tab widget connects these four by name; a rename breaks the swap."""
    for name in ("point_selected", "point_moved", "point_removed", "point_add_requested"):
        assert hasattr(CorrelationCanvasWidget, name), name
        assert hasattr(ImagePointCanvas, name), name


def test_adapter_surface_matches_the_canvas_it_replaces():
    """_CanvasAdapter calls exactly these three on whatever surface it holds."""
    for name in ("set_coordinates", "set_selected", "refresh_coordinate"):
        assert callable(getattr(CorrelationCanvasWidget, name, None)), name
        assert callable(getattr(ImagePointCanvas, name, None)), name


def test_image_methods_match_the_canvas_it_replaces():
    for name in ("set_image", "update_display", "set_pixel_size", "reset_view"):
        assert callable(getattr(CorrelationCanvasWidget, name, None)), name
        assert callable(getattr(ImagePointCanvas, name, None)), name


# ── points through the widget ─────────────────────────────────────────────


def test_coordinates_reach_the_overlay():
    a, b = _coord(10, 10), _coord(20, 20, PointType.SURFACE)
    w = _widget()
    w.set_coordinates([a, b])
    assert w.points.coordinates() == [a, b]
    assert w.points.get_points() == [(10.0, 10.0), (20.0, 20.0)]


def test_selection_round_trips_by_identity():
    a, b = _coord(10, 10), _coord(20, 20)
    w = _widget()
    w.set_coordinates([a, b])
    w.set_selected(b)
    assert w.points.selected_coordinate() is b
    w.set_selected(None)
    assert w.points.selected_coordinate() is None


def test_overlay_signals_are_re_emitted_by_the_widget():
    a = _coord(10, 10)
    w = _widget()
    w.set_coordinates([a])
    seen = {"sel": [], "moved": [], "removed": []}
    w.point_selected.connect(seen["sel"].append)
    w.point_moved.connect(seen["moved"].append)
    w.point_removed.connect(seen["removed"].append)

    w.points.point_selected.emit(0, 10.0, 10.0)
    w.points.point_moved.emit(0, 11.0, 12.0)
    w.points.remove_coordinate(a)

    assert seen["sel"] == [a] and seen["moved"] == [a] and seen["removed"] == [a]


def test_refresh_picks_up_an_external_edit():
    a = _coord(10, 10)
    w = _widget()
    w.set_coordinates([a])
    a.point.x, a.point.y = 40.0, 50.0
    w.refresh_coordinate(a)
    assert w.points.get_points() == [(40.0, 50.0)]


# ── the add menu ──────────────────────────────────────────────────────────


def test_adding_is_off_when_no_types_are_allowed(monkeypatch):
    """correlation_result_widget passes [] to make its overlay read-only."""
    w = _widget(allowed=[])
    called = []
    monkeypatch.setattr(
        "fibsem.ui.correlation.widgets.correlation_canvas_widget.QMenu",
        lambda *a, **k: called.append(1),
    )
    seen = []
    w.point_add_requested.connect(lambda *a: seen.append(a))

    w.points.add_requested.emit(5.0, 6.0)

    assert called == []  # no menu is even built
    assert seen == []


def test_a_dismissed_menu_adds_nothing():
    w = _widget()
    seen = []
    w.point_add_requested.connect(lambda *a: seen.append(a))

    class _Dismissed:
        def __init__(self, *a, **k):
            pass

        def addAction(self, action):
            pass

        def exec_(self, pos):
            return None

    import fibsem.ui.correlation.widgets.correlation_canvas_widget as mod

    real, mod.QMenu = mod.QMenu, _Dismissed
    try:
        w.points.add_requested.emit(5.0, 6.0)
    finally:
        mod.QMenu = real
    assert seen == []


def test_a_chosen_type_is_reported_with_the_click_position():
    w = _widget(allowed=[PointType.FIB, PointType.SURFACE])
    seen = []
    w.point_add_requested.connect(lambda x, y, pt: seen.append((x, y, pt)))

    class _PicksSecond:
        def __init__(self, *a, **k):
            self._actions = []

        def addAction(self, action):
            self._actions.append(action)

        def exec_(self, pos):
            return self._actions[1]

    import fibsem.ui.correlation.widgets.correlation_canvas_widget as mod

    real, mod.QMenu = mod.QMenu, _PicksSecond
    try:
        w.points.add_requested.emit(5.0, 6.0)
    finally:
        mod.QMenu = real

    assert seen == [(5.0, 6.0, PointType.SURFACE)]


def test_the_widget_adds_no_point_itself():
    """The Coordinate is built by the caller, which owns the z. The widget only
    reports x/y and the chosen type -- same contract as today."""
    w = _widget(allowed=[PointType.FIB])

    class _PicksFirst:
        def __init__(self, *a, **k):
            self._actions = []

        def addAction(self, action):
            self._actions.append(action)

        def exec_(self, pos):
            return self._actions[0]

    import fibsem.ui.correlation.widgets.correlation_canvas_widget as mod

    real, mod.QMenu = mod.QMenu, _PicksFirst
    try:
        w.points.add_requested.emit(5.0, 6.0)
    finally:
        mod.QMenu = real

    assert w.points.coordinates() == []


# ── the reason for the migration ──────────────────────────────────────────


def test_contrast_and_gamma_are_available():
    """ImagePointCanvas has neither; on FM data they are the controls an operator
    actually wants while picking points. The shared canvas brings a
    ContrastGammaControl and a toolbar button to raise it."""
    w = _widget()
    assert w.canvas._contrast is not None
    assert callable(w.canvas.toggle_contrast)
    assert w.canvas.btn_contrast is not None
    # and the old canvas genuinely lacks it -- this is a real gain, not a rename
    assert not hasattr(ImagePointCanvas, "toggle_contrast")
