"""CorrelationCanvasWidget: the FIB correlation surface.

Points survive the trip through the widget with their identity intact, and the
widget still answers to everything `correlation_tab_widget` asks of it.

The parity suite that used to live here -- an enumerating diff against
ImagePointCanvas, which caught the set_image and set_scalebar_visible gaps -- is
gone with that class. It guarded a migration that has finished; comparing
against a deleted module proves nothing. What it incidentally pinned is now
asserted directly against the live consumer instead, which is the better test
anyway: `_CanvasAdapter` and the tab widget's four-signal loop.
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

_app = QApplication.instance() or QApplication(sys.argv)


def _coord(x, y, pt=PointType.FIB):
    return Coordinate(PointXYZ(x, y, 0.0), pt)


def _widget(allowed=None):
    w = CorrelationCanvasWidget(allowed_point_types=allowed)
    w.canvas.set_array(np.zeros((64, 64), dtype=np.uint8))
    return w


# ── the contract with the tab widget ──────────────────────────────────────


def test_the_tab_widget_can_connect_its_four_signals():
    """`correlation_tab_widget` wires both surfaces in one loop:

        for canvas in (self._fib_canvas, self._fm_display):
            canvas.point_selected.connect(...)

    so a rename here silently disconnects half the correlation UI.
    """
    for name in (
        "point_selected",
        "point_moved",
        "point_removed",
        "point_add_requested",
    ):
        assert hasattr(CorrelationCanvasWidget, name), name


def test_the_adapter_surface_is_answered():
    """_CanvasAdapter calls exactly these three on whatever surface it holds, and
    it is duck-typed -- nothing else checks they exist until a click does."""
    from fibsem.ui.correlation.widgets.correlation_tab_widget import _CanvasAdapter

    adapter = _CanvasAdapter(CorrelationCanvasWidget(), side="fib")
    for name in ("set_coordinates", "set_selected", "refresh_coordinate"):
        assert callable(getattr(adapter, name)), name
        assert callable(getattr(CorrelationCanvasWidget, name, None)), name


def test_set_image_takes_the_array_its_consumers_pass():
    """The signature check above is only as good as the annotation. This is the
    behaviour: both consumers hold a derived array, never a FibsemImage."""
    w = CorrelationCanvasWidget()
    w.set_image(np.zeros((32, 48), dtype=np.uint8))
    assert w.canvas._img_w == 48 and w.canvas._img_h == 32


def test_the_result_overlay_takes_the_keywords_the_tab_widget_passes():
    """`_overlay_result_on_fib` passes every one of these by keyword; a rename
    turns a correlation run's result markers into a TypeError."""
    import inspect

    params = inspect.signature(CorrelationCanvasWidget.add_overlay_points).parameters
    for key in (
        "color",
        "label_prefix",
        "size",
        "marker",
        "alpha",
        "show_labels",
        "hollow",
        "legend_label",
    ):
        assert key in params, key
    assert callable(getattr(CorrelationCanvasWidget, "clear_overlay", None))


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
    """[] means adding is off, for a canvas that only displays a result.

    The stub is menu-shaped and dismisses itself. A bare lambda lets the
    AttributeError escape the slot when this regresses, and PyQt5 turns that into
    a hard abort (FIB-329) -- taking the whole run down instead of failing here.
    """
    called = []

    class _DismissedMenu:
        def __init__(self, *a, **k):
            called.append(self)

        def addAction(self, action):
            pass

        def exec_(self, pos):
            return None

    monkeypatch.setattr(
        "fibsem.ui.correlation.widgets.correlation_picking.QMenu",
        _DismissedMenu,
    )
    w = _widget(allowed=[])
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

    import fibsem.ui.correlation.widgets.correlation_picking as mod

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

    import fibsem.ui.correlation.widgets.correlation_picking as mod

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

    import fibsem.ui.correlation.widgets.correlation_picking as mod

    real, mod.QMenu = mod.QMenu, _PicksFirst
    try:
        w.points.add_requested.emit(5.0, 6.0)
    finally:
        mod.QMenu = real

    assert w.points.coordinates() == []


# ── the reason for the migration ──────────────────────────────────────────


def test_contrast_and_gamma_are_available():
    """The reason for the whole migration. The canvas correlation used to draw on
    had neither, and on FM data they are exactly the controls an operator wants
    while picking points."""
    w = _widget()
    assert w.canvas._contrast is not None
    assert callable(w.canvas.toggle_contrast)
    assert w.canvas.btn_contrast is not None


# ── chrome delegates to the overlay ───────────────────────────────────────


def test_legend_toggle_reaches_the_overlay():
    w = _widget()
    w.set_coordinates([_coord(10, 10), _coord(20, 20, PointType.POI)])
    assert w.points._legend_entries()[1] == ["FIB", "POI"]
    w.set_legend_visible(False)
    assert w.points._legend_entries() == ([], [])


def test_label_toggle_reaches_the_overlay_without_hiding_points():
    w = _widget()
    w.set_coordinates([_coord(10, 10)])
    w.set_labels_visible(False)
    assert [a.get_visible() for a in w.points._anns if a is not None] == [False]
    assert all(a.get_visible() for a in w.points._artists)


def test_label_toggle_covers_the_result_markers_too():
    """One button, all the text. Leaving the result's E1/P1 numbers behind would
    make the toggle look broken on exactly the image that needs it most."""
    w = _widget()
    w.set_coordinates([_coord(10, 10)])
    w.add_overlay_points([(20.0, 20.0)], color="#ff4444", label_prefix="E")

    w.set_labels_visible(False)

    assert [a.get_visible() for a in w.points._anns if a is not None] == [False]
    assert [a.get_visible() for a in w.results._label_artists] == [False]


def test_result_markers_go_to_the_result_overlay():
    w = _widget()
    w.set_coordinates([_coord(10, 10)])
    w.add_overlay_points(
        [(20.0, 20.0), (30.0, 30.0)], color="#ff4444", legend_label="FM reprojected (E)"
    )

    assert len(w.results._artists) == 2
    assert w.points.get_points() == [(10.0, 10.0)]  # untouched by the result

    w.clear_overlay()
    assert w.results._artists == []


def test_point_labels_have_an_outline():
    """Coloured labels get a dark outline so they stay legible on any image
    background — every point colour is a bright hue, and SURFACE/SURFACE_FM
    vanish over a blown-out image without it.

    Ported from the ImagePointCanvas suite. The font is 8px here against that
    canvas's 9px; the outline is what the test is about.
    """
    w = _widget()
    w.set_coordinates([_coord(5, 5)])

    label = w.points._anns[0]
    assert label.get_path_effects()
    assert label.get_fontsize() == 8
