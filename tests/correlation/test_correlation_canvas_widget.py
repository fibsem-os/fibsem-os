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
    """Signatures, not just callability.

    `set_image` used to take a FibsemImage here while both remaining consumers
    pass a numpy array -- either would have raised AttributeError on the swap. A
    callable-only check sailed straight past it, so this compares each
    parameter's position, annotation and default against the old canvas.
    """
    for name in ("__init__", "set_image", "update_display", "set_coordinates",
                 "set_selected", "refresh_coordinate", "set_pixel_size",
                 "reset_view", "set_legend_visible", "set_labels_visible",
                 "clear_overlay"):
        assert callable(getattr(CorrelationCanvasWidget, name, None)), name
        assert callable(getattr(ImagePointCanvas, name, None)), name
        _assert_accepts_old_calls(name)


def _assert_accepts_old_calls(name: str) -> None:
    """Every call the old canvas accepts, the new one must accept identically.

    Extra *optional* parameters appended on the end are fine -- that is how the
    shared canvas's pixel_size reaches `update_display` -- but a rename, a
    reorder, a changed default or a changed type is a broken swap.
    """
    import inspect

    old = list(inspect.signature(getattr(ImagePointCanvas, name)).parameters.values())
    new = list(inspect.signature(getattr(CorrelationCanvasWidget, name)).parameters.values())
    assert len(new) >= len(old), f"{name}: parameters dropped"

    for want, got in zip(old, new):
        assert got.name == want.name, f"{name}: {want.name} -> {got.name}"
        assert got.kind == want.kind, f"{name}.{want.name}: {want.kind} -> {got.kind}"
        assert got.default == want.default, f"{name}.{want.name}: default changed"
        assert got.annotation == want.annotation, (
            f"{name}.{want.name}: {want.annotation} -> {got.annotation}"
        )

    for extra in new[len(old):]:
        assert extra.default is not inspect.Parameter.empty, (
            f"{name}.{extra.name}: new required parameter breaks existing callers"
        )


def test_set_image_takes_the_array_its_consumers_pass():
    """The signature check above is only as good as the annotation. This is the
    behaviour: both consumers hold a derived array, never a FibsemImage."""
    w = CorrelationCanvasWidget()
    w.set_image(np.zeros((32, 48), dtype=np.uint8))
    assert w.canvas._img_w == 48 and w.canvas._img_h == 32


def test_result_overlay_signature_matches_the_canvas_it_replaces():
    """_overlay_result_on_fib passes these by keyword; a renamed or dropped one
    turns the swap from a construction-site change into a rewrite."""
    import inspect

    for name in ("add_overlay_points", "clear_overlay"):
        assert callable(getattr(CorrelationCanvasWidget, name, None)), name
    old = inspect.signature(ImagePointCanvas.add_overlay_points).parameters
    new = inspect.signature(CorrelationCanvasWidget.add_overlay_points).parameters
    assert set(new) == set(old)
    for key in old:
        if old[key].default is not inspect.Parameter.empty:
            assert new[key].default == old[key].default, key


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
        "fibsem.ui.correlation.widgets.correlation_canvas_widget.QMenu",
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
    w.add_overlay_points([(20.0, 20.0), (30.0, 30.0)], color="#ff4444",
                         legend_label="FM reprojected (E)")

    assert len(w.results._artists) == 2
    assert w.points.get_points() == [(10.0, 10.0)]  # untouched by the result

    w.clear_overlay()
    assert w.results._artists == []
