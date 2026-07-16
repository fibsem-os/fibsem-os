"""Headless tests for CorrelationTabWidget FM-surface / RI-correction behaviour.

Uses PyQt5 directly with the offscreen platform (no pytest-qt dependency),
matching tests/fm/test_autofocus_widget.py.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    PointType,
    PointXYZ,
)
from fibsem.structures import Point


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    """Never hit the network for the zeta LUT during widget construction."""
    import fibsem.correlation.ui.widgets.refractive_index_widget as riw

    monkeypatch.setattr(riw, "_ensure_lut", lambda: None)


def _coord(x=0.0, y=0.0, z=0.0, pt=PointType.FIB) -> Coordinate:
    return Coordinate(point=PointXYZ(x=x, y=y, z=z), point_type=pt)


def _widget(qapp):
    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    return CorrelationTabWidget()


# ---------------------------------------------------------------------------
# Surface exclusivity
# ---------------------------------------------------------------------------


def test_fm_surface_add_replaces_fib_surface(qapp):
    w = _widget(qapp)
    cl = w._coords_tab

    w._on_canvas_add_requested(1.0, 2.0, PointType.SURFACE)
    assert len(cl.surface_list.coordinates) == 1

    w._on_canvas_add_requested(3.0, 4.0, PointType.SURFACE_FM)
    assert cl.surface_list.coordinates == []
    assert len(cl.fm_surface_list.coordinates) == 1

    w._on_canvas_add_requested(5.0, 6.0, PointType.SURFACE)
    assert cl.fm_surface_list.coordinates == []
    assert len(cl.surface_list.coordinates) == 1


def test_fm_surface_add_is_max_one(qapp):
    w = _widget(qapp)
    cl = w._coords_tab
    w._on_canvas_add_requested(1.0, 1.0, PointType.SURFACE_FM)
    w._on_canvas_add_requested(2.0, 2.0, PointType.SURFACE_FM)
    assert len(cl.fm_surface_list.coordinates) == 1
    assert cl.fm_surface_list.coordinates[0].point.x == pytest.approx(2.0)


def test_set_data_prefers_fm_surface_when_both_present(qapp):
    w = _widget(qapp)
    cl = w._coords_tab
    data = CorrelationInputData(
        surface_coordinate=_coord(y=100.0, pt=PointType.SURFACE),
        fm_surface_coordinate=_coord(z=10.0, pt=PointType.SURFACE_FM),
    )
    w.set_data(data)
    assert cl.surface_list.coordinates == []
    assert len(cl.fm_surface_list.coordinates) == 1


# ---------------------------------------------------------------------------
# data property / set_data round trip
# ---------------------------------------------------------------------------


def test_data_includes_fm_surface_and_factor(qapp):
    w = _widget(qapp)
    w._on_canvas_add_requested(3.0, 4.0, PointType.SURFACE_FM)
    w._ri_pre_correction_factor = 1.33
    d = w.data
    assert d.fm_surface_coordinate is not None
    assert d.fm_surface_coordinate.point_type is PointType.SURFACE_FM
    assert d.ri_pre_correction_factor == pytest.approx(1.33)


def test_set_data_restores_factor(qapp):
    w = _widget(qapp)
    data = CorrelationInputData(
        fm_surface_coordinate=_coord(z=10.0, pt=PointType.SURFACE_FM),
        ri_pre_correction_factor=1.47,
    )
    w.set_data(data)
    assert w._ri_pre_correction_factor == pytest.approx(1.47)
    assert w.data.ri_pre_correction_factor == pytest.approx(1.47)


# ---------------------------------------------------------------------------
# RI tab mode
# ---------------------------------------------------------------------------


def test_ri_tab_mode_follows_surface_type(qapp):
    w = _widget(qapp)
    tab = w._ri_tab
    assert tab.mode is None

    w._on_canvas_add_requested(1.0, 200.0, PointType.SURFACE)
    assert tab.mode == "post"
    assert not tab._chk_rerun.isVisible()

    w._on_canvas_add_requested(1.0, 2.0, PointType.SURFACE_FM)
    assert tab.mode == "pre"


def test_tilt_locked_to_zero_in_pre_mode(qapp):
    w = _widget(qapp)
    spin_tilt = w._ri_tab._ri_widget._spin_tilt
    default_tilt = spin_tilt.value()
    assert default_tilt == pytest.approx(15.0)

    w._on_canvas_add_requested(1.0, 2.0, PointType.SURFACE_FM)
    assert spin_tilt.value() == pytest.approx(0.0)
    assert not spin_tilt.isEnabled()

    # removing the FM surface (via replacing with a FIB surface) unlocks tilt
    w._on_canvas_add_requested(1.0, 200.0, PointType.SURFACE)
    assert spin_tilt.value() == pytest.approx(default_tilt)


# ---------------------------------------------------------------------------
# Pre-mode apply
# ---------------------------------------------------------------------------


def _setup_pre_mode(w):
    w._on_canvas_add_requested(1.0, 2.0, PointType.SURFACE_FM)
    w._coords_tab.fm_surface_list.coordinates[0].point.z = 10.0
    w._on_canvas_add_requested(5.0, 6.0, PointType.POI)
    w._coords_tab.poi_list.coordinates[0].point.z = 30.0
    w.data_changed.emit(w.data)  # refresh RI tab with the edited z values


def test_apply_pre_stores_factor_and_reruns(qapp):
    w = _widget(qapp)
    _setup_pre_mode(w)

    runs = []
    w._run = lambda: runs.append(True)

    w._ri_tab._ri_widget.set_factor(1.5)
    w._ri_tab._chk_rerun.setChecked(True)
    w._ri_tab._apply()

    assert w._ri_pre_correction_factor == pytest.approx(1.5)
    assert w.data.ri_pre_correction_factor == pytest.approx(1.5)
    assert runs == [True]


def test_apply_pre_without_rerun_only_stores(qapp):
    w = _widget(qapp)
    _setup_pre_mode(w)

    runs = []
    w._run = lambda: runs.append(True)

    w._ri_tab._ri_widget.set_factor(1.4)
    w._ri_tab._chk_rerun.setChecked(False)
    w._ri_tab._apply()

    assert w._ri_pre_correction_factor == pytest.approx(1.4)
    assert runs == []
    # preview table shows the armed correction for the POI
    assert w._ri_tab._table.rowCount() == 1
    # 10 + (30 - 10) * 1.4 = 38
    assert w._ri_tab._table.item(0, 3).text() == "38.00"


# ---------------------------------------------------------------------------
# Post-mode double-apply guard
# ---------------------------------------------------------------------------


def test_status_line_shows_pre_correction_after_run(qapp):
    """The Done message must survive _update_run_button (it used to be
    overwritten by 'Ready.' immediately)."""
    w = _widget(qapp)
    result = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
        refractive_index_correction_factor=1.5,
        refractive_index_correction_mode="pre",
    )
    w._on_result_ready(result)
    assert w._lbl_status.text() == "Done — RI pre-correction ×1.500 applied."

    with_ghost = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=60.0))],
        poi_uncorrected=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
        refractive_index_correction_factor=1.5,
        refractive_index_correction_mode="pre",
    )
    w._on_result_ready(with_ghost)
    assert (
        w._lbl_status.text()
        == "Done — RI pre-correction ×1.500 applied, POI 1 shifted 40.0 px."
    )

    plain = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
    )
    w._on_result_ready(plain)
    assert w._lbl_status.text() == "Done."


def test_apply_post_creates_ghost_and_status(qapp):
    w = _widget(qapp)
    w._on_canvas_add_requested(1.0, 200.0, PointType.SURFACE)

    result = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=100.0, y=300.0))],
        input_data=w.data,
    )
    w._ri_tab.set_result(result, input_data=w.data)
    w._ri_tab._ri_widget.set_factor(1.5)
    w._ri_tab._apply()

    # 200 + (300 - 200) * 1.5 = 350
    assert result.poi[0].image_px.y == pytest.approx(350.0)
    assert len(result.poi_uncorrected) == 1
    assert result.poi_uncorrected[0].image_px.y == pytest.approx(300.0)
    assert (
        w._lbl_status.text()
        == "RI post-correction ×1.500 applied, POI 1 shifted 50.0 px."
    )


def test_apply_post_blocked_when_already_corrected(qapp):
    w = _widget(qapp)
    w._on_canvas_add_requested(1.0, 200.0, PointType.SURFACE)

    result = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=100.0, y=300.0))],
        input_data=w.data,
        refractive_index_correction_factor=1.4,
        refractive_index_correction_mode="post",
    )
    w._ri_tab.set_result(result, input_data=w.data)

    w._ri_tab._apply()
    # guard: no double correction applied
    assert result.poi[0].image_px.y == pytest.approx(300.0)
    assert result.refractive_index_correction_factor == pytest.approx(1.4)
    assert "already applied" in w._ri_tab._lbl_warning.text()
    # guard message renders in the error style, not the leftover green
    assert "#e07b39" in w._ri_tab._lbl_warning.styleSheet()


def test_apply_post_without_input_data_shows_warning(qapp):
    """A result JSON with input_data: null must warn, not crash."""
    w = _widget(qapp)
    w._on_canvas_add_requested(1.0, 200.0, PointType.SURFACE)

    result = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=100.0, y=300.0))],
        input_data=None,
    )
    w._ri_tab.set_result(result, input_data=w.data)
    w._ri_tab._apply()  # must not raise
    assert result.poi[0].image_px.y == pytest.approx(300.0)
    assert "no input data" in w._ri_tab._lbl_warning.text()


def test_factor_cleared_when_fm_surface_removed(qapp):
    """The pre-correction factor's lifecycle is tied to the FM surface."""
    w = _widget(qapp)
    _setup_pre_mode(w)
    w._ri_pre_correction_factor = 1.5

    # canvas removal
    coord = w._coords_tab.fm_surface_list.coordinates[0]
    w._on_canvas_removed(coord)
    assert w._ri_pre_correction_factor is None
    assert w.data.ri_pre_correction_factor is None

    # replacement by a FIB surface
    w._on_canvas_add_requested(1.0, 2.0, PointType.SURFACE_FM)
    w._ri_pre_correction_factor = 1.5
    w._on_canvas_add_requested(1.0, 200.0, PointType.SURFACE)
    assert w._ri_pre_correction_factor is None

    # list-row removal (generic handler bound to the SURFACE_FM spec)
    w._on_canvas_add_requested(1.0, 2.0, PointType.SURFACE_FM)
    w._ri_pre_correction_factor = 1.5
    coord = w._coords_tab.fm_surface_list.coordinates[0]
    w._coords_tab.fm_surface_list.coordinates = []
    w._on_list_removed(w._point_specs[PointType.SURFACE_FM], coord)
    assert w._ri_pre_correction_factor is None


# ---------------------------------------------------------------------------
# Point-type registry — generic behaviour for every registered type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("point_type", list(PointType))
def test_registry_add_select_remove_for_every_type(qapp, point_type):
    """Every registered point type gets identical add/select/remove plumbing —
    a new point type only needs a registry entry to inherit all of this."""
    w = _widget(qapp)
    spec = w._point_specs[point_type]

    w._on_canvas_add_requested(3.0, 4.0, point_type)
    assert len(spec.list_widget.coordinates) == 1
    coord = spec.list_widget.coordinates[0]
    assert coord.point_type is point_type

    # selecting via the canvas clears every other list's selection
    w._on_canvas_selected(coord)
    for other in w._point_specs.values():
        if other is not spec:
            assert other.list_widget.selected_coordinate is None

    # moving routes to the owning list without error
    coord.point.x = 7.0
    w._on_canvas_moved(coord)

    # removal empties the owning list
    w._on_canvas_removed(coord)
    assert spec.list_widget.coordinates == []


def test_unregistered_point_type_fails_loudly(qapp):
    """Routing must never silently misfile a coordinate (old behaviour sent
    unknown types to the POI list)."""
    w = _widget(qapp)
    w._point_specs.pop(PointType.POI)  # simulate an unregistered type
    with pytest.raises(KeyError):
        w._on_canvas_add_requested(1.0, 2.0, PointType.POI)
    with pytest.raises(KeyError):
        w._on_canvas_moved(_coord(pt=PointType.POI))


def test_set_data_does_not_arm_factor_without_fm_surface(qapp):
    w = _widget(qapp)
    data = CorrelationInputData(ri_pre_correction_factor=1.5)
    w.set_data(data)
    assert w._ri_pre_correction_factor is None


def test_typed_factor_survives_refresh_and_armed_priority(qapp):
    w = _widget(qapp)
    _setup_pre_mode(w)
    tab = w._ri_tab

    # armed input factor (1.6) outranks an older result factor (1.5)
    w._ri_pre_correction_factor = 1.6
    result = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=1.0, y=2.0))],
        refractive_index_correction_factor=1.5,
        refractive_index_correction_mode="pre",
    )
    tab.set_result(result, input_data=w.data)
    assert tab._ri_widget.get_factor() == pytest.approx(1.6)
    assert "stored — applied on the next run" in tab._lbl_warning.text()

    # a factor being typed survives an unrelated refresh (same stored value)
    tab._ri_widget._spin_factor.setValue(1.8)
    tab.set_result(result, input_data=w.data)
    assert tab._ri_widget.get_factor() == pytest.approx(1.8)


def test_tilt_lock_preserves_manual_factor(qapp):
    from fibsem.correlation.ui.widgets.refractive_index_widget import (
        RefractiveIndexWidget,
    )

    rw = RefractiveIndexWidget()
    rw._spin_factor.setValue(1.6)
    rw.set_tilt_locked(True)
    assert rw._spin_tilt.value() == pytest.approx(0.0)
    assert rw.get_factor() == pytest.approx(1.6)
    rw.set_tilt_locked(False)
    assert rw.get_factor() == pytest.approx(1.6)


def test_load_result_keeps_status_message(qapp):
    w = _widget(qapp)
    result = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
        input_data=CorrelationInputData(),
        refractive_index_correction_factor=1.5,
        refractive_index_correction_mode="pre",
    )
    w._load_result(result)
    assert w._lbl_status.text() == "Done — RI pre-correction ×1.500 applied."


def _legend_labels(ax):
    legend = ax.get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend else []


def test_canvas_legend_lists_present_point_types(qapp):
    from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas

    canvas = ImagePointCanvas()
    canvas.set_coordinates(
        [
            _coord(pt=PointType.FIB),
            _coord(pt=PointType.FIB),
            _coord(pt=PointType.SURFACE),
        ]
    )
    assert _legend_labels(canvas._ax) == ["FIB", "SURFACE"]

    # legend follows content
    canvas.set_coordinates([_coord(pt=PointType.FM), _coord(pt=PointType.SURFACE_FM)])
    assert _legend_labels(canvas._ax) == ["FM", "FM-SURFACE"]

    # empty canvas → no legend
    canvas.set_coordinates([])
    assert canvas._ax.get_legend() is None


def test_canvas_legend_overlay_groups_and_toggle(qapp):
    from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas

    canvas = ImagePointCanvas()
    canvas.set_coordinates([_coord(pt=PointType.FIB)])
    canvas.add_overlay_points(
        [(1.0, 2.0)], color="#ff00ff", hollow=True, legend_label="POI uncorrected"
    )
    assert _legend_labels(canvas._ax) == ["FIB", "POI uncorrected"]

    canvas.clear_overlay()
    assert _legend_labels(canvas._ax) == ["FIB"]

    canvas.set_legend_visible(False)
    assert canvas._ax.get_legend() is None
    canvas.set_legend_visible(True)
    assert _legend_labels(canvas._ax) == ["FIB"]


def test_surface_crosshair_keeps_color(qapp):
    """Unfilled '+' markers are drawn by their edge — selection must not turn
    them white, and deselection must not erase them."""
    from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas

    canvas = ImagePointCanvas()
    fib = _coord(x=1.0, y=1.0, pt=PointType.FIB)
    surface = _coord(x=5.0, y=5.0, pt=PointType.SURFACE)
    canvas.set_coordinates([fib, surface])
    fib_artist, surf_artist = canvas._point_artists

    # unselected: crosshair keeps the type colour (was "none" → near-invisible)
    assert surf_artist.get_markeredgecolor() == "#ff9800"

    # selected: still the type colour, shown bolder (was white → colour lost)
    canvas.set_selected(surface)
    assert surf_artist.get_markeredgecolor() == "#ff9800"
    assert surf_artist.get_markeredgewidth() == pytest.approx(3.0)

    # filled circles keep the white-rim selection style
    canvas.set_selected(fib)
    assert fib_artist.get_markeredgecolor() == "white"
    assert surf_artist.get_markeredgecolor() == "#ff9800"
    assert surf_artist.get_markeredgewidth() == pytest.approx(2.0)


def test_fib_surface_line_follows_surface_point(qapp):
    """A dashed datum line spans the canvas at the FIB surface y."""
    from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas

    canvas = ImagePointCanvas()
    canvas.set_coordinates([_coord(x=1.0, y=1.0, pt=PointType.FIB)])
    assert canvas._surface_line is None

    surf = _coord(x=5.0, y=40.0, pt=PointType.SURFACE)
    canvas.set_coordinates([surf])
    assert canvas._surface_line is not None
    assert canvas._surface_line.get_ydata()[0] == pytest.approx(40.0)

    # follows coordinate edits
    surf.point.y = 55.0
    canvas.refresh_coordinate(surf)
    assert canvas._surface_line.get_ydata()[0] == pytest.approx(55.0)

    # removed with the point
    canvas.set_coordinates([])
    assert canvas._surface_line is None


def test_fm_surface_point_gets_no_line(qapp):
    """FM surfaces are z-planes — no in-plane datum line."""
    from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas

    canvas = ImagePointCanvas()
    canvas.set_coordinates([_coord(z=10.0, pt=PointType.SURFACE_FM)])
    assert canvas._surface_line is None


def test_surface_line_exported_by_render_to_axes(qapp):
    from matplotlib.figure import Figure

    from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas

    canvas = ImagePointCanvas()
    canvas.set_coordinates([_coord(x=5.0, y=40.0, pt=PointType.SURFACE)])
    fig = Figure()
    ax = fig.add_subplot(111)
    canvas.render_to_axes(ax)
    dashed = [l for l in ax.get_lines() if l.get_linestyle() == "--"]
    assert len(dashed) == 1
    assert dashed[0].get_ydata()[0] == pytest.approx(40.0)


def test_render_to_axes_replicates_legend(qapp):
    from matplotlib.figure import Figure

    from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas

    canvas = ImagePointCanvas()
    canvas.set_coordinates([_coord(pt=PointType.FIB)])
    canvas.add_overlay_points(
        [(1.0, 2.0)], color="#ff00ff", legend_label="POI (P)"
    )
    fig = Figure()
    ax = fig.add_subplot(111)
    canvas.render_to_axes(ax)
    assert _legend_labels(ax) == ["FIB", "POI (P)"]


def test_ghost_export_preserves_hollow_style(qapp):
    from matplotlib.figure import Figure

    from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas

    canvas = ImagePointCanvas()
    canvas.add_overlay_points(
        [(10.0, 20.0)],
        color="#ff00ff",
        size=13,
        alpha=0.7,
        show_labels=False,
        hollow=True,
    )
    fig = Figure()
    ax = fig.add_subplot(111)
    canvas.render_to_axes(ax)
    lines = ax.get_lines()
    assert len(lines) == 1
    assert lines[0].get_markerfacecolor() == "none"
    assert lines[0].get_alpha() == pytest.approx(0.7)
