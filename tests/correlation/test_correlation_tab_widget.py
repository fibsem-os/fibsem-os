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


def test_status_short_and_result_summary_after_run(qapp):
    """After a run: RMS shows beside Continue; the status line carries a compact
    RI / POI note (or 'Done.') instead of the old verbose sentence — and it
    survives _update_run_button."""
    w = _widget(qapp)
    result = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
        rms_error=1.5,
        refractive_index_correction_factor=1.5,
        refractive_index_correction_mode="pre",
    )
    w._on_result_ready(result)
    assert w._lbl_status.text() == "Done — RI ×1.500"
    assert "RMS 1.50 px" in w._lbl_result.text()

    with_ghost = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=60.0))],
        poi_uncorrected=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
        rms_error=1.5,
        refractive_index_correction_factor=1.5,
        refractive_index_correction_mode="pre",
    )
    w._on_result_ready(with_ghost)
    assert w._lbl_status.text() == "Done — RI ×1.500, POI Δ40.0 px"

    plain = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
        rms_error=0.8,
    )
    w._on_result_ready(plain)
    assert w._lbl_status.text() == "Done."
    assert "RMS 0.80 px" in w._lbl_result.text()


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


def test_canvas_allow_lists_derive_from_registry_map(qapp):
    """The right-click add menus and the registry share one source of truth."""
    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        _POINT_TYPE_SIDES,
    )

    w = _widget(qapp)
    fib_expected = [pt for pt, s in _POINT_TYPE_SIDES.items() if s == "fib"]
    fm_expected = [pt for pt, s in _POINT_TYPE_SIDES.items() if s == "fm"]
    assert w._fib_canvas._allowed_types == fib_expected
    assert w._fm_display.canvas._allowed_types == fm_expected
    # every mapped type has a spec (checked at build time too)
    assert set(w._point_specs) == set(_POINT_TYPE_SIDES)


def test_inconsistent_spec_rejected(qapp):
    """Specs with mismatched side/adapter/fm_fit_role fail at construction."""
    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        _PointTypeSpec,
    )

    w = _widget(qapp)
    fib_list = w._coords_tab.fib_list

    # FIB-side type bound to the FM adapter
    with pytest.raises(ValueError, match="does not match"):
        _PointTypeSpec(PointType.FIB, fib_list, w._fm_adapter)

    # FIB-side spec with an FM fit role
    with pytest.raises(ValueError, match="fm_fit_role"):
        _PointTypeSpec(PointType.FIB, fib_list, w._fib_adapter, fm_fit_role="fid")

    # FM-side spec without a fit role
    with pytest.raises(ValueError, match="fm_fit_role"):
        _PointTypeSpec(PointType.POI, w._coords_tab.poi_list, w._fm_adapter)


def test_on_cleared_fires_only_when_last_point_removed(qapp):
    """The lifecycle hook means 'the spec's last point is gone', not
    'any point was removed' — pinned with a multi-point spec."""
    from dataclasses import replace

    w = _widget(qapp)
    fired = []
    poi_spec = w._point_specs[PointType.POI]
    w._point_specs[PointType.POI] = replace(
        poi_spec, on_cleared=lambda: fired.append(True)
    )

    w._on_canvas_add_requested(1.0, 1.0, PointType.POI)
    w._on_canvas_add_requested(2.0, 2.0, PointType.POI)
    first, second = w._point_specs[PointType.POI].list_widget.coordinates

    w._on_canvas_removed(first)
    assert fired == []  # one point remains
    w._on_canvas_removed(second)
    assert fired == [True]  # last point gone


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
    assert w._lbl_status.text() == "Done — RI ×1.500"


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


def test_canvas_toolbar_buttons_toggle_state(qapp):
    """Each canvas gets reset/scalebar/legend buttons; checkable buttons drive
    only their own canvas and stay in sync with the View-menu master toggle."""
    w = _widget(qapp)
    canvas = w._fib_canvas
    assert len(canvas._overlay_buttons) == 4

    # startup: the menu handler enabled the scalebar on both canvases
    assert canvas._btn_scalebar.isChecked()
    assert w._fm_display.canvas._btn_scalebar.isChecked()
    assert canvas._btn_legend.isChecked()

    # a button toggles only its own canvas
    canvas._btn_scalebar.click()
    assert canvas._show_scalebar is False
    assert w._fm_display.canvas._show_scalebar is True

    canvas._btn_legend.click()
    assert canvas._legend_visible is False
    canvas._btn_legend.click()
    assert canvas._legend_visible is True

    # the View menu still drives both canvases and re-syncs button state
    w._on_scalebar_toggled(True)
    assert canvas._show_scalebar is True
    assert canvas._btn_scalebar.isChecked()


def test_fm_scalebar_pixel_size_corrects_for_resize(qapp):
    """pixel_size_x describes the acquisition resolution; when the displayed
    data was resized without rewriting metadata, the scalebar pixel size must
    scale to the displayed width (no-op when metadata matches the data)."""
    from types import SimpleNamespace

    import numpy as np

    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    def _fm(data_w, res_w, px):
        return SimpleNamespace(
            data=np.zeros((1, 3, data_w, data_w), dtype=np.uint8),
            metadata=SimpleNamespace(pixel_size_x=px, resolution=(res_w, res_w)),
        )

    eff = CorrelationTabWidget._effective_fm_pixel_size
    # matched: unchanged
    assert eff(_fm(512, 512, 150e-9)) == pytest.approx(150e-9)
    # displayed 512 from a 2048 acquisition → 4× larger pixel
    assert eff(_fm(512, 2048, 150e-9)) == pytest.approx(600e-9)
    # missing/zero pixel size → None (no scalebar rather than a wrong one)
    assert eff(_fm(512, 512, None)) is None
    # no resolution metadata → fall back to the raw value
    raw = SimpleNamespace(
        data=np.zeros((1, 1, 512, 512), dtype=np.uint8),
        metadata=SimpleNamespace(pixel_size_x=150e-9, resolution=None),
    )
    assert eff(raw) == pytest.approx(150e-9)


def test_canvas_toolbar_reset_button(qapp):
    w = _widget(qapp)
    canvas = w._fib_canvas
    calls = []
    canvas.reset_view = lambda: calls.append(True)
    canvas._overlay_buttons[0].click()  # reset is the first-added (rightmost)
    assert calls == [True]


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


# ---------------------------------------------------------------------------
# Polish round 2: Z navigation, FM header, save-plot, minimise
# ---------------------------------------------------------------------------


def _fake_fm(n_c=2, n_z=5, filename="/data/BeforeMilling_G1.ome.tiff"):
    from types import SimpleNamespace

    import numpy as np

    chans = [SimpleNamespace(name=f"CH{i}", color=None) for i in range(n_c)]
    return SimpleNamespace(
        data=np.zeros((n_c, n_z, 8, 8), dtype=np.float32),
        metadata=SimpleNamespace(filename=filename, channels=chans),
    )


def _scroll_event(canvas, *, button, shift):
    """A matplotlib-style scroll event carrying a Qt guiEvent with modifiers.

    Mirrors real usage: modifier state is read from ``event.guiEvent.modifiers()``
    (matplotlib's own ``event.key`` is unreliable in an embedded canvas).
    """
    from types import SimpleNamespace

    from PyQt5.QtCore import Qt

    mods = Qt.ShiftModifier if shift else Qt.NoModifier
    gui = SimpleNamespace(modifiers=lambda: mods)
    return SimpleNamespace(
        inaxes=canvas._ax, xdata=1.0, ydata=1.0, button=button, guiEvent=gui
    )


def test_canvas_shift_scroll_emits_z_when_enabled(qapp):
    from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas

    canvas = ImagePointCanvas()
    got = []
    canvas.z_scroll_requested.connect(got.append)

    # Disabled by default: Shift+wheel zooms, never emits.
    canvas._on_scroll(_scroll_event(canvas, button="up", shift=True))
    assert got == []

    # Enabled: Shift+wheel emits +1 (up) / -1 (down), no zoom.
    canvas.set_shift_z_scroll_enabled(True)
    canvas._on_scroll(_scroll_event(canvas, button="up", shift=True))
    canvas._on_scroll(_scroll_event(canvas, button="down", shift=True))
    assert got == [1, -1]

    # Enabled but no Shift held: still zooms, no emit.
    canvas._on_scroll(_scroll_event(canvas, button="up", shift=False))
    assert got == [1, -1]


def test_fm_display_shows_image_name(qapp):
    from fibsem.correlation.ui.widgets.fm_image_display_widget import (
        FMImageDisplayWidget,
    )

    w = FMImageDisplayWidget()
    assert w._name_label.isHidden() is True  # nothing loaded yet

    w.set_fm_image(_fake_fm(filename="/some/dir/BeforeMilling_G1.ome.tiff"))
    assert w._name_label.text() == "BeforeMilling_G1.ome.tiff"
    assert w._name_label.isHidden() is False


def test_fib_header_shows_image_name(qapp):
    from types import SimpleNamespace

    w = _widget(qapp)
    assert w._fib_name_label.isHidden() is True  # nothing loaded yet

    img = SimpleNamespace(
        metadata=SimpleNamespace(
            image_settings=SimpleNamespace(filename="/x/ref_Mill_res_02")
        )
    )
    w._update_fib_name_label(img)
    assert w._fib_name_label.text() == "ref_Mill_res_02"
    assert w._fib_name_label.isHidden() is False


def test_fm_display_z_step_clamps_and_mip_disables(qapp):
    from fibsem.correlation.ui.widgets.fm_image_display_widget import (
        FMImageDisplayWidget,
    )

    w = FMImageDisplayWidget()
    w.set_fm_image(_fake_fm(n_z=5))
    assert w.current_z == 2  # starts mid-stack (n_z // 2)

    w._step_z(1)
    assert w.current_z == 3
    for _ in range(10):
        w._step_z(-1)
    assert w.current_z == 0  # clamped at floor
    for _ in range(10):
        w._step_z(1)
    assert w.current_z == 4  # clamped at n_z - 1

    # MIP disables the slider + step buttons and freezes _step_z.
    w._mip_check.setChecked(True)
    assert not w._z_prev.isEnabled()
    assert not w._z_next.isEnabled()
    assert not w._z_slider.isEnabled()
    frozen = w.current_z
    w._step_z(-1)
    assert w.current_z == frozen


def test_save_plot_in_view_menu_and_test_menu_removed(qapp):
    w = _widget(qapp)
    assert w._action_save_plot.text() == "Save Plot"  # View menu, not the run bar
    assert not hasattr(w, "_btn_save_plot")
    assert not hasattr(w, "_action_test_save_plot")
    # The FM display opts its canvas into Shift+scroll Z stepping.
    assert w._fm_display.canvas._shift_z_enabled is True


def test_dialog_allows_window_minimise(qapp):
    from PyQt5.QtCore import Qt

    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        CorrelationTabDialog,
    )

    d = CorrelationTabDialog()
    assert bool(d.windowFlags() & Qt.WindowMinimizeButtonHint)
    assert hasattr(d, "_min_shortcut")


def test_discover_correlation_files(tmp_path):
    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        _discover_correlation_files,
    )

    # Empty directory → everything None.
    assert _discover_correlation_files(str(tmp_path)) == {
        "fib": None,
        "fm": None,
        "data": None,
        "result": None,
    }

    (tmp_path / "BeforeMilling_G1.ome.tiff").write_bytes(b"")
    (tmp_path / "ref_Mill_ib.tif").write_bytes(b"")
    (tmp_path / "correlation_data.json").write_text("{}")
    (tmp_path / "correlation_result.json").write_text("{}")

    found = _discover_correlation_files(str(tmp_path))
    assert os.path.basename(found["fm"]) == "BeforeMilling_G1.ome.tiff"
    assert os.path.basename(found["fib"]) == "ref_Mill_ib.tif"
    assert os.path.basename(found["data"]) == "correlation_data.json"
    assert os.path.basename(found["result"]) == "correlation_result.json"


def test_discover_fib_falls_back_to_non_ome_tif(tmp_path):
    """With no *_ib.tif, the FIB is the first TIFF that isn't the OME-TIFF."""
    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        _discover_correlation_files,
    )

    (tmp_path / "scene.ome.tif").write_bytes(b"")  # FM — must not be taken as FIB
    (tmp_path / "fib_image.tif").write_bytes(b"")  # plain TIFF → FIB fallback

    found = _discover_correlation_files(str(tmp_path))
    assert os.path.basename(found["fm"]) == "scene.ome.tif"
    assert os.path.basename(found["fib"]) == "fib_image.tif"


def test_load_error_reverts_path_field(qapp, monkeypatch):
    """A failed load reverts the path field to the last-good value, so re-focusing
    the field doesn't re-attempt (and re-warn about) the bad path."""
    from PyQt5.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    tab = _widget(qapp)._images_tab
    tab._fib_loaded_path = "/good/prev_ib.tif"
    tab._fib_path.setText("/good/prev_ib.tif")

    tab._fib_path.setText("/nope/does_not_exist_ib.tif")
    tab._load_fib("/nope/does_not_exist_ib.tif")  # FibsemImage.load raises

    assert tab._fib_path.text() == "/good/prev_ib.tif"
    assert tab._fib_loaded_path == "/good/prev_ib.tif"


def test_rms_color_thresholds():
    from fibsem.correlation.ui.widgets.correlation_tab_widget import _rms_color

    assert _rms_color(1.0) == "#4caf50"  # good → green
    assert _rms_color(3.0) == "#ffb300"  # ok → amber
    assert _rms_color(8.0) == "#e53935"  # poor → red


def test_result_summary_hidden_until_run(qapp):
    w = _widget(qapp)
    assert w._lbl_result.isHidden() is True  # hidden until a run completes

    result = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
        rms_error=1.5,
    )
    w._on_result_ready(result)
    assert w._lbl_result.isHidden() is False
    assert "RMS 1.50 px" in w._lbl_result.text()


def test_advanced_panels_start_collapsed(qapp):
    cl = _widget(qapp)._coords_tab
    # Advanced / set-once panels collapse by default...
    assert cl._surface_panel._btn_collapse.isChecked() is False
    assert cl._fm_surface_panel._btn_collapse.isChecked() is False
    assert cl._fit_panel._btn_collapse.isChecked() is False
    # ...while the everyday fiducial/POI panels stay expanded.
    assert cl._fib_panel._btn_collapse.isChecked() is True
    assert cl._fm_panel._btn_collapse.isChecked() is True
    assert cl._poi_panel._btn_collapse.isChecked() is True


def test_point_labels_have_outline(qapp):
    """Coloured labels get a dark outline (path effect) so they stay legible on
    any image background."""
    from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas

    canvas = ImagePointCanvas()
    canvas.set_coordinates([_coord(5.0, 5.0, 0.0, PointType.FIB)])
    label = canvas._label_artists[0]
    assert label.get_path_effects()  # dark outline applied
    assert label.get_fontsize() == 9


def test_labels_toggle_hides_labels(qapp):
    w = _widget(qapp)
    w._on_canvas_add_requested(5.0, 5.0, PointType.FIB)
    fib = w._fib_canvas
    assert fib._label_artists[0].get_visible() is True  # shown by default

    w._on_labels_toggled(False)  # View → Show Labels off
    assert fib._label_artists[0].get_visible() is False
    assert fib._btn_labels.isChecked() is False

    w._on_labels_toggled(True)
    assert fib._label_artists[0].get_visible() is True
