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
    CorrelationState,
    CorrelationResult,
    PointType,
    PointXYZ,
)
from fibsem.structures import Point


def _coord(x=0.0, y=0.0, z=0.0, pt=PointType.FIB) -> Coordinate:
    return Coordinate(point=PointXYZ(x=x, y=y, z=z), point_type=pt)


def _widget(qapp):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
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


_RI_TAB_INDEX = 3


def test_ri_tab_is_reachable_before_any_run(qapp):
    """The optical parameters, the depth readout and the preview table are what
    a user wants right after placing an FM surface point — gating the whole tab
    on a finished run hid them at exactly that moment."""
    w = _widget(qapp)
    assert w._tabs.isTabEnabled(_RI_TAB_INDEX)

    w._on_canvas_add_requested(1.0, 2.0, PointType.SURFACE_FM)
    assert w._tabs.isTabEnabled(_RI_TAB_INDEX)


def test_apply_is_disabled_until_it_can_do_something(qapp):
    """Apply carries the gating the tab used to, and says why it cannot be
    pressed rather than accepting a press and reporting failure."""
    w = _widget(qapp)
    btn = w._ri_tab._btn_apply

    # no surface at all
    assert not btn.isEnabled()
    assert "Place a surface point" in btn.toolTip()

    # FM surface, no POI to move
    w._on_canvas_add_requested(1.0, 2.0, PointType.SURFACE_FM)
    assert not btn.isEnabled()
    assert "POI" in btn.toolTip()

    # FM surface + POI: usable with no result behind it, which is the point —
    # the pre-mode correction is set up before a run, not after one
    w._on_canvas_add_requested(5.0, 6.0, PointType.POI)
    assert btn.isEnabled()


def test_apply_stays_disabled_in_post_mode_until_there_is_a_result(qapp):
    """Post mode corrects a POI the correlation produced, so unlike pre mode it
    genuinely does need a run first."""
    w = _widget(qapp)
    btn = w._ri_tab._btn_apply

    w._on_canvas_add_requested(1.0, 200.0, PointType.SURFACE)
    w._on_canvas_add_requested(5.0, 6.0, PointType.POI)
    assert not btn.isEnabled()
    assert "Run the correlation first" in btn.toolTip()

    w._on_result_ready(_result_from(w.data))
    assert btn.isEnabled()


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
    # stands in for a runnable set of points; _update_run_button passes the data
    w._can_run = lambda data=None: True

    w._ri_tab._ri_widget.set_factor(1.5)
    w._ri_tab._chk_rerun.setChecked(True)
    w._ri_tab._apply()

    assert w._ri_pre_correction_factor == pytest.approx(1.5)
    assert w.data.ri_pre_correction_factor == pytest.approx(1.5)
    assert runs == [True]


def test_apply_pre_does_not_rerun_what_cannot_run(qapp):
    """The RI tab is reachable before the points can support a correlation, and
    _run does not check — it would start a worker that only errors."""
    w = _widget(qapp)
    _setup_pre_mode(w)

    runs = []
    w._run = lambda: runs.append(True)
    assert not w._can_run()  # no images, no fiducials

    w._ri_tab._ri_widget.set_factor(1.5)
    w._ri_tab._chk_rerun.setChecked(True)
    w._ri_tab._apply()

    assert w._ri_pre_correction_factor == pytest.approx(1.5)  # still stored
    assert runs == []
    assert "run correlation to apply" in w._lbl_status.text()


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
    from fibsem.structures import FibsemImage

    w = _widget(qapp)
    # Correcting needs the image dimensions: px_m is derived from them, and px_m
    # is what Continue commits (FIB-321).
    w.set_fib_image(FibsemImage.generate_blank_image(resolution=(512, 512), hfw=100e-6))
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
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        _POINT_TYPE_SIDES,
    )

    w = _widget(qapp)
    fib_expected = [pt for pt, s in _POINT_TYPE_SIDES.items() if s == "fib"]
    fm_expected = [pt for pt, s in _POINT_TYPE_SIDES.items() if s == "fm"]
    assert w._fib_canvas.picking._allowed_types == fib_expected
    assert w._fm_display.picking._allowed_types == fm_expected
    # every mapped type has a spec (checked at build time too)
    assert set(w._point_specs) == set(_POINT_TYPE_SIDES)


def test_inconsistent_spec_rejected(qapp):
    """Specs with mismatched side/adapter/fm_fit_role fail at construction."""
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
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


def _loaded_surface_without_a_factor(w):
    """The state a saved correlation restores when it was never Applied: an FM
    surface point and a POI, but no armed factor. Placing the point by hand arms
    one, so this is the only route left to the un-armed state."""
    w.set_data(
        CorrelationInputData(
            fm_surface_coordinate=_coord(z=10.0, pt=PointType.SURFACE_FM),
            poi_coordinates=[_coord(z=30.0, pt=PointType.POI)],
        )
    )
    w.data_changed.emit(w.data)


def test_placing_the_fm_surface_arms_the_factor_on_screen(qapp):
    """FIB-590: a plain Run has to apply the correction, so the point arms the
    factor as it lands — whatever the spinbox is showing."""
    w = _widget(qapp)
    w._ri_tab._ri_widget.set_factor(1.42)

    w._on_canvas_add_requested(1.0, 2.0, PointType.SURFACE_FM)

    assert w._ri_pre_correction_factor == pytest.approx(1.42)
    assert w.data.ri_pre_correction_factor == pytest.approx(1.42)
    assert "not applied" not in w._ri_tab._lbl_warning.text()
    # named as it lands; the next edit is free to reclaim the status line
    assert "1.420" in w._lbl_status.text()


def test_auto_arm_does_not_overwrite_a_chosen_factor(qapp):
    """Re-placing the surface must not discard a factor the user applied."""
    w = _widget(qapp)
    _setup_pre_mode(w)
    w._ri_tab._ri_widget.set_factor(1.2)
    w._ri_tab._chk_rerun.setChecked(False)
    w._ri_tab._apply()

    w._on_canvas_add_requested(7.0, 8.0, PointType.SURFACE_FM)  # moved the point
    assert w._ri_pre_correction_factor == pytest.approx(1.2)


def test_a_loaded_surface_without_a_factor_asks_for_apply(qapp):
    """FIB-590: a correlation saved before anyone pressed Apply restores a
    surface point that corrects nothing, and has to say so."""
    w = _widget(qapp)
    _loaded_surface_without_a_factor(w)

    assert w._ri_pre_correction_factor is None
    assert w.data.ri_pre_correction_factor is None  # a run here corrects nothing
    assert "press Apply" in w._ri_tab._lbl_warning.text()
    # and the banner must not claim the run does it by itself
    assert "applied during the run" not in w._ri_tab._lbl_mode.text()


def test_run_with_a_surface_but_no_factor_says_why_nothing_moved(qapp):
    """FIB-590: the RI tab opens only after a run, so the status line has to
    carry this — it is where the user is looking when nothing moves."""
    w = _widget(qapp)
    _loaded_surface_without_a_factor(w)

    result = _result_from(w.data)
    w._on_result_ready(result)

    assert "no RI correction" in w._lbl_status.text()
    assert "Refractive Index" in w._lbl_status.text()


def test_unapplied_factor_edit_is_reported(qapp):
    """FIB-591: Run uses the armed factor, not the spinbox, so a divergence
    between them must be visible rather than silently ignored."""
    w = _widget(qapp)
    _setup_pre_mode(w)
    tab = w._ri_tab

    tab._ri_widget.set_factor(1.5)
    tab._chk_rerun.setChecked(False)
    tab._apply()
    assert "stored — applied on the next run" in tab._lbl_warning.text()

    # typed by hand, never applied: the run would still use 1.5
    tab._ri_widget._spin_factor.setValue(1.8)
    assert "1.800 not applied" in tab._lbl_warning.text()
    assert "runs use 1.500" in tab._lbl_warning.text()
    assert w.data.ri_pre_correction_factor == pytest.approx(1.5)

    # applying it clears the divergence
    tab._apply()
    assert "not applied" not in tab._lbl_warning.text()
    assert w.data.ri_pre_correction_factor == pytest.approx(1.8)


def test_an_optical_parameter_edit_counts_as_an_unapplied_factor(qapp):
    """FIB-591: the parameters feed the factor through zeta, so editing one is
    just as un-applied as typing the number."""
    from fibsem.correlation.refractive_index import _LUT_PATH

    if not _LUT_PATH.exists():
        pytest.skip("LUT CSV not present — the parameter spinboxes are disabled")

    w = _widget(qapp)
    _setup_pre_mode(w)
    tab = w._ri_tab

    tab._ri_widget.set_factor(1.5)
    tab._chk_rerun.setChecked(False)
    tab._apply()

    tab._ri_widget._spin_n2.setValue(1.22)  # moves zeta well clear of 1.5
    assert "not applied" in tab._lbl_warning.text()
    assert "runs use 1.500" in tab._lbl_warning.text()


def test_mirroring_a_stored_factor_is_not_an_unapplied_edit(qapp):
    """set_factor blocks signals precisely so this stays quiet — otherwise every
    refresh that mirrors the stored value would report a divergence."""
    w = _widget(qapp)
    _setup_pre_mode(w)
    tab = w._ri_tab

    w._ri_pre_correction_factor = 1.5950000000000002  # as loaded from JSON
    result = CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=1.0, y=2.0))],
        refractive_index_correction_factor=1.5950000000000002,
        refractive_index_correction_mode="pre",
    )
    tab.set_result(result, input_data=w.data)

    # the spinbox rounds to 3 decimals; that must not read as an edit
    assert tab._ri_widget.get_factor() == pytest.approx(1.595)
    assert "not applied" not in tab._lbl_warning.text()
    assert "Correction applied" in tab._lbl_warning.text()


def test_tilt_lock_preserves_manual_factor(qapp):
    from fibsem.ui.correlation.widgets.refractive_index_widget import (
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


def test_canvas_toolbar_buttons_toggle_state(qapp):
    """Checkable toolbar buttons drive only their own canvas and stay in sync
    with the View-menu master toggle.

    Scalebar and reset come from the shared canvas; legend and labels are
    correlation's own, added by CorrelationPicking because the shared toolbar has
    neither and ImagePointCanvas carried both.
    """
    w = _widget(qapp)
    fib, fm = w._fib_canvas, w._fm_display

    # startup: the menu handler enabled the scalebar on both canvases
    assert fib.canvas.btn_toggle_scalebar.isChecked()
    assert fm.canvas.btn_toggle_scalebar.isChecked()
    assert fib.picking.btn_legend.isChecked()

    # a button toggles only its own canvas
    fib.canvas.btn_toggle_scalebar.click()
    assert fib.canvas._scalebar_visible is False
    assert fm.canvas._scalebar_visible is True

    fib.picking.btn_legend.click()
    assert fib.points._legend_visible is False
    fib.picking.btn_legend.click()
    assert fib.points._legend_visible is True

    # the View menu still drives both canvases and re-syncs button state
    w._on_scalebar_toggled(True)
    assert fib.canvas._scalebar_visible is True
    assert fib.canvas.btn_toggle_scalebar.isChecked()


def test_canvas_toolbar_reset_button(qapp):
    """Asserts the view actually resets, not that a method was called.

    The shared canvas connects the bound `reset_view` at construction, so
    replacing the attribute afterwards -- which worked on ImagePointCanvas, whose
    button went through a late-binding lambda -- intercepts nothing.
    """
    import numpy as np

    from fibsem.structures import FibsemImage

    w = _widget(qapp)
    w.set_fib_image(FibsemImage(data=np.zeros((64, 64), dtype=np.uint8)))
    canvas = w._fib_canvas.canvas
    fitted = canvas._ax.get_xlim()
    canvas._ax.set_xlim(10, 20)  # as a zoom would leave it

    canvas.btn_reset_view.click()

    assert canvas._ax.get_xlim() == fitted


def test_fm_scalebar_pixel_size_corrects_for_resize(qapp):
    """pixel_size_x describes the acquisition resolution; when the displayed
    data was resized without rewriting metadata, the scalebar pixel size must
    scale to the displayed width (no-op when metadata matches the data)."""
    from types import SimpleNamespace

    import numpy as np

    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
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


def _fake_fm(n_c=2, n_z=5, filename="/data/BeforeMilling_G1.ome.tiff"):
    from types import SimpleNamespace

    import numpy as np

    chans = [SimpleNamespace(name=f"CH{i}", color=None) for i in range(n_c)]
    return SimpleNamespace(
        data=np.zeros((n_c, n_z, 8, 8), dtype=np.float32),
        # filepath is where the volume was loaded from and is what names it in the
        # UI (FIB-509); metadata.filename is the name it was acquired under.
        filepath=filename,
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


def _wheel(canvas, *, angle=(0, 0), pixel=(0, 0), shift=True):
    """Deliver a real QWheelEvent, so the Qt -> matplotlib path is exercised.

    The test above hand-builds a matplotlib event and calls _on_scroll directly,
    which is why it kept passing while Z stepping was dead on a real mouse: the
    event never reached _on_scroll at all.
    """
    import numpy as np
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    if canvas._ax.images == []:
        canvas.set_image(np.zeros((64, 64), np.uint8))
        canvas.resize(400, 400)
    pos = QPointF(200.0, 200.0)
    QApplication.sendEvent(
        canvas,
        QWheelEvent(
            pos, pos, QPoint(*pixel), QPoint(*angle), Qt.NoButton,
            Qt.ShiftModifier if shift else Qt.NoModifier, Qt.NoScrollPhase, False,
        ),
    )


def test_fm_header_shows_image_name(qapp):
    """Ported from FMImageDisplayWidget, which owned its own header. The tab
    widget owns both now, through one helper, so the FIB and FM panes cannot
    drift apart the way the two copies could."""
    w = _widget(qapp)
    assert w._fm_name_label.isHidden() is True  # nothing loaded yet

    w.set_fm_image(_fake_fm(filename="/some/dir/BeforeMilling_G1.ome.tiff"))

    assert w._fm_name_label.text() == "BeforeMilling_G1.ome.tiff"
    assert w._fm_name_label.isHidden() is False


def test_fib_header_shows_image_name(qapp):
    """The file it was loaded from, extension and all — see FIB-509 for why the
    acquisition metadata's name is not it."""
    from types import SimpleNamespace

    w = _widget(qapp)
    assert w._fib_name_label.isHidden() is True  # nothing loaded yet

    img = SimpleNamespace(filepath="/x/ref_Mill_res_02.tif")
    w._update_fib_name_label(img)
    assert w._fib_name_label.text() == "ref_Mill_res_02.tif"
    assert w._fib_name_label.isHidden() is False


def test_save_plot_in_view_menu_and_test_menu_removed(qapp):
    w = _widget(qapp)
    assert w._action_save_plot.text() == "Save Plot"  # View menu, not the run bar
    assert not hasattr(w, "_btn_save_plot")
    assert not hasattr(w, "_action_test_save_plot")
    # Shift+scroll Z stepping rides canvas_scrolled now; it needs a loaded
    # stack, so it is tested in test_correlation_fm_canvas_widget.py.


def test_dialog_allows_window_minimise(qapp):
    from PyQt5.QtCore import Qt

    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabDialog,
    )

    d = CorrelationTabDialog()
    assert bool(d.windowFlags() & Qt.WindowMinimizeButtonHint)
    assert hasattr(d, "_min_shortcut")


def test_discover_correlation_files(tmp_path):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        _discover_correlation_files,
    )

    # Empty directory → everything None.
    assert _discover_correlation_files(str(tmp_path)) == {
        "fib": None,
        "fm": None,
        "correlation": None,
        "data": None,
        "result": None,
    }

    (tmp_path / "BeforeMilling_G1.ome.tiff").write_bytes(b"")
    (tmp_path / "ref_Mill_ib.tif").write_bytes(b"")
    (tmp_path / "correlation.json").write_text("{}")
    (tmp_path / "correlation_data.json").write_text("{}")
    (tmp_path / "correlation_result.json").write_text("{}")

    found = _discover_correlation_files(str(tmp_path))
    assert os.path.basename(found["fm"]) == "BeforeMilling_G1.ome.tiff"
    assert os.path.basename(found["fib"]) == "ref_Mill_ib.tif"
    assert os.path.basename(found["correlation"]) == "correlation.json"
    assert os.path.basename(found["data"]) == "correlation_data.json"
    assert os.path.basename(found["result"]) == "correlation_result.json"


def test_discover_fib_falls_back_to_non_ome_tif(tmp_path):
    """With no *_ib.tif, the FIB is the first TIFF that isn't the OME-TIFF."""
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        _discover_correlation_files,
    )

    (tmp_path / "scene.ome.tif").write_bytes(b"")  # FM — must not be taken as FIB
    (tmp_path / "fib_image.tif").write_bytes(b"")  # plain TIFF → FIB fallback

    found = _discover_correlation_files(str(tmp_path))
    assert os.path.basename(found["fm"]) == "scene.ome.tif"
    assert os.path.basename(found["fib"]) == "fib_image.tif"


def test_load_error_reverts_path_field(qapp, monkeypatch):
    """A failed load reverts the picker to the last-good file, so the selection
    never shows an image that isn't actually loaded."""
    from PyQt5.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    tab = _widget(qapp)._images_tab
    tab._fib_loaded_path = "/good/prev_ib.tif"
    tab._fib_picker.show_path("/good/prev_ib.tif")

    tab._fib_picker.show_path("/nope/does_not_exist_ib.tif")
    tab._load_fib("/nope/does_not_exist_ib.tif")  # FibsemImage.load raises

    assert tab._fib_picker.current_path() == "/good/prev_ib.tif"
    assert tab._fib_loaded_path == "/good/prev_ib.tif"


def _fake_fib(pixel_size_m=52.1e-9):
    """FIB image stub carrying just the pixel size the RMS badge needs."""
    from types import SimpleNamespace

    return SimpleNamespace(
        metadata=SimpleNamespace(pixel_size=SimpleNamespace(x=pixel_size_m))
    )


def _result_fit(rms=1.82, n=6, worst=(3.0, 1.0)):
    """A result with n fiducials, one of which is the worst by some margin."""
    deltas = [Point(x=0.1, y=0.1) for _ in range(max(n - 1, 0))]
    if n:
        deltas.append(Point(x=worst[0], y=worst[1]))
    return CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
        rms_error=rms,
        reprojected_3d=[PointXYZ(x=0.0, y=0.0, z=0.0) for _ in range(n)],
        delta_2d=deltas,
    )


def test_rms_never_certifies_a_good_fit():
    """There is deliberately no "good" colour. A residual can show a fit is
    wrong; it cannot show one is right, and green would read as that promise."""
    from fibsem.ui.correlation.widgets.correlation_tab_widget import _rms_concern

    color, reason = _rms_concern(20.0, 8, 1.1)  # about as clean as it gets
    assert color == "#9aa0a6" and reason is None  # neutral, no verdict either way


def test_rms_flags_detectable_problems():
    from fibsem.ui.correlation.widgets.correlation_tab_widget import _rms_concern

    # A minimum-pair fit: residual is small by construction, not by agreement.
    color, reason = _rms_concern(20.0, 4, 1.1)
    assert color == "#ffb300" and "no redundancy" in reason

    # One correspondence dominating the error — the case an average hides.
    color, reason = _rms_concern(20.0, 8, 3.4)
    assert color == "#ffb300" and "3.4× the RMS" in reason

    # Far enough out to be breakage rather than a judgement call.
    color, reason = _rms_concern(1500.0, 8, 1.1)
    assert color == "#e53935" and "not converged" in reason


def test_rms_relative_checks_survive_a_missing_pixel_size():
    """A result loaded from JSON restores no images, so nm is unavailable — but
    pair counts and error ratios are unitless and still apply."""
    from fibsem.ui.correlation.widgets.correlation_tab_widget import _rms_concern

    color, reason = _rms_concern(None, 4, 3.4)
    assert color == "#ffb300"
    assert "no redundancy" in reason and "3.4× the RMS" in reason

    assert _rms_concern(None, 8, 1.1) == ("#9aa0a6", None)


def test_rms_badge_reports_physical_distance(qapp):
    """With a FIB pixel size the badge shows nm, so the number means something
    independent of the HFW the image was taken at."""
    w = _widget(qapp)
    w._fib_image = _fake_fib(pixel_size_m=52.1e-9)
    w._on_result_ready(_result_fit(rms=1.82))

    # 1.82 px x 52.1 nm/px = 95 nm
    assert "95 nm" in w._lbl_result.text()
    assert "#9aa0a6" in w._lbl_result.text()  # nothing wrong detected → no verdict


def test_rms_badge_falls_back_to_px_without_pixel_size(qapp):
    """A result loaded from JSON restores no images, so nm is unavailable — show
    px and stay neutral rather than colouring a figure we cannot interpret."""
    w = _widget(qapp)  # no FIB image
    w._on_result_ready(_result_fit(rms=1.82))

    assert "1.82 px" in w._lbl_result.text()
    assert "#9aa0a6" in w._lbl_result.text()
    assert "pixel size unknown" in w._lbl_result.toolTip().lower()


def test_rms_tooltip_explains_what_is_measured(qapp):
    w = _widget(qapp)
    w._fib_image = _fake_fib(pixel_size_m=52.1e-9)
    w._on_result_ready(_result_fit(rms=1.82, n=6, worst=(3.0, 1.0)))
    tip = w._lbl_result.toolTip()

    assert "1.82 px" in tip and "52.1 nm/px" in tip  # the conversion is shown
    assert "6 fiducial pairs" in tip
    assert "3.16 px" in tip  # worst single fiducial, hypot(3, 1)
    assert "not POI accuracy" in tip


def test_rms_tooltip_flags_a_minimum_fiducial_fit(qapp):
    """At 4 pairs the rigid fit has almost no redundancy, so a small residual is
    structural rather than evidence of a good correlation."""
    w = _widget(qapp)
    w._fib_image = _fake_fib()

    w._on_result_ready(_result_fit(n=4))
    assert "no redundancy" in w._lbl_result.toolTip()

    w._on_result_ready(_result_fit(n=8))
    assert "no redundancy" not in w._lbl_result.toolTip()


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


def test_result_summary_clears_on_edit(qapp):
    """Editing a coordinate invalidates the reported RMS, so the badge must not
    linger beside a status line that has already reset to "Ready."."""
    w = _widget(qapp)
    w._on_result_ready(
        CorrelationResult(
            poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
            rms_error=1.5,
        )
    )
    assert w._lbl_result.isHidden() is False

    # go through the real signal chain rather than calling the handler directly
    w._on_canvas_moved(_coord())
    assert w._lbl_result.isHidden() is True


def _result(rms=1.5) -> CorrelationResult:
    return CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
        rms_error=rms,
    )


def test_continue_gated_on_live_result(qapp):
    """Continue commits result.poi[0].px_m to the protocol editor, so it must only
    be armed while the result still describes the current points — otherwise an
    edit after a run hands the editor a position from the old point set."""
    w = _widget(qapp)
    assert w._btn_continue.isEnabled() is False  # nothing to continue with yet

    w._on_result_ready(_result())
    assert w._btn_continue.isEnabled() is True

    w._on_canvas_moved(_coord())
    assert w._btn_continue.isEnabled() is False


def test_run_continue_emphasis_follows_result(qapp):
    """A successful run makes Continue the primary action; an edit that invalidates
    the result hands primacy back to Run."""
    from fibsem.ui import stylesheets

    w = _widget(qapp)
    assert w._btn_run.styleSheet() == stylesheets.PRIMARY_BUTTON_STYLESHEET
    assert w._btn_continue.styleSheet() == stylesheets.SECONDARY_BUTTON_STYLESHEET

    w._on_result_ready(_result())
    assert w._btn_continue.styleSheet() == stylesheets.PRIMARY_BUTTON_STYLESHEET
    assert w._btn_run.styleSheet() == stylesheets.SECONDARY_BUTTON_STYLESHEET

    w._on_canvas_moved(_coord())
    assert w._btn_run.styleSheet() == stylesheets.PRIMARY_BUTTON_STYLESHEET
    assert w._btn_continue.styleSheet() == stylesheets.SECONDARY_BUTTON_STYLESHEET


def _press_release(canvas, coord, *, drag_to=None):
    """Drive a real press → (optional motion) → release cycle on the canvas."""
    from types import SimpleNamespace

    def _ev(x, y):
        sx, sy = canvas._ax.transData.transform((x, y))
        return SimpleNamespace(
            inaxes=canvas._ax, button=1, x=sx, y=sy, xdata=x, ydata=y
        )

    canvas._on_press(_ev(coord.point.x, coord.point.y))
    if drag_to is not None:
        canvas._on_motion(_ev(*drag_to))
    canvas._on_release(_ev(coord.point.x, coord.point.y))


def _fit_combos(w):
    cl = w._coords_tab
    return [
        cl._fib_method_combo,
        cl._fm_fid_method_combo,
        cl._fm_poi_method_combo,
        cl._fm_fid_ch_combo,
        cl._fm_poi_ch_combo,
    ]


def test_fit_combos_ignore_the_wheel(qapp):
    """Scrolling past the Fit Settings panel must not silently change a fit
    method or channel — the selection has to survive a wheel event."""
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtGui import QWheelEvent

    w = _widget(qapp)
    for combo in _fit_combos(w):
        if combo.count() < 2:
            combo.addItems(["a", "b"])  # channel combos start empty
        combo.setCurrentIndex(0)
        before = combo.currentIndex()
        for _ in range(3):
            qapp.sendEvent(
                combo,
                QWheelEvent(
                    QPoint(5, 5),
                    combo.mapToGlobal(QPoint(5, 5)),
                    QPoint(0, -120),
                    QPoint(0, -120),
                    Qt.NoButton,
                    Qt.NoModifier,
                    Qt.ScrollUpdate,
                    False,
                ),
            )
        assert combo.currentIndex() == before, f"{combo} changed on scroll"


def test_fit_method_combos_keep_their_defaults(qapp):
    """The ValueComboBox swap must not disturb the defaults the fits rely on."""
    cl = _widget(qapp)._coords_tab
    assert cl._fib_method_combo.currentText() == "Hole"
    assert cl._fm_fid_method_combo.currentText() == "None"
    assert cl._fm_poi_method_combo.currentText() == "Gaussian"


def test_z_slider_advertises_shift_scroll(qapp):
    """Shift+scroll-through-Z has no visible affordance, so the slider tooltip is
    the only place a user can discover it."""
    tip = _widget(qapp)._fm_display._z_slider.toolTip()
    assert "Shift" in tip and "scroll" in tip.lower()


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


def test_labels_toggle_hides_labels(qapp):
    import numpy as np

    from fibsem.structures import FibsemImage

    w = _widget(qapp)
    # The overlay draws artists only once the axes has content -- a point
    # added first is kept and drawn when the image arrives, but this test is
    # about the artists, so load one.
    w.set_fib_image(FibsemImage(data=np.zeros((64, 64), dtype=np.uint8)))
    w._on_canvas_add_requested(5.0, 5.0, PointType.FIB)
    fib = w._fib_canvas
    assert fib.points._anns[0].get_visible() is True  # shown by default

    w._on_labels_toggled(False)  # View → Show Labels off
    assert fib.points._anns[0].get_visible() is False
    assert fib.picking.btn_labels.isChecked() is False

    w._on_labels_toggled(True)
    assert fib.points._anns[0].get_visible() is True


# ---------------------------------------------------------------------------
# FIB-263: Load Coordinates on a result file
# ---------------------------------------------------------------------------


def _project(tmp_path, w):
    """A project dir with both auto-saved JSONs, as a real session leaves it."""
    import json

    w.set_project_dir(str(tmp_path))
    data = CorrelationInputData(
        fib_coordinates=[_coord(1.0, 2.0, pt=PointType.FIB)],
        fm_coordinates=[_coord(3.0, 4.0, pt=PointType.FM)],
        poi_coordinates=[_coord(5.0, 6.0, pt=PointType.POI)],
    )
    data_path = tmp_path / "correlation_data.json"
    data_path.write_text(json.dumps(data.to_dict()))

    result_path = tmp_path / "correlation_result.json"
    result_path.write_text(
        json.dumps(
            CorrelationResult(
                poi=[CorrelationPointOfInterest(image_px=Point(x=1.0, y=2.0))],
                rms_error=1.5,
            ).to_dict()
        )
    )
    return str(data_path), str(result_path)


def test_load_coordinates_rejects_a_result_file(qapp, tmp_path):
    """The reported crash: both JSONs sit in the project dir with near-identical
    names, and Load Coordinates on the result one took the app down."""
    w = _widget(qapp)
    _, result_path = _project(tmp_path, w)

    with pytest.raises(ValueError) as exc:
        w.load_data(result_path)
    assert "Load Correlation Result" in str(exc.value)  # says what to do instead


def test_load_coordinates_does_not_wipe_points_on_a_result_file(qapp, tmp_path):
    """Defaulting the missing key to [] instead of rejecting would turn the crash
    into silent data loss: an empty input data is applied, and the auto-save then
    overwrites correlation_data.json with the emptied set."""
    import json

    w = _widget(qapp)
    data_path, result_path = _project(tmp_path, w)
    w.load_data(data_path)
    assert len(w.data.fib_coordinates) == 1

    try:
        w.load_data(result_path)
    except ValueError:
        pass  # rejected, as it should be

    assert len(w.data.fib_coordinates) == 1  # still there, not cleared
    with open(data_path) as f:
        on_disk = json.load(f)
    assert len(on_disk["fib_coordinates"]) == 1  # and not overwritten on disk


def test_load_result_rejects_a_coordinates_file(qapp, tmp_path):
    """The mirror mistake, which produced a plausible-looking empty result."""
    w = _widget(qapp)
    data_path, _ = _project(tmp_path, w)

    with pytest.raises(ValueError) as exc:
        w.load_result(data_path)
    assert "Load Coordinates" in str(exc.value)


def test_menu_load_correlation_accepts_a_legacy_result_file(qapp, tmp_path, monkeypatch):
    """FIB-264: the single Load action dispatches on shape, so selecting the
    legacy result file — the exact FIB-263 misclick — now just loads it."""
    from PyQt5.QtWidgets import QFileDialog, QMessageBox

    w = _widget(qapp)
    _, result_path = _project(tmp_path, w)

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (result_path, ""))
    )
    shown = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(a))
    )

    w._menu_load_correlation()
    assert not shown, "a valid result file should load, not warn"
    assert len(w.result.poi) == 1  # the result was adopted


def test_menu_load_correlation_warns_on_a_foreign_file(qapp, tmp_path, monkeypatch):
    """A genuinely unreadable/foreign JSON still warns rather than crashing."""
    import json

    from PyQt5.QtWidgets import QFileDialog, QMessageBox

    w = _widget(qapp)
    junk = tmp_path / "notes.json"
    junk.write_text(json.dumps({"hello": "world"}))

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(junk), ""))
    )
    shown = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(a))
    )

    w._menu_load_correlation()  # must not raise
    assert shown, "expected a warning dialog rather than an exception"


# ---------------------------------------------------------------------------
# FIB-252: per-point fit confirmation
# ---------------------------------------------------------------------------


def test_coordinate_fitted_roundtrip():
    c = Coordinate(PointXYZ(1.0, 2.0, 3.0), PointType.FIB, fitted=True)
    d = c.to_dict()
    assert d["fitted"] is True
    assert Coordinate.from_dict(d).fitted is True
    # back-compat: a legacy dict without "fitted" defaults to False
    legacy = {"point": {"x": 1.0, "y": 2.0, "z": 3.0}, "point_type": "FIB"}
    assert Coordinate.from_dict(legacy).fitted is False


def test_point_fit_result_classify_and_delta():
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
        FitStatus,
        PointFitResult,
    )

    i = PointXYZ(100.0, 100.0, 5.0)
    assert PointFitResult.classify(i, PointXYZ(103.0, 104.0, 5.0)) is FitStatus.OK
    # exact fallback (fit returned the input) → no change
    assert PointFitResult.classify(i, PointXYZ(100.0, 100.0, 5.0)) is FitStatus.UNCHANGED
    # a 0.1 px refinement is a real move, not "no change"
    assert PointFitResult.classify(i, PointXYZ(100.1, 100.0, 5.0)) is FitStatus.OK
    assert PointFitResult.classify(i, PointXYZ(100.0, 100.0, 7.0)) is FitStatus.OK  # z move
    assert PointFitResult.classify(i, None, error="boom") is FitStatus.ERROR
    assert PointFitResult.classify(i, None) is FitStatus.ERROR

    r = PointFitResult(
        coordinate=_coord(), method="Hole", channel=None,
        initial=i, fitted=PointXYZ(103.0, 104.0, 5.0), status=FitStatus.OK,
    )
    assert round(r.delta_px, 1) == 5.0  # 3-4-5 triangle
    assert r.delta == (3.0, 4.0, 0.0)   # per-axis (dx, dy, dz)


def test_subpixel_change_is_visible_and_flagged():
    """A sub-pixel refinement must read as a real move at 3-decimal precision."""
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
        FitStatus,
        PointFitResult,
        _fmt_delta,
        _fmt_xyz,
    )

    initial = PointXYZ(803.300, 1111.900, 10.500)
    fitted = PointXYZ(803.340, 1111.940, 10.530)
    assert PointFitResult.classify(initial, fitted) is FitStatus.OK
    assert _fmt_xyz(fitted) == "803.340, 1111.940, 10.530"
    assert _fmt_delta(initial, fitted) == "+0.040, +0.040, +0.030"


def test_run_point_fit_returns_none_without_image(qapp):
    w = _widget(qapp)  # no FIB image loaded
    assert w._run_point_fit(_coord(10.0, 10.0, 0.0, PointType.FIB)) is None


def test_apply_fit_result_moves_and_flags(qapp):
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
        FitStatus,
        PointFitResult,
    )

    w = _widget(qapp)
    coord = _coord(10.0, 20.0, 0.0, PointType.FIB)
    w._coords_tab.fib_list.add_coordinate(coord)
    result = PointFitResult(
        coordinate=coord, method="Hole", channel=None,
        initial=PointXYZ(10.0, 20.0, 0.0), fitted=PointXYZ(13.0, 24.0, 0.0),
        status=FitStatus.OK,
    )
    emitted = []
    w.data_changed.connect(lambda d: emitted.append(d))
    w._apply_fit_result(result)

    assert (coord.point.x, coord.point.y) == (13.0, 24.0)
    assert coord.fitted is True
    assert emitted  # data_changed fired


def test_manual_move_clears_fitted_flag(qapp):
    w = _widget(qapp)
    coord = _coord(5.0, 5.0, 0.0, PointType.FIB)
    coord.fitted = True
    w._coords_tab.fib_list.add_coordinate(coord)

    w._on_canvas_moved(coord)              # drag
    assert coord.fitted is False

    coord.fitted = True
    w._on_list_changed(w._point_specs[PointType.FIB], coord, "x", 6.0)  # spinbox
    assert coord.fitted is False


def test_fit_confirmation_dialog_constructs(qapp):
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
        FitConfirmationDialog,
        FitStatus,
        PointFitResult,
    )

    ok = PointFitResult(
        coordinate=_coord(pt=PointType.FIB), method="Hole", channel=None,
        initial=PointXYZ(0.0, 0.0, 0.0), fitted=PointXYZ(2.0, 2.0, 0.0),
        status=FitStatus.OK,
    )
    assert FitConfirmationDialog(ok, show_figure=False) is not None

    err = PointFitResult(
        coordinate=_coord(pt=PointType.FIB), method="Hole", channel=None,
        initial=PointXYZ(0.0, 0.0, 0.0), fitted=None,
        status=FitStatus.ERROR, message="boom",
    )
    assert FitConfirmationDialog(err, show_figure=False) is not None


def test_fit_dialog_sizes_wide_figure_to_aspect(qapp):
    import numpy as np
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    from fibsem.correlation.fit_diagnostics import FitDiagnostic
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
        FitConfirmationDialog,
        FitStatus,
        PointFitResult,
    )

    # a z + XY diagnostic renders wide (9x4.5), like the FM fits
    diag = FitDiagnostic(
        title="t", roi_xy=np.zeros((10, 10)), input_xy=(5.0, 5.0),
        z_axis=np.arange(5), z_signal=np.arange(5.0), z_fit=np.arange(5.0),
        z_input=2.0, z_fitted=2.5,
    )
    r = PointFitResult(
        coordinate=_coord(pt=PointType.FM), method="Hole", channel=0,
        initial=PointXYZ(0.0, 0.0, 0.0), fitted=PointXYZ(2.0, 2.0, 3.0),
        status=FitStatus.OK, diagnostic=diag,
    )
    dialog = FitConfirmationDialog(r, show_figure=True)
    canvases = dialog.findChildren(FigureCanvasQTAgg)
    assert canvases, "diagnostic figure should be embedded"
    canvas = canvases[0]
    assert canvas.minimumWidth() > canvas.minimumHeight()  # wide, not squished
    assert canvas.minimumWidth() <= 900                     # capped


def test_humanize_fit_error_maps_known_modes():
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
        humanize_fit_error,
    )

    conv = humanize_fit_error(RuntimeError(
        "Optimal parameters not found: Number of calls to function has reached "
        "maxfev = 800."
    ))
    assert "converge" in conv and "maxfev" not in conv  # jargon dropped

    edge = humanize_fit_error(IndexError("index 70 is out of bounds"))
    assert "edge" in edge

    generic = humanize_fit_error(TypeError("something odd"))
    assert "unexpectedly" in generic


def test_fit_dialog_shows_channel_for_fm_only(qapp):
    from PyQt5.QtWidgets import QLabel

    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
        FitConfirmationDialog,
        FitStatus,
        PointFitResult,
    )

    def _labels(dialog):
        return {lbl.text() for lbl in dialog.findChildren(QLabel)}

    fm = PointFitResult(
        coordinate=_coord(pt=PointType.FM), method="Hole", channel=1,
        channel_name="reflection",
        initial=PointXYZ(0.0, 0.0, 0.0), fitted=PointXYZ(2.0, 2.0, 3.0),
        status=FitStatus.OK,
    )
    fm_labels = _labels(FitConfirmationDialog(fm, show_figure=False))
    assert "channel" in fm_labels and "reflection" in fm_labels

    fib = PointFitResult(
        coordinate=_coord(pt=PointType.FIB), method="Hole", channel=None,
        initial=PointXYZ(0.0, 0.0, 0.0), fitted=PointXYZ(2.0, 2.0, 0.0),
        status=FitStatus.OK,
    )
    assert "channel" not in _labels(FitConfirmationDialog(fib, show_figure=False))


def test_error_dialog_splits_title_and_wrapped_reason(qapp):
    from PyQt5.QtWidgets import QLabel

    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
        FitConfirmationDialog,
        FitStatus,
        PointFitResult,
    )

    reason = (
        "The fit didn't converge — the feature may be too faint, or not centred "
        "in the search region. Try re-centring your click, or a different "
        "channel / method."
    )
    r = PointFitResult(
        coordinate=_coord(pt=PointType.FM), method="Hole", channel=2,
        channel_name="Feature-2-Active-Reflection",
        initial=PointXYZ(0.0, 0.0, 0.0), fitted=None,
        status=FitStatus.ERROR, message=reason, detail="maxfev = 800",
    )
    dlg = FitConfirmationDialog(r, show_figure=False)
    labels = dlg.findChildren(QLabel)
    texts = [lbl.text() for lbl in labels]

    # The chip is a short title — NOT the long message concatenated in.
    assert "Fit failed" in texts
    assert not any(t.startswith("Fit failed:") for t in texts)
    # The reason lives on its own word-wrapped label (so it wraps, not stretches).
    reason_lbls = [lbl for lbl in labels if lbl.text() == reason]
    assert reason_lbls and reason_lbls[0].wordWrap()


def test_diagnostic_toggle_reveals_and_hides_figure(qapp):
    import numpy as np

    from fibsem.correlation.fit_diagnostics import FitDiagnostic
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
        FitConfirmationDialog,
        FitStatus,
        PointFitResult,
    )

    def _result():
        diag = FitDiagnostic(title="t", roi_xy=np.zeros((10, 10)), input_xy=(5.0, 5.0))
        return PointFitResult(
            coordinate=_coord(pt=PointType.FM), method="Hole", channel=0,
            initial=PointXYZ(0.0, 0.0, 0.0), fitted=PointXYZ(1.0, 1.0, 1.0),
            status=FitStatus.OK, diagnostic=diag,
        )

    # Checkbox unchecked -> figure is still embedded, just hidden; the button
    # offers to reveal it (we always have the diagnostic).
    dlg = FitConfirmationDialog(_result(), show_figure=False)
    assert dlg._canvas is not None
    assert dlg._canvas.isHidden()
    assert dlg._toggle_btn.text() == "Show diagnostic"

    dlg._on_toggle_diagnostic()
    assert not dlg._canvas.isHidden()
    assert dlg._toggle_btn.text() == "Hide diagnostic"

    dlg._on_toggle_diagnostic()
    assert dlg._canvas.isHidden()
    assert dlg._toggle_btn.text() == "Show diagnostic"

    # Checkbox checked -> shown by default.
    shown = FitConfirmationDialog(_result(), show_figure=True)
    assert not shown._canvas.isHidden()
    assert shown._toggle_btn.text() == "Hide diagnostic"


def test_accept_is_default_button_not_the_toggle(qapp):
    import numpy as np
    from PyQt5.QtWidgets import QPushButton

    from fibsem.correlation.fit_diagnostics import FitDiagnostic
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
        FitConfirmationDialog,
        FitStatus,
        PointFitResult,
    )

    diag = FitDiagnostic(title="t", roi_xy=np.zeros((10, 10)), input_xy=(5.0, 5.0))
    result = PointFitResult(
        coordinate=_coord(pt=PointType.FM), method="Hole", channel=0,
        initial=PointXYZ(0.0, 0.0, 0.0), fitted=PointXYZ(1.0, 1.0, 1.0),
        status=FitStatus.OK, diagnostic=diag,
    )
    dlg = FitConfirmationDialog(result, show_figure=True)
    buttons = {b.text(): b for b in dlg.findChildren(QPushButton)}

    assert buttons["Accept"].isDefault()          # Enter accepts the fit...
    assert not dlg._toggle_btn.isDefault()         # ...not toggles the diagnostic
    assert not dlg._toggle_btn.autoDefault()


def test_fitted_icon_reflects_state(qapp):
    from fibsem.ui.correlation.widgets.coordinate_list_widget import (
        CoordinateRowWidget,
    )

    coord = _coord(pt=PointType.FIB)
    coord.fitted = True
    row = CoordinateRowWidget(coord, "FIB-0")
    # Always visible (an aligned status column); state is encoded by colour.
    assert not row.fitted_icon.isHidden()
    assert "confirmed" in row.fitted_icon.toolTip()
    fitted_key = row.fitted_icon.pixmap().cacheKey()

    coord.fitted = False  # a manual edit supersedes the fit
    row.refresh()
    assert not row.fitted_icon.isHidden()          # still shown...
    assert "Manually" in row.fitted_icon.toolTip()  # ...but recoloured
    assert row.fitted_icon.pixmap().cacheKey() != fitted_key


def test_tooltip_labels_have_no_unscoped_background(qapp):
    # An unscoped `background: transparent` on a widget bleeds into its QToolTip
    # (a QLabel), making the tooltip background transparent. The tooltip-bearing
    # labels in a row must not carry a background rule in their own stylesheet.
    from fibsem.ui.correlation.widgets.coordinate_list_widget import (
        CoordinateRowWidget,
    )

    row = CoordinateRowWidget(_coord(pt=PointType.FIB), "FIB-0")
    assert row.fitted_icon.toolTip()               # it does have a tooltip
    assert "background" not in row.fitted_icon.styleSheet()
    assert "background" not in row.name_label.styleSheet()


def test_name_col_width_is_compact_for_short_types(qapp):
    from fibsem.ui.correlation.widgets.coordinate_list_widget import (
        _NAME_FIXED_WIDTH,
        _name_col_width,
    )

    fm = _name_col_width(PointType.FM)
    poi = _name_col_width(PointType.POI)
    surface = _name_col_width(PointType.SURFACE_FM)

    # Short-name lists (FM 8, POI 1) collapse well under the old fixed 100px,
    # killing the empty gap; a surface name needs more room but is still capped
    # at the former width, so surface lists don't regress.
    assert fm < 70 and poi < 70
    assert fm < surface <= _NAME_FIXED_WIDTH


def test_f_hotkey_fits_selected_coordinate(qapp, monkeypatch):
    w = _widget(qapp)
    fitted = []
    monkeypatch.setattr(w, "_on_refit_requested", lambda c: fitted.append(c))

    coord = _coord(10.0, 10.0, 0.0, PointType.FIB)
    lw = w._point_specs[PointType.FIB].list_widget
    lw.coordinates = [coord]
    lw.select_coordinate_silent(coord)

    w._fit_selected_coordinate()
    assert fitted == [coord]


def test_f_hotkey_noop_without_selection(qapp, monkeypatch):
    w = _widget(qapp)
    fitted = []
    monkeypatch.setattr(w, "_on_refit_requested", lambda c: fitted.append(c))

    w._fit_selected_coordinate()  # nothing selected
    assert fitted == []


def test_f_hotkey_skips_while_editing_a_field(qapp, monkeypatch):
    from PyQt5.QtWidgets import QApplication, QLineEdit

    w = _widget(qapp)
    fitted = []
    monkeypatch.setattr(w, "_on_refit_requested", lambda c: fitted.append(c))

    coord = _coord(10.0, 10.0, 0.0, PointType.FIB)
    lw = w._point_specs[PointType.FIB].list_widget
    lw.coordinates = [coord]
    lw.select_coordinate_silent(coord)

    editing = QLineEdit()
    monkeypatch.setattr(QApplication, "focusWidget", lambda: editing)
    w._fit_selected_coordinate()
    assert fitted == []  # don't hijack the key mid-edit


def _fit_result(status, dx=0.0, dy=0.0, dz=0.0, diagnostic=None):
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import PointFitResult

    initial = PointXYZ(100.0, 100.0, 10.0)
    fitted = (
        None
        if status.name == "ERROR"
        else PointXYZ(100.0 + dx, 100.0 + dy, 10.0 + dz)
    )
    return PointFitResult(
        coordinate=_coord(100.0, 100.0, 10.0, PointType.FM),
        method="Hole", channel=0, initial=initial, fitted=fitted,
        status=status, diagnostic=diagnostic,
    )


def test_auto_accept_gating(qapp):
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import FitStatus

    w = _widget(qapp)
    ok = _fit_result(FitStatus.OK, dx=2.0)
    err = _fit_result(FitStatus.ERROR)
    far = _fit_result(FitStatus.OK, dx=100.0)  # surprising jump

    w._coords_tab._auto_accept_check.setChecked(False)
    assert not w._should_auto_accept(ok)   # off -> always confirm

    w._coords_tab._auto_accept_check.setChecked(True)
    assert w._should_auto_accept(ok)        # good, small fit -> apply
    assert not w._should_auto_accept(err)   # failures always surface
    assert not w._should_auto_accept(far)   # outliers fall back to the dialog


def test_auto_accept_applies_without_dialog(qapp, monkeypatch):
    import fibsem.ui.correlation.widgets.correlation_tab_widget as mod
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import FitStatus

    w = _widget(qapp)
    w._coords_tab._auto_accept_check.setChecked(True)

    result = _fit_result(FitStatus.OK, dx=2.0)
    monkeypatch.setattr(w, "_run_point_fit", lambda c: result)
    applied = []
    monkeypatch.setattr(w, "_apply_fit_result", lambda r: applied.append(r))

    def _boom(*a, **k):
        raise AssertionError("confirm dialog must not open in auto-accept")

    monkeypatch.setattr(mod, "FitConfirmationDialog", _boom)

    w._on_refit_requested(_coord(pt=PointType.FM))
    assert applied == [result]


def test_auto_accept_error_still_shows_dialog(qapp, monkeypatch):
    import fibsem.ui.correlation.widgets.correlation_tab_widget as mod
    from PyQt5.QtWidgets import QDialog

    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import FitStatus

    w = _widget(qapp)
    w._coords_tab._auto_accept_check.setChecked(True)

    monkeypatch.setattr(w, "_run_point_fit", lambda c: _fit_result(FitStatus.ERROR))
    constructed = []

    class _FakeDialog:
        def __init__(self, *a, **k):
            constructed.append(True)

        def exec_(self):
            return QDialog.Rejected

    monkeypatch.setattr(mod, "FitConfirmationDialog", _FakeDialog)

    w._on_refit_requested(_coord(pt=PointType.FM))
    assert constructed == [True]  # error surfaced despite auto-accept


def test_run_point_fit_does_not_mutate_coordinate(qapp, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    import fibsem.correlation.util as util
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import FitStatus

    w = _widget(qapp)
    w._fib_image = SimpleNamespace(filtered_data=np.zeros((64, 64), dtype=np.float32))
    diag = object()  # stand-in FitDiagnostic
    monkeypatch.setattr(
        util, "hole_fitting_FIB", lambda img, x, y: (x + 3.0, y + 4.0, diag)
    )

    coord = _coord(20.0, 20.0, 0.0, PointType.FIB)  # FIB method defaults to "Hole"
    result = w._run_point_fit(coord)

    # The fit is only a proposal — the coordinate is untouched until accepted.
    assert (coord.point.x, coord.point.y) == (20.0, 20.0)
    assert coord.fitted is False
    assert result.status is FitStatus.OK
    assert (result.fitted.x, result.fitted.y) == (23.0, 24.0)
    assert round(result.delta_px, 1) == 5.0
    assert result.diagnostic is diag


def test_refit_applies_on_accept_not_on_reject(qapp, monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    from PyQt5.QtWidgets import QDialog

    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw
    import fibsem.correlation.util as util

    w = _widget(qapp)
    w._fib_image = SimpleNamespace(filtered_data=np.zeros((64, 64), dtype=np.float32))
    monkeypatch.setattr(
        util, "hole_fitting_FIB", lambda img, x, y: (x + 3.0, y + 4.0, None)
    )
    coord = _coord(20.0, 20.0, 0.0, PointType.FIB)
    w._coords_tab.fib_list.add_coordinate(coord)

    # Reject → coordinate unchanged.
    monkeypatch.setattr(ctw.FitConfirmationDialog, "exec_", lambda self: QDialog.Rejected)
    w._on_refit_requested(coord)
    assert (coord.point.x, coord.point.y) == (20.0, 20.0)
    assert coord.fitted is False

    # Accept → coordinate moved and flagged.
    monkeypatch.setattr(ctw.FitConfirmationDialog, "exec_", lambda self: QDialog.Accepted)
    w._on_refit_requested(coord)
    assert (coord.point.x, coord.point.y) == (23.0, 24.0)
    assert coord.fitted is True


# ---------------------------------------------------------------------------
# Seam between the fit confirmation (FIB-252) and the result-live state (#143)
# ---------------------------------------------------------------------------


def test_accepting_a_fit_invalidates_the_live_result(qapp):
    """Applying a fit moves a fiducial, so the displayed result no longer
    describes the current points and Continue must not stay armed."""
    from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
        FitStatus,
        PointFitResult,
    )

    w = _widget(qapp)
    coord = _coord(10.0, 10.0, 0.0, PointType.FIB)
    w._on_canvas_add_requested(10.0, 10.0, PointType.FIB)
    coord = w._coords_tab.fib_list.coordinates[0]

    w._on_result_ready(
        CorrelationResult(
            poi=[CorrelationPointOfInterest(image_px=Point(x=1.0, y=2.0))],
            rms_error=1.5,
        )
    )
    assert w._btn_continue.isEnabled() is True

    w._apply_fit_result(
        PointFitResult(
            coordinate=coord,
            method="Hole",
            channel=None,
            initial=PointXYZ(10.0, 10.0, 0.0),
            fitted=PointXYZ(13.0, 14.0, 0.0),
            status=FitStatus.OK,
        )
    )

    assert coord.fitted is True
    assert w._btn_continue.isEnabled() is False  # result is stale now
    assert w._lbl_result.isHidden() is True


def _input(fib=(1.0, 2.0), ri=None) -> CorrelationInputData:
    return CorrelationInputData(
        fib_coordinates=[_coord(*fib, pt=PointType.FIB)],
        ri_pre_correction_factor=ri,
    )


def _result_from(data: CorrelationInputData) -> CorrelationResult:
    return CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
        rms_error=1.5,
        input_data=data,
    )


def test_matches_inputs_detects_a_moved_coordinate():
    result = _result_from(_input(fib=(1.0, 2.0)))
    assert result.matches_inputs(_input(fib=(1.0, 2.0))) is True
    assert result.matches_inputs(_input(fib=(1.0, 9.0))) is False


def test_matches_inputs_tracks_the_ri_factor():
    """The RI pre-correction scales POI z before the fit, so it does change the
    answer — unlike the flags below, it must count as a change."""
    result = _result_from(_input(ri=1.4))
    assert result.matches_inputs(_input(ri=1.4)) is True
    assert result.matches_inputs(_input(ri=1.8)) is False


def test_matches_inputs_ignores_changes_that_cannot_move_the_answer():
    """Scoping guard. If a change the transform never reads marks a good result
    stale, users learn to ignore the warning and it stops carrying information."""
    result = _result_from(_input())

    refit = _input()
    refit.fib_coordinates[0].fitted = True  # provenance, not a position
    assert result.matches_inputs(refit) is True

    other_method = _input()
    other_method.method = "single-point"  # not read by run_correlation_from_data
    assert result.matches_inputs(other_method) is True


def test_matches_inputs_ignores_image_presence():
    """A deserialized result carries no images, so image-derived properties are
    None on one side and populated on the other. Comparing them would report
    every reloaded result as stale."""
    saved = _input()  # no images, as if from JSON
    live = _input()
    live.fib_image = _fake_fib()
    assert _result_from(saved).matches_inputs(live) is True


def test_result_with_no_snapshot_cannot_claim_to_match():
    result = CorrelationResult(poi=[], input_data=None)
    assert result.matches_inputs(_input()) is False


def test_loading_a_stale_result_does_not_arm_continue(qapp):
    """Continue commits result.poi[0].px_m to the protocol editor. The in-session
    guard (_set_result_live on edit) does not survive a reload, so the staleness
    has to be re-derived from the result's own snapshot."""
    w = _widget(qapp)
    w.set_data(_input(fib=(5.0, 5.0)))  # the points as they are now

    stale = _result_from(_input(fib=(1.0, 2.0)))  # fitted to where they were
    w._load_result(stale, adopt_inputs=False)

    assert w._btn_continue.isEnabled() is False
    assert w._lbl_result.isHidden() is True  # no RMS badge for a dead transform
    assert "changed since this run" in w._lbl_status.text()


def test_loading_a_current_result_stays_live(qapp):
    w = _widget(qapp)
    w.set_data(_input(fib=(5.0, 5.0)))

    fresh = _result_from(_input(fib=(5.0, 5.0)))
    w._load_result(fresh, adopt_inputs=False)

    assert w._btn_continue.isEnabled() is True
    assert w._lbl_status.text() == "Done."


def test_load_result_can_keep_the_coordinates_already_loaded(qapp):
    """The destructive half of FIB-295: the result's embedded snapshot used to
    overwrite the lists unconditionally, so reopening a project silently
    discarded every edit made after the last run."""
    w = _widget(qapp)
    w.set_data(_input(fib=(5.0, 5.0)))

    w._load_result(_result_from(_input(fib=(1.0, 2.0))), adopt_inputs=False)

    kept = w._coords_tab.fib_list.coordinates
    assert [(c.point.x, c.point.y) for c in kept] == [(5.0, 5.0)]


def test_load_project_prefers_the_data_file_over_the_result_snapshot(qapp, tmp_path):
    """End-to-end of the reported path: edit after a run, reopen, keep the edit."""
    from fibsem.ui.correlation.widgets.correlation_tab_widget import load_project

    _result_from(_input(fib=(1.0, 2.0))).save(
        str(tmp_path / "correlation_result.json")
    )
    _input(fib=(5.0, 5.0)).save(str(tmp_path / "correlation_data.json"))

    w = _widget(qapp)
    load_project(w, str(tmp_path))

    kept = w._coords_tab.fib_list.coordinates
    assert [(c.point.x, c.point.y) for c in kept] == [(5.0, 5.0)]  # the edit survives
    assert w._btn_continue.isEnabled() is False  # and the old result isn't armed


def test_load_project_adopts_the_snapshot_when_there_is_no_data_file(qapp, tmp_path):
    """With no data file the result's snapshot is the only record of the points."""
    from fibsem.ui.correlation.widgets.correlation_tab_widget import load_project

    _result_from(_input(fib=(1.0, 2.0))).save(
        str(tmp_path / "correlation_result.json")
    )

    w = _widget(qapp)
    load_project(w, str(tmp_path))

    kept = w._coords_tab.fib_list.coordinates
    assert [(c.point.x, c.point.y) for c in kept] == [(1.0, 2.0)]
    assert w._btn_continue.isEnabled() is True  # consistent, so still usable


def test_stale_status_outranks_the_ri_note(qapp):
    """The status line is the only thing explaining why Continue is greyed out,
    so a stale result must not report 'Done — RI ×1.500'."""
    w = _widget(qapp)
    w.set_data(_input(fib=(5.0, 5.0)))

    stale = _result_from(_input(fib=(1.0, 2.0)))
    stale.refractive_index_correction_mode = "pre"
    stale.refractive_index_correction_factor = 1.5
    w._load_result(stale, adopt_inputs=False)

    assert "changed since this run" in w._lbl_status.text()
    assert "Done" not in w._lbl_status.text()


# ---------------------------------------------------------------------------
# FIB-264 — one consolidated correlation.json
# ---------------------------------------------------------------------------


def test_auto_save_writes_one_correlation_json(qapp, tmp_path):
    """Both data_changed and result_changed rewrite a single correlation.json;
    the legacy pair is never written."""
    import json

    w = _widget(qapp)
    w.set_project_dir(str(tmp_path))
    w.set_data(_input(fib=(3.0, 4.0)))
    w.data_changed.emit(w.data)

    single = tmp_path / "correlation.json"
    assert single.exists()
    assert not (tmp_path / "correlation_data.json").exists()
    assert not (tmp_path / "correlation_result.json").exists()

    raw = json.loads(single.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["result"] is None
    assert raw["input_data"]["fib_coordinates"][0]["point"]["x"] == 3.0

    # a run adds the result to the same file
    w._on_result_ready(_result_from(_input(fib=(3.0, 4.0))))
    raw = json.loads(single.read_text(encoding="utf-8"))
    assert raw["result"] is not None
    assert "computed_from" in raw["result"]  # the snapshot, renamed


def test_load_correlation_round_trips_a_consolidated_file(qapp, tmp_path):
    w = _widget(qapp)
    w.set_project_dir(str(tmp_path))
    w.set_data(_input(fib=(5.0, 5.0)))
    w._on_result_ready(_result_from(_input(fib=(5.0, 5.0))))  # live, matches

    fresh = _widget(qapp)
    fresh.load_correlation(str(tmp_path / "correlation.json"))
    assert [(c.point.x, c.point.y) for c in fresh._coords_tab.fib_list.coordinates] == [(5.0, 5.0)]
    assert fresh._btn_continue.isEnabled() is True  # consistent -> usable


def test_load_correlation_flags_a_stale_consolidated_file(qapp, tmp_path):
    """A file saved after a post-run edit carries the new points plus the old
    result; on load, matches_inputs must still flag it (FIB-295 within one file)."""
    import json

    w = _widget(qapp)
    w.set_project_dir(str(tmp_path))
    w.set_data(_input(fib=(1.0, 2.0)))
    w._on_result_ready(_result_from(_input(fib=(1.0, 2.0))))  # result fitted here
    w.set_data(_input(fib=(9.0, 9.0)))  # ...then the points move
    w.data_changed.emit(w.data)         # rewrite: new points, stale result

    raw = json.loads((tmp_path / "correlation.json").read_text(encoding="utf-8"))
    assert raw["input_data"]["fib_coordinates"][0]["point"]["x"] == 9.0
    assert raw["result"]["computed_from"]["fib_coordinates"][0]["point"]["x"] == 1.0

    fresh = _widget(qapp)
    fresh.load_correlation(str(tmp_path / "correlation.json"))
    assert [(c.point.x, c.point.y) for c in fresh._coords_tab.fib_list.coordinates] == [(9.0, 9.0)]
    assert fresh._btn_continue.isEnabled() is False  # stale result not armed


def test_load_project_prefers_the_consolidated_file(qapp, tmp_path):
    """With correlation.json present, the legacy pair is ignored even if it
    disagrees."""
    from fibsem.ui.correlation.widgets.correlation_tab_widget import load_project

    CorrelationState(input_data=_input(fib=(5.0, 5.0))).save(
        str(tmp_path / "correlation.json")
    )
    # a stale legacy pair that must be ignored
    _input(fib=(1.0, 1.0)).save(str(tmp_path / "correlation_data.json"))

    w = _widget(qapp)
    load_project(w, str(tmp_path))
    assert [(c.point.x, c.point.y) for c in w._coords_tab.fib_list.coordinates] == [(5.0, 5.0)]


def test_load_project_falls_back_to_legacy_when_no_consolidated_file(qapp, tmp_path):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import load_project

    _input(fib=(7.0, 7.0)).save(str(tmp_path / "correlation_data.json"))

    w = _widget(qapp)
    load_project(w, str(tmp_path))
    assert [(c.point.x, c.point.y) for c in w._coords_tab.fib_list.coordinates] == [(7.0, 7.0)]


def test_load_correlation_reads_a_legacy_data_file(qapp, tmp_path):
    """A pre-FIB-264 correlation_data.json still loads via the unified entry."""
    p = tmp_path / "correlation_data.json"
    _input(fib=(7.0, 8.0)).save(str(p))

    w = _widget(qapp)
    w.load_correlation(str(p))
    assert [(c.point.x, c.point.y) for c in w._coords_tab.fib_list.coordinates] == [(7.0, 8.0)]
    assert w._result is None  # a bare data file carries no result


def test_load_correlation_file_dispatches_on_every_shape(tmp_path):
    """The one place the shape-sniff lives: container / legacy data / legacy
    result (new + old snapshot key) / junk. This is the 'old files still load'
    guarantee at the unit level."""
    import json

    from fibsem.correlation.structures import load_correlation_file

    # new container
    cpath = tmp_path / "c.json"
    CorrelationState(
        input_data=_input(fib=(1.0, 1.0)), result=_result_from(_input(fib=(1.0, 1.0)))
    ).save(str(cpath))
    st = load_correlation_file(str(cpath))
    assert st.input_data.fib_coordinates[0].point.x == 1.0 and st.result is not None

    # legacy data file
    dpath = tmp_path / "correlation_data.json"
    _input(fib=(2.0, 2.0)).save(str(dpath))
    st = load_correlation_file(str(dpath))
    assert st.result is None and st.input_data.fib_coordinates[0].point.x == 2.0

    # legacy result file (current computed_from key)
    rpath = tmp_path / "correlation_result.json"
    _result_from(_input(fib=(3.0, 3.0))).save(str(rpath))
    st = load_correlation_file(str(rpath))
    assert st.result is not None and st.input_data.fib_coordinates[0].point.x == 3.0

    # legacy result file written with the OLD "input_data" snapshot key
    opath = tmp_path / "old_result.json"
    old = _result_from(_input(fib=(4.0, 4.0))).to_dict()
    old["input_data"] = old.pop("computed_from")
    opath.write_text(json.dumps(old))
    st = load_correlation_file(str(opath))
    assert st.input_data.fib_coordinates[0].point.x == 4.0

    # foreign / junk JSON
    jpath = tmp_path / "junk.json"
    jpath.write_text(json.dumps({"foo": 1}))
    with pytest.raises(ValueError):
        load_correlation_file(str(jpath))


# ---------------------------------------------------------------------------
# FIB-298 — CorrelationConfig passed in on open, read back on close
# ---------------------------------------------------------------------------


def test_widget_config_round_trips(qapp):
    """set_correlation_config -> combos + RI widget -> correlation_config back."""
    from fibsem.correlation.config import CorrelationConfig, FitSettings, RISettings

    w = _widget(qapp)
    cfg = CorrelationConfig(
        fit=FitSettings(fib_method="None", fm_poi_method="Hole", reflection_cutout=4),
        ri=RISettings(na=0.9, wavelength_um=0.488, n2=1.33),
        load_spot_burns=False,
    )
    w.set_correlation_config(cfg)

    cl = w._coords_tab
    assert cl._fib_method_combo.currentText() == "None"
    assert cl._fm_poi_method_combo.currentText() == "Hole"
    p = w._ri_tab._ri_widget.get_params()
    assert (p.NA, p.n2) == (pytest.approx(0.9), pytest.approx(1.33))
    assert p.wavelength_um == pytest.approx(0.488)

    out = w.correlation_config
    assert out.fit.fib_method == "None"
    assert out.fit.reflection_cutout == 4      # UI-less field carried through
    assert out.ri.na == pytest.approx(0.9)
    assert out.load_spot_burns is False        # UI-less field carried through


def test_widget_config_reflects_a_combo_edit(qapp):
    from fibsem.correlation.config import CorrelationConfig

    w = _widget(qapp)
    w.set_correlation_config(CorrelationConfig())
    w._coords_tab._fib_method_combo.setCurrentText("Gaussian")
    assert w.correlation_config.fit.fib_method == "Gaussian"


def test_config_default_matches_current_behaviour(qapp):
    """A widget that was never given a config reports today's defaults."""
    w = _widget(qapp)
    fit = w.correlation_config.fit
    assert (fit.fib_method, fit.fm_fiducial_method, fit.fm_poi_method) == \
        ("Hole", "None", "Gaussian")


def test_channel_by_name_lands_once_the_channel_exists(qapp):
    """A config naming a POI channel applies after the FM image populates the
    combos — stored by name, resolved against the image (not a pinned index)."""
    from fibsem.correlation.config import CorrelationConfig, FitSettings

    w = _widget(qapp)
    # config set before any FM image: channel combo is still empty, no-op
    w.set_correlation_config(
        CorrelationConfig(fit=FitSettings(fm_poi_channel="CH1"))
    )
    assert w._coords_tab._fm_poi_ch_combo.currentText() == ""

    # channels populate; re-apply lands the named selection (as set_fm_image does)
    w._coords_tab.rebuild_channel_combos(_fake_fm(n_c=3))
    w._apply_fit_config()
    assert w._coords_tab._fm_poi_ch_combo.currentText() == "CH1"
    assert w.correlation_config.fit.fm_poi_channel == "CH1"


def _fake_fm_with_optics(names=("Reflection", "GFP"), wl_nm=(488, 520), na=0.85):
    """FM whose channels carry the ζ-seedable metadata (λ, NA)."""
    from types import SimpleNamespace

    import numpy as np

    chans = [
        SimpleNamespace(name=n, color=None, excitation_wavelength=w,
                        objective_numerical_aperture=na)
        for n, w in zip(names, wl_nm)
    ]
    return SimpleNamespace(
        data=np.zeros((len(chans), 5, 8, 8), dtype=np.float32),
        metadata=SimpleNamespace(filename="/data/fm.ome.tiff", channels=chans),
    )


def test_ri_seeds_wavelength_and_na_from_poi_channel(qapp):
    """FIB-277: λ comes from the *POI* channel's excitation wavelength, NA from
    the objective — both read from metadata, not left at the generic default."""
    from fibsem.correlation.config import CorrelationConfig, FitSettings

    w = _widget(qapp)
    fm = _fake_fm_with_optics(names=("Reflection", "GFP"), wl_nm=(488, 520), na=0.85)
    w._coords_tab.rebuild_channel_combos(fm)
    # POI channel = GFP (520 nm)
    w.set_correlation_config(CorrelationConfig(fit=FitSettings(fm_poi_channel="GFP")))
    w._apply_fit_config()

    w._seed_ri_from_fm_metadata(fm)

    p = w._ri_tab._ri_widget.get_params()
    assert p.wavelength_um == pytest.approx(0.520)   # GFP, not Reflection
    assert p.NA == pytest.approx(0.85)


def test_ri_seed_respects_a_manual_edit(qapp):
    """A wavelength the user typed is not clobbered by metadata on FM load."""
    w = _widget(qapp)
    ri = w._ri_tab._ri_widget
    ri._spin_wl.setValue(600.0)
    ri._spin_wl.editingFinished.emit()   # marks it user-edited

    w._seed_ri_from_fm_metadata(_fake_fm_with_optics(wl_nm=(488, 520)))
    assert ri.get_params().wavelength_um == pytest.approx(0.600)  # kept


def test_ri_seed_skips_when_metadata_absent(qapp):
    """A channel with no optics metadata leaves the RI params at their default."""
    w = _widget(qapp)
    before = w._ri_tab._ri_widget.get_params()
    w._seed_ri_from_fm_metadata(_fake_fm(n_c=2))  # plain channels, no λ/NA
    assert w._ri_tab._ri_widget.get_params() == before


# ---------------------------------------------------------------------------
# FIB-299 — seed a new correlation from the previous run
# ---------------------------------------------------------------------------


def _fm_at_zstep(z_step_m, n_z=21):
    from types import SimpleNamespace

    import numpy as np

    return SimpleNamespace(
        data=np.zeros((1, n_z, 8, 8), dtype=np.float32),
        metadata=SimpleNamespace(
            pixel_size_z=z_step_m, pixel_size_x=100e-9, channels=[], filename=""
        ),
    )


def test_stored_fm_pixel_size_z_round_trips():
    d = CorrelationInputData(
        fib_coordinates=[_coord(1.0, 2.0, pt=PointType.FIB)],
        stored_fm_pixel_size_z=500e-9,
    )
    assert d.to_dict()["fm_pixel_size_z"] == 500e-9
    assert CorrelationInputData.from_dict(d.to_dict()).stored_fm_pixel_size_z == 500e-9
    legacy = d.to_dict()
    del legacy["fm_pixel_size_z"]  # a file written before FIB-299
    assert CorrelationInputData.from_dict(legacy).stored_fm_pixel_size_z is None


def test_seed_rescales_fm_z_across_a_different_zstep(qapp):
    """A seed picked at 500 nm/slice, reloaded into a 100 nm/slice (interpolated)
    volume, must land at the same physical depth — index 10 -> 50."""
    w = _widget(qapp)
    w._fm_image = _fm_at_zstep(100e-9, n_z=105)   # current volume, interpolated
    source = CorrelationInputData(
        fm_coordinates=[_coord(2.0, 2.0, z=10.0, pt=PointType.FM)],
        stored_fm_pixel_size_z=500e-9,            # source volume z-step
    )
    w.seed_coordinates(source)

    assert w._coords_tab.fm_list.coordinates[0].point.z == pytest.approx(50.0)
    assert w._result is None                      # previous result not carried in


def test_seed_is_a_noop_when_the_zstep_matches(qapp):
    w = _widget(qapp)
    w._fm_image = _fm_at_zstep(500e-9)
    source = CorrelationInputData(
        fm_coordinates=[_coord(2.0, 2.0, z=10.0, pt=PointType.FM)],
        stored_fm_pixel_size_z=500e-9,
    )
    w.seed_coordinates(source)
    assert w._coords_tab.fm_list.coordinates[0].point.z == pytest.approx(10.0)


def test_seed_without_a_stored_zstep_does_not_rescale(qapp):
    """A legacy seed with no stored z-step is loaded as-is (can't reconcile)."""
    w = _widget(qapp)
    w._fm_image = _fm_at_zstep(100e-9, n_z=105)
    source = CorrelationInputData(
        fm_coordinates=[_coord(2.0, 2.0, z=10.0, pt=PointType.FM)],
        stored_fm_pixel_size_z=None,
    )
    w.seed_coordinates(source)
    assert w._coords_tab.fm_list.coordinates[0].point.z == pytest.approx(10.0)


def test_selecting_a_fitted_point_keeps_its_fitted_flag(qapp):
    """_on_canvas_moved clears `fitted`, and a select-click must not reach it --
    otherwise merely clicking a fitted point unflags it.

    Ported from the ImagePointCanvas suite. The guard now lives in
    PointOverlay._on_release (it only emits when the position actually changed),
    so this is the correlation-level proof that the guard is wired through.
    """
    import numpy as np

    from fibsem.structures import FibsemImage

    w = _widget(qapp)
    w.set_fib_image(FibsemImage(data=np.zeros((100, 100), dtype=np.uint8)))
    coord = _coord(50.0, 50.0, pt=PointType.FIB)
    coord.fitted = True
    w._fib_canvas.set_coordinates([coord])

    moved = []
    w._fib_canvas.point_moved.connect(moved.append)

    _press_release(w._fib_canvas.points, 50.0, 50.0)  # click to select, no drag
    assert moved == []
    assert coord.fitted is True  # selection is not an edit

    _press_release(w._fib_canvas.points, 50.0, 50.0, drag_to=(60.0, 70.0))
    assert moved == [coord]


def _press_release(overlay, x, y, *, drag_to=None):
    """Drive press → (optional motion) → release on the point overlay."""
    from types import SimpleNamespace

    ax = overlay._ax

    def _ev(px, py):
        sx, sy = ax.transData.transform((px, py))
        return SimpleNamespace(
            inaxes=ax, button=1, x=sx, y=sy, xdata=px, ydata=py, dblclick=False
        )

    overlay._on_press(_ev(x, y))
    if drag_to is not None:
        overlay._on_motion(_ev(*drag_to))
    overlay._on_release(_ev(*(drag_to or (x, y))))
