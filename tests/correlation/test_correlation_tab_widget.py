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

    w._on_fib_add_requested(1.0, 2.0, PointType.SURFACE)
    assert len(cl.surface_list.coordinates) == 1

    w._on_fm_add_requested(3.0, 4.0, PointType.SURFACE_FM)
    assert cl.surface_list.coordinates == []
    assert len(cl.fm_surface_list.coordinates) == 1

    w._on_fib_add_requested(5.0, 6.0, PointType.SURFACE)
    assert cl.fm_surface_list.coordinates == []
    assert len(cl.surface_list.coordinates) == 1


def test_fm_surface_add_is_max_one(qapp):
    w = _widget(qapp)
    cl = w._coords_tab
    w._on_fm_add_requested(1.0, 1.0, PointType.SURFACE_FM)
    w._on_fm_add_requested(2.0, 2.0, PointType.SURFACE_FM)
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
    w._on_fm_add_requested(3.0, 4.0, PointType.SURFACE_FM)
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

    w._on_fib_add_requested(1.0, 200.0, PointType.SURFACE)
    assert tab.mode == "post"
    assert not tab._chk_rerun.isVisible()

    w._on_fm_add_requested(1.0, 2.0, PointType.SURFACE_FM)
    assert tab.mode == "pre"


def test_tilt_locked_to_zero_in_pre_mode(qapp):
    w = _widget(qapp)
    spin_tilt = w._ri_tab._ri_widget._spin_tilt
    default_tilt = spin_tilt.value()
    assert default_tilt == pytest.approx(15.0)

    w._on_fm_add_requested(1.0, 2.0, PointType.SURFACE_FM)
    assert spin_tilt.value() == pytest.approx(0.0)
    assert not spin_tilt.isEnabled()

    # removing the FM surface (via replacing with a FIB surface) unlocks tilt
    w._on_fib_add_requested(1.0, 200.0, PointType.SURFACE)
    assert spin_tilt.value() == pytest.approx(default_tilt)


# ---------------------------------------------------------------------------
# Pre-mode apply
# ---------------------------------------------------------------------------


def _setup_pre_mode(w):
    w._on_fm_add_requested(1.0, 2.0, PointType.SURFACE_FM)
    w._coords_tab.fm_surface_list.coordinates[0].point.z = 10.0
    w._on_fm_add_requested(5.0, 6.0, PointType.POI)
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
    w._on_fib_add_requested(1.0, 200.0, PointType.SURFACE)

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
    w._on_fib_add_requested(1.0, 200.0, PointType.SURFACE)

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
