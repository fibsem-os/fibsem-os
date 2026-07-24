"""Tests for the "Set up Correlation" pre-dialog (FIB-302).

Covers the dialog's choices (default source, enable rules, run selection,
interpolation result), the protocol editor's PreviewData builder (coordinate
normalisation), and the interpolation passthrough's target computation.
Headless PyQt5, offscreen; the button handlers are driven directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.config import CorrelationConfig
from fibsem.correlation.history import CorrelationRun, LamellaCorrelation
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationState,
    PointType,
    PointXYZ,
)
from fibsem.correlation.ui.widgets.correlation_setup_dialog import (
    SEED_NONE,
    SEED_PREVIOUS,
    SEED_SPOT_BURNS,
    CorrelationSetupDialog,
    _format_timestamp,
)
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import FibsemImage, Point


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    """The correlation tab widget (built in one test) fetches a LUT — stub it."""
    import fibsem.correlation.ui.widgets.refractive_index_widget as riw

    monkeypatch.setattr(riw, "_ensure_lut", lambda: None)


OLD = "2026-07-20_09-00-00"
NEW = "2026-07-24_14-32-00"


def _run(name, inp=None):
    return CorrelationRun(
        path="/tmp/" + name,
        name=name,
        state=CorrelationState(input_data=inp or CorrelationInputData()),
    )


def _dialog(*, spot_burn_count=0, history=None, config=None):
    return CorrelationSetupDialog(
        lamella_name="Lamella 03",
        fib_options=["a_ib.tif", "b_ib.tif"],
        fib_current="a_ib.tif",
        fm_options=["a.ome.tiff", "b.ome.tiff"],
        fm_current="a.ome.tiff",
        spot_burn_count=spot_burn_count,
        history=history,
        config=config or CorrelationConfig(),
    )


# ---------------------------------------------------------------------------
# starting-coordinates default + enable rules
# ---------------------------------------------------------------------------


def test_default_source_is_previous_when_history_exists(qapp):
    dlg = _dialog(history=LamellaCorrelation(runs=[_run(OLD), _run(NEW)]))
    assert dlg.seed_source() == SEED_PREVIOUS
    dlg._on_accept()
    assert dlg.setup.previous_run.name == NEW  # newest


def test_default_source_is_spot_burns_when_no_history(qapp):
    dlg = _dialog(spot_burn_count=3)
    assert dlg.seed_source() == SEED_SPOT_BURNS
    dlg._on_accept()
    assert dlg.setup.previous_run is None


def test_default_source_is_none_when_nothing_available(qapp):
    dlg = _dialog()
    assert dlg.seed_source() == SEED_NONE


def test_enable_rules_disable_unavailable_sources(qapp):
    dlg = _dialog()  # no burns, no history
    assert not dlg._rb_burns.isEnabled()
    assert not dlg._rb_prev.isEnabled()

    dlg2 = _dialog(spot_burn_count=2, history=LamellaCorrelation(runs=[_run(NEW)]))
    assert dlg2._rb_burns.isEnabled()
    assert dlg2._rb_prev.isEnabled()
    assert dlg2._run_combo.isEnabled()  # previous is the default → dropdown live


def test_run_dropdown_selects_an_older_run(qapp):
    dlg = _dialog(history=LamellaCorrelation(runs=[_run(OLD), _run(NEW)]))
    dlg._run_combo.setCurrentIndex(1)  # newest-first display → index 1 is the older
    dlg._on_accept()
    assert dlg.setup.previous_run.name == OLD


def test_switching_to_none_clears_previous_run(qapp):
    dlg = _dialog(history=LamellaCorrelation(runs=[_run(NEW)]))
    dlg._rb_none.setChecked(True)
    dlg._on_accept()
    assert dlg.setup.seed_source == SEED_NONE
    assert dlg.setup.previous_run is None


# ---------------------------------------------------------------------------
# result: images + interpolation
# ---------------------------------------------------------------------------


def test_setup_reflects_image_selection(qapp):
    dlg = _dialog(spot_burn_count=1)
    dlg._fib_combo.setCurrentText("b_ib.tif")
    dlg._fm_combo.setCurrentText("b.ome.tiff")
    dlg._on_accept()
    assert dlg.setup.fib_filename == "b_ib.tif"
    assert dlg.setup.fm_filename == "b.ome.tiff"


def test_isotropic_yields_no_explicit_target(qapp):
    cfg = CorrelationConfig()
    cfg.interpolation.enabled = True
    cfg.interpolation.isotropic = True
    dlg = _dialog(config=cfg)
    dlg._on_accept()
    assert dlg.setup.interpolate is True
    assert dlg.setup.isotropic is True
    assert dlg.setup.target_z_nm is None  # isotropic → target computed from XY


def test_explicit_target_is_carried(qapp):
    cfg = CorrelationConfig()
    cfg.interpolation.enabled = True
    cfg.interpolation.isotropic = False
    dlg = _dialog(config=cfg)
    dlg._spin_z.setValue(250.0)
    dlg._on_accept()
    assert dlg.setup.isotropic is False
    assert dlg.setup.target_z_nm == pytest.approx(250.0)


def test_interpolation_off_by_default_config(qapp):
    dlg = _dialog()  # default config: interpolation disabled
    dlg._on_accept()
    assert dlg.setup.interpolate is False


def test_format_timestamp_reformats_and_falls_back():
    assert _format_timestamp("2026-07-24_14-30-05") == "2026-07-24 14:30:05"
    assert _format_timestamp("not-a-timestamp") == "not-a-timestamp"


# ---------------------------------------------------------------------------
# PreviewData builder (protocol editor) — coordinate normalisation
# ---------------------------------------------------------------------------


def _fm_image(nz=5):
    data = np.zeros((1, nz, 120, 120), np.uint16)
    yy, xx = np.mgrid[0:120, 0:120]
    for z in range(nz):
        data[0, z] = (300 * np.exp(-((xx - 60) ** 2 + (yy - 55) ** 2) / 128)).astype(
            np.uint16
        )
    ch = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        emission_wavelength=520.0,
        power=0.3,
        exposure_time=0.05,
        gain=1.5,
        offset=50.0,
    )
    meta = FluorescenceImageMetadata(
        acquisition_date=datetime(2026, 7, 24).isoformat(),
        pixel_size_x=40e-9,
        pixel_size_y=40e-9,
        pixel_size_z=200e-9,
        resolution=(120, 120),
        channels=[ch],
    )
    return FluorescenceImage(data=data, metadata=meta)


def test_build_preview_normalises_coordinates(qapp):
    from fibsem.ui.widgets.autolamella_lamella_protocol_editor import (
        AutoLamellaProtocolEditorWidget,
    )

    fib = FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6)
    fm = _fm_image()
    inp = CorrelationInputData(
        fib_coordinates=[
            Coordinate(point=PointXYZ(x=60, y=95, z=0), point_type=PointType.FIB)
        ],
        fm_coordinates=[
            Coordinate(point=PointXYZ(x=30, y=40, z=2), point_type=PointType.FM)
        ],
        poi_coordinates=[
            Coordinate(point=PointXYZ(x=60, y=55, z=2), point_type=PointType.POI)
        ],
    )
    history = LamellaCorrelation(runs=[_run(NEW, inp)])
    cfg = CorrelationConfig()
    cfg.fit.fm_poi_channel = "GFP"

    pv = AutoLamellaProtocolEditorWidget._build_correlation_preview(
        fib, fm, cfg, history, [Point(0.3, 0.5)]
    )
    assert pv.fib_thumb is not None and pv.fm_thumb is not None
    assert pv.fm_caption == "FM · GFP (POI channel) · 5 z"
    assert pv.spot_burns == [(0.3, 0.5)]
    # FIB (60,95)/(300,200); FM (30,40)/(120,120); POI (60,55)/(120,120)
    assert pv.prev_fib[0] == pytest.approx((0.2, 0.475))
    assert pv.prev_fm[0] == pytest.approx((0.25, 40 / 120))
    assert pv.prev_poi == pytest.approx((0.5, 55 / 120))


# ---------------------------------------------------------------------------
# interpolation passthrough — target computation
# ---------------------------------------------------------------------------


def test_start_fm_interpolation_target(qapp, monkeypatch):
    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    w = CorrelationTabWidget()
    w.set_fm_image(_fm_image(nz=5))
    calls = []
    monkeypatch.setattr(w, "_start_fm_interpolation", lambda t, m: calls.append((t, m)))

    w.start_fm_interpolation(isotropic=True, target_z_nm=None, method="linear")
    assert calls[-1] == pytest.approx((40e-9, "linear"))  # isotropic → XY pixel size

    w.start_fm_interpolation(isotropic=False, target_z_nm=250.0, method="cubic")
    assert calls[-1][0] == pytest.approx(250e-9)
    assert calls[-1][1] == "cubic"


def test_start_fm_interpolation_noops_on_single_slice(qapp, monkeypatch):
    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    w = CorrelationTabWidget()
    w.set_fm_image(_fm_image(nz=1))  # single slice → not interpolatable
    calls = []
    monkeypatch.setattr(w, "_start_fm_interpolation", lambda t, m: calls.append((t, m)))

    w.start_fm_interpolation(isotropic=True, target_z_nm=None, method="linear")
    assert calls == []
