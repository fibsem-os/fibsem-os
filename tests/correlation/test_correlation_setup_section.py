"""Tests for the lamella setup section in the correlation window (FIB-302).

The section is a chooser: it emits ``seed_requested`` and the widget applies it to
the live canvas. Covers the default source precedence, enable rules, run
selection, the payload each source yields, and the widget-side seeding + the
manual-edit guard. Headless PyQt5, offscreen.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
from fibsem.structures import FibsemImage, Point
from fibsem.ui.correlation.widgets.correlation_setup_section import (
    SEED_NONE,
    SEED_PREVIOUS,
    SEED_SPOT_BURNS,
    CorrelationSetupSection,
    format_run_timestamp,
)


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    """Never hit the network for the zeta LUT during widget construction."""
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw

    monkeypatch.setattr(riw, "_ensure_lut", lambda: None)


OLD = "2026-07-20_09-00-00"
NEW = "2026-07-24_14-32-00"


def _run(name, inp=None):
    return CorrelationRun(
        path="/tmp/" + name,
        name=name,
        state=CorrelationState(input_data=inp or CorrelationInputData()),
    )


def _section(*, spot_burns=None, runs=None, config=None):
    return CorrelationSetupSection(
        spot_burns=spot_burns,
        history=LamellaCorrelation(runs=runs or []),
        config=config or CorrelationConfig(),
    )


def _emissions(section):
    """Collect (source, payload) pairs emitted by the section."""
    seen = []
    section.seed_requested.connect(lambda s, p: seen.append((s, p)))
    return seen


# ---------------------------------------------------------------------------
# default source precedence + enable rules
# ---------------------------------------------------------------------------


def test_default_is_previous_when_history_exists(qapp):
    s = _section(spot_burns=[Point(0.1, 0.2)], runs=[_run(OLD), _run(NEW)])
    assert s.seed_source() == SEED_PREVIOUS
    assert s.selected_run().name == NEW  # newest first


def test_default_is_spot_burns_without_history(qapp):
    s = _section(spot_burns=[Point(0.1, 0.2)])
    assert s.seed_source() == SEED_SPOT_BURNS


def test_default_is_none_without_context(qapp):
    s = _section()
    assert s.seed_source() == SEED_NONE
    assert not s.rb_burns.isEnabled()
    assert not s.rb_prev.isEnabled()


def test_run_combo_enabled_only_for_previous(qapp):
    s = _section(spot_burns=[Point(0.1, 0.2)], runs=[_run(NEW)])
    assert s.run_combo.isEnabled()  # previous is the default
    s.rb_burns.setChecked(True)
    assert not s.run_combo.isEnabled()


# ---------------------------------------------------------------------------
# emissions drive the seeding
# ---------------------------------------------------------------------------


def test_switching_source_emits_seed_request(qapp):
    s = _section(spot_burns=[Point(0.3, 0.4)], runs=[_run(NEW)])
    seen = _emissions(s)

    s.rb_burns.setChecked(True)
    assert seen[-1][0] == SEED_SPOT_BURNS
    assert seen[-1][1] == [Point(0.3, 0.4)]

    s.rb_none.setChecked(True)
    assert seen[-1] == (SEED_NONE, None)


def test_changing_run_re_emits_that_runs_data(qapp):
    inp_old = CorrelationInputData(
        fib_coordinates=[Coordinate(point=PointXYZ(x=1.0), point_type=PointType.FIB)]
    )
    inp_new = CorrelationInputData(
        fib_coordinates=[Coordinate(point=PointXYZ(x=9.0), point_type=PointType.FIB)]
    )
    s = _section(runs=[_run(OLD, inp_old), _run(NEW, inp_new)])
    seen = _emissions(s)

    s.run_combo.setCurrentIndex(1)  # newest-first display → index 1 is the older run
    source, payload = seen[-1]
    assert source == SEED_PREVIOUS
    assert payload.fib_coordinates[0].point.x == 1.0


def test_emit_current_seed_applies_the_default(qapp):
    s = _section(spot_burns=[Point(0.5, 0.5)])
    seen = _emissions(s)
    s.emit_current_seed()
    assert seen == [(SEED_SPOT_BURNS, [Point(0.5, 0.5)])]


def test_run_timestamp_formatting_falls_back():
    assert format_run_timestamp("2026-07-24_14-30-05") == "2026-07-24 14:30:05"
    assert format_run_timestamp("not-a-timestamp") == "not-a-timestamp"


# ---------------------------------------------------------------------------
# widget side: install + seeding + the manual-edit guard
# ---------------------------------------------------------------------------


def _widget():
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    return CorrelationTabWidget()


def test_add_lamella_setup_installs_and_renames_tab(qapp):
    w = _widget()
    w.add_lamella_setup(
        spot_burns=[Point(0.3, 0.5)],
        history=LamellaCorrelation(runs=[_run(NEW)]),
        config=CorrelationConfig(),
        fib_options=["/lam/a_ib.tif", "/lam/b_ib.tif"],
        fib_current="/lam/a_ib.tif",
        fm_options=["/lam/a.ome.tiff"],
        fm_current="/lam/a.ome.tiff",
    )
    assert w._tabs.tabText(0) == "Setup"
    picker = w._images_tab._fib_picker
    # combo shows basenames but carries full paths, and browse lives on the row
    assert picker.combo.currentText() == "a_ib.tif"
    assert picker.current_path() == "/lam/a_ib.tif"
    assert [picker.combo.itemText(i) for i in range(picker.combo.count())] == [
        "a_ib.tif",
        "b_ib.tif",
    ]
    assert w._images_tab._fm_picker.current_path() == "/lam/a.ome.tiff"


def test_seed_request_places_spot_burns_on_the_canvas(qapp):
    w = _widget()
    w.set_fib_image(FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6))
    section = w.add_lamella_setup(
        spot_burns=[Point(0.25, 0.5)], config=CorrelationConfig()
    )
    section.emit_current_seed()

    fib = w.data.fib_coordinates
    assert len(fib) == 1
    assert fib[0].point.x == pytest.approx(0.25 * 300)
    assert fib[0].point.y == pytest.approx(0.5 * 200)


def test_seed_request_none_clears_coordinates(qapp):
    w = _widget()
    w.set_fib_image(FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6))
    section = w.add_lamella_setup(
        spot_burns=[Point(0.25, 0.5)], config=CorrelationConfig()
    )
    section.emit_current_seed()
    assert w.data.fib_coordinates

    section.rb_none.setChecked(True)
    assert w.data.fib_coordinates == []


def test_manual_edits_prompt_before_reseeding(qapp, monkeypatch):
    """A hand-moved point must not be silently replaced by a re-seed."""
    from PyQt5.QtWidgets import QMessageBox

    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    w = _widget()
    w.set_fib_image(FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6))
    section = w.add_lamella_setup(
        spot_burns=[Point(0.25, 0.5)], config=CorrelationConfig()
    )
    section.emit_current_seed()

    # user drags the seeded point
    w.data.fib_coordinates[0].point.x = 111.0
    assert w._has_manual_edits()

    monkeypatch.setattr(
        ctw.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Cancel)
    )
    section.rb_none.setChecked(True)  # would clear — but the user cancels
    assert w.data.fib_coordinates[0].point.x == 111.0


# ---------------------------------------------------------------------------
# FM panel: voxel size + the Interpolate action
# ---------------------------------------------------------------------------


def _fm_image(nz=11, pixel_size_z=200e-9):
    from datetime import datetime

    import numpy as np

    from fibsem.fm.structures import (
        FluorescenceChannelMetadata,
        FluorescenceImage,
        FluorescenceImageMetadata,
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
        pixel_size_z=pixel_size_z,
        resolution=(64, 64),
        channels=[ch],
    )
    return FluorescenceImage(data=np.zeros((1, nz, 64, 64), np.uint16), metadata=meta)


def test_fm_pixel_size_reports_xy_z_and_anisotropy(qapp):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        _format_fm_pixel_size,
    )

    text = _format_fm_pixel_size(_fm_image().metadata)
    assert "40.0 nm xy" in text and "200.0 nm z" in text
    assert "5.0× anisotropic" in text  # 200/40 — the ratio that decides interpolation

    # an isotropic stack shouldn't be labelled anisotropic
    assert "anisotropic" not in _format_fm_pixel_size(
        _fm_image(pixel_size_z=40e-9).metadata
    )


def test_fm_pixel_size_handles_missing_metadata(qapp):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        _format_fm_pixel_size,
    )

    class _Bare:
        pass

    assert _format_fm_pixel_size(_Bare()) == "—"


def test_loading_fm_through_the_picker_enables_interpolate(qapp, monkeypatch):
    """The picker's load path used to skip the enable, leaving Interpolate greyed
    out for any volume opened through it."""
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    tab = _widget()._images_tab
    monkeypatch.setattr(
        ctw.FluorescenceImage, "load", staticmethod(lambda p: _fm_image())
    )

    tab._load_fm("/lam/some.ome.tiff")
    assert tab._btn_interpolate.isEnabled()
    assert "200.0 nm z" in tab._lbl_fm_px.text()
