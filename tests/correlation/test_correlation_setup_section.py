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
from fibsem.correlation.ui.widgets.correlation_setup_section import (
    SEED_NONE,
    SEED_PREVIOUS,
    SEED_SPOT_BURNS,
    CorrelationSetupSection,
    format_run_timestamp,
)
from fibsem.structures import FibsemImage, Point


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    """Never hit the network for the zeta LUT during widget construction."""
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
    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    return CorrelationTabWidget()


def test_add_lamella_setup_installs_and_renames_tab(qapp):
    w = _widget()
    w.add_lamella_setup(
        spot_burns=[Point(0.3, 0.5)],
        history=LamellaCorrelation(runs=[_run(NEW)]),
        config=CorrelationConfig(),
        fib_options=["a_ib.tif", "b_ib.tif"],
        fib_current="a_ib.tif",
        fm_options=["a.ome.tiff"],
        fm_current="a.ome.tiff",
    )
    assert w._tabs.tabText(0) == "Setup"
    assert w.fib_quick_pick.currentText() == "a_ib.tif"
    assert w.fm_quick_pick.currentText() == "a.ome.tiff"


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

    import fibsem.correlation.ui.widgets.correlation_tab_widget as ctw

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
