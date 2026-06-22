"""Offscreen tests for the cryo GIS/sputter grid task config editor widgets."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.applications.autolamella.workflows.tasks.grid import (  # noqa: E402
    CryoDepositionGridTaskConfig,
    CryoSputterGridTaskConfig,
)
from fibsem.ui.widgets.grid_task_config_widgets import (  # noqa: E402
    CryoDepositionGridConfigWidget,
    CryoSputterGridConfigWidget,
    get_grid_config_widget,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_deposition_widget_loads_and_roundtrips(qapp):
    cfg = CryoDepositionGridTaskConfig(task_name="gis", orientation="FIB",
                                       deposition_time=45.0)
    w = get_grid_config_widget(cfg)
    assert isinstance(w, CryoDepositionGridConfigWidget)  # registered, not placeholder
    # loads from config
    assert w.orientation_combo.currentText() == "FIB"
    assert w.time_spin.value() == 45.0
    # edits write back
    w.orientation_combo.setCurrentText("SEM")
    w.time_spin.setValue(20.0)
    out = w.get_config()
    assert out.orientation == "SEM"
    assert out.deposition_time == 20.0


def test_sputter_widget_loads_and_roundtrips(qapp):
    cfg = CryoSputterGridTaskConfig(task_name="sp", orientation="SEM",
                                    sputter_time=30.0, sputter_current=0.01)
    w = get_grid_config_widget(cfg)
    assert isinstance(w, CryoSputterGridConfigWidget)
    assert w.time_spin.value() == 30.0
    assert w.current_combo.value() == 0.01  # 10 mA preset matched exactly
    # edits write back
    w.time_spin.setValue(90.0)
    w.current_combo.set_value(0.02)
    out = w.get_config()
    assert out.sputter_time == 90.0
    assert out.sputter_current == 0.02


def test_config_changed_emitted_on_edit(qapp):
    w = get_grid_config_widget(CryoSputterGridTaskConfig(task_name="sp"))
    seen = []
    w.config_changed.connect(lambda: seen.append(True))
    w.time_spin.setValue(123.0)
    assert seen  # editing a control emits config_changed
