"""Offscreen tests for the grid protocol editor (per-task config widgets + host)."""

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.applications.autolamella.structures import Experiment  # noqa: E402
from fibsem.applications.autolamella.workflows.tasks.grid_tasks import (  # noqa: E402
    GRID_TASK_REGISTRY,
    AcquireImageGridTaskConfig,
    AcquireOverviewImageGridTaskConfig,
    CryoCleaningGridTaskConfig,
)


@pytest.fixture(scope="module")
def qapp():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Qt unavailable: {e}")
    return app


# --- factory + per-task widgets ---

def test_factory_maps_task_types(qapp):
    from fibsem.ui.widgets.grid_task_config_widgets import (
        AcquireImageGridConfigWidget,
        OverviewGridConfigWidget,
        _PlaceholderConfigWidget,
        get_grid_config_widget,
    )

    ov = get_grid_config_widget(AcquireOverviewImageGridTaskConfig(task_name="x"))
    ai = get_grid_config_widget(AcquireImageGridTaskConfig(task_name="x"))
    cc = get_grid_config_widget(CryoCleaningGridTaskConfig(task_name="x"))
    assert isinstance(ov, OverviewGridConfigWidget)
    assert isinstance(ai, AcquireImageGridConfigWidget)
    assert isinstance(cc, _PlaceholderConfigWidget)  # not built yet


def test_acquire_image_widget_roundtrip_and_signal(qapp):
    from fibsem.ui.widgets.grid_task_config_widgets import AcquireImageGridConfigWidget

    cfg = AcquireImageGridTaskConfig(task_name="x", orientation="SEM", voltage=5000)
    w = AcquireImageGridConfigWidget(cfg)
    seen = []
    w.config_changed.connect(lambda: seen.append(True))

    w.voltage_combo.set_value(10000)
    w.orientation_combo.setCurrentText("FIB")
    w.current_combo.set_value(10e-9)

    out = w.get_config()
    assert out.voltage == 10000 and out.orientation == "FIB"
    assert out.beam_current == 10e-9
    assert out.image_settings is not None  # acquisition settings preserved
    assert seen  # edits emitted config_changed


def test_acquire_image_config_serialization_roundtrip(qapp):
    from fibsem.applications.autolamella.workflows.tasks.grid_tasks import (
        load_grid_task_config,
    )

    cfg = AcquireImageGridTaskConfig(
        task_name="ACQUIRE_IMAGE_GRID", voltage=8000, beam_current=3e-9
    )
    restored = load_grid_task_config(cfg.to_dict())
    assert restored.beam_current == 3e-9
    assert restored.voltage == 8000
    assert restored.image_settings.hfw == cfg.image_settings.hfw


def test_overview_widget_roundtrip(qapp):
    from fibsem.ui.widgets.grid_task_config_widgets import OverviewGridConfigWidget

    cfg = AcquireOverviewImageGridTaskConfig(task_name="x")
    w = OverviewGridConfigWidget(cfg)
    w.orientation_combo.setCurrentText("MILLING")
    out = w.get_config()
    assert out.orientation == "MILLING"
    assert out.settings is not None  # nested settings preserved


def test_set_config_does_not_emit(qapp):
    from fibsem.ui.widgets.grid_task_config_widgets import AcquireImageGridConfigWidget

    w = AcquireImageGridConfigWidget(AcquireImageGridTaskConfig(task_name="x"))
    seen = []
    w.config_changed.connect(lambda: seen.append(True))
    w.set_config(AcquireImageGridTaskConfig(task_name="x", voltage=9000))
    assert not seen  # programmatic load must not look like a user edit


# --- host ---

@pytest.fixture
def editor(qapp):
    from fibsem.ui.widgets.grid_protocol_editor_widget import GridProtocolEditorWidget

    exp = Experiment.create(path=tempfile.mkdtemp(), name="proto-test")
    w = GridProtocolEditorWidget()
    w.set_experiment(exp)
    return w, exp


def test_host_lists_all_tasks(editor):
    w, _ = editor
    assert w._list.count() == len(GRID_TASK_REGISTRY)


def test_host_persists_edits_to_protocol(editor):
    w, exp = editor
    # default selection is the first task (overview); edit it
    w._editor.orientation_combo.setCurrentText("FIB")
    key = w._task_type
    assert key in exp.grid_protocol.task_config
    assert exp.grid_protocol.task_config[key].orientation == "FIB"


def test_host_reuses_saved_config(editor):
    w, exp = editor
    # pre-seed a saved config for the acquire-image task (preset voltage)
    exp.grid_protocol.task_config["ACQUIRE_IMAGE_GRID"] = AcquireImageGridTaskConfig(
        task_name="ACQUIRE_IMAGE_GRID", voltage=10000
    )
    # select that task in the list
    for i in range(w._list.count()):
        if w._list.item(i).data(0x0100) == "ACQUIRE_IMAGE_GRID":  # Qt.UserRole
            w._list.setCurrentRow(i)
            break
    assert w._editor.get_config().voltage == 10000  # loaded the saved value
