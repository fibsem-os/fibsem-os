"""Offscreen tests for the grid protocol editor (per-task config widgets + host)."""

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.applications.autolamella.structures import Experiment  # noqa: E402
from fibsem.applications.autolamella.workflows.tasks.grid_tasks import (  # noqa: E402
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


# --- add dialog ---

def test_add_dialog_validates_duplicate_name(qapp):
    from fibsem.ui.widgets.grid_protocol_editor_widget import AddGridTaskDialog

    d = AddGridTaskDialog(existing_names={"Acquire Image": object()})
    # default name auto-uniques against existing
    assert d.lineEdit_task_name.text() != "Acquire Image"
    # an explicit duplicate is rejected; a fresh name is accepted
    d.lineEdit_task_name.setText("Acquire Image")
    assert d.validate_task_name() is False
    d.lineEdit_task_name.setText("Acquire Image (lo-mag)")
    assert d.validate_task_name() is True


# --- host ---

@pytest.fixture
def editor(qapp):
    from fibsem.ui.widgets.grid_protocol_editor_widget import GridProtocolEditorWidget

    exp = Experiment.create(path=tempfile.mkdtemp(), name="proto-test")
    w = GridProtocolEditorWidget()
    w.set_experiment(exp)
    return w, exp


def _count(w):
    return w._task_list._list.count()


def test_host_starts_empty(editor):
    w, _ = editor
    # instance-driven: nothing configured until the user adds a task
    assert _count(w) == 0


def test_host_add_persists_instance(editor):
    w, exp = editor
    name = w.add_task("ACQUIRE_OVERVIEW_IMAGE_GRID")
    assert name in exp.grid_protocol.task_config
    w._editor.orientation_combo.setCurrentText("FIB")
    assert exp.grid_protocol.task_config[name].orientation == "FIB"


def test_host_add_generates_unique_names(editor):
    w, _ = editor
    # adding the same type twice without an explicit name must not collide
    n1 = w.add_task("ACQUIRE_IMAGE_GRID")
    n2 = w.add_task("ACQUIRE_IMAGE_GRID")
    assert n1 != n2 and _count(w) == 2


def test_host_add_explicit_collision_does_not_overwrite(editor):
    w, exp = editor
    n1 = w.add_task("ACQUIRE_IMAGE_GRID", "Imaging")
    w._editor.voltage_combo.set_value(20000)
    # an explicit colliding name is uniquified, not overwritten
    n2 = w.add_task("ACQUIRE_OVERVIEW_IMAGE_GRID", "Imaging")
    assert n2 != n1 and _count(w) == 2
    assert exp.grid_protocol.task_config[n1].voltage == 20000  # original intact
    assert exp.grid_protocol.task_config[n1].task_type == "ACQUIRE_IMAGE_GRID"


def test_host_multiple_instances_same_type(editor):
    w, exp = editor
    # two Acquire Image tasks, different voltages — the whole point of the change
    n1 = w.add_task("ACQUIRE_IMAGE_GRID", "Acquire Image — lo-mag")
    w._editor.voltage_combo.set_value(5000)
    n2 = w.add_task("ACQUIRE_IMAGE_GRID", "Acquire Image — hi-mag")
    w._editor.voltage_combo.set_value(30000)
    assert n1 != n2
    assert _count(w) == 2
    cfg = exp.grid_protocol.task_config
    assert cfg[n1].voltage == 5000 and cfg[n2].voltage == 30000
    assert cfg[n1].task_type == cfg[n2].task_type == "ACQUIRE_IMAGE_GRID"


def test_host_duplicate_clones_with_chosen_name(editor):
    w, exp = editor
    n1 = w.add_task("ACQUIRE_IMAGE_GRID")
    w._editor.voltage_combo.set_value(15000)
    n2 = w.duplicate_task("Acquire Image — copy")  # user-chosen name
    assert n2 == "Acquire Image — copy" and _count(w) == 2
    # clone carries the source's edited value, independently
    assert exp.grid_protocol.task_config[n2].voltage == 15000
    w._editor.voltage_combo.set_value(2000)
    assert exp.grid_protocol.task_config[n1].voltage == 15000  # original untouched


def test_duplicate_dialog_locks_type_and_prefills_name(qapp):
    from fibsem.ui.widgets.grid_protocol_editor_widget import AddGridTaskDialog

    d = AddGridTaskDialog(
        existing_names={"Acquire Image": object()},
        task_type="ACQUIRE_IMAGE_GRID",
        lock_task_type=True,
        default_name="Acquire Image (2)",
    )
    assert d.comboBox_task_type.isEnabled() is False  # type locked to source
    assert d.comboBox_task_type.currentData() == "ACQUIRE_IMAGE_GRID"
    assert d.lineEdit_task_name.text() == "Acquire Image (2)"  # editable suggestion
    task_type, name = d.get_task_info()
    assert task_type == "ACQUIRE_IMAGE_GRID" and name == "Acquire Image (2)"


def test_host_remove_instance(editor):
    w, exp = editor
    w.add_task("ACQUIRE_IMAGE_GRID")
    name = w.add_task("CRYO_CLEANING_GRID")
    w.remove_task(name)
    assert name not in exp.grid_protocol.task_config
    assert _count(w) == 1


