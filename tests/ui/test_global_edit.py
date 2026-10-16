"""The protocol editor's Global Edit sets reference imaging and the milling
field of view, and with "Also update existing lamella configurations" sets
those two on each lamella too -- and nothing else (FIB-1071).

It used to copy the protocol's whole task config onto every lamella, keeping
only the pattern positions, so a lamella's own depths, currents and parameters
went back to the protocol's.

The real main window and protocol editor on Demo, with the real dialog; only
its ``exec_`` is answered.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict  # noqa: E402

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui import (  # noqa: E402
    autolamella_task_config_editor as protocol_editor_module,
)
from fibsem.applications.autolamella.workflows.tasks.polishing import (  # noqa: E402
    MillPolishingTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.rough import (  # noqa: E402
    MillRoughTaskConfig,
)
from fibsem.structures import MicroscopeState, Point  # noqa: E402
from fibsem.workflows.status import TaskStatus  # noqa: E402

ROUGH = "Rough Milling"
POLISH = "Polishing"
KEY = "mill_rough"


@pytest.fixture(scope="module")
def window(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    win = module.AutoLamellaSingleWindowUI()
    win.autolamella_ui.system_widget.connect_to_microscope()
    yield win
    if win.autolamella_ui.microscope is not None:
        win.autolamella_ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        win.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture
def experiment(window, tmp_path):
    """Two lamellae; the first tuned on its own, the second without Polishing."""
    exp = Experiment(path=tmp_path, name="global-edit")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.task_protocol.task_config[ROUGH] = MillRoughTaskConfig(task_name=ROUGH)
    exp.task_protocol.task_config[POLISH] = MillPolishingTaskConfig(task_name=POLISH)
    for _ in range(2):
        exp.add_new_lamella(MicroscopeState(), EventedDict())
    tuned, other = exp.positions
    tuned.task_config[ROUGH] = MillRoughTaskConfig(task_name=ROUGH)
    tuned.task_config[POLISH] = MillPolishingTaskConfig(task_name=POLISH)
    other.task_config[ROUGH] = MillRoughTaskConfig(task_name=ROUGH)
    stage = tuned.task_config[ROUGH].milling[KEY].stages[0]
    stage.pattern.depth = 2.5e-6
    stage.pattern.width = 8e-6
    stage.pattern.point = Point(1e-6, -2e-6)
    stage.milling.milling_current = 1e-9
    tuned.task_config[ROUGH].sync_to_poi = False
    window.autolamella_ui.experiment = exp
    return exp


def _global_edit(window, experiment, monkeypatch, update_lamellae=True, said=None):
    """Global Edit of every task: 10 µm wider, and a wider first reference image.
    The message it ends with is appended to *said*."""
    editor = window.task_widget
    editor.set_experiment(experiment)

    def accept(dialog):
        dialog._select_all_tasks()
        dialog.spinbox_milling_fov.setValue(dialog.spinbox_milling_fov.value() + 10)
        settings = dialog.ref_image_params_widget.get_settings()
        settings.field_of_view1 = 200e-6
        dialog.ref_image_params_widget.update_from_settings(settings)
        dialog.checkbox_update_existing.setChecked(update_lamellae)
        return protocol_editor_module.QDialog.Accepted

    monkeypatch.setattr(
        protocol_editor_module.AutoLamellaGlobalTaskEditDialog, "exec_", accept
    )
    said = [] if said is None else said
    monkeypatch.setattr(
        protocol_editor_module.QMessageBox,
        "information",
        lambda *args, **kwargs: said.append(args[2]),
    )
    editor._on_global_edit_clicked()


def test_each_lamella_gets_what_the_dialog_sets_and_keeps_its_own_tuning(
    window, experiment, monkeypatch
):
    tuned = experiment.positions[0]
    fov = tuned.task_config[ROUGH].milling[KEY].field_of_view

    _global_edit(window, experiment, monkeypatch)

    config = tuned.task_config[ROUGH]
    milling = config.milling[KEY]
    stage = milling.stages[0]
    # what the dialog sets
    assert milling.field_of_view == pytest.approx(fov + 10e-6)
    assert config.reference_imaging.field_of_view1 == pytest.approx(200e-6)
    # what the lamella was tuned to
    assert stage.pattern.depth == 2.5e-6
    assert stage.pattern.width == 8e-6
    assert (stage.pattern.point.x, stage.pattern.point.y) == (1e-6, -2e-6)
    assert stage.milling.milling_current == 1e-9
    assert config.sync_to_poi is False
    # and the protocol has the new settings too
    protocol = experiment.task_protocol.task_config[ROUGH]
    assert protocol.milling[KEY].field_of_view == pytest.approx(fov + 10e-6)
    assert protocol.reference_imaging.field_of_view1 == pytest.approx(200e-6)


def test_a_lamella_without_a_task_is_not_given_it(window, experiment, monkeypatch):
    _global_edit(window, experiment, monkeypatch)

    assert POLISH not in experiment.positions[1].task_config


def test_unticked_the_lamellae_are_left_as_they_were(window, experiment, monkeypatch):
    tuned = experiment.positions[0]
    before = tuned.task_config[ROUGH].to_dict()

    _global_edit(window, experiment, monkeypatch, update_lamellae=False)

    assert tuned.task_config[ROUGH].to_dict() == before
    protocol = experiment.task_protocol.task_config[ROUGH]
    assert protocol.reference_imaging.field_of_view1 == pytest.approx(200e-6)


def test_the_task_running_on_a_lamella_is_left_alone(window, experiment, monkeypatch):
    """A running task holds its lamella's config object, so an edit would change
    its settings mid-run: new reference imaging with no save path failed the
    task's next save on the Arctis (2026-10-12)."""
    running, other = experiment.positions
    running.task_state.name = ROUGH
    running.task_state.status = TaskStatus.InProgress
    before = running.task_config[ROUGH].to_dict()
    said = []

    _global_edit(window, experiment, monkeypatch, said=said)

    assert running.task_config[ROUGH].to_dict() == before
    # its other tasks, and the other lamella, still get the edit
    for config in (running.task_config[POLISH], other.task_config[ROUGH]):
        assert config.reference_imaging.field_of_view1 == pytest.approx(200e-6)
    assert f"{running.name} ({ROUGH})" in said[-1]


def test_a_lamella_keeps_its_save_path(window, experiment, monkeypatch):
    tuned = experiment.positions[0]
    tuned.task_config[ROUGH].imaging.path = str(tuned.path)

    _global_edit(window, experiment, monkeypatch)

    assert tuned.task_config[ROUGH].imaging.path == str(tuned.path)
