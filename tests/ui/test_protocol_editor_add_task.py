"""A task added from the Protocol tab reaches every lamella, saving its milling
images to that lamella's own folder."""

import copy
import os
import pathlib
import tempfile

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict  # noqa: E402
from PyQt5.QtWidgets import QDialog, QWidget  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.protocol import (
    protocol_tab as editor_module,  # noqa: E402
)
from fibsem.structures import MicroscopeState  # noqa: E402
from tests.fixtures.app_session import app_session  # noqa: E402


@pytest.fixture
def editor(qapp):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    exp = Experiment(path=pathlib.Path(tempfile.mkdtemp()), name="add-task")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
    for _ in range(2):
        exp.add_new_lamella(
            MicroscopeState(),
            EventedDict(copy.deepcopy(dict(exp.task_protocol.task_config))),
        )
    host = QWidget()
    widget = editor_module.AutoLamellaProtocolTaskConfigEditor(
        session=app_session(microscope, exp), parent=host
    )
    yield widget
    host.close()


def test_an_added_task_saves_its_milling_images_to_each_lamella(editor, monkeypatch):
    class _Dialog:
        chosen = ("MILL_UNDERCUT", "Undercut")

        def __init__(self, *args, **kwargs):
            pass

        def exec_(self):
            return QDialog.Accepted

    monkeypatch.setattr(editor_module, "AddTaskDialog", _Dialog)

    editor._on_add_task_clicked()

    for lamella in editor.experiment.positions:
        milling = lamella.task_config["Undercut"].milling.values()
        paths = [m.acquisition.imaging.path for m in milling]
        assert paths, "the undercut carries a milling config"
        assert all(pathlib.Path(p) == pathlib.Path(lamella.path) for p in paths)


def test_the_add_dialog_numbers_a_taken_name(editor, monkeypatch):
    """The dialog offers the type's name, numbered past the protocol's own."""
    from fibsem.applications.autolamella.workflows.tasks.rough import (
        MillRoughTaskConfig,
    )

    configs = editor.experiment.task_protocol.task_config
    configs["Rough Milling"] = MillRoughTaskConfig(task_name="Rough Milling")
    shown = []
    monkeypatch.setattr(
        editor_module.AddTaskDialog, "exec_", lambda d: shown.append(d) or 0
    )
    editor._on_add_task_clicked()

    (dialog,) = shown
    dialog.type_combo.setCurrentIndex(dialog.type_combo.findData("MILL_ROUGH"))
    assert dialog.name_edit.text() == "Rough Milling 2"


def test_adding_and_removing_a_task_is_recorded_as_the_app_server_s(
    editor, monkeypatch
):
    """The tab's add and remove are workflow_edits', so they are saved and on
    the record as the agent's are."""
    from fibsem.workflows.workflow_edits import WORKFLOW_CHANGED

    class _Dialog:
        chosen = ("MILL_UNDERCUT", "Undercut")

        def __init__(self, *args, **kwargs):
            pass

        def exec_(self):
            return QDialog.Accepted

    recorded = []
    monkeypatch.setattr(editor_module, "AddTaskDialog", _Dialog)
    monkeypatch.setattr(
        editor.session,
        "record_event",
        lambda kind, payload: recorded.append((kind, payload)),
    )
    monkeypatch.setattr(
        editor_module.QMessageBox,
        "question",
        lambda *a, **k: editor_module.QMessageBox.Yes,
    )

    editor._on_add_task_clicked()
    editor.task_list_widget.select("Undercut")
    editor._on_remove_task_clicked()

    assert recorded == [
        (WORKFLOW_CHANGED, {"field": "tasks", "added": "Undercut"}),
        (WORKFLOW_CHANGED, {"field": "tasks", "removed": "Undercut"}),
    ]
    saved = Experiment.load(pathlib.Path(editor.experiment.path) / "experiment.yaml")
    assert "Undercut" not in saved.task_protocol.task_config
