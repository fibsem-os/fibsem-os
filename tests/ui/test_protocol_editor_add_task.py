"""A task added from the Protocol tab reaches every lamella, saving its milling
images to that lamella's own folder."""

import copy
import os
import pathlib
import tempfile

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict  # noqa: E402
from PyQt5.QtWidgets import QApplication, QDialog, QWidget  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui import (  # noqa: E402
    autolamella_task_config_editor as editor_module,
)
from fibsem.structures import MicroscopeState  # noqa: E402


@pytest.fixture
def qapp():
    yield QApplication.instance() or QApplication([])


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
    host.experiment, host.microscope = exp, microscope
    widget = editor_module.AutoLamellaProtocolTaskConfigEditor(parent=host)
    yield widget
    host.close()


def test_an_added_task_saves_its_milling_images_to_each_lamella(editor, monkeypatch):
    class _Dialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec_(self):
            return QDialog.Accepted

        def get_task_info(self):
            return "MILL_UNDERCUT", "Undercut"

    monkeypatch.setattr(editor_module, "AddTaskDialog", _Dialog)

    editor._on_add_task_clicked()

    for lamella in editor.experiment.positions:
        milling = lamella.task_config["Undercut"].milling.values()
        paths = [m.acquisition.imaging.path for m in milling]
        assert paths, "the undercut carries a milling config"
        assert all(pathlib.Path(p) == pathlib.Path(lamella.path) for p in paths)
