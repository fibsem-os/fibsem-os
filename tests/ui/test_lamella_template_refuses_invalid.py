"""The Lamella Template dialog's OK refuses a template whose alignment area
lies outside the image, as the agent's set_lamella_defaults does: both ask
``LamellaDefaultConfig.why_invalid``. Before, OK saved whatever was entered."""

import copy
import os
import pathlib
import tempfile

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict  # noqa: E402
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton, QWidget  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
    LamellaDefaultConfig,
)
from fibsem.applications.autolamella.ui.protocol import (
    protocol_tab as editor_module,  # noqa: E402
)
from fibsem.structures import FibsemRectangle, MicroscopeState  # noqa: E402
from tests.fixtures.app_session import app_session  # noqa: E402

_OUTSIDE = FibsemRectangle(left=0.8, top=0.0, width=0.5, height=0.5)
_INSIDE = FibsemRectangle(left=0.1, top=0.1, width=0.5, height=0.5)


def test_why_invalid_names_an_area_outside_the_image():
    assert LamellaDefaultConfig().why_invalid() == ""
    assert LamellaDefaultConfig(alignment_area=_INSIDE).why_invalid() == ""
    assert (
        "inside the image"
        in LamellaDefaultConfig(alignment_area=_OUTSIDE).why_invalid()
    )


@pytest.fixture
def editor():
    QApplication.instance() or QApplication([])
    microscope, _ = utils.setup_session(manufacturer="Demo")
    exp = Experiment(path=pathlib.Path(tempfile.mkdtemp()), name="template")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
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


@pytest.mark.parametrize("area, accepted", [(_OUTSIDE, False), (_INSIDE, True)])
def test_ok_keeps_the_dialog_open_on_an_invalid_template(
    editor, monkeypatch, area, accepted
):
    class _Entered(editor_module.LamellaDefaultConfigWidget):
        def get_template(self):
            return LamellaDefaultConfig(alignment_area=area)

    def press_ok(dialog):
        ok = [b for b in dialog.findChildren(QPushButton) if b.text() == "OK"]
        ok[0].click()
        return dialog.result()

    warnings = []
    monkeypatch.setattr(editor_module, "LamellaDefaultConfigWidget", _Entered)
    monkeypatch.setattr(editor_module.QDialog, "exec_", press_ok)
    monkeypatch.setattr(
        editor_module.QMessageBox, "warning", lambda *a: warnings.append(a[2])
    )
    saves = []
    monkeypatch.setattr(editor, "_save_experiment", lambda: saves.append(1))
    before = editor.experiment.task_protocol.lamella_defaults

    editor._on_lamella_defaults_clicked()

    defaults = editor.experiment.task_protocol.lamella_defaults
    if accepted:
        assert not warnings and saves == [1]
        assert defaults.alignment_area == area
    else:
        assert "inside the image" in warnings[0]
        assert saves == [] and defaults is before
