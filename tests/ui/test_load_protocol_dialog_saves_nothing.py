"""The Load Protocol dialog's OK hands the chosen protocol back and writes
nothing. It used to save ``experiment.task_protocol`` -- still the old protocol
at that point -- over protocol.yaml, just before the window adopted and saved
the new one. ``experiment_setup.adopt_protocol`` is the one save."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QDialog

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.protocol.load import (
    AutoLamellaLoadTaskProtocolWidget,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_ok_returns_the_new_protocol_and_writes_nothing(qapp, tmp_path, monkeypatch):
    experiment = Experiment(path=tmp_path, name="load")
    (tmp_path / "load").mkdir()
    experiment.task_protocol = AutoLamellaTaskProtocol(name="old")
    experiment.save_protocol()
    saved = tmp_path / "load" / "protocol.yaml"
    before = saved.read_text(encoding="utf-8")

    writes = []
    monkeypatch.setattr(
        AutoLamellaTaskProtocol, "save", lambda self, path: writes.append(path)
    )

    dialog = AutoLamellaLoadTaskProtocolWidget(experiment=experiment)
    chosen = AutoLamellaTaskProtocol(name="new")
    dialog.protocol = chosen
    dialog._protocol_changed = True
    dialog._on_ok_clicked()

    assert dialog.result() == QDialog.Accepted
    assert dialog.get_protocol() is chosen
    assert writes == []
    assert saved.read_text(encoding="utf-8") == before
    assert experiment.task_protocol.name == "old"
