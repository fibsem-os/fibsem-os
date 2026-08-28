"""The experiment panel says each thing once, and the status bar owns the guidance.

The panel used to carry the experiment name and the protocol name in two read-only
line edits -- boxes that look like inputs and refuse to be typed in -- above a
label repeating whichever rung of the same guidance ladder the window's status bar
was already showing. The name now lives in the tab-corner button (see
`test_experiment_header_button.py`) and the guidance in the status bar.

What the label is actually for survives, and is pinned here: the question a running
workflow asks, which the Yes/No buttons underneath it answer.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QMainWindow  # noqa: E402

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (  # noqa: E402
    AutoLamellaSingleWindowUI,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import (  # noqa: E402
    INSTRUCTIONS,
    AutoLamellaUI,
)


@pytest.fixture
def panel(qapp):
    """The real panel. It constructs offscreen; the window that embeds it does not.

    Shown on purpose: a child of a hidden window reports `isVisible() == False`
    whatever its own flag says, so every "the label is gone" assertion here would
    pass without the code doing anything.
    """
    ui = AutoLamellaUI(parent_ui=None)
    ui.show()
    yield ui
    ui.deleteLater()


def test_idle_says_nothing_in_the_panel(panel, tmp_path):
    experiment = Experiment(path=str(tmp_path), name="AutoLamella-2026-08-21-11-04")
    experiment.positions.append(
        Lamella(petname="01-test", path=str(tmp_path / "01-test"), number=1)
    )
    panel.experiment = experiment

    panel.update_ui()

    # Not merely empty -- gone. An empty label still reserves its line.
    assert not panel.label_instructions.isVisible()
    assert panel.label_instructions.text() == ""


def test_a_workflow_prompt_still_shows_with_its_buttons(panel):
    panel.set_instructions_msg("Move to the milling position?", "Yes", "No")

    assert panel.label_instructions.text() == "Move to the milling position?"
    assert panel.label_instructions.isVisible()
    assert panel.pushButton_yes.isVisible()
    assert panel.pushButton_no.isVisible()

    # And clears again when the workflow stops asking.
    panel.set_instructions_msg("")

    assert not panel.label_instructions.isVisible()
    assert not panel.pushButton_yes.isVisible()


def test_the_name_boxes_are_gone(panel):
    """They were read-only QLineEdits: they read as inputs and were not.

    Pinned because the obvious way to restore a name to the panel is to put the
    box back, and the tab-corner button already carries it.
    """
    assert not hasattr(panel, "lineEdit_experiment_name")
    assert not hasattr(panel, "lineEdit_protocol_name")


class _UIStub:
    """The two attributes `_update_instructions` reads, plus the protocol."""

    def __init__(self, microscope=None, experiment=None, protocol=None):
        self.microscope = microscope
        self.experiment = experiment
        self.protocol = protocol


class _StatusHost(QMainWindow):
    _update_instructions = AutoLamellaSingleWindowUI._update_instructions

    def __init__(self):
        super().__init__()
        self.status_bar = self.statusBar()
        self.autolamella_ui = _UIStub()


@pytest.fixture
def host(qapp):
    host = _StatusHost()
    yield host
    host.deleteLater()


def test_status_bar_walks_the_whole_ladder(host, tmp_path):
    """Every rung, including the protocol -- the one only the panel used to have.

    Dropping the panel's copy would otherwise have deleted that state outright:
    an experiment with no protocol can run nothing, and the status bar would have
    said "add a lamella".
    """
    assert host.autolamella_ui.microscope is None
    host._update_instructions()
    assert host.status_bar.currentMessage() == INSTRUCTIONS["NOT_CONNECTED"]

    host.autolamella_ui.microscope = object()
    host._update_instructions()
    assert host.status_bar.currentMessage() == INSTRUCTIONS["NO_EXPERIMENT"]

    experiment = Experiment(path=str(tmp_path), name="ladder")
    host.autolamella_ui.experiment = experiment
    host._update_instructions()
    assert host.status_bar.currentMessage() == INSTRUCTIONS["NO_PROTOCOL"]

    host.autolamella_ui.protocol = AutoLamellaTaskProtocol()
    host._update_instructions()
    assert host.status_bar.currentMessage() == INSTRUCTIONS["NO_LAMELLA"]

    experiment.positions.append(
        Lamella(petname="01-test", path=str(tmp_path / "01-test"), number=1)
    )
    host._update_instructions()
    assert host.status_bar.currentMessage() == INSTRUCTIONS["AUTOLAMELLA_READY"]
