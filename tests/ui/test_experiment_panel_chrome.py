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
    INSTRUCTIONS,
    AutoLamellaSingleWindowUI,
)
from fibsem.ui.widgets.status_bar import FibsemStatusBar  # noqa: E402
from tests.fixtures.main_window import close_window  # noqa: E402


@pytest.fixture
def window(qapp):
    """The real window.

    Shown on purpose: a child of a hidden window reports `isVisible() == False`
    whatever its own flag says, so every "the label is gone" assertion here would
    pass without the code doing anything.
    """
    ui = AutoLamellaSingleWindowUI()
    ui.show()
    yield ui
    close_window(ui)


def test_idle_says_nothing_in_the_panel(window, tmp_path):
    experiment = Experiment(path=str(tmp_path), name="AutoLamella-2026-08-21-11-04")
    experiment.positions.append(
        Lamella(petname="01-test", path=str(tmp_path / "01-test"), number=1)
    )
    window.experiment = experiment

    window._refresh_panel()

    # Not merely empty -- gone. An empty label still reserves its line.
    assert not window.panel.prompt_bar.message.isVisible()
    assert window.panel.prompt_bar.message.text() == ""


def test_a_workflow_prompt_still_shows_with_its_buttons(window):
    window.panel.prompt_bar.ask("Move to the milling position?", "Yes", "No")

    assert window.panel.prompt_bar.message.text() == "Move to the milling position?"
    assert window.panel.prompt_bar.message.isVisible()
    assert window.panel.prompt_bar.yes.isVisible()
    assert window.panel.prompt_bar.no.isVisible()

    # And clears again when the workflow stops asking.
    window.panel.prompt_bar.ask("")

    assert not window.panel.prompt_bar.message.isVisible()
    assert not window.panel.prompt_bar.yes.isVisible()


def test_the_name_boxes_are_gone(window):
    """They were read-only QLineEdits: they read as inputs and were not.

    Pinned because the obvious way to restore a name to the panel is to put the
    box back, and the tab-corner button already carries it.
    """
    assert not hasattr(window.panel, "lineEdit_experiment_name")
    assert not hasattr(window.panel, "lineEdit_protocol_name")


class _StatusHost(QMainWindow):
    _update_instructions = AutoLamellaSingleWindowUI._update_instructions

    def __init__(self):
        super().__init__()
        self.status_bar = FibsemStatusBar(self)
        self.setStatusBar(self.status_bar)
        # What `_update_instructions` reads.
        self.microscope = None
        self.experiment = None
        self.protocol = None


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
    assert host.microscope is None
    host._update_instructions()
    assert host.status_bar.text == INSTRUCTIONS["NOT_CONNECTED"]

    host.microscope = object()
    host._update_instructions()
    assert host.status_bar.text == INSTRUCTIONS["NO_EXPERIMENT"]

    experiment = Experiment(path=str(tmp_path), name="ladder")
    host.experiment = experiment
    host._update_instructions()
    assert host.status_bar.text == INSTRUCTIONS["NO_PROTOCOL"]

    host.protocol = AutoLamellaTaskProtocol()
    host._update_instructions()
    assert host.status_bar.text == INSTRUCTIONS["NO_LAMELLA"]

    experiment.positions.append(
        Lamella(petname="01-test", path=str(tmp_path / "01-test"), number=1)
    )
    host._update_instructions()
    assert host.status_bar.text == INSTRUCTIONS["AUTOLAMELLA_READY"]
