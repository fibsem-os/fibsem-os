"""The Lamella tab's protocol editor shows the selected card, and nothing without one (FIB-1370).

Clicking the selected card again deselects it. The editor kept the lamella it had,
and an edit made there was still written to it.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module  # noqa: E402
from fibsem.structures import FibsemStagePosition  # noqa: E402


@pytest.fixture(autouse=True)
def _preferences(tmp_path, monkeypatch):
    import fibsem.config as cfg

    monkeypatch.setattr(cfg, "USER_PREFERENCES_PATH", str(tmp_path / "prefs.yaml"))


@pytest.fixture
def main_ui(qapp):
    window = module.AutoLamellaSingleWindowUI()
    ui = window.autolamella_ui
    ui.system_widget.connect_to_microscope()
    yield window
    window.lamella_widget._save_timer.stop()
    ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture
def experiment(main_ui, tmp_path):
    ui = main_ui.autolamella_ui
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    ui.experiment = exp
    main_ui._on_experiment_update()
    for i, name in enumerate(("one", "two")):
        ui.add_new_lamella(
            stage_position=FibsemStagePosition(x=i * 1e-4, y=0, z=0, r=0, t=0),
            name=name,
        )
    return exp


def _click(main_ui, lamella):
    QTest.mouseClick(main_ui.lamella_card_container._cards[lamella.id], Qt.LeftButton)


def _shown(main_ui):
    """The lamella the protocol page offers for editing, or None for the empty page."""
    if main_ui._protocol_page.currentIndex() == 0:
        return None
    return main_ui.lamella_widget._selected_lamella


def _edit_description(main_ui, text):
    editor = main_ui.lamella_widget
    editor.line_edit_description.setText(text)
    editor.line_edit_description.editingFinished.emit()


def test_selecting_a_card_shows_it(main_ui, experiment):
    two = experiment.positions[1]
    _click(main_ui, two)

    assert _shown(main_ui) is two
    assert main_ui.lamella_widget.label_lamella_name.text() == "two"


def test_deselecting_the_card_empties_the_editor(main_ui, experiment):
    two = experiment.positions[1]
    _click(main_ui, two)
    _click(main_ui, two)

    assert main_ui.lamella_card_container._selected_id is None
    assert _shown(main_ui) is None
    assert main_ui.lamella_widget._selected_lamella is None


def test_an_edit_after_deselecting_reaches_no_lamella(main_ui, experiment):
    two = experiment.positions[1]
    _click(main_ui, two)
    _click(main_ui, two)

    _edit_description(main_ui, "written after deselecting")

    assert [lamella.description for lamella in experiment.positions] == ["", ""]


def test_a_refresh_with_nothing_selected_selects_nothing(main_ui, experiment):
    """The refresh button and every experiment refresh used to fall back to the
    first lamella."""
    two = experiment.positions[1]
    _click(main_ui, two)
    _click(main_ui, two)

    main_ui.lamella_widget.set_experiment()
    main_ui.lamella_widget._refresh_experiment_positions()

    assert main_ui.lamella_widget._selected_lamella is None


def test_an_experiment_opens_with_nothing_selected(main_ui, experiment):
    main_ui._rebuild_lamella_list()
    main_ui.lamella_widget.set_experiment()

    assert main_ui.lamella_card_container._selected_id is None
    assert _shown(main_ui) is None
