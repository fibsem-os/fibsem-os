"""While a run is on, the window edits every lamella, the one it is on included
(only the running task's config waits, FIB-1505), as the App Server does; the
stage and objective stay the run's, and the lamella it is on is not removed."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.ui.window import AutoLamellaWindow  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402
from tests.fixtures.main_window import close_window  # noqa: E402
from tests.run_state import StubRunner, run_working_on  # noqa: E402


@pytest.fixture
def window(qapp, tmp_path):
    ui = AutoLamellaWindow()
    experiment = Experiment(path=str(tmp_path), name="run-edits")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(experiment.path, exist_ok=True)
    for number, name in enumerate(("01-busy", "02-free"), start=1):
        lamella = Lamella(petname=name, path=str(tmp_path / name), number=number)
        lamella.milling_pose = MicroscopeState()
        experiment.positions.append(lamella)
    ui.experiment = experiment
    ui.session.runner = StubRunner(ui.session)
    ui.panel.experiment_tab.update_lamella_combobox()  # listed before the run
    yield ui
    close_window(ui)


def _select(window, name):
    tab = window.panel.experiment_tab
    tab.lamella_list.select(name)
    tab.update_lamella_ui()
    return tab


def test_the_selected_lamella_is_editable_while_the_run_is_on_it(window):
    with run_working_on(window.session.runner, "01-busy"):
        for name in ("01-busy", "02-free"):
            tab = _select(window, name)
            widget = tab.selected_lamella_widget
            assert widget.isEnabled()
            assert not widget._action_use_current_obj_pos.isEnabled()
            assert not widget._action_move_to_obj_pos.isEnabled()

    tab.refresh()
    assert tab.selected_lamella_widget.isEnabled()
    assert tab.selected_lamella_widget._action_use_current_obj_pos.isEnabled()


def test_the_lamella_a_run_is_on_is_not_removed(window):
    experiment = window.experiment
    busy, free = experiment.positions
    with run_working_on(window.session.runner, busy.name):
        window._on_lamella_remove_requested(busy)
        assert busy in experiment.positions
        window._on_lamella_remove_requested(free)
    assert free not in experiment.positions
