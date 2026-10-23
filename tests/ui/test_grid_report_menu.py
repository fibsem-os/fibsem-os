"""Tools → Reporting → Generate Grid Screening Report (FIB-1057).

The action lives with the lamella report and hands the work to the Grids tab,
which does the writing and says how it went on its strip.
"""

import os

from tests.fixtures.main_window import main_window_fixture

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.workflows.grids import GridRecord

window = main_window_fixture(connect=True, name="window")


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    return exp


def _reporting_actions(window):
    tools = next(
        m
        for m in window.menuBar().findChildren(type(window.menuBar().addMenu("")))
        if m.title() == "Tools"
    )
    reporting = next(a.menu() for a in tools.actions() if a.text() == "Reporting")
    return [a.text() for a in reporting.actions() if a.isVisible()]


def test_it_sits_with_the_lamella_report(window):
    assert _reporting_actions(window) == [
        "Generate Report",
        "Generate Grid Screening Report",
        "Generate Overview Plot",
    ]


def test_it_refuses_with_nothing_to_report(window, experiment, monkeypatch):
    toasts = []
    monkeypatch.setattr(window, "show_toast", lambda msg, *a, **k: toasts.append(msg))
    window.menu_bar.action_generate_grid_report.trigger()
    assert toasts == ["Open an experiment to report on."]
    window.experiment = experiment
    window._on_experiment_update()
    window.menu_bar.action_generate_grid_report.trigger()
    assert "No grids in this experiment" in toasts[-1]


def test_it_shows_the_grids_tab_and_writes_there(window, experiment, monkeypatch):
    experiment.add_grid(GridRecord(name="grid-oak"))
    window.experiment = experiment
    window._on_experiment_update()
    tab = window.grids_tab
    tab._synchronous = True
    opened = []
    monkeypatch.setattr(tab, "open_report", opened.append)
    window.menu_bar.action_generate_grid_report.trigger()
    assert window.tab_widget.currentWidget() is tab
    (path,) = opened
    assert path == os.path.join(
        str(experiment.path), "reporting", "grid-screening-report.html"
    )
    assert tab.status_label.text() == "Report written: grid-screening-report.html"
