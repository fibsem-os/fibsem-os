"""Tools → Reporting → Generate Grid Screening Report (FIB-1057).

The action lives with the lamella report, shows with the Grids tab by the same
flag, and hands the work to that tab, which does the writing and says how it
went on its strip.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module


@pytest.fixture
def window(qapp):
    window = module.AutoLamellaSingleWindowUI()
    window._preferences.features.grid_workflow = True
    window._apply_grid_workflow_visibility()
    ui = window.autolamella_ui
    ui.system_widget.connect_to_microscope()
    yield window
    ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


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


def test_it_follows_the_grid_workflow_flag(window):
    window._preferences.features.grid_workflow = False
    window._apply_grid_workflow_visibility()
    assert "Generate Grid Screening Report" not in _reporting_actions(window)


def test_it_refuses_with_nothing_to_report(window, experiment, monkeypatch):
    toasts = []
    monkeypatch.setattr(window, "show_toast", lambda msg, *a, **k: toasts.append(msg))
    window.action_generate_grid_report.trigger()
    assert toasts == ["Open an experiment to report on."]
    window.autolamella_ui.experiment = experiment
    window._on_experiment_update()
    window.action_generate_grid_report.trigger()
    assert "No grids in this experiment" in toasts[-1]


def test_it_shows_the_grids_tab_and_writes_there(window, experiment, monkeypatch):
    pytest.importorskip("reportlab")
    experiment.add_grid(GridRecord(name="grid-oak"))
    window.autolamella_ui.experiment = experiment
    window._on_experiment_update()
    tab = window.grids_tab
    tab._synchronous = True
    opened = []
    monkeypatch.setattr(tab, "open_report", opened.append)
    window.action_generate_grid_report.trigger()
    assert window.tab_widget.currentWidget() is tab
    (path,) = opened
    assert path == os.path.join(str(experiment.path), "grid-screening-report.pdf")
    assert tab.status_label.text() == "Report written: grid-screening-report.pdf"
