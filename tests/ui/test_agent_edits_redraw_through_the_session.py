"""An agent's edit through the App Server context redraws the window through
the session: the context never talks to the window, and the window has no
hooks for it. It hears ``session.changed`` (and the experiment's own events)
and redraws, as it does for anything else that changes the experiment."""

import ast
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict

import fibsem.applications.autolamella.server.context as context_module
import fibsem.server.app_context as app_context_module
from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.applications.autolamella.server.context import item_fields_version
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui import window as window_module
from fibsem.applications.autolamella.ui.window import AutoLamellaWindow
from fibsem.structures.settings import MicroscopeState
from fibsem.workflows.grids import GridRecord
from fibsem.workflows.session import EXPERIMENT
from tests.fixtures.main_window import close_window

# What the window used to offer the context, and must not any more.
WINDOW_HOOKS = (
    "after_agent_edit",
    "before_config_edit",
    "after_config_edit",
    "adopt_experiment",
    "adopt_protocol",
    "stop_task_workflow",
    "stop_current_task",
    "grids_changed",
    "experiment_update_signal",
)


@pytest.fixture
def ui(qapp):
    window = AutoLamellaWindow()
    try:
        yield window
    finally:
        close_window(window)
        window.deleteLater()
        qapp.processEvents()


@pytest.fixture
def experiment(ui, tmp_path):
    exp = Experiment(path=tmp_path / "exp", name="redraw-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    exp.grids.append(GridRecord(name="G1"))
    ui.experiment = exp
    ui.session.changed.emit(EXPERIMENT)
    return exp


def test_the_window_has_no_hooks_for_the_context():
    for hook in WINDOW_HOOKS:
        assert not hasattr(AutoLamellaWindow, hook), hook


def test_an_agent_edit_redraws_the_panel_through_the_session(
    ui, experiment, monkeypatch
):
    redraws = []
    monkeypatch.setattr(ui.panel, "refresh", lambda: redraws.append("panel"))
    shown = []
    monkeypatch.setattr(
        window_module.notification_service,
        "show",
        lambda message, *args, **kwargs: shown.append(message),
    )
    lamella = experiment.positions[0]

    result = AutoLamellaAppContext(ui.session).apply_item_patch(
        lamella.name, {"description": "from the agent"}, item_fields_version(lamella)
    )

    assert result["applied"] is True
    assert redraws, "the panel heard the edit on session.changed"
    # The edit is on the web Agent page, not a toast.
    assert shown == []


def test_an_agent_grid_edit_redraws_the_grid_views(ui, experiment, monkeypatch):
    redraws = []
    monkeypatch.setattr(ui, "_refresh_after_naming", lambda: redraws.append("grids"))

    result = AutoLamellaAppContext(ui.session).set_grid_note("G1", "cracked")

    assert result["applied"] is True
    assert redraws == ["grids"]


def test_an_agent_protocol_edit_redraws_the_protocol_views(ui, experiment, monkeypatch):
    redraws = []
    monkeypatch.setattr(ui, "_on_experiment_update", lambda: redraws.append("tabs"))

    result = AutoLamellaAppContext(ui.session).set_lamella_defaults(
        {"name_prefix": "agent-"}
    )

    assert result["applied"] is True
    assert redraws == ["tabs"]


@pytest.mark.parametrize("module", [context_module, app_context_module])
def test_the_context_does_not_reference_the_window(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    arguments = {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    assert not attributes & {"_window", *WINDOW_HOOKS}
    assert "AppWindow" not in names
    assert "AutoLamellaWindow" not in names
    assert "window" not in arguments
