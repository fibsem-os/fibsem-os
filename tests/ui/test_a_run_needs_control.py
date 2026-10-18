"""The app's run start takes control of the microscope (FIB-1270): a run is in
control from beams-on to its end, and Run is refused, by name, while someone
else is in control -- a command from the App Server, say."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.ui import AutoLamellaUI as module
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.workflows.grid.manager import (
    GridTaskManager,
)


@pytest.fixture
def ui(qapp, monkeypatch, tmp_path):
    import fibsem.config as fibsem_config

    config = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    widget = AutoLamellaUI(parent_ui=None)
    monkeypatch.setattr(
        widget.system_widget,
        "load_configuration",
        lambda configuration_name=None: config,
    )
    widget.system_widget.connect_to_microscope()
    widget.experiment = Experiment(path=tmp_path, name="control")
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


@pytest.fixture
def toasts(monkeypatch):
    shown = []
    monkeypatch.setattr(
        module.notification_service,
        "show_toast",
        lambda message, *args, **kwargs: shown.append(message),
    )
    return shown


@pytest.mark.parametrize(
    "start",
    [
        lambda ui: ui._start_run_workflow_thread(["Trench"], ["lamella-01"]),
        lambda ui: ui._start_run_grid_workflow_thread(["Overview"], ["Grid-01"]),
    ],
    ids=["lamella", "grid"],
)
def test_run_is_refused_by_name_while_someone_else_is_in_control(ui, toasts, start):
    with ui.microscope.control.take("agent", "POST /move_stage_relative"):
        start(ui)

    assert ui._task_worker_thread is None, "nothing started"
    assert toasts == [
        "Can't start the workflow: the agent is in control (POST /move_stage_relative)."
    ]


@pytest.mark.parametrize(
    "manager, worker",
    [
        (TaskManager, lambda ui: ui._run_tasks_worker([], [])),
        (GridTaskManager, lambda ui: ui._run_grid_tasks_worker([], [], False)),
    ],
    ids=["lamella", "grid"],
)
def test_the_run_is_in_control_for_its_whole_length(ui, monkeypatch, manager, worker):
    seen = []
    monkeypatch.setattr(
        manager,
        "run",
        lambda self, *args, **kwargs: seen.append(self.microscope.control.current),
    )

    worker(ui)

    assert seen[0].by == "workflow"
    assert ui.microscope.control.current is None


@pytest.mark.parametrize(
    "manager, worker",
    [
        (TaskManager, lambda ui: ui._run_tasks_worker([], [])),
        (GridTaskManager, lambda ui: ui._run_grid_tasks_worker([], [], False)),
    ],
    ids=["lamella", "grid"],
)
def test_a_run_that_loses_the_race_for_control_does_not_start(
    ui, monkeypatch, manager, worker
):
    # Past the check on Run, someone took control first: the worker is refused
    # at its entry point and ends like any run that never began.
    ran = []
    monkeypatch.setattr(manager, "run", lambda self, *a, **k: ran.append(True))

    with ui.microscope.control.take("agent", "POST /move_stage_relative"):
        worker(ui)

    assert not ran
    assert ui._task_manager is None


@pytest.mark.parametrize(
    "worker",
    [
        lambda ui: ui._run_tasks_worker(["Trench"], []),
        lambda ui: ui._run_grid_tasks_worker(["Overview"], [], False),
    ],
    ids=["lamella", "grid"],
)
def test_a_run_that_dies_in_set_up_ends_with_workflow_failed(ui, monkeypatch, worker):
    """FIB-1353: start_workflow has answered started by the time the worker
    turns the beams on; a failure there ends the run with an event."""
    from fibsem.hooks import FunctionHook, HookEvent, HookManager

    fired = []
    hooks = HookManager()
    hooks.register(
        FunctionHook(name="seen", events=list(HookEvent), callback=fired.append)
    )
    monkeypatch.setattr(ui, "setup_hooks", lambda: hooks)
    monkeypatch.setattr(ui.microscope, "is_on", lambda beam_type: False)

    def beam_fails(beam_type):
        raise RuntimeError("the beam would not turn on")

    monkeypatch.setattr(ui.microscope, "turn_on", beam_fails)

    worker(ui)

    assert [c.event for c in fired] == ["workflow_failed"]
    assert fired[0].error == "the beam would not turn on"
    assert ui.microscope.control.current is None
