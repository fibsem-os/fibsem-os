"""A run takes control of the microscope (FIB-1270): it is in control from
beams-on to its end, and a start is refused, by name, while someone else is in
control -- a command from the App Server, say. Both through the session's
runner, with no window: what any caller, headless included, gets."""

import os

import pytest

from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.hooks import FunctionHook, HookEvent, HookManager
from fibsem.workflows.grid.manager import GridTaskManager
from fibsem.workflows.runner import GRIDS, ITEMS, RunRequest
from fibsem.workflows.session import AppSession


@pytest.fixture(scope="module")
def microscope():
    import fibsem.config as fibsem_config

    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    config = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    microscope, _ = utils.setup_session(config_path=config)
    yield microscope
    microscope.disconnect()


@pytest.fixture
def experiment(tmp_path):
    experiment = Experiment(path=tmp_path, name="control")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    return experiment


@pytest.fixture
def session(microscope, experiment):
    session = AppSession()
    session.microscope = microscope
    session.experiment = experiment
    return session


@pytest.mark.parametrize(
    "request_",
    [
        RunRequest(kind=ITEMS, task_names=("Trench",), names=("lamella-01",)),
        RunRequest(kind=GRIDS, task_names=("Overview",), names=("Grid-01",)),
    ],
    ids=["lamella", "grid"],
)
def test_a_start_is_refused_by_name_while_someone_else_is_in_control(
    session, experiment, request_
):
    with session.microscope.control.take("agent", "POST /move_stage_relative"):
        outcome = session.runner.start(experiment, request_)

    assert not session.runner.is_running, "nothing started"
    assert outcome.as_dict() == {
        "started": False,
        "reason": "Can't start the workflow: the agent is in control "
        "(POST /move_stage_relative).",
    }


def test_a_run_starts_and_ends_with_no_window(session, experiment, monkeypatch):
    """The runner runs the app's manager on its own thread, and keeps the
    summary after the manager is gone."""
    experiment.task_protocol.workflow_config.tasks.clear()
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskDescription,
    )

    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(name="Trench", required=True)
    )
    ran, ended = [], []
    monkeypatch.setattr(
        TaskManager, "run", lambda self, *args: ran.append((args, self.parent_ui))
    )
    session.runner.finished.connect(ended.append)

    outcome = session.runner.start(
        experiment, RunRequest(kind=ITEMS, task_names=("Trench",))
    )

    assert outcome.started, outcome.reason
    assert session.runner.join(10)
    assert ran == [((["Trench"], []), None)]
    assert ended == [False]
    assert session.runner.manager is None
    assert session.runner.last_summary is not None


@pytest.mark.parametrize(
    "manager", [TaskManager, GridTaskManager], ids=["lamella", "grid"]
)
def test_the_run_is_in_control_for_its_whole_length(
    microscope, experiment, monkeypatch, manager
):
    seen = []
    monkeypatch.setattr(
        manager,
        "run",
        lambda self, *args, **kwargs: seen.append(self.microscope.control.current),
    )

    manager(microscope, experiment).execute([], [])

    assert seen[0].by == "workflow"
    assert microscope.control.current is None


@pytest.mark.parametrize(
    "manager", [TaskManager, GridTaskManager], ids=["lamella", "grid"]
)
def test_a_run_that_loses_the_race_for_control_does_not_start(
    microscope, experiment, monkeypatch, manager
):
    # Past the check on start, someone took control first: the run is refused
    # at its entry point and ends like any run that never began.
    ran, fired = [], []
    monkeypatch.setattr(manager, "run", lambda self, *a, **k: ran.append(True))
    hooks = HookManager()
    hooks.register(
        FunctionHook(name="seen", events=list(HookEvent), callback=fired.append)
    )

    with microscope.control.take("agent", "POST /move_stage_relative"):
        manager(microscope, experiment, hook_manager=hooks).execute([], [])

    assert not ran
    assert [c.event for c in fired] == ["workflow_failed"]


@pytest.mark.parametrize(
    "manager, task",
    [(TaskManager, "Trench"), (GridTaskManager, "Overview")],
    ids=["lamella", "grid"],
)
def test_a_run_that_dies_in_set_up_ends_with_workflow_failed(
    microscope, experiment, monkeypatch, manager, task
):
    """FIB-1353: the start has answered started by the time the run turns the
    beams on; a failure there ends the run with an event."""
    fired = []
    hooks = HookManager()
    hooks.register(
        FunctionHook(name="seen", events=list(HookEvent), callback=fired.append)
    )
    monkeypatch.setattr(microscope, "is_on", lambda beam_type: False)

    def beam_fails(beam_type):
        raise RuntimeError("the beam would not turn on")

    monkeypatch.setattr(microscope, "turn_on", beam_fails)

    manager(microscope, experiment, hook_manager=hooks).execute([task], [])

    assert [c.event for c in fired] == ["workflow_failed"]
    assert fired[0].error == "the beam would not turn on"
    assert microscope.control.current is None
