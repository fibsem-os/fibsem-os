"""A run takes control of the microscope (FIB-1270): it is in control from
beams-on to its end, and a start is refused, by name, while someone else is in
control -- a command from the App Server, say. Both through the session's
runner, with no window: what any caller, headless included, gets."""

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.hooks import FunctionHook, HookEvent, HookManager
from fibsem.workflows.grid.manager import GridTaskManager
from fibsem.workflows.runner import GRIDS, ITEMS, RunRequest
from fibsem.workflows.session import AppSession
from tests.fixtures.microscopes import demo_microscope
from tests.protocols import add_planned, planned

microscope = demo_microscope("sim-arctis-configuration.yaml")


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

    add_planned(experiment.task_protocol, planned(name="Trench", required=True))
    ran, ended = [], []
    monkeypatch.setattr(
        TaskManager, "run", lambda self, *args: ran.append((args, self.responder))
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


def test_a_run_that_dies_in_set_up_says_it_failed(
    session, experiment, microscope, monkeypatch
):
    """Not a clean finish: the runner, the App Server's status and its
    ``failed`` line all say the run failed, and why."""
    from fibsem.applications.autolamella.server import AutoLamellaAppContext
    from fibsem.workflows.runner import RunOutcome

    add_planned(experiment.task_protocol, planned(name="Trench", required=True))
    monkeypatch.setattr(microscope, "is_on", lambda beam_type: False)

    def beam_fails(beam_type):
        raise RuntimeError("the beam would not turn on")

    monkeypatch.setattr(microscope, "turn_on", beam_fails)

    started = session.runner.start(
        experiment, RunRequest(kind=ITEMS, task_names=("Trench",))
    )
    assert started.started, started.reason
    assert session.runner.join(10)

    assert session.runner.last_outcome == RunOutcome(
        "failed", "the beam would not turn on"
    )
    workflow = AutoLamellaAppContext(session).status().workflow
    assert workflow.phase == "idle"
    assert (workflow.last_run.ended, workflow.last_run.why) == (
        "failed",
        "the beam would not turn on",
    )
    assert (workflow.failed.what, workflow.failed.message) == (
        "Run failed",
        "the beam would not turn on",
    )


def test_a_stop_reads_as_stopping_until_the_task_unwinds(
    session, experiment, monkeypatch
):
    import threading

    from fibsem.workflows.runner import RunPhase

    add_planned(experiment.task_protocol, planned(name="Trench", required=True))
    inside, release = threading.Event(), threading.Event()

    def run(self, *args):
        inside.set()
        release.wait(10)

    monkeypatch.setattr(TaskManager, "run", run)
    assert session.runner.phase is RunPhase.IDLE
    session.runner.start(experiment, RunRequest(kind=ITEMS, task_names=("Trench",)))
    assert inside.wait(10)
    assert session.runner.phase is not RunPhase.STOPPING

    session.runner.stop()
    assert session.runner.phase is RunPhase.STOPPING
    release.set()
    assert session.runner.join(10)
    assert session.runner.phase is RunPhase.IDLE


def test_the_phase_follows_the_run(session):
    from types import SimpleNamespace

    from fibsem.workflows.runner import RunPhase

    runner = session.runner
    manager = SimpleNamespace(is_stopped=False, began=False, hold=None, waiting=False)
    runner.manager, runner._thread = manager, object()
    try:
        assert runner.phase is RunPhase.STARTING
        manager.began = True
        assert runner.phase is RunPhase.RUNNING
        manager.waiting = True
        assert runner.phase is RunPhase.WAITING
        manager.hold = object()
        assert runner.phase is RunPhase.HELD
        manager.is_stopped = True
        assert runner.phase is RunPhase.STOPPING
    finally:
        runner.manager, runner._thread = None, None


def test_how_a_run_ended_comes_from_its_end_event():
    from types import SimpleNamespace

    from fibsem.workflows.runner import RunOutcome, run_outcome

    def ended(event, error=""):
        return SimpleNamespace(
            ended=event, ended_error=error, closing_note=lambda: "1 task(s) cannot run"
        )

    assert run_outcome(ended(None)) is None
    assert run_outcome(ended(HookEvent.WORKFLOW_COMPLETED)) == RunOutcome("completed")
    assert run_outcome(ended(HookEvent.WORKFLOW_CANCELLED)) == RunOutcome("cancelled")
    assert run_outcome(ended(HookEvent.WORKFLOW_STALLED)) == RunOutcome(
        "stalled", "1 task(s) cannot run"
    )
    assert run_outcome(ended(HookEvent.WORKFLOW_FAILED, "boom")) == RunOutcome(
        "failed", "boom"
    )
