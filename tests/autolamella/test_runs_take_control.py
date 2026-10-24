"""A run is in control of the microscope for its whole length (FIB-1270), the
same headless as in the app: a command from the App Server or a script is
refused beside it, and a run does not start beside them. Every run executes
through ``BaseTaskManager.execute``, whoever starts it."""

import pytest

from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.workflows.grid.manager import GridTaskManager


@pytest.fixture
def experiment(tmp_path):
    return Experiment(path=tmp_path, name="control")


@pytest.mark.parametrize(
    "manager, start",
    [
        (TaskManager, lambda m, e: TaskManager(m, e).execute([])),
        (GridTaskManager, lambda m, e: GridTaskManager(m, e).execute([])),
    ],
    ids=["lamella", "grid"],
)
def test_a_headless_run_is_in_control_while_it_runs(
    microscope, experiment, monkeypatch, manager, start
):
    seen = []
    monkeypatch.setattr(
        manager,
        "run",
        lambda self, *args, **kwargs: seen.append(self.microscope.control.current),
    )

    start(microscope, experiment)

    assert seen[0].by == "workflow"
    assert microscope.control.current is None


@pytest.mark.parametrize(
    "manager, start",
    [
        (TaskManager, lambda m, e: TaskManager(m, e).execute([])),
        (GridTaskManager, lambda m, e: GridTaskManager(m, e).execute([])),
    ],
    ids=["lamella", "grid"],
)
def test_a_headless_run_is_refused_while_someone_else_is_in_control(
    microscope, experiment, monkeypatch, manager, start
):
    ran = []
    monkeypatch.setattr(manager, "run", lambda self, *a, **k: ran.append(True))

    with microscope.control.take("agent", "POST /move_stage_relative"):
        start(microscope, experiment)
        assert microscope.control.current.by == "agent"

    assert not ran


@pytest.mark.parametrize("turn_beams_off", [True, False])
@pytest.mark.parametrize(
    "manager", [TaskManager, GridTaskManager], ids=["lamella", "grid"]
)
def test_a_headless_run_turns_the_beams_off_when_the_protocol_says_so(
    microscope, experiment, monkeypatch, manager, turn_beams_off
):
    from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
    from fibsem.structures.settings import BeamType

    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.task_protocol.options.turn_beams_off = turn_beams_off
    monkeypatch.setattr(manager, "run", lambda self, *a, **k: None)

    manager(microscope, experiment).execute([])

    for beam in (BeamType.ELECTRON, BeamType.ION):
        assert microscope.is_on(beam) is not turn_beams_off
