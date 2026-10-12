"""A run is in control of the microscope for its whole length (FIB-1270), the
same headless as in the app: a command from the App Server or a script is
refused beside it, and a run is refused beside them."""

import os

import pytest

from fibsem import utils
from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    GridTaskManager,
    run_grid_tasks,
)
from fibsem.applications.autolamella.workflows.tasks.manager import (
    TaskManager,
    run_tasks,
)
from fibsem.control import MicroscopeBusy


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo")
    yield microscope
    microscope.disconnect()


@pytest.fixture
def experiment(tmp_path):
    return Experiment(path=tmp_path, name="control")


@pytest.mark.parametrize(
    "manager, start",
    [
        (TaskManager, lambda m, e: run_tasks(m, e, [])),
        (GridTaskManager, lambda m, e: run_grid_tasks(m, e, task_names=[])),
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
        (TaskManager, lambda m, e: run_tasks(m, e, [])),
        (GridTaskManager, lambda m, e: run_grid_tasks(m, e, task_names=[])),
    ],
    ids=["lamella", "grid"],
)
def test_a_headless_run_is_refused_while_someone_else_is_in_control(
    microscope, experiment, monkeypatch, manager, start
):
    ran = []
    monkeypatch.setattr(manager, "run", lambda self, *a, **k: ran.append(True))

    with microscope.control.take("agent", "POST /move_stage_relative"):
        with pytest.raises(MicroscopeBusy, match="the agent is in control"):
            start(microscope, experiment)

    assert not ran
