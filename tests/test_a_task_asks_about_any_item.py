"""A task asks about whatever item it runs on, not only a lamella.

``ask`` is on the shared task base: a grid task asks about its grid the way a
lamella task asks about its lamella, waits for the answer, and Stop wakes the
wait at once.
"""

import threading
import time

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.cancellation import StopEvent
from fibsem.workflows.attention import Attention
from fibsem.workflows.grid import BeamOverviewGridTaskConfig, GridTask
from fibsem.workflows.grid.manager import GridTaskManager
from fibsem.workflows.proposals import STATE, Decision, DecisionOutcome
from tests.fixtures.microscopes import demo_microscope

TASK = "overview_sem"


class _AsksAboutTheGrid(GridTask):
    asks = (STATE,)

    def _run(self) -> None:
        pass


microscope = demo_microscope("sim-arctis-configuration.yaml", scope="function")


@pytest.fixture
def task(tmp_path, microscope):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    config = BeamOverviewGridTaskConfig(task_name=TASK)
    exp.grid_protocol.add(config)
    exp.grid_protocol.set_attention(TASK, Attention.supervised)
    exp.sync_grids_from_inventory(microscope.sample_handling)
    manager = GridTaskManager(microscope, exp, deciders=lambda: ())
    return _AsksAboutTheGrid(
        microscope, config, exp.grids[0], experiment=exp, task_manager=manager
    )


def _asking(task):
    seen = {}

    def ask():
        try:
            seen["decision"] = task.ask(
                STATE, {"stage_position": task.microscope.get_stage_position()}
            )
        except InterruptedError as error:
            seen["error"] = error

    thread = threading.Thread(target=ask, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while (question := task.grid.proposal(TASK)) is None or not question.pending:
        assert time.monotonic() < deadline, "the question was never asked"
        time.sleep(0.01)
    return thread, seen, question


def test_a_grid_task_asks_about_its_grid_and_waits_for_the_answer(task):
    thread, seen, question = _asking(task)
    assert question.asking and thread.is_alive()

    result = task.experiment.decide(
        task.grid.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            actor="agent:x",
            proposal_id=question.id,
            values={"stage_position": task.microscope.get_stage_position()},
        ),
    )
    assert result.applied, result.reason
    thread.join(timeout=10)
    assert seen["decision"].outcome is DecisionOutcome.Confirmed


def test_stop_wakes_the_wait_at_once(task):
    task._stop_event = StopEvent()
    thread, seen, question = _asking(task)

    stopped = time.monotonic()
    task._stop_event.set()
    thread.join(timeout=10)

    assert time.monotonic() - stopped < 0.5, "Stop did not wake the wait"
    assert "error" in seen
    assert question.current.outcome is DecisionOutcome.Withdrawn
