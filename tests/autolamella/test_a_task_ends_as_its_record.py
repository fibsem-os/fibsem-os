"""A finished task is its history and its outputs: it proposes nothing when it
ends, in any mode, and nothing waits on it. What it asked while it ran stays on
the record.

Runs a real milling task class against the Demo microscope through a real
TaskManager, with _run stubbed to the one thing the record needs: the final
reference images the task writes into task_state.outputs.
"""

import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.polishing import (
    MillPolishingTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTask,
    MillRoughTaskConfig,
)
from fibsem.structures import Point
from fibsem.workflows.attention import Attention
from fibsem.workflows.proposals import (
    DETECTION,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.workflows.status import TaskStatus

ROUGH = "Rough Milling"
POLISH = "Polishing"
CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


def _experiment(
    tmp_path: Path, microscope, attention: Attention = Attention.automated
) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=ROUGH, required=True, attention=attention
                ),
                AutoLamellaTaskDescription(
                    name=POLISH, required=True, requires=[ROUGH]
                ),
            ]
        )
    )
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        EventedDict(
            {
                ROUGH: MillRoughTaskConfig(task_name=ROUGH),
                POLISH: MillPolishingTaskConfig(task_name=POLISH),
            }
        ),
    )
    exp.positions[0].path.mkdir(parents=True, exist_ok=True)
    return exp


def _task(microscope, exp: Experiment, body=None) -> MillRoughTask:
    manager = TaskManager(microscope=microscope, experiment=exp)
    lamella = exp.positions[0]
    task = MillRoughTask(
        microscope=microscope,
        config=lamella.task_config[ROUGH],
        lamella=lamella,
        task_manager=manager,
    )

    def _run():
        # what a real run leaves behind: the final set, recorded by role
        lamella.task_state.outputs["final_fib"] = [f"ref_{ROUGH}_final_res_01_ib.tif"]
        lamella.task_state.outputs["final_sem"] = [f"ref_{ROUGH}_final_res_01_eb.tif"]
        if body is not None:
            body()

    task._run = _run  # type: ignore[method-assign]
    return task


def test_a_supervised_task_proposes_nothing_when_it_ends(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    task = _task(microscope, exp)
    lamella = exp.positions[0]

    task.run()

    assert lamella.proposals == [], "its result is its history and outputs"
    run = lamella.task_history[-1]
    assert run.status is TaskStatus.Completed
    assert run.outputs["final_fib"] == [f"ref_{ROUGH}_final_res_01_ib.tif"]
    assert lamella.latest_run_completed(ROUGH), "finished: nothing waits on it"
    assert exp.pending_proposals() == []
    assert task.task_manager._defer_reason(lamella, POLISH) is None


def test_an_automated_task_proposes_nothing_when_it_ends(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    task = _task(microscope, exp)
    lamella = exp.positions[0]

    task.run()

    assert lamella.proposals == []
    assert task.task_manager._defer_reason(lamella, POLISH) is None


def test_a_failed_task_proposes_nothing_and_keeps_its_outputs(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]

    def boom():
        raise RuntimeError("stage timeout")

    task = _task(microscope, exp, body=boom)
    with pytest.raises(RuntimeError):
        task.run()

    assert lamella.proposals == []
    run = lamella.task_history[-1]
    assert run.status is TaskStatus.Failed and run.status_message == "stage timeout"
    assert run.outputs["final_fib"], "what was acquired before it failed"


def test_a_proposal_decided_during_the_run_is_what_the_run_leaves(microscope, tmp_path):
    """A proposal asked mid-run (FIB-1025) stays on the record with its
    decision, and nothing is added after it when the task ends."""
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]
    asked = {}

    def ask_and_be_answered():
        proposal = Proposal(
            kind=DETECTION,
            values={"features": [{"name": "LamellaCentre", "px": Point(10, 20)}]},
            provenance={"task_id": lamella.task_state.task_id},
        )
        assert exp.ask_proposal(lamella.id, ROUGH, proposal)
        result = exp.decide(
            lamella.id,
            ROUGH,
            Decision(
                outcome=DecisionOutcome.Confirmed,
                actor="operator:op",
                values=dict(proposal.values),
                proposal_id=proposal.id,
            ),
        )
        assert result.applied, result.reason
        asked["proposal"] = proposal

    _task(microscope, exp, body=ask_and_be_answered).run()

    assert lamella.proposals_from(ROUGH) == [asked["proposal"]]
    assert asked["proposal"].current.outcome is DecisionOutcome.Confirmed


# ---------------------------------------------------------------------------
# A finished run that cannot be recorded (FIB-1374)
# ---------------------------------------------------------------------------


def _record_events(task) -> list:
    from fibsem.hooks import FunctionHook, HookEvent, HookManager

    fired = []
    hooks = HookManager()
    hooks.register(
        FunctionHook(name="seen", events=list(HookEvent), callback=fired.append)
    )
    task.task_manager.hook_manager = hooks
    return fired


def _block_the_save(exp: Experiment) -> str:
    """A directory where experiment.yaml goes: every save fails, whoever runs
    the test (IsADirectoryError on POSIX, PermissionError on Windows)."""
    blocker = os.path.join(exp.path, "experiment.yaml")
    os.makedirs(blocker)
    return blocker


def test_a_task_whose_end_fails_ends_failed(microscope, tmp_path):
    """What follows a finished run (post_task) failing: the run was recorded
    Completed, so it is taken back to Failed, with one history entry and
    task_failed, and the file says so."""
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]
    task = _task(microscope, exp)
    fired = _record_events(task)
    finish = task.post_task

    def finish_then_fail():
        finish()
        raise OSError("disk full")

    task.post_task = finish_then_fail  # type: ignore[method-assign]
    with pytest.raises(OSError):
        task.run()

    assert lamella.task_state.status is TaskStatus.Failed
    assert "could not be recorded" in lamella.task_state.status_message
    runs = [t for t in lamella.task_history if t.task_id == task.task_id]
    assert [t.status for t in runs] == [TaskStatus.Failed], "one entry"
    assert [c.event for c in fired] == ["task_started", "task_failed"]

    exp.save()
    saved = Experiment.load(os.path.join(exp.path, "experiment.yaml"))
    last = saved.positions[0].task_history[-1]
    assert last.task_id == task.task_id
    assert last.status is TaskStatus.Failed, "the file says it failed"


def test_a_task_the_run_cannot_save_after_ends_failed(
    microscope, tmp_path, monkeypatch
):
    """The run's own save after a finished task: failing, the task is not
    left Completed, and task_failed says so before the run ends."""
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]
    task = _task(microscope, exp)
    manager = task.task_manager
    fired = _record_events(task)

    from fibsem.applications.autolamella.workflows.tasks import manager as module

    def run_then_block(**_kwargs):
        task.run()
        saved = os.path.join(exp.path, "experiment.yaml")
        if os.path.exists(saved):
            os.remove(saved)
        _block_the_save(exp)  # the run's save after it fails

    monkeypatch.setattr(module, "run_task", run_then_block)
    with pytest.raises(OSError):
        manager._run_single_task(ROUGH, lamella)

    assert lamella.task_state.status is TaskStatus.Failed
    assert "could not be saved" in lamella.task_state.status_message
    assert lamella.task_history[-1].status is TaskStatus.Failed
    assert [c.event for c in fired] == ["task_started", "task_completed", "task_failed"]
