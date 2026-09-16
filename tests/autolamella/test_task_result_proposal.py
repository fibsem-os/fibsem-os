"""Any task type that records a result leaves a task_result proposal, from the
base class, in every mode, so a gate works on any task in the protocol
without the task knowing. Recorded on failure too.

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
from fibsem.applications.autolamella.proposals import (
    MILLING_SETUP,
    TASK_RESULT,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
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

ROUGH = "Rough Milling"
POLISH = "Polishing"
CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


def _experiment(tmp_path: Path, microscope, review) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=ROUGH, supervise=False, required=True, review=review
                ),
                AutoLamellaTaskDescription(
                    name=POLISH, supervise=False, required=True, requires=[ROUGH]
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
    manager = TaskManager(microscope=microscope, experiment=exp, parent_ui=None)
    manager.review_enabled = True
    lamella = exp.positions[0]
    task = MillRoughTask(
        microscope=microscope,
        config=lamella.task_config[ROUGH],
        lamella=lamella,
        parent_ui=None,
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


def test_a_gated_task_records_its_result_and_the_consumer_waits(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, review=True)
    task = _task(microscope, exp)
    lamella = exp.positions[0]

    task.run()

    proposal = lamella.proposals[ROUGH]
    assert proposal.kind == TASK_RESULT and proposal.pending
    assert proposal.values == {}
    p = proposal.provenance
    assert p["proposer"] == ROUGH and p["task_name"] == ROUGH
    assert p["status"] == "Completed" and p["failure"] == ""
    assert p["reference_image"] == f"ref_{ROUGH}_final_res_01_ib.tif"
    assert p["reference_image_eb"] == f"ref_{ROUGH}_final_res_01_eb.tif"
    assert p["ended_at"] >= p["started_at"] > 0
    assert lamella.has_completed_task(ROUGH)
    assert task.task_manager._defer_reason(lamella, POLISH) == "awaiting_review"

    result = exp.decide(
        lamella.id,
        ROUGH,
        Decision(outcome=DecisionOutcome.Confirmed, author="human:op", values={}),
    )
    assert result.applied and result.synced_tasks == []
    assert task.task_manager._defer_reason(lamella, POLISH) is None


def test_a_failed_task_records_its_result_with_the_failure(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, review=True)
    lamella = exp.positions[0]

    def boom():
        raise RuntimeError("stage timeout")

    task = _task(microscope, exp, body=boom)
    with pytest.raises(RuntimeError):
        task.run()

    proposal = lamella.proposals[ROUGH]
    assert proposal.kind == TASK_RESULT and proposal.pending
    assert proposal.provenance["status"] == "Failed"
    assert proposal.provenance["failure"] == "stage timeout"
    assert proposal.provenance["reference_image"], "what was acquired before it failed"
    assert lamella.task_state.status is AutoLamellaTaskStatus.Failed


def test_not_gated_the_result_is_recorded_and_the_run_goes_on(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, review=False)
    task = _task(microscope, exp)
    lamella = exp.positions[0]

    task.run()

    proposal = lamella.proposals[ROUGH]
    assert proposal.kind == TASK_RESULT
    assert proposal.to_check and proposal.current.author == f"auto:{ROUGH}"
    assert task.task_manager._defer_reason(lamella, POLISH) is None
    assert [t for _i, t, _p in exp.proposals_to_check()] == [ROUGH]


def test_without_the_flag_the_result_is_recorded_but_never_gates(microscope, tmp_path):
    """The flag hides the Review surface, not the record."""
    exp = _experiment(tmp_path, microscope, review=True)
    task = _task(microscope, exp)
    task.task_manager.review_enabled = False
    task.run()
    lamella = exp.positions[0]
    proposal = lamella.proposals[ROUGH]
    assert proposal.kind == TASK_RESULT and not proposal.pending
    assert proposal.current.author == f"auto:{ROUGH}"
    assert task.task_manager._defer_reason(lamella, POLISH) is None


def test_what_a_task_type_proposes_is_declared_on_the_class(microscope, tmp_path):
    """A run records outputs; a task proposes a kind. Which kind is the task
    type's say, in code, not the protocol's. Every shipped task leaves images,
    so every shipped task proposes; None is reserved for a type with nothing
    to look at, and is exercised here by hand."""
    from fibsem.applications.autolamella.workflows.tasks.fiducial import (
        MillFiducialTask,
    )
    from fibsem.applications.autolamella.workflows.tasks.reference_image import (
        AcquireReferenceImageTask,
    )
    from fibsem.applications.autolamella.workflows.tasks.select_position import (
        SelectMillingPositionTask,
    )

    assert MillFiducialTask.proposal_kind == TASK_RESULT, "a bad fiducial is gateable"
    assert AcquireReferenceImageTask.proposal_kind == TASK_RESULT
    assert MillRoughTask.proposal_kind == TASK_RESULT
    assert SelectMillingPositionTask.proposal_kind == MILLING_SETUP
    exp = _experiment(tmp_path, microscope, review=True)
    task = _task(microscope, exp)
    type(task).proposal_kind = None
    try:
        task.run()
    finally:
        type(task).proposal_kind = TASK_RESULT
    assert exp.positions[0].proposals == {}


def test_a_task_type_with_its_own_kind_is_left_alone(microscope, tmp_path):
    """Setup proposes the milling position itself; the base class records a
    task_result only for types that declare that kind, so it never papers over
    a richer proposal, pending or already decided inline."""
    exp = _experiment(tmp_path, microscope, review=True)
    lamella = exp.positions[0]

    def own():
        p = Proposal(kind=MILLING_SETUP, values={}, provenance={"proposer": "me"})
        p.decisions.append(
            Decision(
                outcome=DecisionOutcome.Confirmed, author="human:op", via="workflow"
            )
        )
        lamella.proposals[ROUGH] = p

    task = _task(microscope, exp, body=own)
    type(task).proposal_kind = MILLING_SETUP
    try:
        task.run()
    finally:
        type(task).proposal_kind = TASK_RESULT
    assert lamella.proposals[ROUGH].kind == MILLING_SETUP
    assert lamella.proposals[ROUGH].current.via == "workflow"


def test_a_rerun_supersedes_a_decided_result(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, review=True)
    lamella = exp.positions[0]
    _task(microscope, exp).run()
    exp.decide(
        lamella.id,
        ROUGH,
        Decision(outcome=DecisionOutcome.Confirmed, author="human:op", values={}),
    )
    decided = lamella.proposals[ROUGH]

    _task(microscope, exp).run()

    fresh = lamella.proposals[ROUGH]
    assert fresh is not decided and fresh.pending
    assert fresh.superseded == [decided]
