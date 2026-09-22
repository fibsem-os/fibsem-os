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
    DETECTION,
    POINT_OF_INTEREST,
    TASK_RESULT,
    Decision,
    DecisionOutcome,
    Proposal,
    TaskResultProposer,
)
from fibsem.applications.autolamella.structures import (
    Attention,
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
from fibsem.structures import Point

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
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    task = _task(microscope, exp)
    lamella = exp.positions[0]

    task.run()

    proposal = lamella.proposal(ROUGH)
    assert proposal.kind == TASK_RESULT and proposal.pending
    assert proposal.values == {}
    p = proposal.provenance
    assert p["proposer"] == ROUGH and p["task_name"] == ROUGH
    assert p["status"] == "Completed" and p["failure"] == ""
    assert p["reference_image"] == f"ref_{ROUGH}_final_res_01_ib.tif"
    assert p["reference_image_eb"] == f"ref_{ROUGH}_final_res_01_eb.tif"
    assert p["ended_at"] >= p["started_at"] > 0
    assert lamella.is_awaiting_decision(ROUGH), "ran, not finished"
    assert task.task_manager._defer_reason(lamella, POLISH) == "awaiting_decision"

    result = exp.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            task_id=lamella.proposal(ROUGH).task_id,
        ),
    )
    assert result.applied and result.synced_tasks == []
    assert task.task_manager._defer_reason(lamella, POLISH) is None


def test_the_result_images_mapping_decides_which_roles_the_proposal_points_at(
    microscope, tmp_path
):
    """A task names its result images by provenance key and output role; the
    proposal takes the last file under each mapped role, and carries only the
    keys the mapping names. How a grid task points at its overview."""
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]

    def body():
        lamella.task_state.outputs["overview_sem"] = ["wide.tif", "tight.tif"]

    task = _task(microscope, exp, body=body)
    task.result_images = {"reference_image": "overview_sem"}

    task.run()

    p = lamella.proposal(ROUGH).provenance
    assert p["reference_image"] == "tight.tif", "the last file under the role"
    assert "reference_image_eb" not in p, "only the keys the mapping names"


def test_a_failed_task_records_its_result_with_the_failure(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]

    def boom():
        raise RuntimeError("stage timeout")

    task = _task(microscope, exp, body=boom)
    with pytest.raises(RuntimeError):
        task.run()

    proposal = lamella.proposal(ROUGH)
    assert proposal.kind == TASK_RESULT and proposal.pending
    assert proposal.provenance["status"] == "Failed"
    assert proposal.provenance["failure"] == "stage timeout"
    assert proposal.provenance["reference_image"], "what was acquired before it failed"
    assert lamella.task_state.status is AutoLamellaTaskStatus.Failed


def test_not_gated_the_result_is_recorded_and_the_run_goes_on(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    task = _task(microscope, exp)
    lamella = exp.positions[0]

    task.run()

    proposal = lamella.proposal(ROUGH)
    assert proposal.kind == TASK_RESULT
    # Nobody decides it: open, listed to check, and nothing waits on it.
    assert proposal.pending and not proposal.to_check
    assert task.task_manager._defer_reason(lamella, POLISH) is None
    assert [t for _i, t, _p in exp.proposals_to_check()] == [ROUGH]
    assert exp.pending_proposals() == []
    # Its consumer starting closes it as used-unreviewed, by the producer's name.
    assert exp.expire_open(lamella.id, ROUGH, f"{POLISH} started") == 1
    assert proposal.unreviewed and proposal.to_check
    assert str(proposal.current.author) == f"auto:{ROUGH}"
    assert f"{POLISH} started" in proposal.current.reason


def test_without_the_flag_the_result_is_recorded_but_never_gates(microscope, tmp_path):
    """The flag hides the Review surface, not the record."""
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    task = _task(microscope, exp)
    task.task_manager.review_enabled = False
    task.run()
    lamella = exp.positions[0]
    proposal = lamella.proposal(ROUGH)
    assert proposal.kind == TASK_RESULT and proposal.pending, "open, not decided"
    assert task.task_manager._defer_reason(lamella, POLISH) is None
    assert not lamella.is_awaiting_decision(ROUGH)


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

    for cls in (MillFiducialTask, AcquireReferenceImageTask, MillRoughTask):
        assert cls.proposer.kind == TASK_RESULT, "a bad fiducial is gateable"
    assert SelectMillingPositionTask.proposer.kind == POINT_OF_INTEREST
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    task = _task(microscope, exp)
    type(task).proposer = None
    try:
        task.run()
    finally:
        type(task).proposer = TaskResultProposer()
    assert exp.positions[0].proposals == {}


class _SitePicker:
    """A proposer swapped onto a task type: its kind, its values, its name."""

    kind = POINT_OF_INTEREST
    name = "site-picker"
    version = 7

    def __init__(self, values):
        self._values = values

    def propose(self, task):
        return Proposal(
            kind=self.kind,
            values=self._values,
            confidence=0.5,
            provenance={"model": "m"},
        )


def test_a_swapped_proposer_records_its_kind_and_the_base_fills_the_result(
    microscope, tmp_path
):
    """Swapping the proposer changes what is proposed and nothing else: the
    kind and values are the proposer's, the result part -- task, status,
    times, images -- is the base's for every kind, and the producer's own
    confirmation is signed with the proposer's name."""
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]
    task = _task(microscope, exp)
    type(task).proposer = _SitePicker({"poi": Point(1e-6, 2e-6)})
    try:
        task.run()
    finally:
        type(task).proposer = TaskResultProposer()
    proposal = lamella.proposal(ROUGH)
    assert proposal.kind == POINT_OF_INTEREST
    assert proposal.values == {"poi": Point(1e-6, 2e-6)} and proposal.confidence == 0.5
    p = proposal.provenance
    assert p["proposer"] == "site-picker" and p["version"] == 7 and p["model"] == "m"
    assert p["task_name"] == ROUGH and p["status"] == "Completed"
    assert p["reference_image"] == f"ref_{ROUGH}_final_res_01_ib.tif"
    assert proposal.pending, "open until something uses it"
    assert lamella.poi == Point(1e-6, 2e-6), "and live from the moment it was proposed"


def test_a_proposer_may_not_carry_a_value_its_kind_does_not_register(
    microscope, tmp_path
):
    exp = _experiment(tmp_path, microscope)
    task = _task(microscope, exp)
    type(task).proposer = _SitePicker({"sites": []})
    try:
        with pytest.raises(ValueError, match="does not carry"):
            task.run()
    finally:
        type(task).proposer = TaskResultProposer()


def test_a_question_answered_during_the_run_is_not_logged_as_a_rerun(
    microscope, tmp_path, caplog
):
    """A question asked mid-run (FIB-1025) shares the slot with the run's own
    result, so at the end of the task it goes under the result like anything
    else the slot held. It is kept -- but it is the same run, and a log that
    says "re-run" puts a run in the record that never happened."""
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]
    asked = {}

    def ask_and_be_answered():
        question = Proposal(
            kind=DETECTION,
            values={"features": [{"name": "LamellaCentre", "px": Point(10, 20)}]},
            provenance={"task_id": lamella.task_state.task_id, "in_run": True},
        )
        assert exp.ask_proposal(lamella.id, ROUGH, question)
        result = exp.decide(
            lamella.id,
            ROUGH,
            Decision(
                outcome=DecisionOutcome.Confirmed,
                author="human:op",
                values=dict(question.values),
                proposal_id=question.id,
            ),
        )
        assert result.applied, result.reason
        asked["question"] = question

    with caplog.at_level("INFO"):
        _task(microscope, exp, body=ask_and_be_answered).run()

    result = lamella.proposal(ROUGH)
    assert result.kind == TASK_RESULT and result.pending
    assert lamella.proposals[ROUGH] == [asked["question"], result], "the answer is kept"
    assert "asked a question during this run" in caplog.text
    assert "re-run" not in caplog.text


def test_a_rerun_supersedes_a_decided_result(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]
    _task(microscope, exp).run()
    exp.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            task_id=lamella.proposal(ROUGH).task_id,
        ),
    )
    decided = lamella.proposal(ROUGH)

    _task(microscope, exp).run()

    fresh = lamella.proposal(ROUGH)
    assert fresh is not decided and fresh.pending
    assert lamella.proposals[ROUGH] == [decided, fresh]
