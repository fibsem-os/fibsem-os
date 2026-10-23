"""Deciding a proposal, on an experiment whose items are not lamellae."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from fibsem.acting import Actor
from fibsem.workflows.deciding import ProposalDecisions
from fibsem.workflows.experiment import BaseExperiment
from fibsem.workflows.item import ExperimentItem
from fibsem.workflows.proposals import (
    Decision,
    DecisionOutcome,
    Proposal,
    ProposalKind,
    register_proposal_kind,
)
from fibsem.workflows.status import TaskState, TaskStatus

TASK = "Mill"
RUN = "run-1"
# A kind with nothing to carry: a yes or no.
LOOK = register_proposal_kind(ProposalKind(name="look", values=())).name


@dataclass
class Position(ExperimentItem):
    name: str
    id: str = "p1"
    description: str = ""
    task_state: TaskState = field(default_factory=TaskState)
    task_history: List[TaskState] = field(default_factory=list)
    proposals: List[Proposal] = field(default_factory=list)


class Volume(BaseExperiment, ProposalDecisions):
    app = "volume"

    def __init__(self, path: Path) -> None:
        super().__init__(path=path, name="volume")
        self.positions: List[Any] = []

    def items(self):
        return list(self.positions)

    def save(self, save_protocol: bool = False) -> None:
        pass


def _finished(tmp_path) -> tuple:
    """A position whose Mill run finished with a proposal on it not decided."""
    experiment = Volume(tmp_path)
    position = Position("P1")
    run = TaskState(name=TASK, task_id=RUN, status=TaskStatus.Completed)
    position.task_history.append(run)
    proposal = position.record_proposal(
        TASK, Proposal(kind=LOOK, provenance={"task_id": RUN})
    )
    experiment.positions.append(position)
    return experiment, position, proposal


def _decision(proposal: Proposal, outcome: DecisionOutcome, reason: str = ""):
    return Decision(
        outcome=outcome,
        actor=Actor.operator("Ada"),
        reason=reason,
        proposal_id=proposal.id,
    )


def test_confirming_records_the_decision_and_leaves_the_run_as_it_was(tmp_path):
    experiment, position, proposal = _finished(tmp_path)
    decided = []
    experiment.decided.connect(lambda *args: decided.append(args))

    result = experiment.decide(
        position.id, TASK, _decision(proposal, DecisionOutcome.Confirmed)
    )

    assert result.applied, result.reason
    assert position.task_history[-1].status is TaskStatus.Completed
    assert proposal.decisions[-1].actor.name == "Ada"
    assert decided == [(position.id, TASK)]
    assert not proposal.pending


def test_a_reject_needs_a_reason(tmp_path):
    experiment, position, proposal = _finished(tmp_path)

    result = experiment.decide(
        position.id, TASK, _decision(proposal, DecisionOutcome.Rejected)
    )

    assert not result.applied
    assert proposal.pending
    assert position.task_history[-1].status is TaskStatus.Completed


def test_a_decision_runs_where_the_session_edits_the_experiment(tmp_path):
    """No Qt hop: once a session adopts the experiment, the decision goes
    through the session's executor."""
    from fibsem.workflows.session import AppSession, LockedExecutor

    experiment, position, proposal = _finished(tmp_path)
    ran = []
    locked = LockedExecutor()

    def executor(call):
        ran.append(call)
        return locked(call)

    session = AppSession(executor=executor)
    Path(experiment.path).mkdir(parents=True, exist_ok=True)
    session.adopt(experiment)
    decided = []
    experiment.decided.connect(lambda *args: decided.append(args))

    result = experiment.decide(
        position.id, TASK, _decision(proposal, DecisionOutcome.Confirmed)
    )

    assert result.applied, result.reason
    assert len(ran) == 1
    assert decided == [(position.id, TASK)]
