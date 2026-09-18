"""A task may be answered while it runs, when it is the one asking.

``Experiment.decide`` refuses a decision on a task that is running, twice over:
once because the proposal is from the run in progress, and once because the
decision carries values that running task may be reading (FIB-1008). Both are
right for a result somebody wandered in and decided under a live run.

They are wrong for the one case they also catch: an in-run question, where the
task raised the proposal itself and is stopped on a future until someone
answers it. It is not reading those values -- it is waiting to be told them.

So ``Experiment.asking`` marks the proposal for as long as the question is up,
and the refusals stand down for that proposal alone. These tests are mostly
about how narrow "that proposal alone" is.
"""

import os

import pytest
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.proposals import (
    POINT_OF_INTEREST,
    TASK_RESULT,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
)
from fibsem.structures import MicroscopeState, Point

TASK = "Setup Lamella Position"
OTHER = "Mill Rough"


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="asking-exp")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    return exp


def _running(experiment, task_name=TASK, task_id="run-1"):
    """The task is mid-flight, as it is when it raises an in-run question."""
    lamella = experiment.positions[0]
    lamella.task_state.name = task_name
    lamella.task_state.task_id = task_id
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    return lamella


def _proposal(lamella, task_name=TASK, task_id="run-1", kind=POINT_OF_INTEREST):
    proposal = Proposal(
        kind=kind,
        values={"poi": Point(0.0, 0.0)} if kind is POINT_OF_INTEREST else {},
        provenance={"task_id": task_id, "proposer": "detection"},
    )
    lamella.proposals[task_name] = proposal
    return proposal


def _answer(experiment, lamella, task_name=TASK, values=None, task_id="run-1"):
    return experiment.decide(
        lamella.id,
        task_name,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0.0)} if values is None else values,
            task_id=task_id,
        ),
    )


# ---------------------------------------------------------------------------
# The carve-out
# ---------------------------------------------------------------------------


def test_a_running_task_can_be_answered_while_it_is_asking(experiment):
    lamella = _running(experiment)
    _proposal(lamella)

    with experiment.asking(lamella.id, TASK):
        result = _answer(experiment, lamella)

    assert result.applied, result.reason


def test_without_the_mark_the_same_decision_is_refused(experiment):
    """The refusal is not gone, it stands down for a question that is up."""
    lamella = _running(experiment)
    _proposal(lamella)

    result = _answer(experiment, lamella)

    assert not result.applied and result.running


def test_a_question_carrying_no_values_is_allowed_on_the_same_grounds(experiment):
    """Phrased about the waiting, not about the values -- which is what keeps
    the door open for the yes/no prompts that are out of scope for now."""
    lamella = _running(experiment)
    _proposal(lamella, kind=TASK_RESULT)

    with experiment.asking(lamella.id, TASK):
        result = experiment.decide(
            lamella.id,
            TASK,
            Decision(
                outcome=DecisionOutcome.Confirmed, author="human:op", task_id="run-1"
            ),
        )

    assert result.applied, result.reason


# ---------------------------------------------------------------------------
# How narrow it is
# ---------------------------------------------------------------------------


def test_another_proposal_on_the_same_item_stays_refused(experiment):
    """The carve-out is on the proposal, not the item: the task will resume
    and may read anything else on this lamella."""
    lamella = _running(experiment, task_name=OTHER, task_id="run-2")
    _proposal(lamella, task_name=OTHER, task_id="run-2")
    _proposal(lamella, task_name=TASK, task_id="run-1")

    with experiment.asking(lamella.id, OTHER):
        result = _answer(experiment, lamella, task_name=TASK)

    assert not result.applied and result.running
    assert "stop it before writing" in result.reason


def test_a_mark_left_on_an_earlier_run_does_not_excuse_the_current_one(experiment):
    """The task re-ran. A decision naming the old run is stale anyway, but the
    mark must not be what lets it through."""
    lamella = _running(experiment, task_id="run-2")
    proposal = _proposal(lamella, task_id="run-1")
    proposal.asking = True

    result = _answer(experiment, lamella, task_id="run-1")

    assert not result.applied and result.running


def test_the_mark_is_cleared_when_the_ask_ends(experiment):
    lamella = _running(experiment)
    proposal = _proposal(lamella)

    with experiment.asking(lamella.id, TASK):
        assert proposal.asking

    assert not proposal.asking


def test_an_ask_that_raises_still_closes_the_question(experiment):
    """The part that matters. An abort mid-question must not leave behind a
    mark that lets a later decision land on a task that really is reading."""
    lamella = _running(experiment)
    proposal = _proposal(lamella)

    with pytest.raises(RuntimeError):
        with experiment.asking(lamella.id, TASK):
            raise RuntimeError("the operator stopped the run")

    assert not proposal.asking
    assert not _answer(experiment, lamella).applied


def test_answering_closes_the_question_immediately(experiment):
    """Cleared as the decision lands, not only when the asker unwinds, so
    there is no window where a second decision is still excused."""
    lamella = _running(experiment)
    proposal = _proposal(lamella)

    with experiment.asking(lamella.id, TASK):
        assert _answer(experiment, lamella).applied
        assert not proposal.asking


def test_withdrawing_closes_it_too(experiment):
    lamella = _running(experiment)
    proposal = _proposal(lamella)

    with experiment.asking(lamella.id, TASK):
        experiment.withdraw_proposal(lamella.id, TASK, "the run was stopped")
        assert not proposal.asking


def test_asking_about_something_that_is_not_there_is_not_an_error(experiment):
    """The caller is asking either way; a missing record is not a reason to
    refuse to raise the question."""
    lamella = experiment.positions[0]

    with experiment.asking(lamella.id, "No Such Task") as proposal:
        assert proposal is None
    with experiment.asking("no-such-item", TASK) as proposal:
        assert proposal is None


# ---------------------------------------------------------------------------
# It is a question, not a record
# ---------------------------------------------------------------------------


def test_the_mark_is_never_written_to_the_file(experiment):
    """A question exists only while something waits on it. A mark that
    survived a reload would claim a waiter that is gone."""
    lamella = _running(experiment)
    proposal = _proposal(lamella)

    with experiment.asking(lamella.id, TASK):
        assert "asking" not in proposal.to_dict()
        assert not Proposal.from_dict(proposal.to_dict()).asking
