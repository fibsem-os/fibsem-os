"""A task may be answered while it runs, when it is the one asking.

``Experiment.decide`` refuses a decision on a task that is running, twice over:
once because the proposal is from the run in progress, and once because the
decision carries values that running task may be reading (FIB-1008). Both are
right for a result somebody wandered in and decided under a live run.

They are wrong for the one case they also catch: an in-run question, where the
task raised the proposal itself and is stopped on a future until someone
answers it. It is not reading those values -- it is waiting to be told them.

So ``Experiment.ask_proposal`` marks the proposal it records as the one the run
is parked on, an answer or a withdrawal clears the mark, and the refusals stand
down for that proposal alone. These tests are mostly about how narrow "that
proposal alone" is.
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
    lamella.proposals[task_name] = [proposal]
    return proposal


def _ask(experiment, lamella, task_name=TASK, task_id="run-1", kind=POINT_OF_INTEREST):
    """The same proposal, recorded the way an in-run question is: through
    ``ask_proposal``, which is the one place the mark is set."""
    proposal = _proposal(lamella, task_name, task_id, kind)
    del lamella.proposals[task_name]
    assert experiment.ask_proposal(lamella.id, task_name, proposal)
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
    _ask(experiment, lamella)

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
    _ask(experiment, lamella, kind=TASK_RESULT)

    result = experiment.decide(
        lamella.id,
        TASK,
        Decision(outcome=DecisionOutcome.Confirmed, author="human:op", task_id="run-1"),
    )

    assert result.applied, result.reason


# ---------------------------------------------------------------------------
# How narrow it is
# ---------------------------------------------------------------------------


def test_another_proposal_on_the_same_item_stays_refused(experiment):
    """The carve-out is on the proposal, not the item: the task will resume
    and may read anything else on this lamella."""
    lamella = _running(experiment, task_name=OTHER, task_id="run-2")
    _ask(experiment, lamella, task_name=OTHER, task_id="run-2")
    _proposal(lamella, task_name=TASK, task_id="run-1")

    result = _answer(experiment, lamella, task_name=TASK)

    assert not result.applied and result.running
    assert "stop it before writing" in result.reason


def test_an_open_value_confirmed_as_it_stands_lands_while_the_item_is_busy(experiment):
    """Setup's point was written through when Setup ended; the item is now
    running Mill Fiducial. Confirming the point as it stands writes nothing
    again, so it is a look and lands. Moving it is a write and waits."""
    lamella = _running(experiment, task_name=OTHER, task_id="run-2")
    proposal = _proposal(lamella, task_name=TASK, task_id="run-1")
    assert experiment._is_open(lamella, TASK, proposal)

    look = _answer(experiment, lamella, task_name=TASK, values=dict(proposal.values))
    assert look.applied, look.reason
    assert proposal.current.outcome is DecisionOutcome.Confirmed
    assert lamella.poi == Point(0.0, 0.0), "nothing written"

    proposal = _proposal(lamella, task_name=TASK, task_id="run-1")
    moved = _answer(experiment, lamella, task_name=TASK)
    assert not moved.applied and moved.running
    assert "stop it before writing" in moved.reason


def test_a_mark_left_on_an_earlier_run_does_not_excuse_the_current_one(experiment):
    """The task re-ran. A decision naming the old run is stale anyway, but the
    mark must not be what lets it through."""
    lamella = _running(experiment, task_id="run-2")
    proposal = _proposal(lamella, task_id="run-1")
    proposal.asking = True

    result = _answer(experiment, lamella, task_id="run-1")

    assert not result.applied and result.running


def test_a_question_taken_back_is_closed_to_a_later_decision(experiment):
    """The part that matters. A run that stops mid-question must not leave
    behind a mark that lets a later decision land on a task that really is
    reading -- nor an open question that anybody can still answer."""
    lamella = _running(experiment)
    proposal = _ask(experiment, lamella)

    experiment.withdraw_proposal(lamella.id, TASK, "the operator stopped the run")

    assert not proposal.asking
    result = _answer(experiment, lamella)
    assert not result.applied and result.error_type == "stale_review"


def test_answering_closes_the_question_immediately(experiment):
    """Cleared as the decision lands, not only when the asker unwinds, so
    there is no window where a second decision is still excused."""
    lamella = _running(experiment)
    proposal = _ask(experiment, lamella)

    assert _answer(experiment, lamella).applied
    assert not proposal.asking


def test_withdrawing_closes_it_too(experiment):
    lamella = _running(experiment)
    proposal = _ask(experiment, lamella)

    experiment.withdraw_proposal(lamella.id, TASK, "the run was stopped")

    assert not proposal.asking


def test_a_question_about_an_item_that_is_not_there_is_not_recorded(experiment):
    """Said, not raised: the asker decides what a question it could not record
    means, and nothing is left half-written."""
    proposal = Proposal(kind=POINT_OF_INTEREST, values={"poi": Point(0.0, 0.0)})

    assert not experiment.ask_proposal("no-such-item", TASK, proposal)
    assert not proposal.asking


# ---------------------------------------------------------------------------
# It is a question, not a record
# ---------------------------------------------------------------------------


def test_the_mark_is_never_written_to_the_file(experiment):
    """A question exists only while something waits on it. A mark that
    survived a reload would claim a waiter that is gone."""
    lamella = _running(experiment)
    proposal = _ask(experiment, lamella)

    assert proposal.asking
    assert "asking" not in proposal.to_dict()
    assert not Proposal.from_dict(proposal.to_dict()).asking
