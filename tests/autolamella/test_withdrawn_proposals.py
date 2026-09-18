"""A question can be closed without being answered.

``Confirmed`` and ``Rejected`` are answers: a decider looked and said
something. ``Withdrawn`` is not -- whatever asked the question is gone, so
there is nothing to read and nothing to compare a proposal against. It exists
because an in-run question (FIB-1025) is raised mid-task and has to be closed
when that task fails, the run stops or the operator aborts.

The risk is that it quietly reads as an answer, so most of these are about
what must *not* treat it as one.

Real Experiment and real Lamella; no microscope is needed to decide or to
withdraw.
"""

import os

import pytest
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.proposals import (
    POINT_OF_INTEREST,
    AuthorKind,
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


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="withdraw-exp")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    return exp


def _asked(experiment, values=None) -> Proposal:
    """A proposal recorded by a task that is still running, as an in-run
    question is: the run it names is the run in progress."""
    lamella = experiment.positions[0]
    lamella.task_state.name = TASK
    lamella.task_state.task_id = "run-1"
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    proposal = Proposal(
        kind=POINT_OF_INTEREST,
        values=values if values is not None else {"poi": Point(0.0, 0.0)},
        provenance={"task_id": "run-1", "proposer": "detection"},
    )
    lamella.proposals[TASK] = proposal
    return proposal


# ---------------------------------------------------------------------------
# Withdrawing
# ---------------------------------------------------------------------------


def test_withdrawing_closes_a_question_nobody_answered(experiment):
    lamella = experiment.positions[0]
    proposal = _asked(experiment)

    result = experiment.withdraw_proposal(lamella.id, TASK, "Mill Rough failed")

    assert result.applied
    assert not proposal.pending, "it is closed"
    assert proposal.withdrawn
    assert proposal.current.outcome is DecisionOutcome.Withdrawn
    assert proposal.current.reason == "Mill Rough failed"


def test_it_is_authored_automatically_because_nobody_decided_it(experiment):
    lamella = experiment.positions[0]
    proposal = _asked(experiment)

    experiment.withdraw_proposal(lamella.id, TASK, "the run was stopped")

    assert proposal.current.author.kind is AuthorKind.automated


def test_withdrawing_works_while_the_task_is_running(experiment):
    """The whole point: a question is withdrawn exactly when its task is
    failing, which is the state ``decide`` refuses a decision in."""
    lamella = experiment.positions[0]
    _asked(experiment)
    assert lamella.task_state.status is AutoLamellaTaskStatus.InProgress

    assert experiment.withdraw_proposal(lamella.id, TASK, "aborted").applied


def test_an_answered_proposal_cannot_be_withdrawn(experiment):
    lamella = experiment.positions[0]
    proposal = _asked(experiment)
    lamella.task_state.status = AutoLamellaTaskStatus.Completed
    assert experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(0.0, 0.0)},
            task_id="run-1",
        ),
    ).applied

    result = experiment.withdraw_proposal(lamella.id, TASK, "too late")

    assert not result.applied and "already decided" in result.reason
    assert proposal.current.outcome is DecisionOutcome.Confirmed, "the answer stands"


def test_withdrawing_twice_is_refused(experiment):
    lamella = experiment.positions[0]
    proposal = _asked(experiment)
    experiment.withdraw_proposal(lamella.id, TASK, "first")

    assert not experiment.withdraw_proposal(lamella.id, TASK, "second").applied
    assert len(proposal.decisions) == 1


# ---------------------------------------------------------------------------
# What must not treat it as an answer
# ---------------------------------------------------------------------------


def test_a_withdrawn_question_cannot_be_answered_afterwards(experiment):
    """Answering now would be written against a run that is over."""
    lamella = experiment.positions[0]
    _asked(experiment)
    experiment.withdraw_proposal(lamella.id, TASK, "Mill Rough failed")
    lamella.task_state.status = AutoLamellaTaskStatus.Failed

    result = experiment.decide(
        lamella.id,
        TASK,
        Decision(outcome=DecisionOutcome.Confirmed, author="human:op", task_id="run-1"),
    )

    assert not result.applied and result.error_type == "stale_review"
    assert "withdrawn" in result.reason


def test_it_is_never_to_check(experiment):
    """Nobody should be asked to acknowledge a question that was taken back:
    a withdrawal is authored automatically, which is otherwise exactly what
    puts a proposal in the to-check group."""
    lamella = experiment.positions[0]
    proposal = _asked(experiment)

    experiment.withdraw_proposal(lamella.id, TASK, "the run was stopped")

    assert not proposal.to_check
    assert proposal not in [p for _, _, p in experiment.proposals_to_check()]


def test_it_leaves_the_waiting_list(experiment):
    lamella = experiment.positions[0]
    proposal = _asked(experiment)
    assert proposal in [p for _, _, p in experiment.pending_proposals()]

    experiment.withdraw_proposal(lamella.id, TASK, "aborted")

    assert proposal not in [p for _, _, p in experiment.pending_proposals()]


def test_it_carries_no_delta_to_compare(experiment):
    """Anything measuring how often a decider changed the proposal has to skip
    these: there is no decided value, and counting one as agreement inflates
    the number meant to detect rubber-stamping."""
    lamella = experiment.positions[0]
    proposal = _asked(experiment, values={"poi": Point(1e-6, 2e-6)})

    experiment.withdraw_proposal(lamella.id, TASK, "aborted")

    assert proposal.delta() == {}
    assert proposal.applied is None, "nothing was written through"


def test_it_does_not_satisfy_what_requires_the_task(experiment):
    """Gating reads the *task's* status, never the proposal's outcome -- so a
    withdrawal cannot unblock anything by itself. Pinned here because the
    guarantee is structural and easy to lose: a later reader who gates on
    "the proposal is no longer pending" would break it silently."""
    lamella = experiment.positions[0]
    _asked(experiment)

    experiment.withdraw_proposal(lamella.id, TASK, "Mill Rough failed")
    lamella.task_state.status = AutoLamellaTaskStatus.Failed
    lamella.set_task_status(TASK, AutoLamellaTaskStatus.Failed, "failed")

    assert not lamella.latest_run_completed(TASK)
    assert not lamella.is_awaiting_decision(TASK)


def test_it_does_not_touch_the_task_status(experiment):
    """The task has its own ending; the withdrawal is a consequence of it, not
    a cause. A confirm or a reject finishes a task that ended AwaitingDecision
    -- this must not."""
    lamella = experiment.positions[0]
    _asked(experiment)
    lamella.set_task_status(TASK, AutoLamellaTaskStatus.InProgress)

    experiment.withdraw_proposal(lamella.id, TASK, "aborted")

    assert lamella.task_state.status is AutoLamellaTaskStatus.InProgress


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_a_withdrawal_round_trips_through_the_file(experiment):
    lamella = experiment.positions[0]
    proposal = _asked(experiment)
    experiment.withdraw_proposal(lamella.id, TASK, "the run was stopped")

    back = Proposal.from_dict(proposal.to_dict())

    assert back.withdrawn
    assert back.current.reason == "the run was stopped"
    assert back.current.outcome is DecisionOutcome.Withdrawn


def test_a_pending_proposal_is_not_withdrawn(experiment):
    assert not _asked(experiment).withdrawn
