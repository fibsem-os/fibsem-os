"""A decision names the proposal it decides, not only the run it came from.

``Experiment.decide`` refuses a stale decision -- one made on something that is
no longer what the record holds. It used to tell by the run: the decision's
``task_id`` against the proposal's. That is enough while a task proposes once
per run, and it is not once a task asks questions during a run (FIB-1025): an
undercut asks four or more detections, every one of them from the same run. A
decision naming the run would be accepted against any of them.

So a proposal has an id of its own, and a decision that carries it is checked
against that. One that does not -- every record and every caller written before
this -- is checked by the run, exactly as before.
"""

import os

import pytest
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.proposals import (
    DETECTION,
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

TASK = "Mill Undercut"
RUN = "run-1"


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="id-exp")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    lamella = exp.positions[0]
    lamella.task_state.name = TASK
    lamella.task_state.task_id = RUN
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    return exp


def _question(x: float = 10.0) -> Proposal:
    return Proposal(
        kind=DETECTION,
        values={"features": [{"name": "LamellaCentre", "px": Point(x, 20)}]},
        provenance={"task_id": RUN, "proposer": "segmentation model"},
    )


def _ask(experiment, proposal: Proposal) -> Proposal:
    assert experiment.ask_proposal(experiment.positions[0].id, TASK, proposal)
    return proposal


def _confirm(experiment, px: Point, **named) -> "object":
    lamella = experiment.positions[0]
    return experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"features": [{"name": "LamellaCentre", "px": px}]},
            **named,
        ),
    )


# ---------------------------------------------------------------------------
# The id
# ---------------------------------------------------------------------------


def test_every_proposal_has_a_name_of_its_own():
    first, second = _question(), _question()

    assert first.id and second.id
    assert first.id != second.id, "the same values from the same run, twice"


def test_the_id_is_on_the_record(experiment):
    proposal = _ask(experiment, _question())

    again = Proposal.from_dict(proposal.to_dict())

    assert again.id == proposal.id


def test_a_record_saved_before_ids_gets_the_same_one_every_time():
    """Derived, not minted: the app and a monitor, this session and the next,
    have to agree on what an old proposal is called without either saving."""
    stored = _question().to_dict()
    del stored["id"]

    first, second = Proposal.from_dict(stored), Proposal.from_dict(stored)

    assert first.id and first.id == second.id
    other = dict(stored, created_at=stored["created_at"] + 1.0)
    assert Proposal.from_dict(other).id != first.id


def test_a_decision_records_the_proposal_it_named(experiment):
    proposal = _ask(experiment, _question())

    assert _confirm(experiment, Point(12, 26), proposal_id=proposal.id).applied

    assert proposal.current.proposal_id == proposal.id
    assert proposal.current.task_id == RUN, "and the run, which it did not pass"
    assert Decision.from_dict(proposal.current.to_dict()).proposal_id == proposal.id


# ---------------------------------------------------------------------------
# Two questions in one run
# ---------------------------------------------------------------------------


def test_a_decision_on_the_first_question_does_not_land_on_the_second(experiment):
    """The case the run could not tell apart. Somebody is shown the first
    question; it is withdrawn and the task asks again, same run; their answer
    arrives. It is about a picture that is no longer the one on the record."""
    first = _ask(experiment, _question(10))
    experiment.withdraw_proposal(experiment.positions[0].id, TASK, "asking again")
    second = _ask(experiment, _question(30))

    result = _confirm(experiment, Point(11, 21), proposal_id=first.id)

    assert not result.applied and result.error_type == "stale_review"
    assert second.pending, "the question that is up was not answered by it"


def test_naming_the_run_alone_cannot_tell_them_apart(experiment):
    """Why the id exists, pinned: the same late answer, named by the run as
    every caller used to, is accepted against the second question."""
    _ask(experiment, _question(10))
    experiment.withdraw_proposal(experiment.positions[0].id, TASK, "asking again")
    second = _ask(experiment, _question(30))

    result = _confirm(experiment, Point(11, 21), task_id=RUN)

    assert result.applied
    assert second.current.values["features"][0]["px"] == Point(11, 21)


def test_the_question_that_is_up_is_answered_by_its_own_id(experiment):
    _ask(experiment, _question(10))
    experiment.withdraw_proposal(experiment.positions[0].id, TASK, "asking again")
    second = _ask(experiment, _question(30))

    assert _confirm(experiment, Point(31, 41), proposal_id=second.id).applied


# ---------------------------------------------------------------------------
# What came before still works, and still refuses what it refused
# ---------------------------------------------------------------------------


def test_a_decision_that_names_neither_is_refused(experiment):
    _ask(experiment, _question())

    result = _confirm(experiment, Point(12, 26))

    assert not result.applied and result.error_type == "missing_field"


def test_a_decision_naming_another_run_is_still_stale(experiment):
    _ask(experiment, _question())

    result = _confirm(experiment, Point(12, 26), task_id="run-0")

    assert not result.applied and result.error_type == "stale_review"


def test_the_id_wins_when_both_are_given(experiment):
    """A caller that passes the run as well is not excused by it."""
    first = _ask(experiment, _question(10))
    experiment.withdraw_proposal(experiment.positions[0].id, TASK, "asking again")
    _ask(experiment, _question(30))

    result = _confirm(experiment, Point(11, 21), proposal_id=first.id, task_id=RUN)

    assert not result.applied and result.error_type == "stale_review"
    assert "asked again since you looked" in result.reason, "same run: not a re-run"


def test_a_withdrawal_names_the_proposal_it_closed(experiment):
    proposal = _ask(experiment, _question())

    experiment.withdraw_proposal(experiment.positions[0].id, TASK, "stopped")

    assert proposal.current.proposal_id == proposal.id


# ---------------------------------------------------------------------------
# Withdrawn is not something a decider can say
# ---------------------------------------------------------------------------


def test_withdrawn_is_refused_as_a_decision(experiment):
    """``decide`` takes an outcome by name from the wire, and Withdrawn is a
    name. Let through, its ending would fail the task with "Rejected by ..."
    -- a rejection nobody made."""
    lamella = experiment.positions[0]
    proposal = _ask(experiment, _question())

    result = experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Withdrawn,
            author="agent:x",
            reason="taking it back",
            proposal_id=proposal.id,
        ),
    )

    assert not result.applied and result.error_type == "invalid_value"
    assert proposal.pending
    assert lamella.task_state.status is AutoLamellaTaskStatus.InProgress


def test_a_plain_result_is_named_the_same_way(experiment):
    """Not only questions: the id is on every proposal, so the Review tab and
    the producer's own decision name it too."""
    lamella = experiment.positions[0]
    lamella.task_state.status = AutoLamellaTaskStatus.Completed
    proposal = Proposal(kind=TASK_RESULT, provenance={"task_id": RUN})
    lamella.proposals[TASK] = [proposal]

    result = experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            proposal_id=proposal.id,
        ),
    )

    assert result.applied, result.reason
