"""A proposal about the experiment itself, not one of its items (an agent's
request): asked, decided by id and saved like any other; one still waiting
when the app closes is withdrawn when the experiment is read again."""

import os

import pytest

from fibsem.acting import Actor
from fibsem.workflows.event_recording import ProposalTap
from fibsem.workflows.events import EventBuffer
from fibsem.workflows.experiment import FibsemExperiment
from fibsem.workflows.proposals import ACTIONS, Decision, DecisionOutcome, Proposal

CALLS = [{"tool": "stage_move_relative", "arguments": {"dx": 1e-6}}]


@pytest.fixture
def experiment(tmp_path):
    exp = FibsemExperiment.create(tmp_path, "plain")
    assert os.path.isdir(exp.path)
    return exp


def _ask(experiment):
    proposal = Proposal(
        kind=ACTIONS,
        values={"actions": CALLS},
        provenance={"reason": "the stage drifted"},
        asked_by=Actor.agent("claude"),
    )
    assert experiment.ask_proposal(None, "", proposal)
    return proposal


def _decide(experiment, proposal, outcome, **kw):
    return experiment.decide(
        None,
        "",
        Decision(
            outcome=outcome,
            actor=Actor.operator("op"),
            proposal_id=proposal.id,
            **kw,
        ),
    )


def test_it_is_decided_by_id_and_once(experiment):
    proposal = _ask(experiment)

    assert _decide(experiment, proposal, DecisionOutcome.Confirmed).applied
    again = _decide(experiment, proposal, DecisionOutcome.Rejected, reason="no")

    assert proposal.current.outcome is DecisionOutcome.Confirmed
    assert (again.applied, again.error_type) == (False, "stale_proposal")


def test_it_is_confirmed_as_it_stands(experiment):
    proposal = _ask(experiment)

    edited = _decide(
        experiment, proposal, DecisionOutcome.Confirmed, values={"actions": []}
    )

    assert not edited.applied and proposal.pending


def test_a_withdrawn_one_takes_no_decision(experiment):
    proposal = _ask(experiment)
    experiment.withdraw_experiment_proposal(
        proposal.id, "withdrawn by the agent: moved", Actor.agent("claude")
    )

    late = _decide(experiment, proposal, DecisionOutcome.Confirmed)

    assert proposal.withdrawn and not late.applied


def test_it_is_saved_and_one_still_waiting_is_withdrawn_on_load(experiment):
    decided = _ask(experiment)
    _decide(experiment, decided, DecisionOutcome.Rejected, reason="not now")
    waiting = _ask(experiment)
    experiment.save()

    read = FibsemExperiment.load(experiment.path)

    by_id = {p.id: p for p in read.proposals}
    assert by_id[decided.id].current.reason == "not now"
    assert by_id[waiting.id].withdrawn
    assert by_id[waiting.id].values["actions"] == CALLS


def test_asking_and_deciding_are_on_the_record_as_the_asker(experiment):
    buffer = EventBuffer()
    tap = ProposalTap(buffer, experiment)  # noqa: F841 - its slots are weak
    proposal = _ask(experiment)
    _decide(experiment, proposal, DecisionOutcome.Confirmed)

    events = buffer.events_since(0)["events"]

    asked, decided = events
    assert (asked["kind"], asked["actor"]) == ("proposal_asked", "agent")
    assert asked["payload"]["item"] is None
    assert asked["payload"]["reason"] == "the stage drifted"
    assert decided["kind"] == "proposal_decided"
    assert decided["payload"]["outcome"] == "Confirmed"
