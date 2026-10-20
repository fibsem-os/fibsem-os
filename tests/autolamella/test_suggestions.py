"""A suggestion on a pending proposal: put forward, approved as the decider's
decision naming it, or declined with a reason. One per proposal, saved with it."""

import os

import pytest
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.structures import MicroscopeState, Point
from fibsem.workflows.event_recording import ProposalTap
from fibsem.workflows.events import EventBuffer
from fibsem.workflows.proposals import (
    POINT_OF_INTEREST,
    STATE,
    DecisionOutcome,
    Proposal,
    Suggestion,
)
from fibsem.workflows.status import TaskStatus

TASK = "Setup Lamella Position"


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="suggest-exp")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    return exp


def _ask(experiment, kind=POINT_OF_INTEREST):
    lamella = experiment.positions[0]
    lamella.task_state.name = TASK
    lamella.task_state.task_id = "run-1"
    lamella.task_state.status = TaskStatus.InProgress
    proposal = Proposal(
        kind=kind,
        values={"poi": Point(0.0, 0.0)} if kind == POINT_OF_INTEREST else {},
        provenance={"task_id": "run-1", "proposer": "detection"},
    )
    assert experiment.ask_proposal(lamella.id, TASK, proposal)
    return lamella, proposal


def _suggestion(**kw):
    return Suggestion(
        actor="agent:claude",
        reason=kw.pop("reason", "the lamella edge is 1 µm right"),
        values=kw.pop("values", {"poi": Point(1e-6, 0.0)}),
        **kw,
    )


def test_a_suggestion_is_put_on_the_proposal_and_decides_nothing(experiment):
    lamella, proposal = _ask(experiment)

    result = experiment.suggest(lamella.id, TASK, proposal.id, _suggestion())

    assert result.applied, result.reason
    assert proposal.pending
    assert proposal.suggestion.reason == "the lamella edge is 1 µm right"


def test_approving_decides_as_suggested_and_names_the_suggestion(experiment):
    lamella, proposal = _ask(experiment)
    suggestion = _suggestion()
    experiment.suggest(lamella.id, TASK, proposal.id, suggestion)

    result = experiment.approve_suggestion(
        lamella.id, TASK, proposal.id, actor="operator:op", via="workflow"
    )

    assert result.applied, result.reason
    decision = proposal.current
    assert decision.outcome is DecisionOutcome.Confirmed
    assert str(decision.actor) == "operator:op"
    assert decision.suggestion_id == suggestion.id
    assert decision.values["poi"] == Point(1e-6, 0.0)


def test_declining_keeps_the_proposal_pending_with_the_reason(experiment):
    lamella, proposal = _ask(experiment)
    experiment.suggest(lamella.id, TASK, proposal.id, _suggestion())

    result = experiment.decline_suggestion(
        lamella.id, TASK, proposal.id, reason="that is the trench"
    )

    assert result.applied
    assert proposal.pending
    assert proposal.suggestion.declined
    assert proposal.suggestion.declined_reason == "that is the trench"
    refused = experiment.approve_suggestion(
        lamella.id, TASK, proposal.id, actor="operator:op", via="workflow"
    )
    assert not refused.applied


def test_a_new_suggestion_replaces_the_one_there(experiment):
    lamella, proposal = _ask(experiment)
    experiment.suggest(lamella.id, TASK, proposal.id, _suggestion())
    second = _suggestion(reason="on second look, 2 µm")

    experiment.suggest(lamella.id, TASK, proposal.id, second)

    assert proposal.suggestion is second


def test_one_that_could_not_be_approved_is_refused_when_made(experiment):
    lamella, proposal = _ask(experiment)

    foreign = experiment.suggest(
        lamella.id, TASK, proposal.id, _suggestion(values={"milling_config": {}})
    )
    no_reason = experiment.suggest(
        lamella.id, TASK, proposal.id, _suggestion(reason="")
    )
    stale = experiment.suggest(lamella.id, TASK, "not-this-one", _suggestion())

    assert (foreign.applied, foreign.error_type) == (False, "invalid_value")
    assert (no_reason.applied, no_reason.error_type) == (False, "missing_field")
    assert (stale.applied, stale.error_type) == (False, "stale_proposal")
    assert proposal.suggestion is None


def test_a_decided_proposal_takes_no_suggestion(experiment):
    lamella, proposal = _ask(experiment, kind=STATE)
    experiment.suggest(
        lamella.id, TASK, proposal.id, _suggestion(values={}, reason="centred")
    )
    experiment.approve_suggestion(
        lamella.id, TASK, proposal.id, actor="operator:op", via="workflow"
    )

    late = experiment.suggest(
        lamella.id, TASK, proposal.id, _suggestion(values={}, reason="again")
    )

    assert not late.applied and late.error_type == "stale_proposal"


def test_a_rejection_can_be_suggested(experiment):
    lamella, proposal = _ask(experiment, kind=STATE)
    experiment.suggest(
        lamella.id,
        TASK,
        proposal.id,
        _suggestion(values={}, outcome=DecisionOutcome.Rejected, reason="tilted"),
    )

    experiment.approve_suggestion(
        lamella.id, TASK, proposal.id, actor="operator:op", via="workflow"
    )

    assert proposal.current.outcome is DecisionOutcome.Rejected
    assert proposal.current.reason == "tilted"


def test_the_suggestion_and_its_credit_are_saved(experiment):
    lamella, proposal = _ask(experiment)
    suggestion = _suggestion(image="Lamella/ref_ib.tif")
    experiment.suggest(lamella.id, TASK, proposal.id, suggestion)
    experiment.approve_suggestion(
        lamella.id, TASK, proposal.id, actor="operator:op", via="workflow"
    )

    read = Proposal.from_dict(proposal.to_dict())

    assert read.suggestion.to_dict() == suggestion.to_dict()
    assert read.current.suggestion_id == suggestion.id


def test_a_record_from_before_suggestions_loads_with_none():
    doc = Proposal(kind=STATE, provenance={"task_id": "run-1"}).to_dict()
    assert "suggestion" not in doc

    assert Proposal.from_dict(doc).suggestion is None


def test_suggesting_declining_and_approving_are_on_the_record(experiment):
    buffer = EventBuffer()
    tap = ProposalTap(buffer, experiment)  # noqa: F841 - its slots are weak
    lamella, proposal = _ask(experiment)
    experiment.suggest(lamella.id, TASK, proposal.id, _suggestion())
    experiment.decline_suggestion(lamella.id, TASK, proposal.id, reason="no")
    second = _suggestion(reason="try this")
    experiment.suggest(lamella.id, TASK, proposal.id, second)
    experiment.approve_suggestion(
        lamella.id, TASK, proposal.id, actor="operator:op", via="workflow"
    )

    events = [
        (e["kind"], e["payload"])
        for e in buffer.events_since(0)["events"]
        if e["kind"] != "proposal_asked"
    ]

    kinds = [kind for kind, _ in events]
    assert kinds == [
        "proposal_suggested",
        "suggestion_declined",
        "proposal_suggested",
        "proposal_decided",
    ]
    assert events[1][1]["reason"] == "no"
    assert events[3][1]["suggestion_id"] == second.id
    assert events[3][1]["suggested_by"] == "agent:claude"


def test_a_suggestion_to_keep_a_position_reads_in_words(experiment):
    """Round 10 (FIB-1492): "Continue with stage position name none, x −35.5 µm,
    ..., coordinate system RAW", under the task's prompt to the operator."""
    from fibsem.ai.agent.recent import RecentActions
    from fibsem.structures import FibsemStagePosition
    from fibsem.ui.agent.suggestions import suggestion_card

    lamella = experiment.positions[0]
    lamella.task_state.name = TASK
    lamella.task_state.task_id = "run-1"
    lamella.task_state.status = TaskStatus.InProgress
    here = FibsemStagePosition(x=-35.5e-6, y=-27.9e-6, z=0, r=0, t=0, name="asked")
    proposal = Proposal(
        kind=STATE,
        values={"stage_position": here},
        provenance={
            "task_id": "run-1",
            "proposer": TASK,
            "message": "Double click the image to move. Press Continue when done.",
        },
    )
    buffer = EventBuffer()
    tap = ProposalTap(buffer, experiment)  # noqa: F841 - its slots are weak
    assert experiment.ask_proposal(lamella.id, TASK, proposal)
    kept = _suggestion(reason="it is centred", values={"stage_position": here})
    assert experiment.suggest(lamella.id, TASK, proposal.id, kept).applied

    card = suggestion_card(lamella, TASK, proposal, str(experiment.path))
    assert card["decision"] == "Keep the position: x −35.5 µm, y −27.9 µm, flat"
    assert "Press Continue" not in card["proposal"]
    recent = RecentActions()
    for event in buffer.events_since(0)["events"]:
        recent.add(event)
    (row,) = [r for r in recent.rows if r.text.startswith("Suggested")]
    assert "it is centred" not in row.text  # the reason is the card's Why
    assert "keep the position x −35.5 µm" in row.text
