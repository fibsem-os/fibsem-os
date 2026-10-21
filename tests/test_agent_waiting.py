"""Headless round 12 H3: /chat says which decision a run waits on, and who can
make it."""

from types import SimpleNamespace

from fibsem.acting import OPERATOR
from fibsem.ai.agent.levels import AgentLevel
from fibsem.ai.agent.session import AgentSession
from fibsem.workflows.proposals import ALIGNMENT_AREA, STATE, Proposal, Suggestion


def _agent(level, pending=(), deciders=(), experiment=True):
    experiment = (
        SimpleNamespace(name="r12", pending_proposals=lambda: list(pending))
        if experiment
        else None
    )
    host = SimpleNamespace(running=True, auth=SimpleNamespace(level=level))
    session = SimpleNamespace(
        experiment=experiment, app_server=host, deciders=lambda: set(deciders)
    )
    return AgentSession(session)


def test_the_waiting_decision_and_who_can_make_it():
    item = SimpleNamespace(name="01-meet-tapir")
    proposal = Proposal(kind=ALIGNMENT_AREA, provenance={"task_id": "run-1"})
    pending = [(item, "Setup", proposal)]

    assert _agent(AgentLevel.WATCH).waiting() is None
    waiting = _agent(AgentLevel.WATCH, pending).waiting()
    assert waiting["proposal"] == "01-meet-tapir · Setup · Alignment area"
    assert waiting["who"].startswith("No one here can decide it")
    assert (
        _agent(AgentLevel.WATCH, pending, {OPERATOR}).waiting()["who"]
        == "Decide it in the window."
    )
    assert _agent(AgentLevel.DRIVE, pending).waiting()["who"] == (
        "The agent decides it."
    )
    assert _agent(AgentLevel.ASSIST, pending).waiting()["who"] == (
        "The agent will suggest a decision."
    )
    proposal.suggestion = Suggestion(actor="agent:claude", reason="looks right")
    assert _agent(AgentLevel.ASSIST, pending).waiting()["who"] == (
        "The agent suggested a decision."
    )


def test_a_tilt_asked_as_a_position_reads_as_the_tilt():
    item = SimpleNamespace(name="01-meet-tapir")
    tilt = Proposal(kind=STATE, provenance={"task_id": "run-1", "label": "Tilt"})
    waiting = _agent(AgentLevel.DRIVE, [(item, "Setup", tilt)]).waiting()
    assert waiting["proposal"] == "01-meet-tapir · Setup · Tilt"
    # The header names the open experiment, never the session's plain record.
    assert _agent(AgentLevel.DRIVE).experiment_name() == "r12"
    assert _agent(AgentLevel.DRIVE, experiment=False).experiment_name() is None
