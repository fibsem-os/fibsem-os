"""The window puts the agent's suggestion on a pending proposal on its card;
Approve decides the proposal as suggested, as the operator, naming the
suggestion, and Decline keeps the proposal up with the reason recorded."""

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.proposals import POINT_OF_INTEREST  # noqa: E402
from fibsem.applications.autolamella.workflows.tasks.select_position import (  # noqa: E402
    SelectMillingPositionTaskConfig,
)
from fibsem.structures import Point  # noqa: E402
from fibsem.workflows.proposals import (  # noqa: E402
    DecisionOutcome,
    Proposal,
    Suggestion,
)
from fibsem.workflows.status import TaskStatus  # noqa: E402
from tests.fixtures.main_window import main_window  # noqa: E402

TASK = "Setup Lamella Position"


@pytest.fixture
def window(qapp, tmp_path):
    with main_window(
        qapp,
        tmp_path,
        "suggest-exp",
        {TASK: SelectMillingPositionTaskConfig(task_name=TASK)},
    ) as win:
        yield win


def _ask_and_suggest(win, qapp):
    experiment = win.autolamella_ui.experiment
    lamella = experiment.positions[0]
    lamella.task_state.name = TASK
    lamella.task_state.task_id = "run-1"
    lamella.task_state.status = TaskStatus.InProgress
    proposal = Proposal(
        kind=POINT_OF_INTEREST,
        values={"poi": Point(0.0, 0.0)},
        provenance={"task_id": "run-1"},
    )
    experiment.ask_proposal(lamella.id, TASK, proposal)
    suggestion = Suggestion(
        actor="agent:claude",
        reason="the edge is 1 µm right",
        values={"poi": Point(1e-6, 0.0)},
    )
    assert experiment.suggest(lamella.id, TASK, proposal.id, suggestion).applied
    qapp.processEvents()
    return proposal, suggestion


def test_approve_decides_as_suggested_naming_the_suggestion(window, qapp):
    proposal, suggestion = _ask_and_suggest(window, qapp)
    chip = window.agent_chip
    assert chip.suggestion["reason"] == "the edge is 1 µm right"
    assert chip.text() == "Agent suggests"

    chip.card.approve_suggestion_button.click()
    qapp.processEvents()

    decision = proposal.current
    assert decision.outcome is DecisionOutcome.Confirmed
    assert decision.actor.kind.value == "operator"
    assert decision.suggestion_id == suggestion.id
    assert decision.values["poi"] == Point(1e-6, 0.0)
    assert chip.suggestion is None


def test_decline_keeps_the_proposal_up_with_the_reason(window, qapp):
    proposal, _ = _ask_and_suggest(window, qapp)
    card = window.agent_chip.card

    card.decline_button.click()
    card.reason_edit.setText("that is the trench")
    card.decline_button.click()
    qapp.processEvents()

    assert proposal.pending
    assert proposal.suggestion.declined_reason == "that is the trench"
    assert window.agent_chip.suggestion is None
