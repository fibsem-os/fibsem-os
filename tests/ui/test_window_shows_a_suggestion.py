"""The window shows the agent's suggestion on a pending proposal on its chip and
card, until it is decided on the Agent page (the session's approve)."""

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.workflows.tasks.select_position import (  # noqa: E402
    SelectMillingPositionTaskConfig,
)
from fibsem.structures.settings import Point  # noqa: E402
from fibsem.workflows.proposals import (  # noqa: E402
    POINT_OF_INTEREST,  # noqa: E402
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
    experiment = win.experiment
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


def test_the_suggestion_shows_until_it_is_approved(window, qapp):
    proposal, suggestion = _ask_and_suggest(window, qapp)
    chip = window.agent_host.chip
    assert chip.suggestion["reason"] == "the edge is 1 µm right"
    assert chip.text() == "Agent suggests"
    assert chip.card.waiting_word.text() == "Suggests"

    assert window.session.agent.approve_suggestion(proposal.id).applied
    qapp.processEvents()

    decision = proposal.current
    assert decision.outcome is DecisionOutcome.Confirmed
    assert decision.suggestion_id == suggestion.id
    assert chip.suggestion is None
    assert not chip.card.waiting_row.isVisibleTo(chip.card)
