"""The agent card's Recent list (FIB-1414): what the agent did lately, one line
each with how it ended, read from the session's record."""

from fibsem.acting import AGENT, OPERATOR, TASK
from fibsem.applications.autolamella.ui.agent_recent import (
    ERROR,
    MUTED,
    OK,
    WAITING,
    RecentActions,
)

_seq = iter(range(1, 10_000))


def _record(kind, payload=None, actor=AGENT):
    return {
        "seq": next(_seq),
        "t": "2026-10-15T16:08:12.000",
        "kind": kind,
        "actor": actor,
        "payload": payload or {},
    }


def _rows(recent):
    return [(row.text, row.outcome, row.tone) for row in recent.rows]


def test_at_drive_a_move_a_mill_and_an_answer_each_make_a_row():
    recent = RecentActions()
    recent.add(_record("stage_moved", {"move": "move_relative"}))
    recent.add(
        _record(
            "milling",
            {"milling_task_name": "agent test mill", "status": "completed"},
        )
    )
    recent.add(
        _record(
            "prompt_raised",
            {
                "question_id": "q1",
                "message": "Is the lamella centred?",
                "positive": "Yes",
            },
            actor=TASK,
        )
    )
    recent.add(
        _record(
            "prompt_answered",
            {"question_id": "q1", "response": True, "answered_by": AGENT},
        )
    )
    assert _rows(recent) == [
        ("Answered 'Is the lamella centred?': Yes", "done", OK),
        ("Milled 'agent test mill'", "done", OK),
        ("Moved the stage: move relative", "done", OK),
    ]
    assert recent.rows[0].when == "16:08"


def test_only_the_agent_s_own_actions_make_rows():
    recent = RecentActions()
    assert not recent.add(_record("stage_moved", {"move": "x"}, actor=OPERATOR))
    assert not recent.add(_record("milling", {"status": "completed"}, actor=TASK))
    assert not recent.add(_record("image_acquired", {}, actor=TASK))
    assert recent.rows == []


def test_a_request_is_one_row_that_follows_it_to_its_end():
    recent = RecentActions()
    actions = [
        {"tool": "stage_move_relative", "arguments": {"dx": 5e-6}},
        {"tool": "beam_acquire_image", "arguments": {"beam_type": "ION"}},
    ]
    recent.add(_record("request_raised", {"request_id": "r1", "actions": actions}))
    assert _rows(recent) == [
        ("Asked: Stage · Move by x 5 µm, and 1 more", "waiting", WAITING)
    ]
    recent.add(
        _record(
            "request_decided", {"request_id": "r1", "outcome": "approved"}, OPERATOR
        )
    )
    # What the approved request does is the request's row, not rows of its own.
    assert not recent.add(_record("stage_moved", {"move": "move_relative"}))
    assert not recent.add(_record("image_acquired", {"beam_type": "ION"}))
    recent.add(_record("request_finished", {"request_id": "r1", "status": "applied"}))
    assert _rows(recent) == [
        ("Asked: Stage · Move by x 5 µm, and 1 more", "applied", OK)
    ]
    recent.add(_record("request_raised", {"request_id": "r2", "actions": actions}))
    recent.add(
        _record(
            "request_decided", {"request_id": "r2", "outcome": "rejected"}, OPERATOR
        )
    )
    assert recent.rows[0].outcome == "rejected" and recent.rows[0].tone == MUTED


def test_a_suggestion_is_taken_or_dropped():
    recent = RecentActions()
    for question in ("q1", "q2"):
        recent.add(
            _record(
                "prompt_raised",
                {"question_id": question, "message": "Mill?", "positive": "Mill"},
                actor=TASK,
            )
        )
        recent.add(
            _record("prompt_suggested", {"question_id": question, "response": True})
        )
    recent.add(
        _record(
            "prompt_answered",
            {
                "question_id": "q1",
                "response": True,
                "answered_by": OPERATOR,
                "suggested_by": AGENT,
            },
            actor=OPERATOR,
        )
    )
    recent.add(
        _record(
            "prompt_answered",
            {"question_id": "q2", "response": False, "answered_by": OPERATOR},
            actor=OPERATOR,
        )
    )
    assert _rows(recent) == [
        ("Suggested Mill for 'Mill?'", "dropped", MUTED),
        ("Suggested Mill for 'Mill?'", "taken", OK),
    ]


def test_a_long_command_is_one_row_and_a_refusal_says_why():
    recent = RecentActions()
    recent.add(_record("command_started", {"service": "imaging", "run_id": "a"}))
    assert not recent.add(_record("stage_moved", {"move": "tile"}))
    assert not recent.add(_record("image_acquired", {"beam_type": "ELECTRON"}))
    recent.add(
        _record(
            "command_ended",
            {
                "service": "imaging",
                "command": "tiled_overview",
                "run_id": "a",
                "state": "done",
            },
        )
    )
    recent.add(
        _record(
            "agent_refused",
            {"action": "workflow_stop", "error_type": "ask_first", "level": "assist"},
        )
    )
    assert _rows(recent) == [
        ("Tried to workflow stop", "refused: needs your approval", ERROR),
        ("Tiled overview (imaging)", "done", OK),
    ]


def test_the_agent_s_queue_edit_is_a_row_and_the_operator_s_is_not():
    recent = RecentActions()
    edit = {"item_name": "01-full-magpie", "task_name": "Trench Milling"}
    recent.add(_record("queue_changed", dict(edit, action="remove")))
    assert not recent.add(
        _record("queue_changed", dict(edit, action="move_up"), actor=OPERATOR)
    )
    recent.add(_record("queue_changed", dict(edit, action="requeue_front")))
    assert _rows(recent) == [
        ("Queued again, first: 01-full-magpie · Trench Milling", "done", OK),
        ("Took out of the queue: 01-full-magpie · Trench Milling", "done", OK),
    ]


def test_it_keeps_the_last_ten():
    recent = RecentActions()
    for index in range(12):
        recent.add(_record("agent_note", {"text": f"note {index}"}))
    assert len(recent.rows) == 10
    assert recent.rows[0].text == "Note: note 11"
