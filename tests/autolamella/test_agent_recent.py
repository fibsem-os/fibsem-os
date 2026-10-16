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


def _called(action, status=200, level="drive", **extra):
    """The server's record of a direct call (FIB-1426)."""
    return _record(
        "agent_called",
        dict(action=action, status=status, ok=status < 400, level=level, **extra),
    )


def test_at_drive_a_move_a_mill_and_an_answer_each_make_a_row():
    recent = RecentActions()
    recent.add(_called("Stage · Move by x 2 µm"))
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
        ("Stage · Move by x 2 µm", "done", OK),
    ]
    assert recent.rows[0].when == "16:08"
    # Each at the level it was done at: the mill's is the agent's last seen.
    assert [row.level for row in recent.rows] == ["drive", "drive", "drive"]


def test_only_the_agent_s_own_actions_make_rows():
    recent = RecentActions()
    assert not recent.add(
        _record("agent_called", {"action": "x", "ok": True}, actor=OPERATOR)
    )
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
    assert _rows(recent)[0][1:] == ("approved", WAITING)
    # What the approved request does is the request's row, not rows of its own.
    assert not recent.add(_called("Stage · Move by x 5 µm", level="assist"))
    recent.add(_record("request_applied", {"request_id": "r1", "index": 0, "ok": True}))
    assert _rows(recent)[0][1:] == ("applying 2 of 2", WAITING)
    recent.add(_record("request_applied", {"request_id": "r1", "index": 1, "ok": True}))
    recent.add(_record("request_finished", {"request_id": "r1", "status": "applied"}))
    assert _rows(recent) == [
        ("Asked: Stage · Move by x 5 µm, and 1 more", "applied", OK)
    ]
    assert recent.rows[0].level == "assist"
    recent.add(_record("request_raised", {"request_id": "r2", "actions": actions}))
    recent.add(
        _record(
            "request_decided",
            {"request_id": "r2", "outcome": "rejected", "reason": "bad"},
            OPERATOR,
        )
    )
    assert _rows(recent)[0][1:] == ("rejected: bad", MUTED)


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
    overview = {
        "service": "imaging",
        "command": "beam_acquire_overview",
        "run_id": "a",
        "action": "FIB · Overview, 3 × 3 tiles",
        "level": "drive",
    }
    recent.add(_record("command_started", dict(overview, state="running")))
    assert _rows(recent) == [("FIB · Overview, 3 × 3 tiles", "running", WAITING)]
    recent.add(_record("command_ended", dict(overview, state="done")))
    recent.add(
        _record(
            "agent_refused",
            {"action": "workflow_stop", "error_type": "ask_first", "level": "assist"},
        )
    )
    assert _rows(recent) == [
        ("Tried to workflow stop", "refused: needs your approval", ERROR),
        ("FIB · Overview, 3 × 3 tiles", "done", OK),
    ]
    assert [row.level for row in recent.rows] == ["assist", "drive"]


def test_a_direct_call_says_how_it_ended():
    recent = RecentActions()
    recent.add(_called("Stage · Move by x 2 µm", 409, error_type="microscope_busy"))
    recent.add(_called("FIB · Take an image", 500))
    # A mill started and left running is its mill's row.
    assert not recent.add(
        _called(
            "Milling · Start", route="POST /microscope/services/milling/commands/start"
        )
    )
    recent.add(
        _record(
            "agent_stop",
            {"action": "milling stop", "what": "the mill", "reason": "through"},
        )
    )
    assert _rows(recent) == [
        ("Stopped the mill: through", "done", OK),
        ("FIB · Take an image", "failed", ERROR),
        ("Stage · Move by x 2 µm", "refused: busy", ERROR),
    ]


def test_it_keeps_the_last_ten():
    recent = RecentActions()
    for index in range(12):
        recent.add(_record("agent_note", {"text": f"note {index}"}))
    assert len(recent.rows) == 10
    assert recent.rows[0].text == "Note: note 11"


def test_the_operator_s_message_and_the_agent_s_reply_are_rows():
    # FIB-1423: the operator writes from the card; the agent answers with a note.
    recent = RecentActions()
    recent.add(_record("agent_level_set", {"level": "drive"}, actor=OPERATOR))
    recent.add(
        _record("operator_message", {"id": "ab12", "text": "skip 03"}, actor=OPERATOR)
    )
    recent.add(_record("agent_note", {"text": "skipping 03", "reply_to": "ab12"}))
    assert _rows(recent) == [
        ("Reply: skipping 03", "", MUTED),
        ("You: skip 03", "", MUTED),
    ]
    # The operator's line is at no level; the agent's reply at its own.
    assert [row.level for row in recent.rows] == ["drive", ""]
