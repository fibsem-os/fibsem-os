"""The agent card's Recent list (FIB-1414): what the agent did lately, one line
each with how it ended, read from the session's record, oldest first."""

from fibsem.acting import AGENT, OPERATOR, TASK
from fibsem.ai.agent.recent import (
    ERROR,
    MUTED,
    OK,
    RUNNING,
    SIZE,
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


def _ended(recent):
    """How the newest row ended: its word, why, and its tone."""
    row = recent.rows[-1]
    return row.outcome, row.why, row.tone


def _called(action, status=200, level="drive", **extra):
    """The server's record of a direct call (FIB-1426)."""
    return _record(
        "agent_called",
        dict(action=action, status=status, ok=status < 400, level=level, **extra),
    )


def test_at_drive_a_move_and_a_mill_each_make_a_row():
    recent = RecentActions()
    recent.add(_called("Stage · Move 2 µm in x"))
    recent.add(
        _record(
            "milling",
            {"milling_task_name": "agent test mill", "status": "completed"},
        )
    )
    assert _rows(recent) == [
        ("Stage · Move 2 µm in x", "done", OK),
        ("Milled 'agent test mill'", "done", OK),
    ]
    assert recent.rows[0].when == "16:08"
    # Each at the level it was done at: the mill's is the agent's last seen.
    assert [row.level for row in recent.rows] == ["drive", "drive"]


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
        ("Asked: Stage · Move 5 µm in x, and 1 more", "waiting", WAITING)
    ]
    recent.add(
        _record(
            "request_decided", {"request_id": "r1", "outcome": "approved"}, OPERATOR
        )
    )
    assert _ended(recent) == ("approved", "", RUNNING)
    # What the approved request does is the request's row, not rows of its own.
    assert not recent.add(_called("Stage · Move 5 µm in x", level="assist"))
    recent.add(_record("request_applied", {"request_id": "r1", "index": 0, "ok": True}))
    assert _ended(recent) == ("2 of 2", "", RUNNING)
    recent.add(_record("request_applied", {"request_id": "r1", "index": 1, "ok": True}))
    recent.add(_record("request_finished", {"request_id": "r1", "status": "applied"}))
    assert _rows(recent) == [("Asked: Stage · Move 5 µm in x, and 1 more", "done", OK)]
    assert recent.rows[0].level == "assist"
    recent.add(_record("request_raised", {"request_id": "r2", "actions": actions}))
    recent.add(
        _record(
            "request_decided",
            {"request_id": "r2", "outcome": "rejected", "reason": "bad"},
            OPERATOR,
        )
    )
    assert _ended(recent) == ("rejected", "bad", MUTED)


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
    assert _rows(recent) == [("FIB · Overview, 3 × 3 tiles", "running", RUNNING)]
    recent.add(_record("command_ended", dict(overview, state="done")))
    recent.add(
        _record(
            "agent_refused",
            {"action": "workflow_stop", "error_type": "ask_first", "level": "assist"},
        )
    )
    assert _rows(recent) == [
        ("FIB · Overview, 3 × 3 tiles", "done", OK),
        ("Tried to workflow stop", "refused", ERROR),
    ]
    assert recent.rows[-1].why == "needs your approval"
    assert [row.level for row in recent.rows] == ["drive", "assist"]


def test_a_direct_call_says_how_it_ended():
    recent = RecentActions()
    recent.add(_called("Stage · Move 2 µm in x", 409, error_type="microscope_busy"))
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
        ("Stage · Move 2 µm in x", "refused", ERROR),
        ("FIB · Take an image", "failed", ERROR),
        ("Stopped the mill: through", "done", OK),
    ]
    assert recent.rows[0].why == "busy"


def test_it_keeps_the_last_rows_oldest_first():
    recent = RecentActions()
    for index in range(SIZE + 2):
        recent.add(_record("agent_note", {"text": f"note {index}"}))
    assert len(recent.rows) == SIZE
    assert recent.rows[0].text == "note 2"
    assert recent.rows[-1].text == f"note {SIZE + 1}"


def test_a_message_is_kept_whole_and_an_action_on_one_line():
    """Agent UI notes 2026-10-20: long replies were cut at 300 characters and
    their lists run together on one line."""
    recent = RecentActions()
    reply = "Five cells look worth a lamella:\n\n- 01\n- 02\n\n" + "x" * 400
    recent.add(_record("agent_note", {"text": reply}))
    recent.add(_called("Stage ·\n Move 2 µm in x"))
    note, call = recent.rows
    assert note.text == reply and note.kind == "note" and note.conversation
    assert call.text == "Stage · Move 2 µm in x" and not call.conversation


def test_the_agent_s_own_decision_reads_as_done():
    """Agent UI notes 2026-10-20: a decision the agent made itself at Drive
    read "Confirm the proposal", an order to the operator."""
    recent = RecentActions()
    for outcome, extra in (
        ("Confirmed", {}),
        ("Rejected", {"reason": "drift above 0.5 µm"}),
    ):
        recent.add(
            _called(
                "02-alive-skunk · Mill Rough · Confirm the proposal",
                tool="proposal_decide",
                arguments={"item_id": "a1", "outcome": outcome, **extra},
            )
        )
    assert [row.text for row in recent.rows] == [
        "Confirmed Mill Rough on 02-alive-skunk",
        "Rejected Mill Rough on 02-alive-skunk: drift above 0.5 µm",
    ]
    # Refused, it was never done: it keeps the words it was asked in.
    recent.add(
        _called(
            "02-alive-skunk · Mill Rough · Confirm the proposal",
            403,
            tool="proposal_decide",
            arguments={"item_id": "a1", "outcome": "Confirmed"},
        )
    )
    assert recent.rows[-1].text.endswith("Confirm the proposal")


def test_a_request_row_reads_as_its_card_did():
    """Sim session 2026-10-16 F7: "Set hfw to 0.0003" where the card said the
    field of view in µm, old -> new."""
    recent = RecentActions()
    actions = [
        {
            "tool": "device_set_parameter",
            "arguments": {"device": "ion", "parameter": "hfw", "value": 3e-4},
            "what": "Ion",
            "change": "Set Field of View 150 µm → 300 µm",
        }
    ]
    recent.add(_record("request_raised", {"request_id": "r1", "actions": actions}))
    assert _rows(recent)[0][0] == "Asked: Ion · Set Field of View 150 µm → 300 µm"


def test_a_long_line_is_kept_whole_for_the_card_to_wrap():
    recent = RecentActions()
    reason = "the trench is through the lamella and the next pass would cut the top"
    recent.add(_called(f"Workflow · Stop the run: {reason}"))
    assert _rows(recent)[0][0].endswith(reason)


def test_a_refusal_gives_its_reason_not_an_exception_s_name():
    """Sim session 2026-10-16: "refused: StageLimitError", "refused: not at
    drive"."""
    recent = RecentActions()
    recent.add(
        _called(
            "Stage · Move to x 500 mm",
            422,
            error_type="StageLimitError",
            message="x 500 mm is outside the stage limits\nlimits: ...",
        )
    )
    recent.add(
        _record(
            "agent_refused",
            {"action": "request create", "error_type": "level", "level": "drive"},
        )
    )
    recent.add(
        _record(
            "agent_refused",
            {"action": "stage move", "error_type": "level", "level": "watch"},
        )
    )
    assert [(row.outcome, row.why) for row in recent.rows] == [
        ("refused", "x 500 mm is outside the stage limits"),
        ("refused", "at Drive the agent acts itself"),
        ("refused", "Watch is read-only"),
    ]


def test_a_request_the_agent_took_back_reads_withdrawn():
    recent = RecentActions()
    actions = [{"tool": "stage_move_relative", "arguments": {"dx": 5e-6}}]
    for request_id, withdrawn in (("r1", True), ("r2", False)):
        recent.add(
            _record("request_raised", {"request_id": request_id, "actions": actions})
        )
        payload = {"request_id": request_id, "why": "the operator stopped the agent"}
        if withdrawn:
            payload["withdrawn"] = True
        recent.add(_record("request_cancelled", payload))
    # A cancel says who ended it (sim session 2026-10-19).
    assert [(row.outcome, row.why) for row in recent.rows] == [
        ("withdrawn", ""),
        ("cancelled", "the operator stopped the agent"),
    ]


def test_a_failed_request_says_why():
    recent = RecentActions()
    actions = [{"tool": "stage_move_relative", "arguments": {"dx": 5e-6}}]
    recent.add(_record("request_raised", {"request_id": "r1", "actions": actions}))
    error = {"error_type": "out_of_range", "message": "target outside the stage limits"}
    recent.add(
        _record(
            "request_applied",
            {"request_id": "r1", "index": 0, "ok": False, "error": error},
        )
    )
    recent.add(_record("request_finished", {"request_id": "r1", "status": "failed"}))
    assert _ended(recent) == ("failed", "target outside the stage limits", ERROR)


def test_the_operator_s_message_and_the_agent_s_reply_are_rows():
    # FIB-1423: the operator writes from the card; the agent answers with a note.
    recent = RecentActions()
    recent.add(_record("agent_level_set", {"level": "drive"}, actor=OPERATOR))
    recent.add(
        _record("operator_message", {"id": "ab12", "text": "skip 03"}, actor=OPERATOR)
    )
    recent.add(_record("agent_note", {"text": "skipping 03", "reply_to": "ab12"}))
    assert _rows(recent) == [("skip 03", "", MUTED), ("skipping 03", "", MUTED)]
    assert [row.kind for row in recent.rows] == ["you", "reply"]
    # The operator's line is at no level; the agent's reply at its own.
    assert [row.level for row in recent.rows] == ["", "drive"]


def test_a_request_on_the_record_as_a_proposal_is_one_row():
    recent = RecentActions()
    actions = [{"tool": "stage_move_relative", "arguments": {"dx": 5e-6}}]

    def asked(proposal_id):
        payload = {"proposal_id": proposal_id, "kind": "actions"}
        return _record("proposal_asked", {**payload, "proposed": {"actions": actions}})

    def decided(proposal_id, outcome, reason="", actor="operator:op"):
        payload = dict(proposal_id=proposal_id, kind="actions", outcome=outcome)
        return _record(
            "proposal_decided", {**payload, "reason": reason, "actor": actor}, OPERATOR
        )

    recent.add(asked("r1"))
    assert _rows(recent) == [("Asked: Stage · Move 5 µm in x", "waiting", WAITING)]
    recent.add(decided("r1", "Confirmed"))
    assert _ended(recent) == ("approved", "", RUNNING)
    recent.add(asked("r2"))
    recent.add(decided("r2", "Rejected", "not now"))
    assert _ended(recent) == ("rejected", "not now", MUTED)
    recent.add(asked("r3"))
    recent.add(decided("r3", "Withdrawn", "withdrawn by the agent: moved", "agent:c"))
    assert _ended(recent) == ("withdrawn", "", MUTED)
    recent.add(asked("r4"))
    recent.add(decided("r4", "Withdrawn", "the operator stopped the agent", "task:x"))
    assert _ended(recent) == ("cancelled", "the operator stopped the agent", MUTED)


def test_actions_and_notes_on_the_record_make_the_same_rows():
    recent = RecentActions()
    recent.add(
        _record(
            "action",
            {"what": "Stage · Move 2 µm in x", "ok": True, "level": "drive"},
        )
    )
    recent.add(
        _record(
            "action",
            {
                "what": "Stop the mill",
                "ok": True,
                "stops": "milling stop",
                "target": "the mill",
                "reason": "off target",
                "level": "drive",
            },
        )
    )
    recent.add(
        _record(
            "action",
            {
                "what": "move relative",
                "ok": False,
                "refused": True,
                "error_type": "ask_first",
                "level": "assist",
            },
        )
    )
    recent.add(_record("note", {"text": "skip 03", "id": "ab12"}, actor=OPERATOR))
    recent.add(_record("note", {"text": "skipping 03", "reply_to": "ab12"}))
    # The operator's note on a grid is not to the agent.
    assert not recent.add(
        _record("note", {"text": "thin ice", "item_name": "G1"}, actor=OPERATOR)
    )
    # The operator's own stop is not the agent's row.
    assert not recent.add(
        _record("action", {"what": "Stop the run", "stops": "workflow"}, OPERATOR)
    )

    assert [text for text, _, _ in _rows(recent)] == [
        "Stage · Move 2 µm in x",
        "Stopped the mill: off target",
        "Tried to move relative",
        "skip 03",
        "skipping 03",
    ]


def test_a_row_carries_the_evidence_its_call_cites():
    recent = RecentActions()
    kept = {"evidence_id": "ab12cd34", "text": "trench off centre", "image": "/e.png"}
    recent.add(_record("evidence", kept))
    assert recent.rows == []  # the call citing it is the row
    recent.add(
        _record(
            "action",
            dict(
                what="Stage · Move 2 µm in x",
                ok=True,
                level="drive",
                evidence="ab12cd34",
            ),
        )
    )
    recent.add(_record("action", dict(what="Stage · Move 1 µm in y", ok=True)))
    cited, uncited = recent.rows
    assert cited.evidence["text"] == "trench off centre"
    assert uncited.evidence is None
