"""What an event record says, in words: one describer for every reader.

`describe` gives a record's label, detail, a quieter note and an error. The App
Server sends it with each record when asked (``/app/events?text=true``), and the
web Events page shows it rather than reading payloads itself.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from fibsem.workflows.events import kinds as k

Text = Dict[str, Any]
# Each legacy question as raised, by "<session>:<question_id>", so its answer can
# say what was asked and which button.
Asked = Dict[str, Mapping[str, Any]]


def note_asked(asked: Asked, record: Mapping[str, Any]) -> None:
    payload = record.get("payload") or {}
    if record.get("kind") == k.PROMPT_RAISED and isinstance(payload, dict):
        asked[f"{record.get('session') or ''}:{payload.get('question_id')}"] = payload


def with_text(
    record: Mapping[str, Any], asked: Optional[Asked] = None
) -> Dict[str, Any]:
    """*record* with its ``text`` and ``group``: a copy, never the buffer's own."""
    return {
        **record,
        "text": describe(record, asked),
        "group": k.group_of(str(record.get("kind") or "")),
    }


def _join(*parts: Any) -> str:
    return " · ".join(
        str(p) for p in parts if p is not None and p != "" and p is not False
    )


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _um(value: Any) -> str:
    return f"{value * 1e6:.2f} µm" if _num(value) else "?"


def _signed(value: Any) -> Optional[str]:
    if not _num(value):
        return None
    return f"{'+' if value >= 0 else '−'}{abs(value * 1e6):.2f} µm"


def _file_name(path: Any) -> str:
    return str(path or "").replace("\\", "/").split("/")[-1]


def _cites(p: Mapping[str, Any]) -> Optional[str]:
    return f"evidence {p['evidence']}" if p.get("evidence") else None


def _position(at: Any) -> str:
    if not isinstance(at, dict):
        return ""
    parts = [f"{a} {_um(at[a])}" for a in ("x", "y", "z") if _num(at.get(a))]
    parts += [f"{a} {math.degrees(at[a]):.1f}°" for a in ("r", "t") if _num(at.get(a))]
    return " · ".join(parts)


def _text(label: str, **fields: Any) -> Text:
    """Leaves out the empty fields, as the page would not show them."""
    out: Text = {"label": label}
    for key, value in fields.items():
        if value not in (None, "", False):
            out[key] = value
    return out


def _stage_move(p: Mapping[str, Any]) -> Text:
    r = p.get("request") or {}
    failed = (
        {"error": f"failed: {p['error']}", "failed": True} if p.get("error") else {}
    )
    move = p.get("move")
    if move in ("stable_move", "vertical_move"):
        return _text(
            "moved stage",
            detail=_join(move.replace("_", " ", 1), r.get("beam_type")),
            note=_join(_signed(r.get("dx")), _signed(r.get("dy"))),
            **failed,
        )
    if move == "move_to_orientation":
        return _text(
            "moved stage", detail=f"to the {r.get('orientation')} orientation", **failed
        )
    if move == "move_to_milling_angle":
        return _text(
            "moved stage",
            detail=f"to milling angle {r.get('milling_angle')}°",
            **failed,
        )
    return _text(
        "moved stage",
        detail=str(move or "move").replace("_", " "),
        note=_position(r.get("position") or r.get("stage_position")),
        **failed,
    )


def _plain(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def describe(record: Mapping[str, Any], asked: Optional[Asked] = None) -> Text:
    """What *record* says: ``label``, and when it has them ``detail``, ``note``,
    ``error`` and ``failed``."""
    kind = str(record.get("kind") or "")
    p = record.get("payload")
    p = p if isinstance(p, dict) else {}
    asked = asked or {}

    if kind == k.PROMPT_RAISED:
        positive, negative = p.get("positive"), p.get("negative")
        choice = (
            f"{positive}{f' or {negative}' if negative else ''}" if positive else None
        )
        return _text("asked", detail=p.get("message") or p.get("type"), note=choice)
    if kind == k.PROMPT_ANSWERED:
        q = asked.get(f"{record.get('session') or ''}:{p.get('question_id')}") or {}
        yes = p.get("response")
        button = (q.get("positive") if yes else q.get("negative")) or (
            "yes" if yes else "no"
        )
        return _text(
            "answered",
            detail=_join(q.get("message") or p.get("type"), button),
            note=_join(
                p.get("adjusted") and "adjusted",
                p.get("suggested_by") and f"suggested by {p.get('suggested_by')}",
            ),
        )
    if kind == k.PROMPT_SUGGESTED:
        return _text(
            "suggested",
            detail=_join(p.get("type"), "yes" if p.get("response") else "no"),
            note=p.get("reason"),
        )
    if kind == k.PROMPT_CANCELLED:
        return _text("withdrawn", detail=p.get("type"))
    if kind == k.PROMPT_SUGGESTION_DISMISSED:
        return _text("suggestion dismissed", detail=p.get("type"), note=p.get("reason"))
    if kind in (k.PROPOSAL_HELD, k.QUESTION_ASKED):
        return _text("proposal held", detail="the run waits on a proposal")
    if kind in (k.PROPOSAL_RELEASED, k.QUESTION_RELEASED):
        return _text("proposal released")
    if kind == k.PROPOSAL_ASKED:
        return _text(
            "proposed",
            detail=p.get("field") or p.get("kind") or p.get("type"),
            note=_cites(p),
        )
    if kind == k.PROPOSAL_DECIDED:
        return _text(
            "decided", detail=_join(p.get("field") or p.get("kind"), p.get("outcome"))
        )
    if kind == k.REVIEW_DECIDED:
        return _text(
            "reviewed",
            detail=_join(p.get("field"), p.get("outcome") or p.get("decision")),
        )
    if kind == k.ACTION:
        if p.get("stops"):
            return _text(
                "stopped",
                detail=f"{p.get('target') or p.get('what')}{' (its own)' if p.get('own') else ''}",
                note=p.get("reason") or p.get("via"),
            )
        if p.get("refused"):
            return _text(
                "refused",
                detail=p.get("what"),
                note=f"{p.get('error_type')} at {p.get('level')}",
            )
        ok = bool(p.get("ok"))
        return _text(
            "did" if ok else "failed",
            detail=p.get("what"),
            note=_join(
                p.get("request_id") and f"request {p.get('request_id')}", _cites(p)
            ),
            error=None if ok else (p.get("error_type") or p.get("status")),
            failed=not ok,
        )
    if kind == k.NOTE:
        message = (
            p.get("to")
            or p.get("reply_to")
            or (record.get("actor") == "operator" and p.get("id"))
        )
        return _text(
            "message" if message else "note",
            detail=_join(p.get("text"), p.get("item_name")),
            note=_cites(p),
        )
    if kind == k.EVIDENCE:
        own = " (the agent's own)" if p.get("uploaded") else ""
        return _text(
            "evidence",
            detail=f"{_file_name(p.get('image'))}{own}",
            note=p.get("evidence_id"),
        )
    if kind == k.AGENT_NOTE:
        return _text("message" if p.get("reply_to") else "note", detail=p.get("text"))
    if kind == k.OPERATOR_MESSAGE:
        return _text("message", detail=p.get("text"))
    if kind == k.AGENT_MESSAGES_READ:
        ids = p.get("ids") or []
        many = isinstance(ids, list) and len(ids) > 1
        return _text(
            "read",
            detail="The agent read your messages"
            if many
            else "The agent read your message",
        )
    if kind == k.AGENT_PLAN_EDITED:
        sections = p.get("sections") or []
        return _text("plan edited", detail=", ".join(map(str, sections)) or None)
    if kind == k.AGENT_SAMPLE_SET:
        was, sample = p.get("was"), p.get("sample")
        return _text("sample", detail=f"{was} → {sample}" if was else sample)
    if kind == k.AGENT_STOP:
        own = " (its own)" if p.get("own") else ""
        return _text(
            "stopped",
            detail=f"{p.get('what') or p.get('action')}{own}",
            note=p.get("reason"),
        )
    if kind == k.AGENT_SESSION_BUDGET:
        budget, spent = p.get("budget_usd"), p.get("spent_usd")
        used = (
            round(100 * spent / budget)
            if _num(budget) and budget and _num(spent)
            else 0
        )
        shown = budget if isinstance(budget, int) else f"{float(budget or 0):.2f}"
        return _text(
            "budget", detail=f"The agent has used {used}% of its ${shown} budget"
        )
    if kind == k.AGENT_CALLED:
        if p.get("ok"):
            return _text("called", detail=p.get("action"), note=f"at {p.get('level')}")
        return _text(
            "call failed",
            detail=p.get("action"),
            note=f"at {p.get('level')}",
            error=p.get("error_type") or p.get("status"),
            failed=True,
        )
    if kind == k.AGENT_REFUSED:
        return _text(
            "refused",
            detail=p.get("action"),
            note=f"{p.get('error_type')} at {p.get('level')}",
        )
    if kind == k.REQUEST_RAISED:
        tools = ", ".join(
            str(a.get("tool")) for a in p.get("actions") or [] if isinstance(a, dict)
        )
        return _text("requested", detail=tools, note=p.get("reason"))
    if kind == k.REQUEST_DECIDED:
        return _text(
            "rejected" if p.get("outcome") == "rejected" else "approved",
            detail=f"request {p.get('request_id')}",
            note=p.get("reason"),
        )
    if kind == k.REQUEST_APPLIED:
        return _text("applied", detail=f"request {p.get('request_id')}")
    if kind == k.REQUEST_FINISHED:
        return _text(
            "finished",
            detail=f"request {p.get('request_id')} · {p.get('status')}",
            note=f"{p.get('applied')} of {p.get('of')} applied",
        )
    if kind == k.REQUEST_CANCELLED:
        return _text(
            "cancelled", detail=f"request {p.get('request_id')}", note=p.get("why")
        )
    if kind == k.TASK_STARTED:
        return _text("started", detail=p.get("task_name"))
    if kind == k.TASK_COMPLETED:
        return _text("completed", detail=p.get("task_name"))
    if kind == k.TASK_FAILED:
        return _text(
            "failed", detail=p.get("task_name"), error=p.get("error"), failed=True
        )
    if kind == k.TASK_CANCELLED:
        return _text("stopped", detail=p.get("task_name"))
    if kind == k.TASK_SKIPPED:
        missing = p.get("missing_prereqs")
        reason = p.get("skip_reason")
        note = f"{reason}: {', '.join(map(str, missing))}" if missing else reason
        return _text("skipped", detail=p.get("task_name"), note=note)
    if kind == k.TASK_STEP:
        return _text("step", detail=p.get("display_message") or p.get("step"))
    if kind == k.WORKFLOW_STARTED:
        return _text("run started")
    if kind == k.WORKFLOW_COMPLETED:
        return _text("run completed")
    if kind == k.WORKFLOW_CANCELLED:
        return _text("run stopped")
    if kind == k.WORKFLOW_STALLED:
        return _text("run stalled", error=p.get("error"), failed=True)
    if kind == k.WORKFLOW_FAILED:
        return _text("run failed", error=p.get("error"), failed=True)
    if kind == k.ITEM_COMPLETED:
        return _text("item done")
    if kind == k.EXPERIMENT_COMPLETED:
        return _text("experiment done")
    if kind == k.WORKFLOW_CHANGED:
        return _text("setting", detail=f"{p.get('field')} → {_plain(p.get('value'))}")
    if kind == k.EDIT:
        item = p.get("item")
        name = item.get("name") if isinstance(item, dict) else None
        return _text(
            "edited",
            detail=_join(name, p.get("task"), p.get("target")),
            note=p.get("via"),
        )
    if kind == k.STAGE_MOVED:
        return _stage_move(p)
    if kind == k.BEAM_SHIFTED:
        return _text("beam shift", detail=p.get("beam_type"))
    if kind in (k.IMAGE_ACQUIRED, k.FM_IMAGE_ACQUIRED):
        hfw = p.get("hfw")
        return _text(
            "image",
            detail=_join(
                p.get("beam_type") or ("FM" if kind == k.FM_IMAGE_ACQUIRED else ""),
                f"{hfw * 1e6:.0f} µm" if _num(hfw) else None,
            ),
            note=_file_name(p.get("path")) if p.get("path") else None,
        )
    if kind == k.MILLING:
        failed = p.get("status") == "failed"
        return _text(
            "milling",
            detail=_join(
                p.get("milling_task_name"), None if failed else p.get("status")
            ),
            note=p.get("reason"),
            error="failed" if failed else None,
            failed=failed,
        )
    if kind == k.MILLING_STAGE_STARTED:
        stage = p.get("stage")
        stage_name = stage.get("name") if isinstance(stage, dict) else None
        return _text(
            "milling stage", detail=p.get("name") or p.get("stage_name") or stage_name
        )
    if kind == k.AUTOFOCUS:
        return _text("autofocus", detail=_join(p.get("beam_type"), p.get("method")))
    if kind == k.ALIGNMENT:
        return _text("alignment", detail=p.get("status"))

    # Anything else: its plain fields, briefly.
    brief = [
        f"{key} {_plain(value)}"
        for key, value in p.items()
        if value is None or isinstance(value, (str, int, float, bool))
    ][:4]
    return _text(kind.replace("_", " "), note=" · ".join(brief))
