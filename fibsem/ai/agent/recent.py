"""What the agent did lately, one line each with how it ended (FIB-1414).

The agent card's Recent list: the agent's requests (asked, then approved,
applying, then applied or failed, or rejected with the operator's why), its
suggestions on proposals (approved, declined with the why, or decided otherwise),
and what it does itself at Drive or was refused. The operator's messages to it are rows too, "You: ...",
and its answers "Reply: ..." (FIB-1423). Read from the session's record, the event stream the event
log shows, so the list and the log never disagree; only the agent's own entries
make a row. Each row says the level it was done at (FIB-1426).

A direct call is the server's ``agent_called``, in the words a request's card
uses for the same tool; a long command is its run, running until it ends. What
an approved request or a long command does on the way (the stage moves of an
overview, a task's images) is that request's or that command's row, not rows of
its own.

No Qt: the window feeds it records and draws `rows`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Dict, List

from fibsem.acting import AGENT, OPERATOR
from fibsem.ai.agent.request_text import describe_action

# Rows kept: the card shows them all.
SIZE = 10

# How a row ended, as the card colours it.
WAITING, OK, ERROR, MUTED = "waiting", "ok", "error", "muted"

# The levels the agent acts at, as rows name them.
ASSIST, DRIVE = "assist", "drive"

# Why the agent was refused, in the operator's words (server.AGENT_REFUSALS).
_REFUSED = {
    "ask_first": "needs your approval",
    "agent_off": "the agent is off",
    "agent_stopped": "the agent was stopped",
    "not_allowed": "never allowed",
}

_RUN_ENDED = {
    "completed": ("done", OK),
    "done": ("done", OK),
    "cancelled": ("stopped", MUTED),
    "failed": ("failed", ERROR),
}


# A mill started and left running: its mill's row says how it went.
_MILL_START = "/services/milling/commands/start"

# A call the agent's level doesn't take, by the level it was at.
_LEVEL_REFUSED = {
    "watch": "Watch is read-only",
    "assist": "not allowed at Assist",
    "drive": "at Drive the agent acts itself",
    "off": "the agent is off",
}


def _first_line(text: Any) -> str:
    """A route's message as a reason: its first line, cut for a row."""
    return _cut(str(text or "").strip().split("\n", 1)[0], 120)


# Why a direct call was refused, beyond the agent's permissions (agent_called).
_CALL_REFUSED = {
    "microscope_busy": "busy",
    "microscope_off": "not served",
}


@dataclass
class RecentRow:
    """One thing the agent did: when (HH:MM), what, and how it ended."""

    when: str
    text: str
    outcome: str
    tone: str
    # The level it was done at, "assist" or "drive"; "" when not known.
    level: str = ""


def _when(record: Dict[str, Any]) -> str:
    t = record.get("t")
    if isinstance(t, str) and "T" in t:
        return t.split("T", 1)[1][:5]
    return ""


def _words(name: Any) -> str:
    return str(name or "").replace("_", " ")


def _why_failed(error: Any) -> str:
    """A failed step's error, as its message alone."""
    if isinstance(error, dict):
        error = error.get("message") or error.get("error_type") or ""
    return _cut(error or "", 120)


def _cut(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _request_text(actions: Any) -> str:
    """A request's first action as the card wrote it, and how many more."""
    actions = [a for a in actions or [] if isinstance(a, dict)]
    if not actions:
        return "Asked"
    first = actions[0]
    if first.get("what") or first.get("change"):
        line = first
    else:
        line = describe_action(str(first.get("tool")), first.get("arguments") or {}, "")
    change = str(line.get("change") or "").split("\n", 1)[0]
    text = f"Asked: {line.get('what')} · {change}" if line.get("what") else change
    if len(actions) > 1:
        text += f", and {len(actions) - 1} more"
    return text


def _command_text(p: Dict[str, Any]) -> str:
    """A long command's line: the call that started it, else its name."""
    if p.get("action"):
        return str(p["action"])
    return f"{_words(p.get('command')).capitalize()} ({_words(p.get('service'))})"


class RecentActions:
    """The agent's last `SIZE` actions, newest first."""

    def __init__(self, size: int = SIZE) -> None:
        self._size = size
        self._rows: "OrderedDict[Any, RecentRow]" = OrderedDict()
        # What the agent's own rows stand for while it runs: an approved
        # request, a long command.
        self._applying: set = set()
        # A request's number of actions, for "applying 2 of 3".
        self._sizes: Dict[Any, int] = {}
        # Why a request's step failed, for its "failed: ..." row.
        self._failures: Dict[Any, str] = {}
        # The level the agent was last seen at, for a row whose record omits it.
        self._level = ""

    @property
    def rows(self) -> List[RecentRow]:
        return list(reversed(self._rows.values()))

    def clear(self) -> None:
        self._rows.clear()
        self._applying.clear()
        self._sizes.clear()
        self._failures.clear()
        self._level = ""

    def add(self, record: Dict[str, Any]) -> bool:
        """Take one record of the session's; True when the rows changed."""
        kind = record.get("kind")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        handler = getattr(self, f"_on_{kind}", None)
        if handler is None:
            return False
        return bool(handler(record, payload, record.get("actor") == AGENT))

    # --- rows ------------------------------------------------------------------

    def _put(
        self,
        key: Any,
        record: Dict[str, Any],
        text: str,
        outcome: str,
        tone: str,
        level: Any = None,
    ):
        level = str(self._level if level is None else level)
        self._rows.pop(key, None)
        self._rows[key] = RecentRow(_when(record), _cut(text), outcome, tone, level)
        while len(self._rows) > self._size:
            self._rows.popitem(last=False)
        return True

    def _end(self, key: Any, outcome: str, tone: str) -> bool:
        row = self._rows.get(key)
        if row is None:
            return False
        # A new row, not the old one changed: the card compares the rows it
        # has with the ones it is given, and the same object never differs.
        self._rows[key] = replace(row, outcome=outcome, tone=tone)
        return True

    # --- requests --------------------------------------------------------------

    def _on_agent_level_set(self, record, p, agent):
        self._level = str(p.get("level") or "")
        return False

    def _on_proposal_asked(self, record, p, agent):
        """The agent's request, on the record as an ``actions`` proposal."""
        if p.get("kind") != "actions":
            return False
        actions = (p.get("proposed") or {}).get("actions")
        return self._on_request_raised(
            record, {"request_id": p.get("proposal_id"), "actions": actions}, agent
        )

    def _request_decided(self, record, p, agent):
        request = {"request_id": p.get("proposal_id"), "reason": p.get("reason")}
        outcome = p.get("outcome")
        if outcome == "Confirmed":
            return self._on_request_decided(record, request, agent)
        if outcome == "Rejected":
            return self._on_request_decided(
                record, {**request, "outcome": "rejected"}, agent
            )
        withdrawn = str(p.get("actor") or "").startswith("agent:")
        return self._on_request_cancelled(
            record, {**request, "why": p.get("reason"), "withdrawn": withdrawn}, agent
        )

    def _on_request_raised(self, record, p, agent):
        key = ("request", p.get("request_id"))
        self._sizes[p.get("request_id")] = len(p.get("actions") or [])
        return self._put(
            key, record, _request_text(p.get("actions")), "waiting", WAITING, ASSIST
        )

    def _on_request_decided(self, record, p, agent):
        request_id = p.get("request_id")
        if p.get("outcome") == "rejected":
            reason = str(p.get("reason") or "").strip()
            outcome = f"rejected: {reason}" if reason else "rejected"
            return self._end(("request", request_id), outcome, MUTED)
        self._applying.add(request_id)
        return self._end(("request", request_id), "approved", WAITING)

    def _on_request_applied(self, record, p, agent):
        """One action of an approved request done: the next one runs."""
        request_id = p.get("request_id")
        of = self._sizes.get(request_id) or 0
        done = int(p.get("index") or 0) + 1
        if not p.get("ok"):
            self._failures[request_id] = _why_failed(p.get("error"))
        if not p.get("ok") or done >= of:
            return False  # request_finished says how it ended
        return self._end(
            ("request", request_id), f"applying {done + 1} of {of}", WAITING
        )

    def _on_request_finished(self, record, p, agent):
        request_id = p.get("request_id")
        self._applying.discard(request_id)
        self._sizes.pop(request_id, None)
        why = self._failures.pop(request_id, "")
        status = p.get("status")
        if status == "applied":
            return self._end(("request", request_id), "applied", OK)
        if status == "failed":
            outcome = f"failed: {why}" if why else "failed"
            return self._end(("request", request_id), outcome, ERROR)
        return self._end(("request", request_id), _words(status), MUTED)

    def _on_request_cancelled(self, record, p, agent):
        self._applying.discard(p.get("request_id"))
        # The agent taking it back says who ended it (sim session 2026-10-16).
        outcome = "withdrawn" if p.get("withdrawn") else "cancelled"
        if not p.get("withdrawn") and p.get("why"):
            outcome = f"cancelled: {_cut(p['why'], 120)}"
        return self._end(("request", p.get("request_id")), outcome, MUTED)

    # --- its suggestions on proposals ---------------------------------------------

    def _on_proposal_suggested(self, record, p, agent):
        if not agent:
            return False
        item = (p.get("item") or {}).get("name") or ""
        text = f"Suggested on {item} · {p.get('task')}: {p.get('reason') or ''}"
        return self._put(
            ("suggestion", p.get("proposal_id")),
            record,
            text.rstrip(": "),
            "waiting",
            WAITING,
            ASSIST,
        )

    def _on_suggestion_declined(self, record, p, agent):
        reason = str(p.get("reason") or "").strip()
        return self._end(
            ("suggestion", p.get("proposal_id")),
            f"declined: {reason}" if reason else "declined",
            MUTED,
        )

    def _on_proposal_decided(self, record, p, agent):
        if p.get("kind") == "actions":
            return self._request_decided(record, p, agent)
        key = ("suggestion", p.get("proposal_id"))
        if p.get("suggestion_id"):
            return self._end(key, "approved", OK)
        row = self._rows.get(key)
        if row is not None and row.tone == WAITING:
            return self._end(key, "decided otherwise", MUTED)
        return False

    # --- what it does itself -----------------------------------------------------

    def _on_agent_called(self, record, p, agent):
        """A direct call, answered: done, refused or failed."""
        if not agent or self._applying:
            return False
        self._level = str(p.get("level") or self._level)
        if str(p.get("route", "")).endswith(_MILL_START):
            return False  # the mill's own row says it, when it ends
        status = p.get("status")
        if p.get("ok"):
            outcome, tone = "done", OK
        elif isinstance(status, int) and status >= 500:
            outcome, tone = "failed", ERROR
        else:
            error = p.get("error_type")
            why = (
                _CALL_REFUSED.get(error)
                or _first_line(p.get("message"))
                or _words(error)
                or f"error {status}"
            )
            outcome, tone = f"refused: {why}", ERROR
        text = str(p.get("action") or "").strip() or "Acted"
        return self._put(
            ("call", record.get("seq")), record, text, outcome, tone, p.get("level")
        )

    def _on_command_started(self, record, p, agent):
        """A long command of the agent's: running until its run ends. A mill is
        its mill's row."""
        if not agent or self._applying:
            return False
        self._level = str(p.get("level") or self._level)
        if p.get("service") == "milling":
            return False
        return self._put(
            ("command", p.get("run_id")),
            record,
            _command_text(p),
            "running",
            WAITING,
            p.get("level"),
        )

    def _on_command_ended(self, record, p, agent):
        if not agent or self._applying or p.get("service") == "milling":
            return False
        outcome, tone = _RUN_ENDED.get(
            str(p.get("state")), (_words(p.get("state")), MUTED)
        )
        key = ("command", p.get("run_id"))
        if key in self._rows:
            return self._end(key, outcome, tone)
        return self._put(key, record, _command_text(p), outcome, tone, p.get("level"))

    def _on_milling(self, record, p, agent):
        if not agent or self._applying:
            return False
        status = str(p.get("status") or "")
        outcome, tone = _RUN_ENDED.get(status, (_words(status), MUTED))
        text = f"Milled '{p.get('milling_task_name') or 'milling'}'"
        return self._put(("mill", p.get("milling_run_id")), record, text, outcome, tone)

    def _on_action(self, record, p, agent):
        """An act on the record (``fibsem.workflows.actions``), as the record
        it replaced reads: a request's step, a stop, a refusal or a call."""
        if p.get("request_id"):
            return self._on_request_applied(record, p, agent)
        if not agent:
            return False
        if p.get("stops"):
            return self._on_agent_stop(record, p, agent)
        if p.get("refused"):
            return self._on_agent_refused(record, {**p, "action": p["what"]}, agent)
        return self._on_agent_called(record, {**p, "action": p.get("what")}, agent)

    def _on_note(self, record, p, agent):
        if agent:
            return self._on_agent_note(record, p, agent)
        if record.get("actor") == OPERATOR:
            return self._on_operator_message(record, p, agent)
        return False

    def _on_agent_note(self, record, p, agent):
        # Its answer to the operator's message (FIB-1423), or an observation.
        head = "Reply" if p.get("reply_to") else "Note"
        return self._put(
            ("note", record.get("seq")),
            record,
            f"{head}: {p.get('text', '')}",
            "",
            MUTED,
        )

    def _on_operator_message(self, record, p, agent):
        # The operator's own line, at no level: the agent answers with a note.
        return self._put(
            ("message", record.get("seq")),
            record,
            f"You: {p.get('text', '')}",
            "",
            MUTED,
            level="",
        )

    def _on_agent_stop(self, record, p, agent):
        # An agent_stop said what it stopped as ``what``; an action, ``target``.
        stopped = p.get("target") if "stops" in p else p.get("what")
        verb = "Ended" if p.get("stops") == "end" else "Stopped"
        text = f"{verb} {stopped or _words(p.get('stops') or p.get('action'))}"
        if p.get("reason"):
            text += f": {p['reason']}"
        return self._put(
            ("stop", record.get("seq")), record, text, "done", OK, p.get("level")
        )

    def _on_agent_refused(self, record, p, agent):
        error = p.get("error_type")
        why = _REFUSED.get(error) or _LEVEL_REFUSED.get(
            str(p.get("level")), f"not at {p.get('level')}"
        )
        text = p.get("line") or f"Tried to {_words(p.get('action'))}"
        return self._put(
            ("refused", record.get("seq")),
            record,
            text,
            f"refused: {why}",
            ERROR,
            p.get("level"),
        )
