"""What the agent did lately, one line each with how it ended (FIB-1414).

The agent card's Recent list: the agent's requests (asked, then approved or
rejected, then applied or failed), its suggested answers (taken or dropped), and
what it does itself at Drive (moves, mills, answers, stops) or was refused. Read
from the session's record, the event stream the event log shows, so the list and
the log never disagree; only the agent's own entries make a row.

What an approved request or a long command of the agent's does on the way (the
stage moves of an overview, a task's images) is that request's or that
command's row, not rows of its own.

No Qt: the window feeds it records and draws `rows`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List

from fibsem.acting import AGENT
from fibsem.server.request_text import describe_action

# Rows kept: the card shows them all.
SIZE = 10

# How a row ended, as the card colours it.
WAITING, OK, ERROR, MUTED = "waiting", "ok", "error", "muted"

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


# A queue edit, as the timeline's menu names it (queue_changed).
_QUEUE_CHANGES = {
    "remove": "Took out of the queue",
    "move_up": "Moved up the queue",
    "move_down": "Moved down the queue",
    "run_next": "Queued to run next",
    "requeue": "Queued again",
    "requeue_front": "Queued again, first",
}


@dataclass
class RecentRow:
    """One thing the agent did: when (HH:MM), what, and how it ended."""

    when: str
    text: str
    outcome: str
    tone: str


def _when(record: Dict[str, Any]) -> str:
    t = record.get("t")
    if isinstance(t, str) and "T" in t:
        return t.split("T", 1)[1][:5]
    return ""


def _words(name: Any) -> str:
    return str(name or "").replace("_", " ")


def _cut(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _request_text(actions: Any) -> str:
    """A request's first action as the card wrote it, and how many more."""
    actions = [a for a in actions or [] if isinstance(a, dict)]
    if not actions:
        return "Asked"
    first = actions[0]
    line = describe_action(str(first.get("tool")), first.get("arguments") or {}, "")
    change = str(line.get("change") or "").split("\n", 1)[0]
    text = f"Asked: {line.get('what')} · {change}" if line.get("what") else change
    if len(actions) > 1:
        text += f", and {len(actions) - 1} more"
    return text


class RecentActions:
    """The agent's last `SIZE` actions, newest first."""

    def __init__(self, size: int = SIZE) -> None:
        self._size = size
        self._rows: "OrderedDict[Any, RecentRow]" = OrderedDict()
        self._questions: Dict[Any, Dict[str, Any]] = {}
        # What the agent's own rows stand for while it runs: an approved
        # request, a long command.
        self._applying: set = set()
        self._commands = 0

    @property
    def rows(self) -> List[RecentRow]:
        return list(reversed(self._rows.values()))

    def clear(self) -> None:
        self._rows.clear()
        self._questions.clear()
        self._applying.clear()
        self._commands = 0

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
        self, key: Any, record: Dict[str, Any], text: str, outcome: str, tone: str
    ):
        self._rows.pop(key, None)
        self._rows[key] = RecentRow(_when(record), _cut(text), outcome, tone)
        while len(self._rows) > self._size:
            self._rows.popitem(last=False)
        return True

    def _end(self, key: Any, outcome: str, tone: str) -> bool:
        row = self._rows.get(key)
        if row is None:
            return False
        row.outcome, row.tone = outcome, tone
        return True

    def _own(self, agent: bool) -> bool:
        """Whether an agent's record is a row of its own: not a step of an
        approved request or a long command it runs."""
        return agent and not self._applying and not self._commands

    # --- requests --------------------------------------------------------------

    def _on_request_raised(self, record, p, agent):
        key = ("request", p.get("request_id"))
        return self._put(
            key, record, _request_text(p.get("actions")), "waiting", WAITING
        )

    def _on_request_decided(self, record, p, agent):
        request_id = p.get("request_id")
        if p.get("outcome") == "rejected":
            return self._end(("request", request_id), "rejected", MUTED)
        self._applying.add(request_id)
        return self._end(("request", request_id), "approved", WAITING)

    def _on_request_finished(self, record, p, agent):
        request_id = p.get("request_id")
        self._applying.discard(request_id)
        status = p.get("status")
        if status == "applied":
            return self._end(("request", request_id), "applied", OK)
        if status == "failed":
            return self._end(("request", request_id), "failed", ERROR)
        return self._end(("request", request_id), _words(status), MUTED)

    def _on_request_cancelled(self, record, p, agent):
        self._applying.discard(p.get("request_id"))
        return self._end(("request", p.get("request_id")), "cancelled", MUTED)

    # --- the run's questions -----------------------------------------------------

    def _on_prompt_raised(self, record, p, agent):
        self._questions[p.get("question_id")] = p
        return False

    def _button(self, question_id: Any, response: Any) -> str:
        asked = self._questions.get(question_id) or {}
        button = asked.get("positive") if response else asked.get("negative")
        return str(button or ("yes" if response else "no"))

    def _question(self, question_id: Any, p: Dict[str, Any]) -> str:
        asked = self._questions.get(question_id) or {}
        return str(
            asked.get("message") or _words(p.get("type")) or "the run's question"
        )

    def _on_prompt_suggested(self, record, p, agent):
        question_id = p.get("question_id")
        text = (
            f"Suggested {self._button(question_id, p.get('response'))} for "
            f"'{self._question(question_id, p)}'"
        )
        return self._put(("suggestion", question_id), record, text, "waiting", WAITING)

    def _on_prompt_answered(self, record, p, agent):
        question_id = p.get("question_id")
        changed = False
        if p.get("suggested_by"):
            changed = self._end(("suggestion", question_id), "taken", OK)
        elif (
            row := self._rows.get(("suggestion", question_id))
        ) and row.tone == WAITING:
            changed = self._end(("suggestion", question_id), "dropped", MUTED)
        if p.get("answered_by") == AGENT:
            text = (
                f"Answered '{self._question(question_id, p)}': "
                f"{self._button(question_id, p.get('response'))}"
            )
            changed = self._put(("answer", question_id), record, text, "done", OK)
        return changed

    def _on_prompt_cancelled(self, record, p, agent):
        row = self._rows.get(("suggestion", p.get("question_id")))
        if row is not None and row.tone == WAITING:
            return self._end(("suggestion", p.get("question_id")), "dropped", MUTED)
        return False

    # --- what it does itself -----------------------------------------------------

    def _on_command_started(self, record, p, agent):
        if agent:
            self._commands += 1
        return False

    def _on_command_ended(self, record, p, agent):
        if not agent:
            return False
        self._commands = max(0, self._commands - 1)
        if p.get("service") == "milling":
            return False  # its mill's own row says it
        if self._applying:
            return False
        outcome, tone = _RUN_ENDED.get(
            str(p.get("state")), (_words(p.get("state")), MUTED)
        )
        text = f"{_words(p.get('command')).capitalize()} ({_words(p.get('service'))})"
        return self._put(("command", p.get("run_id")), record, text, outcome, tone)

    def _on_milling(self, record, p, agent):
        if not agent or self._applying:
            return False
        status = str(p.get("status") or "")
        outcome, tone = _RUN_ENDED.get(status, (_words(status), MUTED))
        text = f"Milled '{p.get('milling_task_name') or 'milling'}'"
        return self._put(("mill", p.get("milling_run_id")), record, text, outcome, tone)

    def _on_stage_moved(self, record, p, agent):
        if not self._own(agent):
            return False
        failed = bool(p.get("error"))
        text = f"Moved the stage: {_words(p.get('move') or 'move')}"
        return self._put(
            ("move", record.get("seq")),
            record,
            text,
            "failed" if failed else "done",
            ERROR if failed else OK,
        )

    def _on_image_acquired(self, record, p, agent):
        if not self._own(agent):
            return False
        beam = {"ELECTRON": "SEM", "ION": "FIB"}.get(str(p.get("beam_type")), "")
        hfw = p.get("hfw")
        width = f", {hfw * 1e6:.0f} µm" if isinstance(hfw, (int, float)) else ""
        text = f"Took an image{': ' + beam if beam else ''}{width}"
        return self._put(("image", record.get("seq")), record, text, "done", OK)

    def _on_workflow_start(self, record, p, agent):
        if not agent:
            return False
        tasks = ", ".join(str(t) for t in p.get("task_names") or []) or "a run"
        if "grid_names" in p:
            names, every = p.get("grid_names"), "every grid"
        else:
            names, every = p.get("item_names"), "every lamella"
        on = ", ".join(str(n) for n in names) if names else every
        return self._put(
            ("start", record.get("seq")), record, f"Started {tasks} on {on}", "done", OK
        )

    def _on_edit(self, record, p, agent):
        if not self._own(agent):
            return False
        item = (
            (p.get("item") or {}).get("name")
            if isinstance(p.get("item"), dict)
            else None
        )
        what = " · ".join(
            str(part) for part in (item, p.get("task"), p.get("target")) if part
        )
        return self._put(
            ("edit", record.get("seq")),
            record,
            f"Edited {what or 'a setting'}",
            "done",
            OK,
        )

    def _on_queue_changed(self, record, p, agent):
        if not self._own(agent):
            return False
        action = _QUEUE_CHANGES.get(str(p.get("action")), _words(p.get("action")))
        text = f"{action}: {p.get('item_name')} · {p.get('task_name')}"
        return self._put(("queue", record.get("seq")), record, text, "done", OK)

    def _on_agent_note(self, record, p, agent):
        return self._put(
            ("note", record.get("seq")), record, f"Note: {p.get('text', '')}", "", MUTED
        )

    def _on_agent_stop(self, record, p, agent):
        text = f"Stopped {_words(p.get('action'))}"
        if p.get("reason"):
            text += f": {p['reason']}"
        return self._put(("stop", record.get("seq")), record, text, "done", OK)

    def _on_agent_refused(self, record, p, agent):
        error = p.get("error_type")
        why = _REFUSED.get(error) or f"not at {p.get('level')}"
        text = f"Tried to {_words(p.get('action'))}"
        return self._put(
            ("refused", record.get("seq")), record, text, f"refused: {why}", ERROR
        )
