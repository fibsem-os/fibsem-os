"""What the agent did lately, one line each with how it ended.

The agent card's Recent list: the agent's requests (asked, then approved,
applying, then done or failed, or rejected with the operator's why), its
suggestions on proposals (approved, declined with the why, or overruled),
and what it does itself at Drive or was refused. The operator's messages to it
are rows too, and its replies and notes. Read from the session's
record, the event stream the event log shows, so the list and the log never
disagree; only the agent's own entries make a row. Each row says the level it
was done at.

The conversation (the operator's messages, the agent's replies and notes) is
kept whole, newlines and all, for the card to draw as Markdown; an action is one
line, how it ended one word, and why it failed or was refused, declined or
rejected beside it. The rows run oldest first, like a chat.

A direct call is the server's ``agent_called``, in the words a request's card
uses for the same tool; a long command is its run, running until it ends. What
an approved request or a long command does on the way (the stage moves of an
overview, a task's images) is that request's or that command's row, not rows of
its own.

No Qt: the window feeds it records and draws `rows`. The page reads them
with `read_rows`, which reaches back past the live buffer into the record's
file: after an app restart, which empties the buffer, and for Show earlier.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from fibsem.acting import AGENT, OPERATOR
from fibsem.ai.agent.catalog.describe import describe_action, describe_done
from fibsem.util import wording
from fibsem.workflows.actions import is_message
from fibsem.workflows.events.kinds import LIVE_KINDS
from fibsem.workflows.events.recording import read_events_before
from fibsem.workflows.proposals import kind_label

# Rows kept: the card scrolls through them all. The page asks for more, a
# page at a time, up to `MAX_SIZE`.
SIZE = 50
MAX_SIZE = 2000

# How a row ended, as the card colours it.
WAITING, RUNNING, OK, ERROR, MUTED = "waiting", "running", "ok", "error", "muted"

# What a row is: the operator's message, the agent's message or note, or an action.
YOU, MESSAGE, NOTE, ACTION = "you", "message", "note", "action"

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
    # The evidence it cites (its record), or None.
    evidence: Optional[Dict[str, Any]] = None
    # Why it failed, or was refused, declined or rejected; "" otherwise.
    why: str = ""
    # YOU, MESSAGE or NOTE for the conversation, else ACTION.
    kind: str = ACTION
    # When, as the record has it (ISO with its offset): the web page places
    # the agent's transcript between the rows by it. An operator's message
    # the agent read later is placed where it read it.
    at: str = ""
    # An operator's message: its id, and when the agent read it ("" not yet).
    message_id: str = ""
    read: str = ""
    # A side question (the operator's ``/btw`` message, marked ``aside``) or
    # the agent's answer to one: the page shows these in its side chat, not
    # in the conversation.
    aside: bool = False

    @property
    def conversation(self) -> bool:
        return self.kind != ACTION


def _stream_of(record: Dict[str, Any]) -> str:
    """The stream a record was written in (its id up to the seq): a new one
    each time the app starts."""
    stream, _, seq = str(record.get("id") or "").rpartition("-")
    return stream if seq.isdigit() else ""


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


def _suggested(p: Dict[str, Any]) -> str:
    """What a suggestion would decide, in a few words."""
    if p.get("outcome") == "Rejected":
        return "reject it"
    if p.get("action"):
        return _words(p["action"])
    values = p.get("values") or {}
    place = values.get("stage_position") if len(values) == 1 else None
    if isinstance(place, dict):
        kept = values == (p.get("proposed") or {})
        return (
            f"{'keep the position' if kept else 'move to'} {wording.stage_place(place)}"
        )
    if values:
        return wording.values_text(values)
    if p.get("patch"):
        return f"set {wording.values_text(p['patch'])}"
    return "confirm it"


def _command_text(p: Dict[str, Any]) -> str:
    """A long command's line: the call that started it, else its name."""
    if p.get("action"):
        return str(p["action"])
    return f"{_words(p.get('command')).capitalize()} ({_words(p.get('service'))})"


class RecentActions:
    """The agent's last `SIZE` rows, oldest first."""

    def __init__(self, size: int = SIZE) -> None:
        self._size = size
        self._rows: "OrderedDict[Any, RecentRow]" = OrderedDict()
        # What the agent's own rows stand for while it runs: an approved
        # request, a long command.
        self._applying: set = set()
        # A request's number of actions, for "2 of 3".
        self._sizes: Dict[Any, int] = {}
        # Why a request's step failed, for its "failed: ..." row.
        self._failures: Dict[Any, str] = {}
        # The level the agent was last seen at, for a row whose record omits it.
        self._level = ""
        # The evidence kept this session, by id, for the rows that cite it.
        self._evidence: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._citing: Optional[Dict[str, Any]] = None
        # The operator's messages not read yet: their row's key, by id.
        self._unread: Dict[str, Any] = {}
        # The ids of the operator's side questions, for the answers to them.
        self._asides: Set[str] = set()
        # The stream the last record came from: a new one is the app restarted.
        self._stream = ""

    @property
    def rows(self) -> List[RecentRow]:
        return list(self._rows.values())

    def clear(self) -> None:
        self._rows.clear()
        self._applying.clear()
        self._sizes.clear()
        self._failures.clear()
        self._level = ""
        self._evidence.clear()
        self._unread.clear()
        self._asides.clear()
        self._stream = ""

    def add(self, record: Dict[str, Any]) -> bool:
        """Take one record of the session's; True when the rows changed."""
        kind = record.get("kind")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        handler = getattr(self, f"_on_{kind}", None)
        if handler is None:
            return False
        stream = _stream_of(record)
        if stream and self._stream and stream != self._stream:
            self._restarted()
        self._stream = stream or self._stream
        self._citing = self._evidence.get(str(payload.get("evidence")))
        try:
            return bool(handler(record, payload, record.get("actor") == AGENT))
        finally:
            self._citing = None

    def _restarted(self) -> None:
        """The app started again: what was running then ended with it, and no
        request it was applying is applying now."""
        for key, row in list(self._rows.items()):
            if row.tone == RUNNING:
                self._rows[key] = replace(row, outcome="app restarted", tone=MUTED)
        self._applying.clear()
        self._sizes.clear()
        self._failures.clear()

    # --- rows ------------------------------------------------------------------

    def _put(
        self,
        key: Any,
        record: Dict[str, Any],
        text: str,
        outcome: str,
        tone: str,
        level: Any = None,
        why: str = "",
        kind: str = ACTION,
    ):
        level = str(self._level if level is None else level)
        text = str(text).strip() if kind != ACTION else _cut(text)
        self._rows.pop(key, None)
        self._rows[key] = RecentRow(
            _when(record),
            text,
            outcome,
            tone,
            level,
            self._citing,
            _cut(why, 120),
            kind,
            record.get("t") if isinstance(record.get("t"), str) else "",
        )
        while len(self._rows) > self._size:
            self._rows.popitem(last=False)
        return True

    def _end(self, key: Any, outcome: str, tone: str, why: Any = "") -> bool:
        row = self._rows.get(key)
        if row is None:
            return False
        # A new row, not the old one changed: the card compares the rows it
        # has with the ones it is given, and the same object never differs.
        self._rows[key] = replace(
            row, outcome=outcome, tone=tone, why=_cut(str(why or "").strip(), 120)
        )
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
            return self._end(
                ("request", request_id), "rejected", MUTED, p.get("reason")
            )
        self._applying.add(request_id)
        return self._end(("request", request_id), "approved", RUNNING)

    def _on_request_applied(self, record, p, agent):
        """One action of an approved request done: the next one runs."""
        request_id = p.get("request_id")
        of = self._sizes.get(request_id) or 0
        done = int(p.get("index") or 0) + 1
        if not p.get("ok"):
            self._failures[request_id] = _why_failed(p.get("error"))
        if not p.get("ok") or done >= of:
            return False  # request_finished says how it ended
        return self._end(("request", request_id), f"{done + 1} of {of}", RUNNING)

    def _on_request_finished(self, record, p, agent):
        request_id = p.get("request_id")
        self._applying.discard(request_id)
        self._sizes.pop(request_id, None)
        why = self._failures.pop(request_id, "")
        status = p.get("status")
        if status == "applied":
            return self._end(("request", request_id), "done", OK)
        if status == "failed":
            return self._end(("request", request_id), "failed", ERROR, why)
        return self._end(("request", request_id), _words(status), MUTED)

    def _on_request_cancelled(self, record, p, agent):
        self._applying.discard(p.get("request_id"))
        # The agent taking it back says who ended it.
        if p.get("withdrawn"):
            return self._end(("request", p.get("request_id")), "withdrawn", MUTED)
        return self._end(
            ("request", p.get("request_id")), "cancelled", MUTED, p.get("why")
        )

    # --- its suggestions on proposals ---------------------------------------------

    def _on_proposal_suggested(self, record, p, agent):
        if not agent:
            return False
        item = (p.get("item") or {}).get("name") or ""
        # what it suggested: the reason is the card's Why
        what = p.get("label") or kind_label(str(p.get("kind") or ""))
        text = f"Suggested on {item} · {p.get('task')} · {what}: {_suggested(p)}"
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
            ("suggestion", p.get("proposal_id")), "declined", MUTED, reason
        )

    def _on_proposal_decided(self, record, p, agent):
        if p.get("kind") == "actions":
            return self._request_decided(record, p, agent)
        key = ("suggestion", p.get("proposal_id"))
        if p.get("suggestion_id"):
            return self._end(key, "approved", OK)
        row = self._rows.get(key)
        if row is not None and row.tone == WAITING:
            return self._end(key, "overruled", MUTED)
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
        text = str(p.get("action") or "").strip() or "Acted"
        why = ""
        if p.get("ok"):
            outcome, tone = "done", OK
            text = describe_done(p.get("tool"), p.get("arguments"), text) or text
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
            outcome, tone = "refused", ERROR
        return self._put(
            ("call", record.get("seq")),
            record,
            text,
            outcome,
            tone,
            p.get("level"),
            why,
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
            RUNNING,
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

    def _on_evidence(self, record, p, agent):
        # Not a row: the call, request, suggestion or note citing it is.
        if p.get("evidence_id"):
            self._evidence[str(p["evidence_id"])] = dict(p)
            while len(self._evidence) > 4 * self._size:
                self._evidence.popitem(last=False)
        return False

    def _on_note(self, record, p, agent):
        if agent:
            return self._on_agent_note(record, p, agent)
        if record.get("actor") == OPERATOR and is_message(record):
            return self._on_operator_message(record, p, agent)
        return False

    def _on_agent_note(self, record, p, agent):
        # Its message to the operator (an answer to theirs), or a note.
        self._read(record, [p.get("reply_to")])
        key = ("note", record.get("seq"))
        self._put(
            key,
            record,
            p.get("text", ""),
            "",
            MUTED,
            kind=MESSAGE if is_message(record) else NOTE,
        )
        if str(p.get("reply_to") or "") in self._asides:
            self._rows[key] = replace(self._rows[key], aside=True)
        return True

    def _on_operator_message(self, record, p, agent):
        # The operator's own line, at no level: the agent answers with a message.
        key = ("message", record.get("seq"))
        self._put(key, record, p.get("text", ""), "", MUTED, level="", kind=YOU)
        message_id = str(p.get("id") or "")
        if message_id:
            self._rows[key] = replace(self._rows[key], message_id=message_id)
            self._unread[message_id] = key
        if p.get("aside"):
            self._rows[key] = replace(self._rows[key], aside=True)
            if message_id:
                self._asides.add(message_id)
        return True

    def _on_agent_messages_read(self, record, p, agent):
        return self._read(record, p.get("ids") or [])

    def _read(self, record, ids) -> bool:
        """The agent read the messages *ids*: each row moves to where it read
        it, after what the agent did before."""
        at = record.get("t") if isinstance(record.get("t"), str) else ""
        changed = False
        for message_id in ids:
            key = self._unread.pop(str(message_id or ""), None)
            row = self._rows.pop(key, None) if key is not None else None
            if row is not None:
                self._rows[key] = replace(row, read=at or "read", at=at or row.at)
                changed = True
        return changed

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
            "refused",
            ERROR,
            p.get("level"),
            why,
        )


# The kinds that can make a row (or, like evidence, feed one), and the ones a
# read of the file leaves unparsed.
ROW_KINDS = frozenset(
    name[len("_on_") :] for name in dir(RecentActions) if name.startswith("_on_")
)
_UNREAD_KINDS = LIVE_KINDS - ROW_KINDS
# How many records a read of the file takes first, for each row asked, and
# the most it takes in all.
_RECORDS_PER_ROW = 4
_RECORDS_MAX = 200_000


def _rows_of(
    records: Iterable[Dict[str, Any]], size: int, stream: Optional[str] = None
) -> List[RecentRow]:
    rows = RecentActions(size)
    for record in records:
        rows.add(record)
    # The last record from an earlier start of the app's: what was running
    # then ended with it, though nothing of this stream's says so yet.
    if stream and rows._stream and rows._stream != stream:
        rows._restarted()
    return rows.rows


class EarlierRecords:
    """The records before the live buffer, read from the record's file, and
    kept: the page reads the rows again on every event, and what is written
    before the buffer doesn't change."""

    def __init__(self) -> None:
        # The last read's key and what it found, as one: readers share it.
        self._last: Tuple[Any, Tuple[List[Dict[str, Any]], int]] = (None, ([], 0))

    def read(
        self, path: Path, held: Sequence[Dict[str, Any]], limit: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Up to *limit* records that can make a row, written before *held*
        (the buffer, oldest first), oldest first; and where in the file the
        read stopped, 0 for its start."""
        oldest = held[0].get("timestamp") if held else None
        # With nothing held, nothing of this stream's is being written: the
        # file's end is where the read starts, and it moves only if it grows.
        key = (str(path), oldest, limit, None if held else path.stat().st_size)
        last_key, found = self._last
        if key == last_key:
            return found
        ids = {record.get("id") for record in held}

        def keep(record: Dict[str, Any]) -> bool:
            if record.get("kind") not in ROW_KINDS or record.get("id") in ids:
                return False
            stamp = record.get("timestamp")
            return (
                oldest is None
                or not isinstance(stamp, (int, float))
                or (stamp <= oldest)
            )

        found = read_events_before(
            path, limit=limit, skip_kinds=_UNREAD_KINDS, keep=keep
        )
        self._last = (key, found)
        return found


def read_rows(
    held: Sequence[Dict[str, Any]],
    path: Optional[Path],
    size: int = SIZE,
    earlier: Optional[EarlierRecords] = None,
    stream: Optional[str] = None,
) -> Tuple[List[RecentRow], bool]:
    """The last *size* rows, oldest first, and whether there are earlier ones:
    from *held* (the live buffer, oldest first) and, when it hasn't that
    many, from the record's file at *path* before it. The file is read
    backwards and only as far as the rows asked for need. *stream* is the
    buffer's: rows from another were written before the app last started."""
    size = max(1, min(int(size), MAX_SIZE))
    held = list(held)
    rows = _rows_of(held, size + 1, stream)
    if len(rows) > size or path is None or not Path(path).is_file():
        return rows[-size:], len(rows) > size
    earlier = earlier or EarlierRecords()
    limit = _RECORDS_PER_ROW * (size + 1)
    while True:
        older, start = earlier.read(Path(path), held, limit)
        rows = _rows_of(older + held, size + 1, stream)
        if len(rows) > size or start == 0 or limit >= _RECORDS_MAX:
            return rows[-size:], len(rows) > size or start > 0
        limit = min(limit * 4, _RECORDS_MAX)
