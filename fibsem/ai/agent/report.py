"""The agent's session as its harness reports it (``POST /app/agent/session``).

One report says what changed: a field left out (None) keeps what the server
holds. `SessionReport` is what the harness writes and what the server reads
with the codec's strict decode, so an unknown key or a bad value is refused
the same way on both sides. `ReportedSession` is what the server answers: the
reports folded together, with what the server adds itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import Field, StringConstraints

from fibsem.codec import encode

MAX_TEXT = 500
MAX_ITEM = 200
MAX_SKILLS = 20
MAX_CHECKLIST = 50
MAX_HOOKS = 20
MAX_PARTS = 20
MAX_LONG = 2000
MAX_LIST = 50
MAX_SERVERS = 20

Text = Annotated[str, StringConstraints(max_length=MAX_TEXT, strip_whitespace=True)]
Item = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_ITEM, strip_whitespace=True)
]
SkillName = Annotated[str, StringConstraints(max_length=100, strip_whitespace=True)]
Dollars = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
Count = Annotated[int, Field(ge=0, strict=True)]
Flag = Annotated[bool, Field(strict=True)]
# A playbook's line as its file gives it: a reminder or a rule.
Long = Annotated[str, StringConstraints(max_length=MAX_LONG, strip_whitespace=True)]


@dataclass
class Tokens:
    """Tokens its model calls used."""

    input: Optional[Count] = None
    output: Optional[Count] = None
    cache_read: Optional[Count] = None
    cache_write: Optional[Count] = None


@dataclass
class ContextPart:
    """One part of the context and its size, as the CLI's ``/context`` names
    it: "System prompt", "MCP tools", "Messages"."""

    name: Item
    tokens: Count


@dataclass
class ContextUse:
    """How full its context is, in tokens, and of what."""

    used: Optional[Count] = None
    window: Optional[Count] = None
    compact_at: Optional[Count] = None
    # The session's context_suggest and context_ceiling (`handoff.Ceiling`).
    suggest_at: Optional[Count] = None
    due_at: Optional[Count] = None
    parts: Optional[Annotated[List[ContextPart], Field(max_length=MAX_PARTS)]] = None


@dataclass
class ChecklistItem:
    text: Item
    # "always" is a standing rule.
    state: Literal["open", "done", "always"] = "open"


@dataclass
class Hook:
    """A reminder on now."""

    when: Text
    label: Text
    say: Text


@dataclass
class PlaybookSkill:
    skill: str
    state: Literal["not loaded", "loaded", "done"]
    summary: Optional[Text] = None


@dataclass
class PlaybookState:
    """The session's playbook; every field None (``{}``) for none."""

    name: Optional[Text] = None
    hash: Optional[Text] = None
    chosen_by: Optional[Text] = None
    skills: Optional[Annotated[List[PlaybookSkill], Field(max_length=MAX_SKILLS)]] = (
        None
    )

    def __post_init__(self) -> None:
        if not self.name and any(
            value is not None for value in (self.hash, self.chosen_by, self.skills)
        ):
            raise ValueError("a playbook has a name, or is {} for none")


@dataclass
class PlaybookReminder:
    when: Text
    label: Text
    say: Long


@dataclass
class PlaybookOption:
    """An optional skill: read *when* the job needs it, its reminders and
    subagents on while it is loaded."""

    skill: SkillName
    when: Long
    agents: Optional[Annotated[List[SkillName], Field(max_length=MAX_LIST)]] = None
    reminders: Optional[
        Annotated[List[PlaybookReminder], Field(max_length=MAX_HOOKS)]
    ] = None


@dataclass
class PlaybookAbout:
    """The playbook the session runs, as it was loaded (`Playbook.about`):
    what the agent follows, whatever its file says now. Sent when the
    session starts on it or ``playbook_set`` changes it."""

    name: SkillName
    hash: Optional[Text] = None
    description: Optional[Long] = None
    # The playbooks it builds on, the base first.
    extends: Optional[Annotated[List[SkillName], Field(max_length=MAX_LIST)]] = None
    instrument: Optional[Text] = None
    # Where its file was: fibsemOS's own, the lab's copy or a plugin's.
    source: Optional[Literal["shipped", "lab", "plugin"]] = None
    # The skills always read; the optional ones are read when needed.
    skills: Optional[Annotated[List[SkillName], Field(max_length=MAX_LIST)]] = None
    optional: Optional[Annotated[List[PlaybookOption], Field(max_length=MAX_LIST)]] = (
        None
    )
    reminders: Optional[
        Annotated[List[PlaybookReminder], Field(max_length=MAX_LIST)]
    ] = None
    rules: Optional[Annotated[List[Long], Field(max_length=MAX_LIST)]] = None
    agents: Optional[Annotated[List[SkillName], Field(max_length=MAX_LIST)]] = None
    mcp: Optional[Annotated[List[SkillName], Field(max_length=MAX_SERVERS)]] = None
    # The fibsem tools loaded from the first turn.
    tools: Optional[Annotated[List[SkillName], Field(max_length=MAX_LIST)]] = None


@dataclass
class McpServer:
    """An MCP server of the session, as the agent's CLI last said: its
    status (``connected``, ``failed``, ``pending``, ...) and its tools."""

    name: SkillName
    status: Text
    tools: Optional[Count] = None
    about: Optional[Text] = None


@dataclass
class SessionReport:
    session_id: Optional[Text] = None
    model: Optional[Text] = None
    goal: Optional[Text] = None
    # Why it ended, with ``ended``.
    reason: Optional[Text] = None
    harness: Optional[Text] = None
    effort: Optional[Text] = None
    workdir: Optional[Text] = None
    budget_usd: Optional[Dollars] = None
    spent_usd: Optional[Dollars] = None
    turns: Optional[Count] = None
    # How many contexts the session has had: 1, and one more for each fresh
    # context it started (`fibsem.ai.agent.handoff`).
    contexts: Optional[Count] = None
    tokens: Optional[Tokens] = None
    context: Optional[ContextUse] = None
    code: Optional[Flag] = None
    files: Optional[Flag] = None
    resumed: Optional[Flag] = None
    # The skills it has read, in order.
    skills: Optional[Annotated[List[SkillName], Field(max_length=MAX_SKILLS)]] = None
    checklist: Optional[
        Annotated[List[ChecklistItem], Field(max_length=MAX_CHECKLIST)]
    ] = None
    hooks: Optional[Annotated[List[Hook], Field(max_length=MAX_HOOKS)]] = None
    playbook: Optional[PlaybookState] = None
    playbook_about: Optional[PlaybookAbout] = None
    servers: Optional[Annotated[List[McpServer], Field(max_length=MAX_SERVERS)]] = None
    ended: Optional[Flag] = None


@dataclass
class Deadline:
    """A plan Rule that gives a time."""

    text: str
    by: str


@dataclass
class ReportedSession(SessionReport):
    """The session as the server holds it."""

    # The MCP client its sidecar named.
    client: Optional[str] = None
    # When the server first heard of it, ISO 8601 with its offset.
    started: Optional[str] = None
    # Read from its plan, when it keeps one.
    deadlines: Optional[List[Deadline]] = None
    # The budget the operator set (``PUT /app/agent/budget``), which
    # fibsem-agent keeps to; a report can't set it.
    budget_set_usd: Optional[float] = None
    # When the operator last asked for a fresh context (``POST
    # /app/agent/context-reset``), and the context it was asked in.
    reset_asked_at: Optional[str] = None
    reset_asked_in: Optional[int] = None
    # The instrument it is connected to and the experiment open there, as the
    # server knows them: ids, which a rename does not change. The serial
    # number is what the microscope reports, when it reports one.
    instrument_id: Optional[str] = None
    instrument_serial: Optional[str] = None
    experiment_id: Optional[str] = None


def plain(report: SessionReport) -> Dict[str, Any]:
    """*report* as JSON data without the fields it leaves out."""
    return _without_none(encode(report))


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_none(v) for v in value]
    return value
