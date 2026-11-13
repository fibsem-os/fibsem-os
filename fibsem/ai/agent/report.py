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

Text = Annotated[str, StringConstraints(max_length=MAX_TEXT, strip_whitespace=True)]
Item = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_ITEM, strip_whitespace=True)
]
SkillName = Annotated[str, StringConstraints(max_length=100, strip_whitespace=True)]
Dollars = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
Count = Annotated[int, Field(ge=0, strict=True)]
Flag = Annotated[bool, Field(strict=True)]


@dataclass
class Tokens:
    """Tokens its model calls used."""

    input: Optional[Count] = None
    output: Optional[Count] = None
    cache_read: Optional[Count] = None
    cache_write: Optional[Count] = None


@dataclass
class ContextUse:
    """How full its context is, in tokens."""

    used: Optional[Count] = None
    window: Optional[Count] = None
    compact_at: Optional[Count] = None


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


def plain(report: SessionReport) -> Dict[str, Any]:
    """*report* as JSON data without the fields it leaves out."""
    return _without_none(encode(report))


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_none(v) for v in value]
    return value
