"""What the App Server's routes take: a dataclass per request body.

A route declares its body as one of these, so FastAPI reads it and
``/openapi.json`` describes requests as `responses` describes answers. Each
class carries ``STRICT``, so a key it doesn't have is refused, as the codec's
strict decode refuses one: on the wire, a key the route doesn't take is the
caller's mistake. Numbers sent as text read as numbers, as the codec reads them.

A body that doesn't read is answered as every refusal is, ``{error_type,
message}`` (`invalid_body`), with ``missing_field`` when only required fields
are missing. Checks a type can't say (a length, a pattern, text that isn't
blank) stay in the route, with their own messages.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from fastapi import HTTPException
from pydantic import ConfigDict

from fibsem.server.auth import refusal_error

# Unknown keys refused: set as each class's ``__pydantic_config__``.
STRICT = ConfigDict(extra="forbid")


def invalid_body(errors: Sequence[Mapping[str, Any]]) -> HTTPException:
    """FastAPI's validation errors as one refusal: each field by its path,
    what was wrong with it."""
    said = []
    for error in errors:
        loc = [str(part) for part in error.get("loc", ())]
        where = ".".join(loc[1:] if loc[:1] == ["body"] else loc)
        kind = error.get("type")
        if not where:
            said.append(
                "Send a JSON body."
                if kind == "missing"
                else "The body is a JSON object."
            )
        elif kind == "missing":
            said.append(f"{where} is required.")
        elif kind in ("extra_forbidden", "unexpected_keyword_argument"):
            said.append(f"{where} is not a field this route takes.")
        else:
            message = str(error.get("msg", "")).removeprefix("Value error, ")
            said.append(f"{where}: {message}.")
    missing = all(error.get("type") == "missing" for error in errors)
    return refusal_error(422, "missing_field" if missing else "invalid", " ".join(said))


# --- the agent's own routes (`fibsem.server.agent`) ---------------------------


@dataclass
class MarkedEvidence:
    """An image the operator marked for the agent, sent with a message
    (`evidence.save`). ``keep`` keeps it as an example: True, or the
    knowledge section (``sample``, ``playbook``, ``instrument``)."""

    __pydantic_config__ = STRICT

    image: Optional[str] = None
    marks: Optional[Any] = None
    channels: Optional[List[Any]] = None
    answers: Optional[str] = None
    verdict: Optional[str] = None
    judged: Optional[List[Dict[str, Any]]] = None
    keep: Union[bool, str, None] = None
    sample: Optional[str] = None


@dataclass
class MessageBody:
    """POST /app/agent/messages: the operator's message to the agent. ``text``
    may be empty when ``evidence`` says it."""

    __pydantic_config__ = STRICT

    text: str = ""
    evidence: Optional[MarkedEvidence] = None


@dataclass
class LevelBody:
    """PUT /app/agent/level."""

    __pydantic_config__ = STRICT

    level: str


@dataclass
class ReasonBody:
    """A decision with the operator's reason: denying a connection, rejecting
    a request."""

    __pydantic_config__ = STRICT

    reason: str = ""


@dataclass
class AgentRequestBody:
    """POST /app/agent/requests: the agent's actions, each ``{tool,
    arguments}``, and why, for the operator to approve at Assist."""

    __pydantic_config__ = STRICT

    actions: List[Dict[str, Any]]
    reason: str


@dataclass
class WithdrawBody:
    """POST /app/agent/requests/{id}/withdraw."""

    __pydantic_config__ = STRICT

    reason: str


@dataclass
class PlanSectionBody:
    """POST /app/agent/plan/section: one section; empty text removes it."""

    __pydantic_config__ = STRICT

    section: str
    text: str = ""


@dataclass
class PlanBody:
    """PUT /app/agent/plan: the whole plan, its sections under ## headings."""

    __pydantic_config__ = STRICT

    markdown: str


@dataclass
class TranscriptBody:
    """POST /app/agent/transcript: lines fibsem-agent printed."""

    __pydantic_config__ = STRICT

    session_id: str
    entries: List[Any] = field(default_factory=list)


@dataclass
class MessagesReadBody:
    """POST /app/agent/read: the operator's messages the agent has read."""

    __pydantic_config__ = STRICT

    ids: List[str]


@dataclass
class LibraryFileBody:
    """PUT /app/library/{kind}/{name}: the file's ``text``, or its ``data``
    for the library to write."""

    __pydantic_config__ = STRICT

    text: Optional[str] = None
    data: Optional[Any] = None
