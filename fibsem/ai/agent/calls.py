"""The agent's call the server is answering, in the words an operator reads.

The sidecar names the tool it is running and its arguments on each acting call
(`TOOL_HEADER`, `ARGUMENTS_HEADER`); an approved request names its action. The
server describes it as a request's card would ("Stage · Move 2 µm in x") for the
Now line, which shows what holds control of the microscope, and for the record's
``agent_called``, from which the card's Recent list reads the agent's direct
calls. A plain HTTP agent names no tool, and its call reads as its route does.

No FastAPI: the runs read it too.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional

TOOL_HEADER = "X-Agent-Tool"
ARGUMENTS_HEADER = "X-Agent-Arguments"
# The evidence the call rests on, by id (`fibsem.workflows.evidence`).
EVIDENCE_HEADER = "X-Agent-Evidence"

# The arguments are sent only while they fit a header comfortably: a list of
# patterns is left out, and the line says what the tool alone says.
ARGUMENTS_LIMIT = 4096


@dataclass
class AgentCall:
    """One call of the agent's: its tool and arguments when named, the line
    that describes it, and the level it was made at. ``run_id`` is the long
    command it started, whose own events then say how it went; ``recorded``,
    a route that puts the call on the record itself (a stop); ``evidence``, the
    id of the evidence it cites."""

    tool: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    action: Optional[str] = None
    level: Optional[str] = None
    run_id: Optional[str] = None
    recorded: bool = False
    evidence: Optional[str] = None


_current: "ContextVar[Optional[AgentCall]]" = ContextVar(
    "fibsem_agent_call", default=None
)


@contextmanager
def calling(call: AgentCall) -> Iterator[AgentCall]:
    """Serve the calls made inside this block as *call*."""
    token = _current.set(call)
    try:
        yield call
    finally:
        _current.reset(token)


def current_call() -> Optional[AgentCall]:
    """The agent's call being answered, or None."""
    return _current.get()


def describe(tool: Optional[str], arguments: Mapping[str, Any]) -> Optional[str]:
    """*tool* with *arguments* as one line, ``what · change``; None without a
    tool. Reads nothing: what a request's card adds from the live state (a move's
    from) is left out."""
    if not tool:
        return None
    from fibsem.ai.agent.catalog.describe import describe_action

    try:
        line = describe_action(tool, dict(arguments or {}), "")
    except Exception:  # noqa: BLE001 - the route's name stands in
        return None
    return join_line(line.get("what"), line.get("change"))


def join_line(what: Any, change: Any) -> Optional[str]:
    """A described action's ``what`` and its ``change`` on one line: every line
    of it, so a config edit of three stages names all three."""
    change = "; ".join(
        line.strip() for line in str(change or "").split("\n") if line.strip()
    )
    what = str(what or "").strip()
    if what and change:
        return f"{what} · {change}"
    return what or change or None


def from_headers(headers: Mapping[str, str]) -> AgentCall:
    """The call the sidecar named in *headers*; an unnamed one if it named none
    or sent arguments that don't parse."""
    tool = headers.get(TOOL_HEADER.lower()) or headers.get(TOOL_HEADER)
    raw = headers.get(ARGUMENTS_HEADER.lower()) or headers.get(ARGUMENTS_HEADER)
    arguments: Dict[str, Any] = {}
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            arguments = parsed
    evidence = headers.get(EVIDENCE_HEADER.lower()) or headers.get(EVIDENCE_HEADER)
    return AgentCall(
        tool=tool or None,
        arguments=arguments,
        action=describe(tool, arguments),
        evidence=evidence or None,
    )


def headers_for(
    tool: str, arguments: Mapping[str, Any], evidence: Optional[str] = None
) -> Dict[str, str]:
    """The headers that name *tool*, its *arguments* and the *evidence* it
    cites to the server."""
    headers = {TOOL_HEADER: tool}
    if evidence:
        headers[EVIDENCE_HEADER] = str(evidence)
    try:
        encoded = json.dumps(dict(arguments), ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        return headers
    if len(encoded) <= ARGUMENTS_LIMIT:
        headers[ARGUMENTS_HEADER] = encoded
    return headers
