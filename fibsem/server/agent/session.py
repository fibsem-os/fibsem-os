"""The agent's session, as the agent reports it.

What the App Server knows of the agent on its own is its token, when it was last
heard and the client its sidecar names. Its session -- which model, toward what
goal, on what budget, how much spent, how many tokens and how full its context
is -- only the agent's harness knows, so the harness reports it here:
``fibsem-agent`` after every turn, Claude Code through ``server_report_session``. fibsem-agent also says what it can do: code mode and
its folder, effort, skills. Self-reported, like the client's name: a label for
the operator, not an identity. Its checklist and deadlines are read from its
plan (``fibsem.server.agent.plan``) when it keeps one, for the Agent page and
for fibsem-agent's deadline wake-ups. A session starting, its goal changing and
its end go on the record, and so does the spend first reaching 80% of the
budget; the spend and the checklist in between are state, read with GET.

Which of the operator's messages the agent has read goes on the record too
(``POST /app/agent/read``): fibsem-agent says so as a turn carrying them
starts, ``fibsem-mcp watch`` as it prints them for any other agent.
"""

import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from fibsem.ai.agent.levels import AgentLevel
from fibsem.ai.agent.report import SessionReport, plain
from fibsem.ai.agent.session import budget_low
from fibsem.codec import decode
from fibsem.instrument_id import instrument_id
from fibsem.server.auth import AuthConfig, refusal_error, require_level
from fibsem.server.bodies import MessagesReadBody
from fibsem.server.responses import AgentSession

logger = logging.getLogger(__name__)

SESSION_STARTED = "agent_session_started"
SESSION_GOAL = "agent_session_goal"
SESSION_ENDED = "agent_session_ended"
SESSION_BUDGET = "agent_session_budget"
MESSAGES_READ = "agent_messages_read"
# A message's id (`AgentSession.message`), and how many one report names.
_MESSAGE_ID = re.compile(r"^[0-9a-f]{1,32}$")
MAX_READ = 50


def _invalid(message: str) -> HTTPException:
    return refusal_error(422, "invalid", message)


def _checked(body: Dict[str, Any]) -> Dict[str, Any]:
    """The report's fields that it gives, read as a `SessionReport`; anything
    else refused."""
    try:
        fields = plain(decode(SessionReport, body, strict=True))
    except ValueError as exc:
        raise _invalid(f"The report did not read: {exc}") from exc
    fields["ended"] = bool(fields.get("ended"))
    return fields


def report(auth: AuthConfig, body: Dict[str, Any]) -> tuple:
    """Fold a report into the session the server holds: (the session, the
    record kinds it starts). A new ``session_id`` starts a new session."""
    fields = _checked(body)
    ended = fields.pop("ended")
    previous: Optional[Dict[str, Any]] = auth.agent_session
    session_id = fields.get("session_id")
    starts = (
        previous is None
        or previous.get("ended")
        # A session that has no id yet takes the first one reported: the agent
        # may set its checklist before its harness first reports.
        or (
            session_id is not None
            and previous.get("session_id") is not None
            and session_id != previous.get("session_id")
        )
    )
    kinds = []
    if starts:
        started = datetime.now().astimezone().isoformat(timespec="seconds")
        session = {"client": auth.agent_client, "started": started, **fields}
        session = {key: value for key, value in session.items() if value is not None}
        kinds.append(SESSION_STARTED)
    else:
        session = {**previous, **fields}
        if "goal" in fields and fields["goal"] != previous.get("goal"):
            kinds.append(SESSION_GOAL)
    if budget_low(session) and (starts or not budget_low(previous)):
        kinds.append(SESSION_BUDGET)
    if ended:
        session["ended"] = True
        kinds.append(SESSION_ENDED)
    auth.agent_session = session
    return session, kinds


def where(request: Request) -> Dict[str, Optional[str]]:
    """This instrument's id, the serial number its microscope reports, and
    the open experiment's id; None for each the server can't name."""
    microscope = getattr(getattr(request.app.state, "served", None), "current", None)
    serial = getattr(
        getattr(getattr(microscope, "system", None), "info", None),
        "serial_number",
        None,
    )
    experiment = getattr(getattr(microscope, "experiment", None), "id", None)
    return {
        "instrument_id": instrument_id(),
        "instrument_serial": serial if serial not in ("", "Unknown") else None,
        "experiment_id": str(experiment) if experiment else None,
    }


def build_agent_session_router(
    emit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> APIRouter:
    """``/app/agent/session``: the agent reports its session, anyone reads it.
    *emit* puts an event on the record (the app's ``record_event``)."""
    router = APIRouter(
        prefix="/app/agent", dependencies=[Depends(require_level(AgentLevel.WATCH))]
    )

    def with_plan(request: Request, session: Optional[Dict[str, Any]]):
        """With a plan, the checklist and deadlines are read from it."""
        plan = getattr(request.app.state, "agent_plan", None)
        if session is None or plan is None:
            return session
        session_id = session.get("session_id")
        found = plan.checklist(session_id)
        if found is not None:
            session = {**session, "checklist": found}
        deadlines = plan.deadlines(session_id)
        if deadlines:
            session = {**session, "deadlines": deadlines}
        return session

    @router.post(
        "/session", response_model=AgentSession, response_model_exclude_none=True
    )
    def report_session(request: Request, body: Dict[str, Any]):
        auth: AuthConfig = request.app.state.auth
        session, kinds = report(auth, body)
        for key, value in where(request).items():
            if value is None:
                session.pop(key, None)
            else:
                session[key] = value
        for kind in kinds:
            if emit is None:
                break
            try:
                emit(kind, dict(session))
            except Exception:  # noqa: BLE001 - the record never costs the report
                logger.debug("could not record %s", kind, exc_info=True)
        return {"session": with_plan(request, session)}

    @router.post("/read")
    def report_read(body: MessagesReadBody):
        ids = body.ids
        if not 0 < len(ids) <= MAX_READ or not all(_MESSAGE_ID.match(i) for i in ids):
            raise refusal_error(
                422,
                "invalid",
                f"ids lists 1 to {MAX_READ} of the operator's message ids.",
            )
        if emit is not None:
            emit(MESSAGES_READ, {"ids": ids})
        return {"read": ids}

    @router.get(
        "/session", response_model=AgentSession, response_model_exclude_none=True
    )
    def get_session(request: Request):
        auth: AuthConfig = request.app.state.auth
        return {"session": with_plan(request, auth.agent_session)}

    return router
