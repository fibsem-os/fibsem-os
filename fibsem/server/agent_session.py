"""The agent's session, as the agent reports it.

What the App Server knows of the agent on its own is its token, when it was last
heard and the client its sidecar names. Its session -- which model, toward what
goal, on what budget, how much spent -- only the agent's harness knows, so the
harness reports it here: ``fibsem-agent`` after every turn, Claude Code through
``server_report_session``. fibsem-agent also says what it can do: code mode and
its folder, effort, skills. Self-reported, like the client's name: a label for
the operator, not an identity. A session starting, its goal changing and its end
go on the record; the spend in between is state, read with GET.
"""

import logging
import math
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from fibsem.ai.agent.levels import AgentLevel
from fibsem.server.auth import AuthConfig, require_level

SESSION_STARTED = "agent_session_started"
SESSION_GOAL = "agent_session_goal"
SESSION_ENDED = "agent_session_ended"

MAX_TEXT = 500
_TEXT = ("session_id", "model", "goal", "reason", "harness", "effort", "workdir")
_NUMBERS = ("budget_usd", "spent_usd")
MAX_SKILLS = 20


def _refused(message: str) -> HTTPException:
    return HTTPException(
        status_code=422, detail={"error_type": "invalid", "message": message}
    )


def _checked(body: Dict[str, Any]) -> Dict[str, Any]:
    """The report's fields, each checked; anything else refused."""
    known = set(_TEXT) | set(_NUMBERS) | {"turns", "ended", "code", "resumed", "skills"}
    unknown = sorted(set(body) - known)
    if unknown:
        raise _refused(f"Unknown fields {unknown}; the fields are {sorted(known)}.")
    report: Dict[str, Any] = {}
    for key in _TEXT:
        value = body.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or len(value) > MAX_TEXT:
            raise _refused(f"{key} is a string of at most {MAX_TEXT} characters.")
        report[key] = value.strip()
    for key in _NUMBERS:
        value = body.get(key)
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise _refused(f"{key} is a number of US dollars, 0 or more.")
        report[key] = float(value)
    turns = body.get("turns")
    if turns is not None:
        if isinstance(turns, bool) or not isinstance(turns, int) or turns < 0:
            raise _refused("turns is a whole number, 0 or more.")
        report["turns"] = turns
    for key in ("code", "resumed"):
        value = body.get(key)
        if value is not None:
            if not isinstance(value, bool):
                raise _refused(f"{key} is true or false.")
            report[key] = value
    skills = body.get("skills")
    if skills is not None:
        if (
            not isinstance(skills, list)
            or len(skills) > MAX_SKILLS
            or not all(isinstance(s, str) and len(s) <= 100 for s in skills)
        ):
            raise _refused(
                f"skills is a list of at most {MAX_SKILLS} names, each at most "
                "100 characters."
            )
        report["skills"] = [s.strip() for s in skills]
    ended = body.get("ended")
    if ended is not None and not isinstance(ended, bool):
        raise _refused("ended is true or false.")
    report["ended"] = bool(ended)
    return report


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
        or (session_id is not None and session_id != previous.get("session_id"))
    )
    kinds = []
    if starts:
        session = {"client": auth.agent_client, **fields}
        kinds.append(SESSION_STARTED)
    else:
        session = {**previous, **fields}
        if "goal" in fields and fields["goal"] != previous.get("goal"):
            kinds.append(SESSION_GOAL)
    if ended:
        session["ended"] = True
        kinds.append(SESSION_ENDED)
    auth.agent_session = session
    return session, kinds


def build_agent_session_router(
    emit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> APIRouter:
    """``/app/agent/session``: the agent reports its session, anyone reads it.
    *emit* puts an event on the record (the app's ``record_event``)."""
    router = APIRouter(
        prefix="/app/agent", dependencies=[Depends(require_level(AgentLevel.WATCH))]
    )

    @router.post("/session")
    def report_session(request: Request, body: Dict[str, Any]):
        auth: AuthConfig = request.app.state.auth
        session, kinds = report(auth, body)
        for kind in kinds:
            if emit is None:
                break
            try:
                emit(kind, dict(session))
            except Exception:  # noqa: BLE001 - the record never costs the report
                logging.debug("could not record %s", kind, exc_info=True)
        return {"session": session}

    @router.get("/session")
    def get_session(request: Request):
        auth: AuthConfig = request.app.state.auth
        return {"session": auth.agent_session}

    return router
