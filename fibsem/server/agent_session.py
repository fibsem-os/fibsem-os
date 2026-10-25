"""The agent's session, as the agent reports it.

What the App Server knows of the agent on its own is its token, when it was last
heard and the client its sidecar names. Its session -- which model, toward what
goal, on what budget, how much spent -- only the agent's harness knows, so the
harness reports it here: ``fibsem-agent`` after every turn, Claude Code through
``server_report_session``. fibsem-agent also says what it can do: code mode and
its folder, effort, skills. Self-reported, like the client's name: a label for
the operator, not an identity. The agent also keeps the operator's constraints
(``server_set_constraints``) and its checklist (``server_set_checklist``) here,
for the Agent page. A session starting, its goal or constraints changing and
its end go on the record, and so does the spend first reaching 80% of the
budget; the spend and the checklist in between are state, read with GET.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from fibsem.ai.agent.levels import AgentLevel
from fibsem.ai.agent.session import budget_low
from fibsem.server.auth import AuthConfig, refusal_error, require_level
from fibsem.server.responses import AgentSession

SESSION_STARTED = "agent_session_started"
SESSION_GOAL = "agent_session_goal"
SESSION_ENDED = "agent_session_ended"
SESSION_BUDGET = "agent_session_budget"
SESSION_CONSTRAINTS = "agent_session_constraints"

MAX_TEXT = 500
_TEXT = (
    "session_id",
    "model",
    "goal",
    "reason",
    "harness",
    "effort",
    "workdir",
    "profile",
    "profile_hash",
)
_NUMBERS = ("budget_usd", "spent_usd")
MAX_SKILLS = 20
MAX_ITEM = 200
MAX_CHECKLIST = 50
MAX_CONSTRAINTS = 10
MAX_HOOKS = 20
CHECK_STATES = ("open", "done", "always")


def _invalid(message: str) -> HTTPException:
    return refusal_error(422, "invalid", message)


def _checked(body: Dict[str, Any]) -> Dict[str, Any]:
    """The report's fields, each checked; anything else refused."""
    known = (
        set(_TEXT)
        | set(_NUMBERS)
        | {
            "turns",
            "ended",
            "code",
            "files",
            "resumed",
            "skills",
            "checklist",
            "constraints",
            "hooks",
        }
    )
    unknown = sorted(set(body) - known)
    if unknown:
        raise _invalid(f"Unknown fields {unknown}; the fields are {sorted(known)}.")
    report: Dict[str, Any] = {}
    for key in _TEXT:
        value = body.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or len(value) > MAX_TEXT:
            raise _invalid(f"{key} is a string of at most {MAX_TEXT} characters.")
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
            raise _invalid(f"{key} is a number of US dollars, 0 or more.")
        report[key] = float(value)
    turns = body.get("turns")
    if turns is not None:
        if isinstance(turns, bool) or not isinstance(turns, int) or turns < 0:
            raise _invalid("turns is a whole number, 0 or more.")
        report["turns"] = turns
    for key in ("code", "files", "resumed"):
        value = body.get(key)
        if value is not None:
            if not isinstance(value, bool):
                raise _invalid(f"{key} is true or false.")
            report[key] = value
    skills = body.get("skills")
    if skills is not None:
        if (
            not isinstance(skills, list)
            or len(skills) > MAX_SKILLS
            or not all(isinstance(s, str) and len(s) <= 100 for s in skills)
        ):
            raise _invalid(
                f"skills is a list of at most {MAX_SKILLS} names, each at most "
                "100 characters."
            )
        report["skills"] = [s.strip() for s in skills]
    if body.get("checklist") is not None:
        report["checklist"] = _checklist(body["checklist"])
    if body.get("constraints") is not None:
        report["constraints"] = _constraints(body["constraints"])
    if body.get("hooks") is not None:
        report["hooks"] = _hooks(body["hooks"])
    ended = body.get("ended")
    if ended is not None and not isinstance(ended, bool):
        raise _invalid("ended is true or false.")
    report["ended"] = bool(ended)
    return report


def _text(item: Any, what: str) -> str:
    text = item.get("text") if isinstance(item, dict) else None
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_ITEM:
        raise _invalid(f"Each {what} has a text of at most {MAX_ITEM} characters.")
    return text.strip()


def _checklist(items: Any) -> List[Dict[str, str]]:
    """Each item {text, state}: open (the default), done, or always for a
    standing rule."""
    if not isinstance(items, list) or len(items) > MAX_CHECKLIST:
        raise _invalid(f"checklist is a list of at most {MAX_CHECKLIST} items.")
    checked = []
    for item in items:
        state = item.get("state", "open") if isinstance(item, dict) else None
        if state not in CHECK_STATES:
            raise _invalid(f"An item's state is one of {list(CHECK_STATES)}.")
        checked.append({"text": _text(item, "item"), "state": state})
    return checked


def deadline(by: str, now: Optional[datetime] = None) -> datetime:
    """*by* as a time with its offset: "07:00" is its next occurrence here, an
    ISO date and time without an offset is this computer's local time."""
    now = now or datetime.now().astimezone()
    try:
        clock = datetime.strptime(by, "%H:%M")
    except ValueError:
        when = datetime.fromisoformat(by)
        return when if when.tzinfo else when.astimezone()
    when = now.replace(hour=clock.hour, minute=clock.minute, second=0, microsecond=0)
    return when if when > now else when + timedelta(days=1)


def _constraints(items: Any) -> List[Dict[str, str]]:
    """Each {text, by?}: by, when given, is stored as a time with its offset."""
    if not isinstance(items, list) or len(items) > MAX_CONSTRAINTS:
        raise _invalid(f"constraints is a list of at most {MAX_CONSTRAINTS} items.")
    checked = []
    for item in items:
        constraint = {"text": _text(item, "constraint")}
        by = item.get("by")
        if by is not None:
            try:
                constraint["by"] = deadline(str(by)).isoformat(timespec="minutes")
            except ValueError:
                raise _invalid(
                    "by is a time, 07:00, or an ISO date and time, 2026-10-24T07:00."
                ) from None
        checked.append(constraint)
    return checked


def _hooks(items: Any) -> List[Dict[str, str]]:
    """A profile's reminders, each {when, label, say}."""
    if (
        not isinstance(items, list)
        or len(items) > MAX_HOOKS
        or not all(
            isinstance(item, dict)
            and set(item) == {"when", "label", "say"}
            and all(isinstance(v, str) and len(v) <= MAX_TEXT for v in item.values())
            for item in items
        )
    ):
        raise _invalid(
            f"hooks is a list of at most {MAX_HOOKS} reminders, each {{when, "
            f"label, say}} of at most {MAX_TEXT} characters."
        )
    return [
        {key: item[key].strip() for key in ("when", "label", "say")} for item in items
    ]


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
        kinds.append(SESSION_STARTED)
    else:
        session = {**previous, **fields}
        if "goal" in fields and fields["goal"] != previous.get("goal"):
            kinds.append(SESSION_GOAL)
        if "constraints" in fields and fields["constraints"] != previous.get(
            "constraints"
        ):
            kinds.append(SESSION_CONSTRAINTS)
    if budget_low(session) and (starts or not budget_low(previous)):
        kinds.append(SESSION_BUDGET)
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

    @router.post("/session", response_model=AgentSession)
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

    @router.get("/session", response_model=AgentSession)
    def get_session(request: Request):
        auth: AuthConfig = request.app.state.auth
        return {"session": auth.agent_session}

    return router
