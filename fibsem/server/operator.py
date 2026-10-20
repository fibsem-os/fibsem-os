"""The operator's routes: what the agent chip and card do, from outside the window.

The web page (or the operator's own script) messages the agent, sets its level,
stops it, and approves or rejects its request. Each route calls the
:class:`~fibsem.ai.agent.session.AgentSession` function the window's control
calls, so both record the same thing. Behind the operator token only
(``require_operator``): the agent's token is refused, so the agent can't
approve its own request or raise its own level.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from fibsem.ai.agent.levels import AgentLevel
from fibsem.ai.agent.session import MESSAGE_LIMIT, AgentSession
from fibsem.server.auth import require_operator
from fibsem.server.requests import RequestRefused


def _refused(status_code: int, error_type: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error_type": error_type, "message": message},
    )


def _text(body: Dict[str, Any], key: str, required: bool = True) -> str:
    value = body.get(key, "")
    if not isinstance(value, str) or (required and not value.strip()):
        raise _refused(422, "missing_field", f"Pass {key!r} as text.")
    return value.strip()


def build_operator_router(agent: AgentSession) -> APIRouter:
    """The operator's side of the agent, under ``/app/agent``."""
    router = APIRouter(prefix="/app/agent", dependencies=[Depends(require_operator())])

    @router.get("")
    def get_agent():
        """The agent as the chip shows it: level, stopped, last heard, the
        client its sidecar named, and the request waiting, if any."""
        auth = agent.host.auth if agent.host is not None else None
        request = agent.current()
        return {
            "level": agent.level.value if agent.level is not None else None,
            "stopped": bool(auth.stopped) if auth is not None else False,
            "last_heard_s": agent.seconds_since_seen(),
            "client": agent.client(),
            "request": request.to_dict() if request is not None else None,
        }

    @router.post("/messages", status_code=201)
    def send_message(body: Dict[str, Any]):
        text = _text(body, "text")
        if len(text) > MESSAGE_LIMIT:
            raise _refused(
                422,
                "too_long",
                f"A message is at most {MESSAGE_LIMIT} characters.",
            )
        message_id = agent.message(text)
        if message_id is None:
            raise _refused(
                409,
                "not_recording",
                "Nothing is recording, so the agent can't hear a message.",
            )
        return {"id": message_id}

    @router.put("/level")
    def set_level(body: Dict[str, Any]):
        try:
            level = AgentLevel(_text(body, "level").lower())
        except ValueError:
            names = ", ".join(level.value for level in AgentLevel)
            raise _refused(422, "bad_level", f"The level is one of {names}.")
        why_not = agent.set_level(level)
        if why_not is not None:
            raise _refused(409, "level_refused", why_not)
        return {"level": level.value}

    @router.post("/stop")
    def stop_agent():
        """Stop agent: down to Watch, and stop what it is doing on the microscope."""
        stopped = agent.stop()
        return {
            "stopped": stopped,
            "level": agent.level.value if agent.level is not None else None,
        }

    def _store() -> None:
        if agent.requests is None:
            raise _refused(409, "no_server", "The App Server isn't serving the agent.")

    @router.post("/requests/{request_id}/approve")
    def approve_request(request_id: str):
        _store()
        try:
            agent.approve(request_id)
        except RequestRefused as refusal:
            raise HTTPException(status_code=refusal.status_code, detail=refusal.detail)
        return {"id": request_id, "approved": True}

    @router.post("/requests/{request_id}/reject")
    def reject_request(request_id: str, body: Optional[Dict[str, Any]] = None):
        _store()
        reason = _text(body or {}, "reason", required=False)
        try:
            agent.reject(request_id, reason)
        except RequestRefused as refusal:
            raise HTTPException(status_code=refusal.status_code, detail=refusal.detail)
        return {"id": request_id, "approved": False}

    return router
