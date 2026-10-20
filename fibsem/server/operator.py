"""The operator's routes: what the agent chip and card do, from outside the window.

The web page (or the operator's own script) messages the agent, sets its level,
stops it, and approves or rejects its request. Each route calls the
:class:`~fibsem.ai.agent.session.AgentSession` function the window's control
calls, so both record the same thing. Behind the operator token only
(``require_operator``): the agent's token is refused, so the agent can't
approve its own request or raise its own level.
"""

import dataclasses
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from fibsem.ai.agent.levels import AgentLevel
from fibsem.ai.agent.session import MESSAGE_LIMIT, AgentSession
from fibsem.server import files
from fibsem.server.agent_setup import setup_text
from fibsem.server.auth import require_operator
from fibsem.server.requests import RequestRefused
from fibsem.workflows import evidence


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
        path = agent.experiment_path()
        return {
            "experiment": os.path.basename(path) if path else None,
            "level": agent.level.value if agent.level is not None else None,
            "stopped": bool(auth.stopped) if auth is not None else False,
            "last_heard_s": agent.seconds_since_seen(),
            "client": agent.client(),
            "request": request.to_dict() if request is not None else None,
            "suggestion": agent.suggestion(),
        }

    @router.get("/connect")
    def connect():
        """The text the card's "Connect an agent…" copies: pasted into an agent
        on this machine, it registers the sidecar and checks the connection.
        None when the sidecar isn't installed."""
        host = agent.host
        return {"text": setup_text(host.url if host is not None else None)}

    @router.get("/recent")
    def recent():
        """The card's rows: the conversation and the agent's actions, oldest
        first, as the card reads them from the record."""
        return {"rows": [dataclasses.asdict(row) for row in agent.recent()]}

    @router.get("/evidence/{evidence_id}")
    def get_evidence(evidence_id: str):
        """A piece of evidence: its image as a preview, and its marks in the
        image's pixels."""
        root = agent.experiment_path()
        try:
            record = evidence.load(lambda: root, evidence_id)
        except evidence.EvidenceRefused as refusal:
            raise HTTPException(status_code=refusal.status_code, detail=refusal.detail)
        image = record.get("image")
        try:
            relative = os.path.relpath(image, os.path.realpath(root))
            answer = files.read(root, relative)
        except (TypeError, ValueError, files.FileRefusal) as exc:
            raise _refused(404, "not_found", f"The evidence image is missing: {exc}")
        return {"preview": answer.get("preview"), "marks": record.get("marks")}

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

    @router.post("/suggestions/{proposal_id}/approve")
    def approve_suggestion(proposal_id: str):
        """Decide the proposal as the agent suggested, as the card's Approve."""
        return _decided(agent.approve_suggestion(proposal_id), proposal_id)

    @router.post("/suggestions/{proposal_id}/decline")
    def decline_suggestion(proposal_id: str, body: Optional[Dict[str, Any]] = None):
        """Drop the suggestion with the operator's reason, as the card's
        Decline; the proposal stays pending."""
        reason = _text(body or {}, "reason", required=False)
        return _decided(agent.decline_suggestion(proposal_id, reason), proposal_id)

    return router


def _decided(result: Any, proposal_id: str) -> Dict[str, Any]:
    if not result.applied:
        status = 422 if result.error_type == "invalid_value" else 409
        raise _refused(status, result.error_type or "not_decided", result.reason)
    return {"proposal_id": proposal_id, "applied": True}
