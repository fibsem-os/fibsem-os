"""The agent's handoff notes: what it wrote for a fresh context of its session.

When the agent starts a fresh context (``context_reset``,
`fibsem.ai.agent.handoff`), ``fibsem-agent`` posts its note here; only the
running session's are taken. The notes go to ``agent/handoff-<session
id>.md`` in the session's own folder, beside its plan and transcript
(`agent_files`), each after the earlier ones, so
the operator can read why and where the agent started over. Self-reported,
like the transcript.
"""

import re
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, Depends, Request

from fibsem.ai.agent.levels import AgentLevel
from fibsem.server.agent.files import AgentFiles
from fibsem.server.auth import AuthConfig, refusal_error, require_level
from fibsem.server.bodies import HandoffBody

FOLDER = "agent"
PREFIX = "handoff-"
MAX_TEXT = 50_000
_SESSION = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def path_for(folder: Path, session_id: str) -> Path:
    return folder / f"{PREFIX}{session_id}.md"


def build_agent_handoff_router(
    experiment_path: Callable[[], Optional[str]],
) -> APIRouter:
    """``/app/agent/handoff``: fibsem-agent adds a note, anyone reads them."""
    router = APIRouter(
        prefix="/app/agent", dependencies=[Depends(require_level(AgentLevel.WATCH))]
    )

    def _files(request: Request) -> AgentFiles:
        files = getattr(request.app.state, "agent_files", None)
        if files is None:
            files = request.app.state.agent_files = AgentFiles(experiment_path)
        return files

    @router.post("/handoff")
    def post_handoff(request: Request, body: HandoffBody):
        if not _SESSION.match(body.session_id):
            raise refusal_error(
                422,
                "invalid",
                "session_id is the session's id: letters, digits, - and _.",
            )
        # Only the running session's own notes: another caller at Watch can't
        # write into a session that isn't its.
        auth: AuthConfig = request.app.state.auth
        running = auth.agent_session or {}
        if running.get("ended") or running.get("session_id") != body.session_id:
            raise refusal_error(
                409,
                "not_running",
                f"Session {body.session_id} is not the agent session running "
                "here: only it adds handoff notes.",
            )
        text = body.text.strip()
        if not text or len(text) > MAX_TEXT:
            raise refusal_error(
                422, "invalid", f"text is the note, 1 to {MAX_TEXT} characters."
            )
        folder = _files(request).folder(body.session_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = path_for(folder, body.session_id)
        with open(path, "a", encoding="utf-8") as file:
            file.write(text + "\n\n")
        return {"path": path.name}

    @router.get("/handoff")
    def get_handoff(request: Request, session_id: Optional[str] = None):
        """The session's notes, oldest first: *session_id*'s, else the
        running session's."""
        if session_id is None:
            auth: AuthConfig = request.app.state.auth
            session = auth.agent_session or {}
            session_id = session.get("session_id")
        if not isinstance(session_id, str) or not _SESSION.match(session_id):
            return {"session_id": None, "text": ""}
        path = _files(request).find(session_id, path_for(Path(), session_id).name)
        text = path.read_text(encoding="utf-8") if path and path.is_file() else ""
        return {"session_id": session_id, "text": text}

    return router
