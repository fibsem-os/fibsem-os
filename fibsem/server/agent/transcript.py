"""The agent's transcript: what its harness printed, kept beside the experiment.

``fibsem-agent`` posts the lines it prints between the agent's messages (its
words, and one line per step: "Wrote analyse.py", "Ran python analyse.py") so
the web page can show what the agent was doing without a login on the agent's
machine. They go to ``agent/transcript-<session id>.jsonl`` in the session's
experiment, or its own folder until one is open (`agent_files`), not on the
record: the record stays the events the app and later agents read.
Self-reported, like the session.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from fibsem.ai.agent.levels import AgentLevel
from fibsem.server.agent.files import AgentFiles
from fibsem.server.auth import AuthConfig, refusal_error, require_level
from fibsem.server.bodies import TranscriptBody
from fibsem.util.timestamps import iso_from_posix
from fibsem.workflows.events.kinds import AGENT_TRANSCRIPT

log = logging.getLogger(__name__)

FOLDER = "agent"
# context: what the harness says of the session ("Playbook set: ...").
# done: a step finished (its ``step``), so the page stops showing it running.
KINDS = ("said", "step", "done", "failed", "context")
MAX_ENTRIES = 200
MAX_TEXT = 2000
MAX_AGE_S = 24 * 3600
# What GET answers at most: the newest entries.
READ_LIMIT = 500
_SESSION = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def _invalid(message: str) -> HTTPException:
    return refusal_error(422, "invalid", message)


def _entry(raw: Any, now: float) -> Dict[str, Any]:
    """One posted entry, checked, stamped with the server's time."""
    if not isinstance(raw, dict):
        raise _invalid("an entry is {kind, text, verb?, step?, took_s?, age_s?}.")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise _invalid(f"kind is one of {list(KINDS)}.")
    text = raw.get("text")
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise _invalid(f"text is a string of at most {MAX_TEXT} characters.")
    age = raw.get("age_s", 0)
    if isinstance(age, bool) or not isinstance(age, (int, float)):
        raise _invalid("age_s is how many seconds ago it happened.")
    entry: Dict[str, Any] = {
        "t": iso_from_posix(now - min(max(float(age), 0.0), MAX_AGE_S)),
        "kind": kind,
        "text": text.strip(),
    }
    for key in ("seq", "step"):
        value = raw.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise _invalid(f"{key} is a whole number.")
            entry[key] = value
    took = raw.get("took_s")
    if took is not None:
        if isinstance(took, bool) or not isinstance(took, (int, float)) or took < 0:
            raise _invalid("took_s is how many seconds the step took.")
        entry["took_s"] = round(float(took), 1)
    verb = raw.get("verb")
    if verb is not None:
        if not isinstance(verb, str) or len(verb) > 40:
            raise _invalid("verb is a word of at most 40 characters.")
        entry["verb"] = verb
    return entry


def read(folder: Path, since: Optional[str] = None) -> List[Dict[str, Any]]:
    """The transcript entries in an agent/ *folder*, every session's, oldest
    first; from *since* (an ISO time) on when given. The newest `READ_LIMIT`."""
    from datetime import datetime

    entries: List[Dict[str, Any]] = []
    for path in sorted(folder.glob("transcript-*.jsonl")):
        session = path.stem[len("transcript-") :]
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    at = datetime.fromisoformat(entry["t"])
                except (ValueError, KeyError, TypeError):
                    continue
                entries.append({**entry, "session_id": session, "_at": at})
    if since:
        try:
            start = datetime.fromisoformat(since)
        except ValueError:
            raise _invalid("since is an ISO time.")
        entries = [e for e in entries if e["_at"] >= start]
    entries.sort(key=lambda e: e["_at"])
    return [{k: v for k, v in e.items() if k != "_at"} for e in entries[-READ_LIMIT:]]


def build_agent_transcript_router(
    experiment_path: Callable[[], Optional[str]],
    emit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> APIRouter:
    """``/app/agent/transcript``: the agent posts its lines, anyone reads them.
    *emit* puts an ``agent_transcript`` event on the app's events for each post,
    so the web page reads the new lines at once; a live event, left out of a
    run's history."""
    router = APIRouter(
        prefix="/app/agent", dependencies=[Depends(require_level(AgentLevel.WATCH))]
    )

    def _files(request: Request) -> AgentFiles:
        files = getattr(request.app.state, "agent_files", None)
        if files is None:
            files = request.app.state.agent_files = AgentFiles(experiment_path)
        return files

    @router.post("/transcript")
    def post_transcript(request: Request, body: TranscriptBody):
        session_id = body.session_id
        if not _SESSION.match(session_id):
            raise _invalid("session_id is the session's id: letters, digits, - and _.")
        raw = body.entries
        if len(raw) > MAX_ENTRIES:
            raise _invalid(f"entries is a list of at most {MAX_ENTRIES}.")
        now = time.time()
        entries = [_entry(item, now) for item in raw]
        folder = _files(request).folder(session_id)
        folder.mkdir(parents=True, exist_ok=True)
        with open(
            folder / f"transcript-{session_id}.jsonl", "a", encoding="utf-8"
        ) as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if emit is not None and entries:
            try:
                emit(AGENT_TRANSCRIPT, {"session_id": session_id, "kept": len(entries)})
            except Exception:  # noqa: BLE001 - the lines are kept either way
                log.debug("could not record %s", AGENT_TRANSCRIPT, exc_info=True)
        return {"kept": len(entries)}

    @router.get("/transcript")
    def get_transcript(request: Request, since: Optional[str] = None):
        # The running session's folder, else the open experiment's.
        auth: AuthConfig = request.app.state.auth
        session = auth.agent_session or {}
        running = None if session.get("ended") else session.get("session_id")
        folder = _files(request).folder(running if isinstance(running, str) else None)
        if folder is None or not folder.is_dir():
            return {"entries": []}
        return {"entries": read(folder, since)}

    return router
