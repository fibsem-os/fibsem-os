"""The agent's transcript: what its harness printed, kept beside the experiment.

``fibsem-agent`` posts the lines it prints between the agent's messages (its
words, and one line per step: "Wrote analyse.py", "Ran python analyse.py") so
the web page can show what the agent was doing without a login on the agent's
machine. They go to ``agent/transcript-<session id>.jsonl`` in the open
experiment, not on the record: the record stays the events the app and later
agents read. Self-reported, like the session.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from fibsem.ai.agent.levels import AgentLevel
from fibsem.server.auth import refusal_error, require_level
from fibsem.util.timestamps import iso_from_posix

FOLDER = "agent"
KINDS = ("said", "step", "failed")
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
        raise _invalid("an entry is {kind, text, verb?, step?, age_s?}.")
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
    verb = raw.get("verb")
    if verb is not None:
        if not isinstance(verb, str) or len(verb) > 40:
            raise _invalid("verb is a word of at most 40 characters.")
        entry["verb"] = verb
    return entry


def read(experiment: str, since: Optional[str] = None) -> List[Dict[str, Any]]:
    """The experiment's transcript entries, every session's, oldest first;
    from *since* (an ISO time) on when given. The newest `READ_LIMIT`."""
    from datetime import datetime

    folder = Path(experiment) / FOLDER
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
) -> APIRouter:
    """``/app/agent/transcript``: the agent posts its lines, anyone reads them."""
    router = APIRouter(
        prefix="/app/agent", dependencies=[Depends(require_level(AgentLevel.WATCH))]
    )

    def _experiment() -> str:
        path = experiment_path()
        if not path or not os.path.isdir(path):
            raise refusal_error(
                409, "no_experiment", "No experiment is open to keep it in."
            )
        return path

    @router.post("/transcript")
    def post_transcript(request: Request, body: Dict[str, Any]):
        session_id = body.get("session_id")
        if not isinstance(session_id, str) or not _SESSION.match(session_id):
            raise _invalid("session_id is the session's id: letters, digits, - and _.")
        raw = body.get("entries")
        if not isinstance(raw, list) or len(raw) > MAX_ENTRIES:
            raise _invalid(f"entries is a list of at most {MAX_ENTRIES}.")
        now = time.time()
        entries = [_entry(item, now) for item in raw]
        folder = Path(_experiment()) / FOLDER
        folder.mkdir(exist_ok=True)
        with open(
            folder / f"transcript-{session_id}.jsonl", "a", encoding="utf-8"
        ) as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return {"kept": len(entries)}

    @router.get("/transcript")
    def get_transcript(since: Optional[str] = None):
        path = experiment_path()
        if not path or not os.path.isdir(path):
            return {"entries": []}
        return {"entries": read(path, since)}

    return router
