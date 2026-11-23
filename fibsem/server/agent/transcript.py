"""The agent's transcript: what its harness printed, kept beside the experiment.

``fibsem-agent`` posts the lines it prints between the agent's messages (its
words, and one line per step: "Wrote analyse.py", "Ran python analyse.py") so
the web page can show what the agent was doing without a login on the agent's
machine. They go to ``agent/transcript-<session id>.jsonl`` in the session's
experiment, or its own folder until one is open (`agent_files`), not on the
record: the record stays the events the app and later agents read.
Self-reported, like the session.

A picture a lab's MCP server answered with comes as a ``picture`` line, its
image in it (``image``, base64, and ``media_type``). It is saved beside the
transcript, ``agent/pictures/<session id>-<line>.<ext>``, the line keeps its
``path`` instead, and ``GET /app/agent/transcript/pictures/<name>`` serves it.

A subagent's lines carry ``run``, the ``seq`` of the Agent call that started
it; that call's line carries ``subagent``, the subagent's name, and its ``done``
keeps the subagent's answer, with ``verdict`` (agree or concern) and
``attention`` (true when someone should look) when it gave them.

A session resumed in a new ``fibsem-agent`` numbers its lines from 1 again, in
the same file. Read back, each later run's numbers follow on from the run
before (`_runs`), so a line names one step, and a step the run ended before
answering (stopped, crashed or killed) carries ``ended``: the page shows it
stopped, not running for ever.
"""

import base64
import binascii
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from fibsem.ai.agent.levels import AgentLevel
from fibsem.server.agent.files import PICTURES, AgentFiles
from fibsem.server.auth import AuthConfig, refusal_error, require_level
from fibsem.server.bodies import TranscriptBody
from fibsem.util.timestamps import iso_from_posix
from fibsem.workflows.events.kinds import AGENT_TRANSCRIPT

log = logging.getLogger(__name__)

FOLDER = "agent"
# context: what the harness says of the session ("Playbook set: ...").
# done: a step finished (its ``step``), so the page stops showing it running.
# picture: an image a step answered with (its ``step``), saved beside.
KINDS = ("said", "step", "done", "failed", "context", "picture")
# What a subagent's done line says of a decision it was asked to check.
VERDICTS = ("agree", "concern")
# A picture's kinds of image, by media type: the file's extension.
PICTURE_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}
MAX_PICTURE_BYTES = 5 * 1024 * 1024
_PICTURE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,100}-\d{1,9}\.(png|jpg|gif|webp)$")
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
        raise _invalid(
            "an entry is {kind, text, verb?, step?, run?, subagent?, verdict?, attention?, took_s?, age_s?}."
        )
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
    # run: the line of the Agent call whose subagent printed this line.
    for key in ("seq", "step", "run"):
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
    verdict = raw.get("verdict")
    if verdict is not None:
        if verdict not in VERDICTS:
            raise _invalid(f"verdict is one of {list(VERDICTS)}.")
        entry["verdict"] = verdict
    attention = raw.get("attention")
    if attention is not None:
        if not isinstance(attention, bool):
            raise _invalid("attention is true or false.")
        entry["attention"] = attention
    subagent = raw.get("subagent")
    if subagent is not None:
        if not isinstance(subagent, str) or len(subagent) > 60:
            raise _invalid("subagent is the subagent's name, at most 60 characters.")
        entry["subagent"] = subagent
    if kind == "picture":
        entry["_image"] = _picture(raw)
        if "seq" not in entry:
            raise _invalid("a picture line has its seq.")
    return entry


def _picture(raw: Dict[str, Any]) -> tuple:
    """A picture line's image: (its extension, its bytes)."""
    media_type = raw.get("media_type")
    if media_type not in PICTURE_TYPES:
        raise _invalid(f"media_type is one of {sorted(PICTURE_TYPES)}.")
    data = raw.get("image")
    if not isinstance(data, str):
        raise _invalid("image is the picture, base64.")
    try:
        image = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise _invalid("image is the picture, base64.")
    if not image or len(image) > MAX_PICTURE_BYTES:
        raise _invalid(f"a picture is at most {MAX_PICTURE_BYTES} bytes.")
    return PICTURE_TYPES[media_type], image


def _keep_picture(folder: Path, session_id: str, entry: Dict[str, Any]) -> None:
    """Save a picture line's image beside the transcript; the line keeps its
    path, relative to the agent/ folder."""
    extension, image = entry.pop("_image")
    name = f"{session_id}-{entry['seq']}.{extension}"
    (folder / PICTURES).mkdir(parents=True, exist_ok=True)
    (folder / PICTURES / name).write_bytes(image)
    entry["path"] = f"{PICTURES}/{name}"


# The line numbers a line names: its own, its step's, its subagent run's.
_NUMBERS = ("seq", "step", "run")


def _runs(entries: List[Dict[str, Any]]) -> None:
    """One session's *entries*, in file order, numbered apart by run: a line
    numbered no higher than one before it starts a new run, whose numbers then
    follow on from the run before. A step a run ended without answering gets
    ``ended``; the last run's steps may still be running."""
    offset, top, steps, answered = 0, 0, [], set()

    def close() -> None:
        for step in steps:
            if step["seq"] not in answered:
                step["ended"] = True

    for entry in entries:
        seq = entry.get("seq")
        if isinstance(seq, int) and seq + offset <= top:
            close()
            offset, steps, answered = top, [], set()
        for key in _NUMBERS:
            if isinstance(entry.get(key), int):
                entry[key] += offset
        if isinstance(seq, int):
            top = max(top, entry["seq"])
        if entry["kind"] == "step" and isinstance(seq, int):
            steps.append(entry)
        elif entry["kind"] in ("done", "failed") and isinstance(entry.get("step"), int):
            answered.add(entry["step"])


def read(folder: Path, since: Optional[str] = None) -> List[Dict[str, Any]]:
    """The transcript entries in an agent/ *folder*, every session's, oldest
    first; from *since* (an ISO time) on when given. The newest `READ_LIMIT`."""
    from datetime import datetime

    entries: List[Dict[str, Any]] = []
    for path in sorted(folder.glob("transcript-*.jsonl")):
        session = path.stem[len("transcript-") :]
        first = len(entries)
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    at = datetime.fromisoformat(entry["t"])
                except (ValueError, KeyError, TypeError):
                    continue
                entries.append({**entry, "session_id": session, "_at": at})
        _runs(entries[first:])
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
        for entry in entries:
            if "_image" in entry:
                _keep_picture(folder, session_id, entry)
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

    def _folder(request: Request) -> Optional[Path]:
        # The running session's folder, else the open experiment's.
        auth: AuthConfig = request.app.state.auth
        session = auth.agent_session or {}
        running = None if session.get("ended") else session.get("session_id")
        return _files(request).folder(running if isinstance(running, str) else None)

    @router.get("/transcript")
    def get_transcript(request: Request, since: Optional[str] = None):
        folder = _folder(request)
        if folder is None or not folder.is_dir():
            return {"entries": []}
        return {"entries": read(folder, since)}

    @router.get("/transcript/pictures/{name}")
    def get_picture(request: Request, name: str):
        folder = _folder(request)
        path = None if folder is None else folder / PICTURES / name
        if not _PICTURE_NAME.match(name) or path is None or not path.is_file():
            raise refusal_error(404, "not_found", f"No picture {name}.")
        media_type = next(
            kind for kind, ext in PICTURE_TYPES.items() if name.endswith("." + ext)
        )
        return FileResponse(path, media_type=media_type)

    return router
