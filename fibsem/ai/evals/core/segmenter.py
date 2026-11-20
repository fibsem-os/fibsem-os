"""The segmenter: fibsem-agent, run offline on an episode on the
``eval-segmenter`` playbook, proposing its segments (``segments.yaml``) for
a person to review.

It runs as any fibsem-agent session with no instrument: an agent host of its
own (``fibsem-agent-host``, on a free port, with tokens made for the run) and
``fibsem-agent --host --files <episode>``, so it has the harness's prompt,
log, session record and budget. It reads the episode's record (events, the
agent's transcripts, images) and the case catalogue, and writes
``segments.yaml``; :func:`segments.check` then checks it against the record,
and the problems go to the agent as one operator message, for one turn to
fix them. It only reads the episode and writes that one file (anything else
it changes is reported): it never reaches a microscope, and nothing it
writes goes back into a run.
"""

import hashlib
import json
import logging
import os
import secrets
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from fibsem.ai.agent import manager, sources
from fibsem.ai.evals.core import catalogue, segments
from fibsem.ai.evals.core.episode import Episode
from fibsem.ai.evals.core.runner import _free_port, _turn_ended, _wait_for
from fibsem.client.transport import HOST_TOKEN_ENV
from fibsem.util import timestamps
from fibsem.util.process import end

log = logging.getLogger(__name__)

FOLDER = "segmenter"
EVENTS_DIGEST = "events.txt"
CATALOGUE = "catalogue.yaml"
GOAL = "goal.md"
AGENT_LOG = "agent.log"
HOST_LOG = "host.log"
PLAYBOOK = "eval-segmenter"
SKILL = "eval-segmenter"
PAYLOAD_CHARS = 240
HOST_OPERATOR_TOKEN_ENV = "FIBSEM_HOST_OPERATOR_TOKEN"
HOST_WAIT_S = 60.0
STOP_WAIT_S = 60.0
# What the agent may leave in the episode besides segments.yaml: its skills
# (files mode copies them in), its caches, its notebook and the segmenter's
# own folder.
OWN = {".claude", ".cache", "notebook", FOLDER, segments.FILE, "segments.previous.yaml"}

GOAL_TEXT = """Segment the episode in your working folder ({episode}) into the decisions an eval case could score, following the eval-segmenter skill, and write segments.yaml there."""


def _short(value: Any) -> str:
    text = json.dumps(value, default=str, separators=(",", ":"))
    return text if len(text) <= PAYLOAD_CHARS else text[: PAYLOAD_CHARS - 3] + "..."


def events_digest(events: List[Dict[str, Any]]) -> str:
    lines = []
    for event in events:
        lines.append(
            "\t".join(
                str(x)
                for x in (
                    event.get("id"),
                    event.get("t") or event.get("timestamp"),
                    event.get("actor"),
                    event.get("kind"),
                    event.get("item") or "-",
                    _short(event.get("payload") or {}),
                )
            )
        )
    return "\n".join(lines) + "\n"


def prepare(episode: Episode) -> Path:
    """The segmenter's folder in the episode: the events digest, the
    catalogue and the goal, written fresh."""
    folder = episode.path / FOLDER
    folder.mkdir(exist_ok=True)
    (folder / EVENTS_DIGEST).write_text(events_digest(episode.events()))
    (folder / CATALOGUE).write_text(
        yaml.safe_dump(catalogue.catalogue(), sort_keys=False, width=100)
    )
    (folder / GOAL).write_text(goal(episode) + "\n")
    return folder


def goal(episode: Episode) -> str:
    return GOAL_TEXT.format(episode=episode.name)


def instructions(checkout: Optional[Path] = None) -> str:
    """The playbook and the skill the segmenter runs on, as one text."""
    root = Path(checkout) if checkout else sources.find_root(None, SKILL)
    parts = []
    for path in (
        root / sources.CLAUDE / "fibsem-agent" / "playbooks" / f"{PLAYBOOK}.yaml",
        root / sources.CLAUDE / "skills" / SKILL / "SKILL.md",
    ):
        parts.append(path.read_text() if path.is_file() else "")
    return "\n".join(parts)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class SegmenterError(RuntimeError):
    pass


def fix_text(problems: List[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "The check of segments.yaml against the record found these problems. "
        "Fix them in segments.yaml (drop a segment you cannot fix), then say "
        f"what you changed.\n\n{listed}"
    )


def agent_command() -> List[str]:
    return manager.agent_command()


def host_command() -> List[str]:
    return [sys.executable, "-m", "fibsem.server.agent.host"]


def _snapshot(episode: Episode) -> Dict[str, tuple]:
    """Each file of the episode the agent must leave alone, with its size
    and time."""
    found = {}
    for path in episode.path.rglob("*"):
        rel = path.relative_to(episode.path)
        if rel.parts[0] in OWN or not path.is_file():
            continue
        stat = path.stat()
        found[rel.as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return found


def _changed(before: Dict[str, tuple], after: Dict[str, tuple]) -> List[str]:
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


def _tail(path: Path, lines: int = 5) -> str:
    try:
        return " | ".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _turns(log_path: Path) -> int:
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return 0
    return text.count("Turn ended:")


def _serving(url: str) -> Callable[[], bool]:
    def check() -> bool:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as answer:
                return answer.status == 200
        except OSError:
            return False

    return check


def _message(url: str, token: str, text: str) -> None:
    """An operator message to the agent, through its host."""
    request = urllib.request.Request(
        f"{url}/app/agent/messages",
        data=json.dumps({"text": text}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30):
        pass


def _spent(session_id: str) -> Dict[str, Any]:
    record = manager.read_json(manager.session_folder(session_id) / manager.RECORD)
    last = (record.get("runs") or [{}])[-1]
    return {k: last[k] for k in ("spent_usd", "turns", "reason") if k in last}


def run(
    episode: Episode,
    model: Optional[str] = None,
    budget_usd: float = 3.0,
    effort: Optional[str] = None,
    timeout_s: float = 20 * 60,
    agent: Optional[List[str]] = None,
    host: Optional[List[str]] = None,
) -> segments.Check:
    """Propose the episode's segments with fibsem-agent; the check of what it
    wrote. A segments.yaml already there is kept as segments.previous.yaml.
    Raises SegmenterError when the agent fails (it can't sign in, say),
    writes no segments or changes a file of the episode."""
    folder = prepare(episode)
    existing = segments.path(episode)
    if existing.is_file():
        existing.replace(episode.path / "segments.previous.yaml")
    agent_token, operator_token = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        HOST_TOKEN_ENV: agent_token,
        HOST_OPERATOR_TOKEN_ENV: operator_token,
    }
    agent_env = {k: v for k, v in env.items() if k != HOST_OPERATOR_TOKEN_ENV}
    session_id = str(uuid.uuid4())
    log_path = folder / AGENT_LOG
    started = time.monotonic()
    before = _snapshot(episode)
    command = (agent or agent_command()) + [
        "--host",
        url,
        "--playbook",
        PLAYBOOK,
        "--files",
        str(episode.path),
        "--goal",
        goal(episode),
        "--max-budget-usd",
        str(budget_usd),
        "--session-id",
        session_id,
    ]
    for flag, value in (("--model", model), ("--effort", effort)):
        if value:
            command += [flag, value]

    host_log = open(folder / HOST_LOG, "w")
    server = subprocess.Popen(
        (host or host_command())
        + ["--port", str(port), "--folder", str(folder / "host")],
        env=env,
        stdout=host_log,
        stderr=subprocess.STDOUT,
    )
    proc = None
    fixed = False
    try:
        if not _wait_for(_serving(url), HOST_WAIT_S, server):
            raise SegmenterError(
                f"the agent host did not start: {_tail(folder / HOST_LOG)}"
            )
        with open(log_path, "w") as agent_log:
            proc = subprocess.Popen(
                command, env=agent_env, stdout=agent_log, stderr=subprocess.STDOUT
            )

            def written() -> bool:
                return existing.is_file() and _turn_ended(log_path)

            _wait_for(written, timeout_s, proc)
            if not existing.is_file():
                raise SegmenterError(
                    "the agent wrote no segments.yaml: "
                    + (_tail(log_path) or "it printed nothing")
                )
            result = segments.check(episode)
            if not result.ok and proc.poll() is None:
                turns = _turns(log_path)
                _message(url, operator_token, fix_text(result.problems))
                fixed = True

                def answered() -> bool:
                    return _turns(log_path) > turns and _turn_ended(log_path)

                left = max(60.0, timeout_s - (time.monotonic() - started))
                _wait_for(answered, left, proc)
    finally:
        if proc is not None and proc.poll() is None:
            session = manager.session_folder(session_id)
            session.mkdir(parents=True, exist_ok=True)
            (session / manager.STOP).touch()
            try:
                proc.wait(STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                end(proc, 10)
        end(server, 30)
        host_log.close()

    changed = _changed(before, _snapshot(episode))
    info: Dict[str, Any] = {
        "harness": "fibsem-agent",
        "playbook": PLAYBOOK,
        "session_id": session_id,
        "model": model or "default",
        "effort": effort,
        "prompt_hash": prompt_hash(instructions()),
        "at": timestamps.now().isoformat(),
        "seconds": round(time.monotonic() - started, 1),
        "fix_turn": fixed,
    }
    info.update(_spent(session_id))
    data = segments.load(episode)
    if data:
        data["segmenter"] = info
        segments.save(episode, data)
    if changed:
        raise SegmenterError(
            "the agent changed files of the episode it may only read: "
            + ", ".join(changed[:10])
        )
    return segments.check(episode)
