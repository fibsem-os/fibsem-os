"""Where each ``fibsem-agent`` session keeps its record, and the manager that
starts, watches, restarts and ends agents for the gateway and the hub.

Every session has a folder, ``~/.fibsem/agent/sessions/<date>-<session id>/``:
``session.json`` (what it runs with, and each run's start and end, written by
``fibsem-agent``), its log, and ``manager.json`` when a manager started it.
A resumed session runs again in the same folder.

The manager starts each agent detached, so agents keep running when the
manager stops, and a new manager takes back the ones still running. It stops
an agent with a ``stop`` file in its folder, which works on every platform
and for an agent it did not start itself.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fibsem import paths
from fibsem.util import timestamps

log = logging.getLogger("fibsem.agent.manager")

RECORD = "session.json"
MANAGED = "manager.json"
LOG = "fibsem-agent.log"
STOP = "stop"

# fibsem-agent's exit codes.
STOPPED, GONE, BUDGET = 0, 1, 2
# Restarts per microscope in any hour: after the app went away (exit 1), and
# after a crash (no exit recorded).
RESTARTS_PER_HOUR = {"app": 3, "crash": 1}
HOUR_S = 3600.0
END_WAIT_S = 60.0


def sessions_dir() -> Path:
    return paths.user_dir() / "agent" / "sessions"


def folder_name(session_id: str, started: Optional[float] = None) -> str:
    when = timestamps.from_posix(started if started is not None else time.time())
    return f"{when.strftime('%Y-%m-%d-%H%M%S')}-{session_id}"


def find_folder(session_id: str, root: Optional[Path] = None) -> Optional[Path]:
    """The folder of session *session_id*, or None."""
    root = root or sessions_dir()
    found = sorted(root.glob(f"*-{session_id}")) if root.is_dir() else []
    return found[-1] if found else None


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict) -> None:
    """Whole or not at all: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(partial, path)


def folders(root: Optional[Path] = None) -> List[Path]:
    """Every session's folder, newest first."""
    root = root or sessions_dir()
    if not root.is_dir():
        return []
    found = [path for path in root.iterdir() if (path / RECORD).is_file()]
    return sorted(found, key=lambda path: path.name, reverse=True)


def last_session(cwd: Path, root: Optional[Path] = None) -> Optional[str]:
    """The newest session run in *cwd* (the work folder, else the checkout),
    or None: the CLI keeps a session's transcript under its folder, so one
    run elsewhere can't be resumed here."""
    for folder in folders(root):
        record = read_json(folder / RECORD)
        if record.get("cwd") == str(cwd) and record.get("session_id"):
            return record["session_id"]
    return None


class SessionFolder:
    """The record ``fibsem-agent`` keeps of one session, in its folder."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    @classmethod
    def open(
        cls, session_id: str, about: dict, root: Optional[Path] = None
    ) -> "SessionFolder":
        """The session's folder, made when it is new; a run begins in it.
        *about* is what the session runs with (cwd, url, goal, model...)."""
        root = root or sessions_dir()
        folder = find_folder(session_id, root)
        if folder is None:
            folder = root / folder_name(session_id)
        folder.mkdir(parents=True, exist_ok=True)
        # A stop left from an earlier run is not for this one.
        (folder / STOP).unlink(missing_ok=True)
        session = cls(folder)
        record = session.record()
        runs = record.get("runs", [])
        runs.append({"pid": os.getpid(), "started": timestamps.now().isoformat()})
        # A resumed run says only what it was given: the goal stays the first run's.
        given = {key: value for key, value in about.items() if value is not None}
        record.update({"session_id": session_id, **given, "runs": runs})
        session._write(record)
        return session

    @property
    def log_path(self) -> Path:
        return self.folder / LOG

    def record(self) -> dict:
        return read_json(self.folder / RECORD)

    def stop_requested(self) -> bool:
        return (self.folder / STOP).exists()

    def end(
        self,
        code: int,
        reason: str,
        spent_usd: Optional[float] = None,
        turns: Optional[int] = None,
        resumable: bool = True,
    ) -> None:
        """How this run ended. *resumable* is False when resuming can't help
        (the model can't sign in)."""
        record = self.record()
        runs = record.get("runs") or [{}]
        runs[-1].update(
            ended=timestamps.now().isoformat(),
            exit_code=code,
            reason=reason,
            resumable=resumable,
            **({} if spent_usd is None else {"spent_usd": spent_usd}),
            **({} if turns is None else {"turns": turns}),
        )
        record["runs"] = runs
        self._write(record)

    def _write(self, record: dict) -> None:
        try:
            write_json(self.folder / RECORD, record)
        except OSError as error:
            log.warning("Couldn't write %s: %s", self.folder / RECORD, error)


def watch_for_stop(
    session: SessionFolder, stop: Callable[[], None], poll_s: float = 1.0
) -> threading.Thread:
    """Call *stop* once a ``stop`` file appears in the session's folder, on a
    daemon thread."""

    def run():
        while not session.stop_requested():
            time.sleep(poll_s)
        stop()

    thread = threading.Thread(target=run, name="fibsem-agent-stop", daemon=True)
    thread.start()
    return thread


def _pid_alive(pid) -> bool:
    from fibsem.server.discovery import _pid_alive as alive

    return alive(pid)


class AgentRunning(Exception):
    """The microscope already has an agent this manager runs."""


@dataclass
class Agent:
    """One microscope's agent, as the manager sees it."""

    microscope: str
    session_id: str
    folder: Path
    # "running", "waiting" (for its app, to resume) or "ended".
    state: str = "running"
    ended: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    # Added to the manager's environment for each run; not written down.
    env: Optional[Dict[str, str]] = None


def _detached() -> dict:
    """Popen arguments for a child that outlives its parent."""
    if sys.platform == "win32":
        flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        return {"creationflags": flags}
    return {"start_new_session": True}


def agent_command() -> List[str]:
    return [sys.executable, "-m", "fibsem.ai.agent.runner"]


class AgentSessionManager:
    """Starts ``fibsem-agent`` for a microscope, resumes it after its app
    comes back within a restart budget, and ends it. One agent per
    microscope. ``poll()`` acts on what has changed; the host calls it.

    *reachable(microscope)* says whether that microscope's App Server
    answers; a session whose app went away is resumed only once it does.
    *env_for(microscope)* gives the environment to resume with when the
    agent was started by an earlier manager (its token): the environment is
    never written down. *on_resume(microscope, about)* runs before each
    resume, with what the host recorded at start.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        command: Optional[List[str]] = None,
        reachable: Callable[[str], bool] = lambda microscope: True,
        clock: Callable[[], float] = time.time,
        end_wait_s: float = END_WAIT_S,
        env_for: Callable[[str], Dict[str, str]] = lambda microscope: {},
        on_resume: Callable[[str, dict], None] = lambda microscope, about: None,
    ) -> None:
        self._root = root or sessions_dir()
        self._command = command or agent_command()
        self._reachable = reachable
        self._clock = clock
        self._end_wait_s = end_wait_s
        self._env_for = env_for
        self._on_resume = on_resume
        self._lock = threading.RLock()
        self._agents: Dict[str, Agent] = {}
        self._adopt()

    def start(
        self,
        microscope: str,
        session_id: str,
        args: List[str],
        goal: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        about: Optional[dict] = None,
    ) -> Agent:
        """Start a new session *session_id* for *microscope*. *args* are
        ``fibsem-agent``'s flags (``--url``, ``--profile``, ...), passed again
        on each resume; *env* is added to this process's environment (the
        server token: never on the command line); *about* is the host's own
        record of the start (who, the level), kept in ``manager.json``."""
        with self._lock:
            current = self._agents.get(microscope)
            if current is not None and current.state != "ended":
                raise AgentRunning(microscope)
            folder = self._root / folder_name(session_id, self._clock())
            managed = {
                "microscope": microscope,
                "args": list(args),
                "restarts": [],
                **({"about": about} if about else {}),
            }
            write_json(folder / MANAGED, managed)
            launch = args + ["--session-id", session_id]
            if goal:
                launch += ["--goal", goal]
            agent = Agent(microscope, session_id, folder, env=dict(env or {}))
            agent.process = self._spawn(launch, env)
            self._agents[microscope] = agent
            log.info("Started the agent for %s (session %s).", microscope, session_id)
            return agent

    def end(self, microscope: str) -> Optional[Agent]:
        """Stop *microscope*'s agent: its stop file, then a kill once it has
        had ``end_wait_s``. Returns the agent, or None when there was none."""
        with self._lock:
            agent = self._agents.get(microscope)
            if agent is None:
                return None
            if agent.state == "waiting":
                self._finish(agent, "ended by the operator")
                return agent
            if agent.state == "ended":
                return agent
            (agent.folder / STOP).touch()
        deadline = time.monotonic() + self._end_wait_s
        while time.monotonic() < deadline and self._alive(agent):
            time.sleep(0.2)
        with self._lock:
            if self._alive(agent):
                log.warning("The agent for %s didn't stop; killing it.", microscope)
                self._kill(agent)
                self._finish(agent, "killed: it didn't stop")
            else:
                self._finish(agent, "ended by the operator")
            return agent

    def agent(self, microscope: str) -> Optional[Agent]:
        with self._lock:
            return self._agents.get(microscope)

    def poll(self) -> None:
        """Notice agents that exited, and resume the ones that may."""
        with self._lock:
            for agent in list(self._agents.values()):
                if agent.state == "running" and not self._alive(agent):
                    self._exited(agent)
                if agent.state == "waiting" and self._reachable(agent.microscope):
                    self._resume(agent)

    def sessions(self, limit: int = 20) -> List[dict]:
        """The newest sessions, run here or not: each folder's record, with
        the manager's part when it has one."""
        rows = []
        for folder in folders(self._root)[:limit]:
            row = read_json(folder / RECORD)
            managed = read_json(folder / MANAGED)
            if managed:
                row["manager"] = managed
            rows.append(row)
        return rows

    def _spawn(
        self, args: List[str], env: Optional[Dict[str, str]]
    ) -> subprocess.Popen:
        child_env = {**os.environ, **(env or {})}
        return subprocess.Popen(
            self._command + args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=child_env,
            **_detached(),
        )

    def _alive(self, agent: Agent) -> bool:
        if agent.process is not None:
            return agent.process.poll() is None
        runs = read_json(agent.folder / RECORD).get("runs") or [{}]
        return "ended" not in runs[-1] and _pid_alive(runs[-1].get("pid"))

    def _kill(self, agent: Agent) -> None:
        if agent.process is not None:
            agent.process.kill()
            agent.process.wait(10)
            return
        pid = (read_json(agent.folder / RECORD).get("runs") or [{}])[-1].get("pid")
        if _pid_alive(pid):
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False)
                else:
                    import signal

                    os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    def _exited(self, agent: Agent) -> None:
        """The agent's process ended on its own: what its record says decides
        whether it is resumed."""
        runs = read_json(agent.folder / RECORD).get("runs") or [{}]
        run = runs[-1]
        code = run.get("exit_code")
        if code is None:
            kind = "crash"
        elif code == GONE and run.get("resumable", True):
            kind = "app"
        else:
            reason = run.get("reason") or f"exited ({code})"
            self._finish(agent, reason)
            return
        if not self._may_restart(agent, kind):
            self._finish(agent, "restarts used up")
            return
        agent.state = "waiting"
        log.info(
            "The agent for %s %s; it resumes once its app answers.",
            agent.microscope,
            "crashed" if kind == "crash" else "lost its app",
        )
        self._note_restart(agent, kind)

    def _may_restart(self, agent: Agent, kind: str) -> bool:
        since = self._clock() - HOUR_S
        restarts = read_json(agent.folder / MANAGED).get("restarts", [])
        recent = [r for r in restarts if r.get("kind") == kind and r["at"] >= since]
        return len(recent) < RESTARTS_PER_HOUR[kind]

    def _note_restart(self, agent: Agent, kind: str) -> None:
        managed = read_json(agent.folder / MANAGED)
        managed.setdefault("restarts", []).append({"at": self._clock(), "kind": kind})
        write_json(agent.folder / MANAGED, managed)

    def _resume(self, agent: Agent) -> None:
        managed = read_json(agent.folder / MANAGED)
        args = list(managed.get("args", [])) + ["--resume", agent.session_id]
        env = agent.env if agent.env is not None else self._env_for(agent.microscope)
        try:
            self._on_resume(agent.microscope, managed.get("about") or {})
        except Exception:  # noqa: BLE001 - the host's step never stops the resume
            log.exception("Before resuming the agent for %s", agent.microscope)
        agent.process = self._spawn(args, env)
        agent.state = "running"
        log.info(
            "Resumed the agent for %s (session %s).", agent.microscope, agent.session_id
        )

    def _finish(self, agent: Agent, why: str) -> None:
        agent.state, agent.ended = "ended", why
        managed = read_json(agent.folder / MANAGED)
        managed["ended"] = why
        write_json(agent.folder / MANAGED, managed)
        log.info("The agent for %s ended: %s.", agent.microscope, why)

    def _adopt(self) -> None:
        """Take back the agents a manager started that haven't ended: running
        ones are watched again; ones that exited while no manager ran are
        dealt with by the next ``poll()``."""
        for folder in reversed(folders(self._root)):
            managed = read_json(folder / MANAGED)
            if not managed or "ended" in managed:
                continue
            record = read_json(folder / RECORD)
            microscope = managed.get("microscope")
            if not microscope or not record.get("session_id"):
                continue
            agent = Agent(microscope, record["session_id"], folder)
            self._agents[microscope] = agent
            log.info(
                "Took back the agent for %s (session %s).", microscope, agent.session_id
            )
