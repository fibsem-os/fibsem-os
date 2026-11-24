"""Where each ``fibsem-agent`` session keeps its record, and the manager that
starts, watches, restarts and ends agents for the gateway and the hub.

Every session has a folder, ``~/.fibsem/agent/sessions/<date>-<session id>/``:
``session.json`` (what it runs with, and each run's start and end, written by
``fibsem-agent``), its log, its model calls' tokens (``usage.jsonl``), the
Claude Code transcript (``transcript.jsonl``, and ``transcript/`` for the
specialists), the agent's handoff notes (``handoff.md``) once it has started
a fresh context, and ``manager.json`` when a manager started it.
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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fibsem import paths
from fibsem.util import timestamps
from fibsem.util.files import write_atomic
from fibsem.util.process import kill_pid, pid_alive, watch_stop_file

log = logging.getLogger("fibsem.agent.manager")

RECORD = "session.json"
MANAGED = "manager.json"
LOG = "fibsem-agent.log"
# fibsem-agent's console: what it says before its log opens, such as a refusal.
OUTPUT = "fibsem-agent-output.txt"
STOP = "stop"
# In the record: the newest event id the agent was given, for a resume.
EVENT_AFTER = "event_after"
# In the record: each fresh context after the session's first, oldest first:
# its CLI conversation and the session's spend when it started.
CONTEXTS = "contexts"
# The agent's handoff notes (`handoff`), newest last.
HANDOFF = "handoff.md"
# The wake policy the agent set itself, in the record (`SessionFolder.keep_wake_on`).
WAKE_ON = "wake_on"
# The agent's own files, when it was given no folder.
WORK = "work"
# Without a work folder a session runs in run/, which holds only its skills.
RUN = "run"

# fibsem-agent's exit codes.
STOPPED, GONE, BUDGET = 0, 1, 2
# Restarts per microscope in any hour: after the app went away (exit 1), and
# after a crash (no exit recorded).
RESTARTS_PER_HOUR = {"app": 3, "crash": 1}
HOUR_S = 3600.0
END_WAIT_S = 60.0
# A session found unfinished when a manager starts is resumed by itself only
# if it was heard from this recently, and its manager wasn't stopped on
# purpose; else it waits for the operator's Resume.
RESUME_YOUNG_S = 15 * 60.0


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


def session_folder(session_id: str, root: Optional[Path] = None) -> Path:
    """The folder of session *session_id*: its own, else the one a new
    session gets, not made yet."""
    root = root or sessions_dir()
    return find_folder(session_id, root) or root / folder_name(session_id)


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict) -> None:
    """Whole or not at all: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(data, indent=2))


def folders(root: Optional[Path] = None) -> List[Path]:
    """Every session's folder, newest first."""
    root = root or sessions_dir()
    if not root.is_dir():
        return []
    found = [path for path in root.iterdir() if (path / RECORD).is_file()]
    return sorted(found, key=lambda path: path.name, reverse=True)


def last_session(
    cwd: Optional[Path], root: Optional[Path] = None, own: str = WORK
) -> Optional[str]:
    """The newest session run in *cwd* (a work folder, or the checkout a
    session from before ``run/`` ran in), or with *cwd* None the newest that
    ran in its own folder's *own* (``work/`` or ``run/``); None if there is
    none: the CLI keeps a session's transcript under its folder, so one run
    elsewhere can't be resumed here."""
    for folder in folders(root):
        record = read_json(folder / RECORD)
        here = folder / own if cwd is None else cwd
        if record.get("cwd") == str(here) and record.get("session_id"):
            return record["session_id"]
    return None


class SessionRunning(RuntimeError):
    """The session is running in another process already."""


# Held while a run opens the session's record; one older than this was left
# by a process that died holding it.
OPENING = ".opening"
OPENING_STALE_S = 30.0
OPENING_WAIT_S = 10.0


@contextmanager
def _opening(folder: Path, wait_s: float = OPENING_WAIT_S):
    """Hold the session's opening lock: one run at a time reads its record
    and adds itself. Raises `SessionRunning` if it isn't free within *wait_s*."""
    lock = folder / OPENING
    deadline = time.monotonic() + wait_s
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > OPENING_STALE_S:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                continue
            if time.monotonic() > deadline:
                raise SessionRunning(
                    f"Another run of this session is starting ({lock} is held)."
                ) from None
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


class SessionFolder:
    """The record ``fibsem-agent`` keeps of one session, in its folder."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    @classmethod
    def open(
        cls,
        session_id: str,
        about: dict,
        root: Optional[Path] = None,
        folder: Optional[Path] = None,
    ) -> "SessionFolder":
        """The session's folder (*folder*, from `session_folder`, if given),
        made when it is new; a run begins in it. *about* is what the session
        runs with (cwd, url, goal, model...)."""
        folder = folder or session_folder(session_id, root)
        folder.mkdir(parents=True, exist_ok=True)
        session = cls(folder)
        # One run at a time: two starting together would both rebuild the
        # session's folder and each take the record as the other wrote it.
        with _opening(folder):
            record = session.record()
            runs = record.get("runs", [])
            last = runs[-1] if runs else {}
            pid = last.get("pid")
            if "ended" not in last and pid != os.getpid() and pid_alive(pid):
                raise SessionRunning(
                    f"Session {session_id} is already running (process {pid}): "
                    "stop it, or wait for it to end, before running it again."
                )
            # A stop left from an earlier run is not for this one.
            (folder / STOP).unlink(missing_ok=True)
            runs.append({"pid": os.getpid(), "started": timestamps.now().isoformat()})
            # A resumed run says only what it was given: the goal stays the first run's.
            given = {key: value for key, value in about.items() if value is not None}
            # The playbook a session keeps (`keep_playbook`, with its optional
            # skills' state) is never written over by the one it is launched with.
            if "playbook" in record:
                given.pop("playbook", None)
                given.pop("playbook_hash", None)
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
        tokens: Optional[Dict[str, int]] = None,
    ) -> None:
        """How this run ended. *resumable* is False when resuming can't help
        (the model can't sign in). *tokens* is what its model calls used."""
        record = self.record()
        runs = record.get("runs") or [{}]
        runs[-1].update(
            ended=timestamps.now().isoformat(),
            exit_code=code,
            reason=reason,
            resumable=resumable,
            **({} if spent_usd is None else {"spent_usd": spent_usd}),
            **({} if turns is None else {"turns": turns}),
            **({} if tokens is None else {"tokens": tokens}),
        )
        record["runs"] = runs
        self._write(record)

    def keep_playbook(self, state: dict) -> None:
        """The session's playbook and its optional skills
        (`playbooks.Reminders.state`), so a resume picks them up."""
        record = self.record()
        record["playbook"] = state.get("playbook")
        self._write(record)

    def keep_where(self, reported: dict) -> None:
        """The instrument and the experiment the session works on, as the App
        Server names them in its answer to a report: written when either
        changes, and kept when the server stops naming one."""
        record = self.record()
        found = {
            key: reported[key]
            for key in ("instrument_id", "instrument_serial", "experiment_id")
            if reported.get(key) and reported[key] != record.get(key)
        }
        if found:
            record.update(found)
            self._write(record)

    def keep_wake_on(self, state: Optional[dict]) -> None:
        """The wake policy the agent set itself (`wake_policy.WakeOn.state`),
        so a resume keeps it; None for the playbook's default."""
        record = self.record()
        if state is None:
            record.pop(WAKE_ON, None)
        else:
            record[WAKE_ON] = state
        self._write(record)

    def keep_event_cursor(self, event_id: str) -> None:
        """The id of the newest event the agent has been given, so a resume
        watches from after it (`watch.watch` ``after``)."""
        record = self.record()
        record[EVENT_AFTER] = event_id
        self._write(record)

    def keep_context(self, conversation_id: str, spent_before_usd: float) -> None:
        """A fresh context started (`handoff`): its CLI conversation, which a
        resume carries on, and the session's spend before it."""
        record = self.record()
        record.setdefault(CONTEXTS, []).append(
            {
                "conversation_id": conversation_id,
                "started": timestamps.now().isoformat(),
                "spent_before_usd": spent_before_usd,
            }
        )
        self._write(record)

    def keep_handoff(self, text: str) -> None:
        """A handoff note, after the earlier ones."""
        with open(self.folder / HANDOFF, "a", encoding="utf-8") as file:
            file.write(text if text.endswith("\n") else text + "\n")
            file.write("\n")

    def _write(self, record: dict) -> None:
        try:
            write_json(self.folder / RECORD, record)
        except OSError as error:
            log.warning("Couldn't write %s: %s", self.folder / RECORD, error)


def conversation(record: dict) -> Optional[str]:
    """The CLI conversation a resume of the session carries on: its newest
    fresh context's, else the session's own id."""
    contexts = [c for c in record.get(CONTEXTS) or [] if isinstance(c, dict)]
    if contexts and contexts[-1].get("conversation_id"):
        return contexts[-1]["conversation_id"]
    return record.get("session_id")


def context_spend(record: dict) -> float:
    """What the session had spent before its newest context: the CLI's total
    for that conversation starts there."""
    contexts = [c for c in record.get(CONTEXTS) or [] if isinstance(c, dict)]
    spent = contexts[-1].get("spent_before_usd") if contexts else None
    return float(spent) if isinstance(spent, (int, float)) else 0.0


def watch_for_stop(
    session: SessionFolder, stop: Callable[[], None], poll_s: float = 1.0
) -> threading.Thread:
    """Call *stop* once a ``stop`` file appears in the session's folder, on a
    daemon thread."""

    return watch_stop_file(
        session.folder / STOP, stop, poll_s=poll_s, name="fibsem-agent-stop"
    )


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
        # A hidden console, not none (DETACHED_PROCESS): the Claude CLI the
        # agent starts is a console program and would open a window of its own.
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        return {"creationflags": flags}
    return {"start_new_session": True}


def agent_command() -> List[str]:
    return [sys.executable, "-m", "fibsem.ai.agent.harness"]


def last_output(folder: Path, limit: int = 300) -> Optional[str]:
    """The last line fibsem-agent printed in *folder*, or None."""
    try:
        lines = (folder / OUTPUT).read_text(errors="replace").splitlines()
    except OSError:
        return None
    said = next((line.strip() for line in reversed(lines) if line.strip()), None)
    return said[:limit] if said else None


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
        ``fibsem-agent``'s flags (``--url``, ``--playbook``, ...), passed again
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
            agent.process = self._spawn(launch, env, folder)
            self._agents[microscope] = agent
            log.info("Started the agent for %s (session %s).", microscope, session_id)
            return agent

    def resume(
        self,
        microscope: str,
        session_id: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
        about: Optional[dict] = None,
    ) -> Agent:
        """Carry on the ended session *session_id* for *microscope*, in its own
        folder, whoever started it. *args* as for `start` (``--resume`` is
        added); the manager's record starts again: new args, no restarts."""
        with self._lock:
            current = self._agents.get(microscope)
            if current is not None and current.state != "ended":
                raise AgentRunning(microscope)
            folder = find_folder(session_id, self._root)
            if folder is None or not (folder / RECORD).is_file():
                raise FileNotFoundError(f"No session {session_id} on this PC.")
            managed = {
                "microscope": microscope,
                "args": list(args),
                "restarts": [],
                **({"about": about} if about else {}),
            }
            write_json(folder / MANAGED, managed)
            agent = Agent(microscope, session_id, folder, env=dict(env or {}))
            agent.process = self._spawn(args + ["--resume", session_id], env, folder)
            self._agents[microscope] = agent
            log.info("Resumed session %s for %s.", session_id, microscope)
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

    def folder(self, session_id: str) -> Optional[Path]:
        """Session *session_id*'s folder, or None."""
        return find_folder(session_id, self._root)

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
        self, args: List[str], env: Optional[Dict[str, str]], folder: Path
    ) -> subprocess.Popen:
        child_env = {**os.environ, **(env or {})}
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / OUTPUT, "ab") as output:
            return subprocess.Popen(
                self._command + args,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=child_env,
                **_detached(),
            )

    def _alive(self, agent: Agent) -> bool:
        if agent.process is not None:
            return agent.process.poll() is None
        runs = read_json(agent.folder / RECORD).get("runs") or [{}]
        return "ended" not in runs[-1] and pid_alive(runs[-1].get("pid"))

    def _kill(self, agent: Agent) -> None:
        if agent.process is not None:
            agent.process.kill()
            agent.process.wait(10)
            return
        pid = (read_json(agent.folder / RECORD).get("runs") or [{}])[-1].get("pid")
        if pid_alive(pid):
            kill_pid(pid)

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
        agent.process = self._spawn(args, env, agent.folder)
        agent.state = "running"
        log.info(
            "Resumed the agent for %s (session %s).", agent.microscope, agent.session_id
        )

    def _finish(self, agent: Agent, why: str) -> None:
        agent.state, agent.ended = "ended", why
        managed = read_json(agent.folder / MANAGED)
        managed["ended"] = why
        said = last_output(agent.folder)
        if said:
            managed["output"] = said
        write_json(agent.folder / MANAGED, managed)
        log.info("The agent for %s ended: %s.", agent.microscope, why)

    def left(self) -> None:
        """This manager's host is being stopped on purpose: the agents it
        leaves unfinished wait for the operator's Resume, not the next start."""
        with self._lock:
            for agent in self._agents.values():
                if agent.state == "ended":
                    continue
                managed = read_json(agent.folder / MANAGED)
                managed["left"] = self._clock()
                write_json(agent.folder / MANAGED, managed)

    def note(self, microscope: str, **about) -> None:
        """Add *about* to the host's record of *microscope*'s running agent
        (``manager.json``'s ``about``), which each resume is given."""
        with self._lock:
            agent = self._agents.get(microscope)
            if agent is None or agent.state == "ended":
                return
            managed = read_json(agent.folder / MANAGED)
            if all(managed.get("about", {}).get(k) == v for k, v in about.items()):
                return
            managed.setdefault("about", {}).update(about)
            write_json(agent.folder / MANAGED, managed)

    def _heard(self, folder: Path) -> float:
        """When session *folder*'s agent last wrote its record or printed."""
        times = []
        for name in (RECORD, OUTPUT):
            try:
                times.append((folder / name).stat().st_mtime)
            except OSError:
                continue
        return max(times, default=0.0)

    def _stale(self, agent: Agent, managed: dict) -> Optional[str]:
        """Why an unfinished session found at start, whose agent no longer
        runs, must not resume by itself; None when it may."""
        if managed.get("left") is not None:
            return "did not finish (the gateway closed)"
        quiet = self._clock() - self._heard(agent.folder)
        if quiet > RESUME_YOUNG_S:
            return (
                f"did not finish (last heard {quiet / 60:.0f} min before the "
                "gateway started)"
            )
        return None

    def _adopt(self) -> None:
        """Take back the agents a manager started that haven't ended: running
        ones are watched again; ones that exited while no manager ran are
        dealt with by the next ``poll()``, if they may resume by themselves
        (`_stale`); else they end, and the operator's Resume carries them on."""
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
            if self._alive(agent):
                if managed.pop("left", None) is not None:
                    # Still running when its gateway came back: carry on.
                    write_json(folder / MANAGED, managed)
            else:
                why = self._stale(agent, managed)
                if why is not None:
                    self._finish(agent, why)
                    continue
            log.info(
                "Took back the agent for %s (session %s).", microscope, agent.session_id
            )
