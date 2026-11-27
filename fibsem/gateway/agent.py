"""The gateway's agent: one ``fibsem-agent`` for this PC's microscope, started
and ended from the gateway's page or window, kept by an `AgentSessionManager`.

Start works with the app running, with or without a microscope connected,
or with no app while the Agent Server runs (`fibsem.server.agent.process`): the
agent then talks, plans and keeps its record there, and reaches the
microscope once the app answers. It pairs the agent's
token straight into the shared ``paired.json`` (the operator starting it is
the Allow), sets the App Server's level, then starts
``fibsem-agent --playbook NAME``, with the gateway's own address: the agent
reaches the App Server through the gateway's port. The token and the API key
go only in the agent's environment. When the app comes back after a restart the manager
resumes the agent, and the gateway sets the Start level again: the App Server
starts at Watch on every launch. End stops the agent and unpairs it.
Resume carries an ended session on, whoever started it, as its record says.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem import records
from fibsem.ai.agent.levels import AGENT_LISTENING_S, AgentLevel
from fibsem.ai.agent.manager import (
    END_WAIT_S,
    MANAGED,
    RECORD,
    RUN,
    RUN_BUDGET,
    WORK,
    AgentRunning,
    AgentSessionManager,
    read_json,
    sessions_dir,
)
from fibsem.ai.agent.report import MAX_GOAL
from fibsem.gateway import slot as slot_mod
from fibsem.gateway.gateway import RUNNING, Gateway, GatewayRefused
from fibsem.gateway.health import STARTING
from fibsem.gateway.host import call_host
from fibsem.keystore import stored_secret
from fibsem.server.discovery import read_discovery_file
from fibsem.server.pairing import AGENT, PairedTokens
from fibsem.util import timestamps

logger = logging.getLogger(__name__)

__all__ = ["AGENT_NAME", "API_KEY", "GatewayAgent", "AgentState", "SessionsPage"]

API_KEY = "ANTHROPIC_API_KEY"
KEY_KEPT_S = 30.0
# The agent's pairing, as the App Server's list of paired computers shows it.
AGENT_NAME = "fibsem-agent (gateway)"
SESSIONS_SHOWN = 5
# History pages through every session: this many at a time, at most this many
# in one read.
SESSIONS_PAGE = 20
SESSIONS_MAX = 1000
# While the agent runs, the App Server's level is noted this often, so a
# resume comes back at the level the operator last set.
LEVEL_READ_S = 5.0
LEVELS = {level.value for level in AgentLevel}


@dataclass
class AgentState:
    """The agent the gateway runs, or none, and the latest sessions."""

    running: Optional[Dict[str, Any]] = None
    # An agent the gateway didn't start, connected to the app: its name, when
    # the app last heard from it, and what to check when that was long ago.
    connected: Optional[Dict[str, Any]] = None
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    api_key: bool = False
    microscope: str = ""
    # Who holds the instrument's one agent slot (`fibsem.gateway.slot`).
    slot: Dict[str, Any] = field(default_factory=dict)
    # Whether the Agent Server runs: the agent works with no app through it.
    agent_host: bool = False


@dataclass
class SessionsPage:
    """The newest sessions run on this PC, newest first, and whether there
    are older ones."""

    sessions: List[Dict[str, Any]] = field(default_factory=list)
    more: bool = False


def read_playbooks() -> List[dict]:
    """The playbooks, as ``fibsem-agent --list-playbooks`` lists them."""
    from fibsem.ai.agent.harness import SetupRefused, playbook_list

    try:
        return playbook_list()
    except SetupRefused as refused:
        raise GatewayRefused(422, "bad_playbook", str(refused)) from None


def read_configuration(name: str) -> dict:
    """What fibsem-agent would run with for playbook *name*
    (``--show-config``), or `GatewayRefused`."""
    from fibsem.ai.agent.harness import SetupRefused, playbook_configuration

    try:
        return playbook_configuration(name)
    except SetupRefused as refused:
        raise GatewayRefused(422, "bad_playbook", str(refused)) from None


def set_level(url: str, level: str) -> None:
    """Set the App Server's agent level as the operator, or raise `GatewayRefused`."""
    try:
        call_host(url, "PUT", "/app/agent/level", {"level": level})
    except GatewayRefused as refusal:
        if refusal.error_type == "no_app":
            raise
        raise GatewayRefused(409, "level_refused", str(refusal)) from None


def answers(url: str) -> bool:
    try:
        call_host(url, "GET", "/health", timeout=3.0)
        return True
    except (GatewayRefused, ValueError):
        return False


def heard(url: str) -> dict:
    """The App Server's ``/app/agent``, as the operator: when it last heard
    from the agent."""
    return call_host(url, "GET", "/app/agent", timeout=3.0)


def quiet_note(seconds: Optional[float]) -> Optional[str]:
    """What the page says about an agent the app hasn't heard from in a
    while, None while it is listening."""
    if seconds is None:
        return "Not heard from since the app started."
    if seconds < AGENT_LISTENING_S:
        return None
    return (
        f"Not heard from for {int(seconds // 60)} min. If it runs on another "
        "computer, its SSH tunnel may be down."
    )


def _spent(record: dict) -> Optional[float]:
    return records.session_spent(record.get("runs") or [])


# Whether each working folder was there, and when that was read: the page
# lists the sessions every 2 s, and a folder on a slow drive can take seconds.
FOLDER_KEPT_S = 60.0
_folders_seen: Dict[str, Tuple[float, bool]] = {}


def _folder_there(cwd: str) -> bool:
    """Whether *cwd* is a folder, as read in the last `FOLDER_KEPT_S`."""
    seen = _folders_seen.get(cwd)
    if seen is None or time.monotonic() - seen[0] > FOLDER_KEPT_S:
        seen = (time.monotonic(), Path(cwd).is_dir())
        _folders_seen[cwd] = seen
    return seen[1]


def _why_not(
    record: dict, platform: str, is_dir: Callable[[str], bool] = os.path.isdir
) -> Optional[str]:
    """Why the session can't be resumed here, or None when it can."""
    if not record.get("session_id"):
        return "It has no session id."
    if record.get("mode") == "code" and platform == "win32":
        return "It ran code, which has no sandbox on Windows."
    cwd = record.get("cwd")
    if cwd and not is_dir(cwd) and Path(cwd).name != RUN:
        return "Its working folder is gone."
    return None


def _playbook(record: dict) -> Tuple[Optional[str], Optional[str]]:
    """The record's playbook name and hash: a name at start, then
    ``{name, hash, chosen_by, skills}`` once the session keeps its playbook
    (`SessionFolder.keep_playbook`)."""
    playbook = record.get("playbook")
    if isinstance(playbook, dict):
        return playbook.get("name"), playbook.get("hash")
    return playbook, record.get("playbook_hash")


def _last_level(about: dict) -> Optional[str]:
    for key in ("level_now", "level"):
        if about.get(key) in LEVELS:
            return about[key]
    return None


def _last_budget(record: dict, about: dict) -> Optional[float]:
    kept = record.get(RUN_BUDGET)
    usd = kept.get("usd") if isinstance(kept, dict) else None
    if usd is None:
        usd = about.get("budget_usd")
    try:
        return None if usd is None else float(usd)
    except (TypeError, ValueError):
        return None


def _row(record: dict, platform: str = sys.platform) -> Dict[str, Any]:
    """One session as the page lists it."""
    managed = record.get("manager") or {}
    runs = record.get("runs") or [{}]
    ended = managed.get("ended")
    playbook, playbook_hash = _playbook(record)
    if ended is None and "ended" in runs[-1]:
        ended = runs[-1].get("reason") or "ended"
    return {
        "session_id": record.get("session_id"),
        "started": runs[0].get("started")
        or (managed.get("about") or {}).get("started"),
        "playbook": playbook or (managed.get("about") or {}).get("playbook"),
        "playbook_hash": playbook_hash,
        "goal": record.get("goal"),
        "spent_usd": _spent(record),
        "ended": ended,
        "gateway": bool(managed),
        "resumed": bool((managed.get("about") or {}).get("resumed")),
        # The level the operator last set while it ran (Stop agent as Watch),
        # else its start level: Resume's first choice.
        "last_level": _last_level(managed.get("about") or {}),
        # The budget it last ran with (one the operator changed included),
        # else its start budget: Resume's first choice.
        "last_budget_usd": _last_budget(record, managed.get("about") or {}),
        "cannot_resume": _why_not(record, platform, _folder_there),
        # what the agent last printed, when it ended before its log said why
        "output": managed.get("output"),
    }


class GatewayAgent:
    """Start, end and show this PC's agent. One per gateway: the microscope
    is this PC. The App Server is found by its discovery file."""

    def __init__(
        self,
        gateway: Gateway,
        tokens: PairedTokens,
        root: Optional[Path] = None,
        command: Optional[List[str]] = None,
        end_wait_s: float = END_WAIT_S,
        discovery: Callable[[], Optional[dict]] = read_discovery_file,
        level: Callable[[str, str], None] = set_level,
        reachable: Callable[[str], bool] = answers,
        last_heard: Callable[[str], dict] = heard,
        playbooks: Callable[[], List[dict]] = read_playbooks,
        configuration: Callable[[str], dict] = read_configuration,
        environ: Optional[Dict[str, str]] = None,
        stored: Callable[[str], Optional[str]] = stored_secret,
        platform: str = sys.platform,
        microscope: Optional[str] = None,
        own_url: Optional[str] = None,
        slot_file: Optional[Path] = None,
        host: Callable[[], Optional[dict]] = lambda: None,
    ) -> None:
        self._gateway = gateway
        self._root = root
        self.slot = slot_mod.SlotRecord(
            slot_file or (root or sessions_dir()).parent / "slot.json"
        )
        # The gateway's own address: the agent reaches the App Server through
        # it (`fibsem.gateway.passthrough`), else directly.
        self._own_url = own_url
        # The Agent Server's discovery file (`fibsem.server.agent.process`): the
        # agent works through it while no app answers.
        self._host = host
        self._tokens = tokens
        self._discovery = discovery
        self._level = level
        self._answers = reachable
        self._heard = last_heard
        self._read_playbooks = playbooks
        self._read_configuration = configuration
        self._environ = os.environ if environ is None else environ
        self._stored = stored
        # Whether there is a key, and when that was read (the page's state).
        self._key_seen: Optional[Tuple[float, bool]] = None
        self._platform = platform
        self.microscope = microscope or socket.gethostname()
        self._lock = threading.Lock()
        self._manager = AgentSessionManager(
            root=root,
            command=command,
            end_wait_s=end_wait_s,
            reachable=lambda _: self._url() is not None,
            env_for=lambda _: self._pair(),
            on_resume=lambda _, about: self._again(about),
        )
        gateway.before_stop.append(self._for_switch)
        gateway.watchers.append(self._unpark)
        gateway.watchers.append(self._manager.poll)
        gateway.watchers.append(self._follow_level)
        gateway.watchers.append(self._after_end)
        gateway.watchers.append(self._follow_slot)
        self._level_read = float("-inf")
        # The session this gateway saw running, until its end sets Watch.
        self._ran: Optional[str] = None
        # The app a parked agent waits out (`_for_switch`).
        self._parked_pid: Optional[int] = None
        self._parked_on: Tuple[Optional[str], Optional[str]] = (None, None)
        # Whether no app answered since it was parked (its pid unread then).
        self._parked_gone = False
        # Whether the agent carried on on the Agent Server when its app stopped
        # (not parked): the next app is checked as a parked agent's is.
        self._moved = False
        self._allowed_read = float("-inf")
        self._allowed_at = float("-inf")
        self._lost_holder()

    def _lost_holder(self) -> None:
        """A gateway agent the slot names that no session carries on (a
        start that waited for its app when the last gateway stopped): the
        slot is free again."""
        held = self.slot.read()
        if held.get("holder") != AGENT_NAME or held.get("state") == slot_mod.ENDED:
            return
        agent = self._manager.agent(self.microscope)
        if agent is None or agent.state == "ended":
            self.slot.change(
                state=slot_mod.ENDED, why="did not carry on (the gateway closed)"
            )
            # Its pairing too: nothing of the gateway's agent runs.
            paired = self._tokens.agent()
            if paired is not None and paired.name == AGENT_NAME:
                self._unpair()

    # -- reading -----------------------------------------------------------

    def running(self) -> Optional[str]:
        """The agent's state ("running", "waiting"), None when there is none:
        cheap, read every second by the gateway's window."""
        agent = self._manager.agent(self.microscope)
        return agent.state if agent is not None and agent.state != "ended" else None

    def state(self) -> AgentState:
        agent = self._manager.agent(self.microscope)
        running = None
        rows = self._rows(SESSIONS_SHOWN + 1)
        if agent is not None and agent.state != "ended":
            # Read here: the agent may not have written its record yet.
            managed = read_json(agent.folder / MANAGED)
            record = {**read_json(agent.folder / RECORD), "manager": managed}
            running = {
                **_row(record, self._platform),
                **(managed.get("about") or {}),
                "session_id": agent.session_id,
                "state": agent.state,
                "restarts": managed.get("restarts", []),
            }
            rows = [r for r in rows if r["session_id"] != agent.session_id]
        return AgentState(
            running=running,
            connected=None if running is not None else self._connected(),
            sessions=rows[:SESSIONS_SHOWN],
            api_key=self._has_key(),
            microscope=self.microscope,
            slot=self.slot.read(),
            agent_host=self._host_running(),
        )

    def sessions(self, limit: int = SESSIONS_PAGE) -> SessionsPage:
        """The newest *limit* sessions, the running one too, for History to
        page through them all."""
        limit = max(1, min(int(limit), SESSIONS_MAX))
        rows = self._rows(limit + 1)
        return SessionsPage(sessions=rows[:limit], more=len(rows) > limit)

    def _rows(self, limit: int) -> List[Dict[str, Any]]:
        """The newest *limit* sessions as the page lists them, each with its
        status and experiment from the shared list (`fibsem.records`)."""
        rows = []
        for found in records.list_sessions(limit, root=self._root):
            folder = Path(found.folder)
            record = {"session_id": found.session_id, **read_json(folder / RECORD)}
            managed = read_json(folder / MANAGED)
            if managed:
                record["manager"] = managed
            rows.append(
                {
                    **_row(record, self._platform),
                    "status": found.status,
                    "status_reason": found.reason,
                    "last_active": found.last_active_at,
                    "experiment": (
                        asdict(found.experiment) if found.experiment else None
                    ),
                }
            )
        return rows

    def _connected(self) -> Optional[Dict[str, Any]]:
        """The agent paired with the app that the gateway didn't start, as
        the app last heard from it; None when there is none or the app isn't
        running."""
        paired = self._tokens.agent()
        if paired is None or paired.name == AGENT_NAME:
            return None
        url = self._url()
        if url is None:
            return None
        try:
            seconds = self._heard(url).get("last_heard_s")
        except (GatewayRefused, AttributeError):
            return None
        return {
            "name": paired.name,
            "last_heard_s": seconds,
            "quiet": quiet_note(seconds),
        }

    def own_running(self) -> Optional[str]:
        """The agent the gateway started, as the card names it while it runs
        ("the agent started from the gateway at 17:30"), or None."""
        agent = self._manager.agent(self.microscope)
        if agent is None or agent.state == "ended":
            return None
        about = read_json(agent.folder / MANAGED).get("about") or {}
        started = about.get("started")
        at = f" at {started[11:16]}" if isinstance(started, str) else ""
        return f"the agent started from the gateway{at}"

    def in_use(self) -> Optional[str]:
        """The agent holding this instrument's one slot and in use: the one
        the gateway started while it runs, else one paired another way that
        the app heard from in the last `AGENT_LISTENING_S`. None when the slot
        is free or its agent has gone quiet."""
        own = self.own_running()
        if own is not None:
            return own
        connected = self._connected()
        seconds = connected and connected.get("last_heard_s")
        if seconds is not None and seconds < AGENT_LISTENING_S:
            return connected["name"]
        return None

    def playbooks(self) -> List[dict]:
        return self._read_playbooks()

    def configuration(self, name: str) -> dict:
        return self._read_configuration(name)

    # -- acting ------------------------------------------------------------

    def start(
        self,
        playbook: str,
        goal: Optional[str],
        level: str,
        who: str,
        via: str,
        budget_usd: Optional[float] = None,
        replace: bool = False,
    ) -> AgentState:
        """Pair, set the level, start ``fibsem-agent``; each failure undoes
        what came before it. Raises `GatewayRefused`."""
        if goal and len(goal.strip()) > MAX_GOAL:
            raise GatewayRefused(
                422,
                "goal_too_long",
                f"The goal is {len(goal.strip())} characters: keep it to {MAX_GOAL} "
                "(the details can go to the agent in a message).",
            )
        with self._lock:
            level, url = self._ready(level, playbook, replace, wait=True)
            env = self._pair()
            # With no app yet, the agent waits; its level is set as it starts.
            if url is not None:
                try:
                    self._level(url, level)
                except GatewayRefused:
                    self._unpair()
                    raise
            args = ["--playbook", playbook, "--url", self._own_url or url]
            if budget_usd is not None:
                args += ["--max-budget-usd", f"{float(budget_usd):g}"]
            about = {
                "started": timestamps.now().isoformat(),
                "playbook": playbook,
                "goal": goal or None,
                "level": level,
                "who": who,
                "via": via,
                "budget_usd": budget_usd,
                "app_pid": self._server_pid(),
            }
            try:
                started = self._manager.start(
                    self.microscope,
                    str(uuid.uuid4()),
                    args,
                    goal=goal or None,
                    env=env,
                    about=about,
                    wait=url is None,
                )
            except (AgentRunning, OSError) as error:
                self._undo(url)
                raise GatewayRefused(409, "agent_not_started", str(error)) from None
            self.slot.take(
                AGENT_NAME,
                slot_mod.START,
                f"started by {who} ({via})",
                session_id=started.session_id,
                level=level,
            )
            logger.info(
                "Gateway started the agent (%s, %s) for %s", playbook, level, who
            )
            return self.state()

    def resume(
        self,
        session_id: str,
        level: str,
        who: str,
        via: str,
        budget_usd: Optional[float] = None,
        replace: bool = False,
    ) -> AgentState:
        """Carry on an ended session, started here or with ``fibsem-agent`` by
        hand, as it ran: its playbook, mode and folder from its record, with a
        fresh pairing and the App Server's url now. Checks and undoes as
        Start does. Raises `GatewayRefused`."""
        with self._lock:
            folder = self._manager.folder(session_id)
            record = read_json(folder / RECORD) if folder else {}
            if not record:
                raise GatewayRefused(
                    404, "no_session", f"No session {session_id} on this PC."
                )
            why = _why_not(record, self._platform)
            if why:
                raise GatewayRefused(409, "cannot_resume", why)
            started = read_json(folder / MANAGED).get("about") or {}
            playbook = _playbook(record)[0] or started.get("playbook")
            mode = record.get("mode")
            level, url = self._ready(level, playbook, replace, mode)
            env = self._pair()
            try:
                self._level(url, level)
            except GatewayRefused:
                self._unpair()
                raise
            args = (["--playbook", playbook] if playbook else []) + [
                "--url",
                self._own_url or url,
            ]
            cwd = record.get("cwd")
            if mode in ("code", "files"):
                # Its own work/ is the flag alone; any other folder is named.
                own = cwd == str(folder / WORK)
                args += [f"--{mode}"] + ([] if own else [cwd])
            elif mode == "none" and cwd and cwd != str(folder / RUN):
                # From before run/: it ran in a checkout, which the CLI needs
                # again to find its transcript.
                args += ["--checkout", cwd]
            for key in ("model", "effort"):
                if record.get(key):
                    args += [f"--{key}", str(record[key])]
            if budget_usd is not None:
                args += ["--max-budget-usd", f"{float(budget_usd):g}"]
            about = {
                "started": timestamps.now().isoformat(),
                "playbook": playbook,
                "goal": record.get("goal"),
                "level": level,
                "who": who,
                "via": via,
                "budget_usd": budget_usd,
                "resumed": True,
                "app_pid": self._server_pid(),
            }
            try:
                self._manager.resume(
                    self.microscope, session_id, args, env=env, about=about
                )
            except (AgentRunning, OSError) as error:
                self._undo(url)
                raise GatewayRefused(409, "agent_not_started", str(error)) from None
            self.slot.take(
                AGENT_NAME,
                slot_mod.RESUME,
                f"resumed by {who} ({via})",
                session_id=session_id,
                level=level,
            )
            logger.info(
                "Gateway resumed session %s (%s) for %s", session_id, level, who
            )
            return self.state()

    def _app_starting(self) -> bool:
        """Whether the gateway's health watch says the app is starting."""
        watch = getattr(self._gateway, "health", None)
        try:
            return watch is not None and watch.state().app.status == STARTING
        except Exception:  # noqa: BLE001 - a read never refuses a start
            return False

    def _ready(
        self,
        level: str,
        playbook: Optional[str],
        replace: bool,
        mode: Optional[str] = None,
        wait: bool = False,
    ) -> Tuple[str, Optional[str]]:
        """Start's and Resume's checks, before anything is done: the level
        and the App Server's url. *mode* is a resumed session's own. With
        *wait* (Start), no app is no refusal: the url is None and the agent
        waits for the app."""
        try:
            level = AgentLevel(str(level).lower()).value
        except ValueError:
            names = ", ".join(lvl.value for lvl in AgentLevel)
            raise GatewayRefused(422, "bad_level", f"The level is one of {names}.")
        waits = wait and self._own_url is not None
        starting = self._app_starting()
        server = self._server()
        if server is not None and not server["host"]:
            if self._gateway.state().mode != RUNNING or starting:
                # Not an app to start on yet: the Agent Server, if it runs.
                server = self._host_server()
        url = server["url"] if server is not None else None
        if url is None and not waits:
            if starting:
                raise GatewayRefused(
                    409,
                    "app_starting",
                    "The app is still starting: try again when its health shows ok.",
                )
            if self._gateway.state().mode != RUNNING:
                raise GatewayRefused(
                    409, "no_app", "Run the app first: the agent acts through it."
                )
            raise GatewayRefused(
                409, "no_app", "The app's server isn't up yet: try again shortly."
            )
        current = self._manager.agent(self.microscope)
        if current is not None and current.state != "ended":
            raise GatewayRefused(
                409, "agent_running", "The agent is already running: end it first."
            )
        if not self._api_key():
            raise GatewayRefused(
                409,
                "no_api_key",
                f"No API key: run fibsem-cli key set {API_KEY} on this PC.",
            )
        config = self.configuration(playbook) if playbook else {}
        if "code" in (config.get("mode"), mode) and self._platform == "win32":
            raise GatewayRefused(
                409,
                "code_mode_windows",
                f"Playbook {playbook} runs code, which has no sandbox on Windows: "
                "pick a playbook without code.",
            )
        paired = self._tokens.agent()
        if paired is not None and paired.name != AGENT_NAME and not replace:
            raise GatewayRefused(
                409,
                "agent_paired",
                f"An agent is already connected ({paired.name}). "
                "Starting this agent disconnects it.",
            )
        return level, url

    def end(
        self,
        who: str,
        via: str,
        stop_app: bool = False,
        force: bool = False,
        window: bool = False,
    ):
        """End the agent and unpair it; with *stop_app*, then stop the app
        (closing its window too with *window*, the operator's)."""
        ended = self._manager.end(self.microscope)
        if ended is not None:
            self._unpair()
            logger.info("Gateway ended the agent for %s", who)
            self._follow_slot()
            self._after_end()
        if stop_app:
            self._gateway.stop(who, via, force, window=window)
        return self.state()

    # -- the manager's and the gateway's calls -------------------------------

    def _for_switch(self) -> None:
        """The app is about to stop: the agent it serves stops first and
        waits, carrying on once a new app answers (`_unpark`). End ends it."""
        agent = self._manager.agent(self.microscope)
        if agent is not None and agent.state != "ended":
            self._parked_pid = self._app_pid()
            self._parked_on = self._app_and_experiment()
            self._parked_gone = False
            if agent.state == "running" and self._host_server() is not None:
                # It carries on with no app, on the Agent Server: its calls
                # pass there once the app has stopped.
                self._moved = True
                return
            parked = self._manager.park(self.microscope)
            if parked is not None and parked.state == "ended":
                # It didn't stop and was killed: it no longer holds the slot.
                self._unpair()

    def _app_and_experiment(self) -> Tuple[Optional[str], Optional[str]]:
        state = self._gateway.state()
        return state.app, state.experiment

    def _unpark(self) -> None:
        """A parked agent carries on once the app it waited out is gone and
        another answers: never on the app that was being stopped. A new app
        or experiment ends it instead: a session is one experiment's."""
        agent = self._manager.agent(self.microscope)
        if agent is None or agent.state == "ended":
            self._moved = False
            return
        if not agent.parked and not self._moved:
            return
        pid = self._app_pid()
        if pid is None:
            self._parked_gone = True
            return
        if pid == self._parked_pid or (
            self._parked_pid is None and not self._parked_gone
        ):
            return
        self._moved = False
        if self._app_and_experiment() != self._parked_on:
            self.end("gateway", "gateway")
            return
        if agent.parked:
            self._manager.unpark(self.microscope)

    def left(self) -> None:
        """The gateway is quitting on purpose: an agent it leaves running
        waits for the operator's Resume, not the next gateway start. A start
        still waiting for its app ends: it never ran."""
        self._manager.left()
        agent = self._manager.agent(self.microscope)
        if agent is not None and agent.state == "ended":
            self._unpair()
            self._follow_slot()

    def _app_pid(self) -> Optional[int]:
        found = self._discovery()
        return found.get("pid") if found else None

    def allowed(self, name: str, who: str, replacing: Optional[str]) -> None:
        """An agent allowed on the gateway's page now holds the slot."""
        why = f"allowed by {who}" + (f", replacing {replacing}" if replacing else "")
        self.slot.take(name, slot_mod.ALLOW, why)
        self._allowed_at = time.monotonic()
        self._allowed_read = float("-inf")

    def _follow_slot(self) -> None:
        """The gateway's own agent's state, as the slot record says it; an
        allowed agent's, from its pairing and when the app last heard it."""
        agent = self._manager.agent(self.microscope)
        if agent is None or (
            agent.state == "ended" and self.slot.read().get("came_in") == slot_mod.ALLOW
        ):
            self._follow_allowed()
            return
        if agent.state == "ended":
            changes = {"state": slot_mod.ENDED, "why": agent.ended or "ended"}
        elif agent.state == "waiting":
            changes = {"state": slot_mod.WAITING, "why": "waiting for the app"}
        else:
            changes = {"state": slot_mod.RUNNING}
        self.slot.change(session_id=agent.session_id, **changes)

    def _follow_allowed(self) -> None:
        """An agent that came in by Allow holds the slot while it stays paired
        and the app has heard from it in the last `AGENT_LISTENING_S`; asked
        every `LEVEL_READ_S`."""
        held = self.slot.read()
        if held.get("came_in") != slot_mod.ALLOW:
            return
        if time.monotonic() - self._allowed_read < LEVEL_READ_S:
            return
        self._allowed_read = time.monotonic()
        paired = self._tokens.agent()
        if paired is None or paired.name != held.get("holder"):
            self.slot.change(state=slot_mod.ENDED, why="no longer paired")
            return
        connected = self._connected()
        seconds = connected.get("last_heard_s") if connected else None
        if seconds is not None and seconds < AGENT_LISTENING_S:
            self.slot.change(state=slot_mod.RUNNING, why=held.get("why"))
        elif time.monotonic() - self._allowed_at < AGENT_LISTENING_S:
            return  # just allowed: give it time to call
        elif held.get("state") != slot_mod.ENDED:
            self.slot.change(
                state=slot_mod.ENDED,
                why=(
                    f"not heard from for {AGENT_LISTENING_S / 60:.0f} min"
                    if self._url() is not None
                    else "the app is not answering"
                ),
            )

    def _after_end(self) -> None:
        """Once an agent this gateway saw running has ended (End, replaced
        by an Allow, a resume that failed, restarts used up), the App
        Server goes back to Watch. Only the App Server: the session's record
        keeps the level the operator last set, for its Resume."""
        agent = self._manager.agent(self.microscope)
        if agent is not None and agent.state != "ended":
            self._ran = agent.session_id
            return
        if self._ran is None or (agent is not None and agent.session_id != self._ran):
            self._ran = None
            return
        self._ran = None
        url = self._url()
        if url is None:
            return  # a restarted app is at Watch already
        try:
            self._level(url, AgentLevel.WATCH.value)
        except GatewayRefused as refusal:
            logger.warning("The level wasn't set back to Watch: %s", refusal)

    def _follow_level(self) -> None:
        """While the agent runs, note the App Server's level (every
        `LEVEL_READ_S`) in its session's record: what the operator last set,
        Stop agent as Watch. A resume comes back at it."""
        agent = self._manager.agent(self.microscope)
        if agent is None or agent.state != "running":
            return
        if time.monotonic() - self._level_read < LEVEL_READ_S:
            return
        self._level_read = time.monotonic()
        server = self._server()
        if server is None:
            return
        url, pid = server["url"], server.get("pid")
        about = read_json(agent.folder / MANAGED).get("about") or {}
        if about.get("app_pid") is not None and pid != about["app_pid"]:
            # The app restarted under a running agent, or it moved between
            # the app and the Agent Server: that server is at Watch, so the
            # level last set is set back first.
            self._again(about)
            self._manager.note(self.microscope, app_pid=pid)
            return
        try:
            level = call_host(url, "GET", "/capabilities").get("level")
        except (GatewayRefused, AttributeError):
            return
        if level in LEVELS:
            self._manager.note(self.microscope, level_now=level, app_pid=pid)
            self.slot.change(session_id=agent.session_id, level=level)

    def _again(self, about: dict) -> None:
        """Before a resume by itself. An App Server that kept running holds
        the level the operator last set (a lowering, a Stop agent): it
        stands. A restarted one is at Watch again: the level the operator
        last set while the agent ran is set back, else the Start level."""
        url = self._url()
        if url is None:
            return
        if about.get("app_pid") is not None and about["app_pid"] == self._server_pid():
            return
        level = _last_level(about)
        if not level or level in (AgentLevel.WATCH.value, AgentLevel.OFF.value):
            return
        try:
            self._level(url, level)
        except GatewayRefused as refusal:
            logger.warning("The agent's level wasn't set again: %s", refusal)

    def _url(self) -> Optional[str]:
        """The server the agent works through now: the App Server, else the
        Agent Server; None when neither answers."""
        server = self._server()
        return server["url"] if server is not None else None

    def _server(self) -> Optional[dict]:
        """The App Server's ``{url, pid, host: False}`` while it answers,
        else the Agent Server's (``host: True``), else None."""
        found = self._discovery()
        if found is not None and self._answers(found["url"]):
            return {"url": found["url"], "pid": found.get("pid"), "host": False}
        return self._host_server()

    def _host_server(self) -> Optional[dict]:
        try:
            found = self._host()
        except Exception:  # noqa: BLE001 - an unreadable file is no host
            found = None
        if found is None or not self._answers(found["url"]):
            return None
        return {"url": found["url"], "pid": found.get("pid"), "host": True}

    def _host_running(self) -> bool:
        """Whether the Agent Server's discovery file is live: read every time
        the page asks, so no call to it."""
        try:
            return self._host() is not None
        except Exception:  # noqa: BLE001 - an unreadable file is no host
            return False

    def _server_pid(self) -> Optional[int]:
        server = self._server()
        return server.get("pid") if server is not None else None

    def _api_key(self) -> Optional[str]:
        """The environment's key, else the stored one: read at each start."""
        return self._environ.get(API_KEY) or self._stored(API_KEY)

    def _has_key(self) -> bool:
        """Whether there is a key, read again after `KEY_KEPT_S`: the stored
        one is a keyring lookup, too slow for every refresh of the page."""
        seen = self._key_seen
        if seen is None or time.monotonic() - seen[0] > KEY_KEPT_S:
            seen = self._key_seen = (time.monotonic(), bool(self._api_key()))
        return seen[1]

    def _pair(self) -> Dict[str, str]:
        """A fresh agent token, replacing any agent paired before, and the
        API key: the agent's environment only."""
        _, token = self._tokens.add(AGENT_NAME, role=AGENT)
        env = {"FIBSEM_SERVER_TOKEN": token}
        key = self._api_key()
        if key:
            env[API_KEY] = key
        return env

    def _unpair(self) -> None:
        paired = self._tokens.agent()
        if paired is not None and paired.name == AGENT_NAME:
            self._tokens.remove(paired.id)

    def _undo(self, url: Optional[str]) -> None:
        self._unpair()
        if url is None:
            return
        try:
            self._level(url, AgentLevel.WATCH.value)
        except GatewayRefused:
            pass
