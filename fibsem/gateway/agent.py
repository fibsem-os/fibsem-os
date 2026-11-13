"""The gateway's agent: one ``fibsem-agent`` for this PC's microscope, started
and ended from the gateway's page or window, kept by an `AgentSessionManager`.

Start needs the app running, with or without a microscope connected. It pairs the agent's
token straight into the shared ``paired.json`` (the operator starting it is
the Allow), sets the App Server's level, then starts
``fibsem-agent --playbook NAME``. The token and the API key go only in the
agent's environment. When the app comes back after a restart the manager
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.ai.agent.levels import AGENT_LISTENING_S, AgentLevel
from fibsem.ai.agent.manager import (
    END_WAIT_S,
    MANAGED,
    RECORD,
    RUN,
    WORK,
    AgentRunning,
    AgentSessionManager,
    read_json,
)
from fibsem.gateway.gateway import RUNNING, Gateway, GatewayRefused
from fibsem.gateway.host import call_host
from fibsem.keystore import stored_secret
from fibsem.server.discovery import read_discovery_file
from fibsem.server.pairing import AGENT, PairedTokens
from fibsem.util import timestamps

__all__ = ["AGENT_NAME", "API_KEY", "GatewayAgent", "AgentState"]

API_KEY = "ANTHROPIC_API_KEY"
KEY_KEPT_S = 30.0
# The agent's pairing, as the App Server's list of paired computers shows it.
AGENT_NAME = "fibsem-agent (gateway)"
SESSIONS_SHOWN = 5


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
    import httpx

    try:
        return httpx.get(f"{url}/health", timeout=3.0).status_code == 200
    except httpx.HTTPError:
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
    spent = [run["spent_usd"] for run in record.get("runs", []) if "spent_usd" in run]
    return round(sum(spent), 2) if spent else None


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
    ) -> None:
        self._gateway = gateway
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
        gateway.watchers.append(self._manager.poll)

    # -- reading -----------------------------------------------------------

    def running(self) -> Optional[str]:
        """The agent's state ("running", "waiting"), None when there is none:
        cheap, read every second by the gateway's window."""
        agent = self._manager.agent(self.microscope)
        return agent.state if agent is not None and agent.state != "ended" else None

    def state(self) -> AgentState:
        agent = self._manager.agent(self.microscope)
        running = None
        records = self._manager.sessions(SESSIONS_SHOWN + 1)
        rows = [_row(record, self._platform) for record in records]
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
        )

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
        with self._lock:
            level, url = self._ready(level, playbook, replace)
            env = self._pair()
            try:
                self._level(url, level)
            except GatewayRefused:
                self._unpair()
                raise
            args = ["--playbook", playbook, "--url", url]
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
            }
            try:
                self._manager.start(
                    self.microscope,
                    str(uuid.uuid4()),
                    args,
                    goal=goal or None,
                    env=env,
                    about=about,
                )
            except (AgentRunning, OSError) as error:
                self._undo(url)
                raise GatewayRefused(409, "agent_not_started", str(error)) from None
            logging.info(
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
            args = (["--playbook", playbook] if playbook else []) + ["--url", url]
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
            }
            try:
                self._manager.resume(
                    self.microscope, session_id, args, env=env, about=about
                )
            except (AgentRunning, OSError) as error:
                self._undo(url)
                raise GatewayRefused(409, "agent_not_started", str(error)) from None
            logging.info(
                "Gateway resumed session %s (%s) for %s", session_id, level, who
            )
            return self.state()

    def _ready(
        self,
        level: str,
        playbook: Optional[str],
        replace: bool,
        mode: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Start's and Resume's checks, before anything is done: the level
        and the App Server's url. *mode* is a resumed session's own."""
        try:
            level = AgentLevel(str(level).lower()).value
        except ValueError:
            names = ", ".join(lvl.value for lvl in AgentLevel)
            raise GatewayRefused(422, "bad_level", f"The level is one of {names}.")
        if self._gateway.state().mode != RUNNING:
            raise GatewayRefused(
                409, "no_app", "Run the app first: the agent acts through it."
            )
        url = self._url()
        if url is None:
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
            logging.info("Gateway ended the agent for %s", who)
        if stop_app:
            self._gateway.stop(who, via, force, window=window)
        return self.state()

    # -- the manager's and the gateway's calls -------------------------------

    def _for_switch(self) -> None:
        """The app is about to stop: the agent it serves ends first."""
        agent = self._manager.agent(self.microscope)
        if agent is not None and agent.state != "ended":
            self._manager.end(self.microscope)
            self._unpair()

    def _again(self, about: dict) -> None:
        """Before a resume: the restarted App Server is at Watch again."""
        url = self._url()
        level = about.get("level")
        if url is None or not level or level == AgentLevel.WATCH.value:
            return
        try:
            self._level(url, level)
        except GatewayRefused as refusal:
            logging.warning("The agent's level wasn't set again: %s", refusal)

    def _url(self) -> Optional[str]:
        found = self._discovery()
        if found is None or not self._answers(found["url"]):
            return None
        return found["url"]

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

    def _undo(self, url: str) -> None:
        self._unpair()
        try:
            self._level(url, AgentLevel.WATCH.value)
        except GatewayRefused:
            pass
