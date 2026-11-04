"""The gateway's agent: one ``fibsem-agent`` for this PC's microscope, started
and ended from the gateway's page or tray, kept by an `AgentSessionManager`.

Start needs the app running with a microscope connected. It pairs the agent's
token straight into the shared ``paired.json`` (the operator starting it is
the Allow), sets the App Server's level, then starts
``fibsem-agent --profile NAME``. The token and the API key go only in the
agent's environment. When the app comes back after a restart the manager
resumes the agent, and the gateway sets the Start level again: the App Server
starts at Watch on every launch. End stops the agent and unpairs it.
Resume carries an ended session on, whoever started it, as its record says.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.ai.agent.levels import AgentLevel
from fibsem.ai.agent.manager import (
    END_WAIT_S,
    MANAGED,
    RECORD,
    RUN,
    WORK,
    AgentRunning,
    AgentSessionManager,
    agent_command,
    read_json,
)
from fibsem.gateway.gateway import RUNNING, Gateway, GatewayRefused
from fibsem.gateway.host import call_host, microscope_connected
from fibsem.keystore import stored_secret
from fibsem.server.discovery import read_discovery_file
from fibsem.server.pairing import AGENT, PairedTokens
from fibsem.util import timestamps

__all__ = ["AGENT_NAME", "API_KEY", "GatewayAgent", "AgentState"]

API_KEY = "ANTHROPIC_API_KEY"
# The agent's pairing, as the App Server's list of paired computers shows it.
AGENT_NAME = "fibsem-agent (gateway)"
SESSIONS_SHOWN = 5
CONFIG_WAIT_S = 60.0


@dataclass
class AgentState:
    """The agent the gateway runs, or none, and the latest sessions."""

    running: Optional[Dict[str, Any]] = None
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    api_key: bool = False
    microscope: str = ""


def fibsem_agent_json(flags: List[str]) -> Any:
    """``fibsem-agent <flags> --json``'s answer; its error text on failure."""
    done = subprocess.run(
        agent_command() + flags + ["--json"],
        capture_output=True,
        text=True,
        timeout=CONFIG_WAIT_S,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if done.returncode != 0:
        raise GatewayRefused(
            422,
            "bad_profile",
            (done.stderr or done.stdout).strip() or "fibsem-agent failed.",
        )
    return json.loads(done.stdout)


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


def _spent(record: dict) -> Optional[float]:
    spent = [run["spent_usd"] for run in record.get("runs", []) if "spent_usd" in run]
    return round(sum(spent), 2) if spent else None


def _why_not(record: dict, platform: str) -> Optional[str]:
    """Why the session can't be resumed here, or None when it can."""
    if not record.get("session_id"):
        return "It has no session id."
    if record.get("mode") == "code" and platform == "win32":
        return "It ran code, which has no sandbox on Windows."
    cwd = record.get("cwd")
    if cwd and not Path(cwd).is_dir() and Path(cwd).name != RUN:
        return "Its working folder is gone."
    return None


def _row(record: dict, platform: str = sys.platform) -> Dict[str, Any]:
    """One session as the page lists it."""
    managed = record.get("manager") or {}
    runs = record.get("runs") or [{}]
    ended = managed.get("ended")
    if ended is None and "ended" in runs[-1]:
        ended = runs[-1].get("reason") or "ended"
    return {
        "session_id": record.get("session_id"),
        "started": runs[0].get("started")
        or (managed.get("about") or {}).get("started"),
        "profile": record.get("profile") or (managed.get("about") or {}).get("profile"),
        "profile_hash": record.get("profile_hash"),
        "goal": record.get("goal"),
        "spent_usd": _spent(record),
        "ended": ended,
        "gateway": bool(managed),
        "resumed": bool((managed.get("about") or {}).get("resumed")),
        "cannot_resume": _why_not(record, platform),
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
        connected: Callable[[str], bool] = microscope_connected,
        agent_json: Callable[[List[str]], Any] = fibsem_agent_json,
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
        self._connected = connected
        self._agent_json = agent_json
        self._environ = os.environ if environ is None else environ
        self._stored = stored
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
            sessions=rows[:SESSIONS_SHOWN],
            api_key=bool(self._api_key()),
            microscope=self.microscope,
        )

    def profiles(self) -> List[dict]:
        return self._agent_json(["--list-profiles"])

    def configuration(self, name: str) -> dict:
        return self._agent_json(["--profile", name, "--show-config"])

    # -- acting ------------------------------------------------------------

    def start(
        self,
        profile: str,
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
            level, url = self._ready(level, profile, replace)
            env = self._pair()
            try:
                self._level(url, level)
            except GatewayRefused:
                self._unpair()
                raise
            args = ["--profile", profile, "--url", url]
            if budget_usd is not None:
                args += ["--max-budget-usd", f"{float(budget_usd):g}"]
            about = {
                "started": timestamps.now().isoformat(),
                "profile": profile,
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
                "Gateway started the agent (%s, %s) for %s", profile, level, who
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
        hand, as it ran: its profile, mode and folder from its record, with a
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
            profile = record.get("profile") or started.get("profile")
            mode = record.get("mode")
            level, url = self._ready(level, profile, replace, mode)
            env = self._pair()
            try:
                self._level(url, level)
            except GatewayRefused:
                self._unpair()
                raise
            args = (["--profile", profile] if profile else []) + ["--url", url]
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
                "profile": profile,
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
        profile: Optional[str],
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
        if not self._connected(url):
            raise GatewayRefused(
                409,
                "not_connected",
                "Connect the microscope first: the agent works on it.",
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
        config = self.configuration(profile) if profile else {}
        if "code" in (config.get("mode"), mode) and self._platform == "win32":
            raise GatewayRefused(
                409,
                "code_mode_windows",
                f"Profile {profile} runs code, which has no sandbox on Windows: "
                "pick a profile without code.",
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
