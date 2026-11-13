"""The agents a hub runs: on this PC, one per instrument, each reaching its
instrument's App Server through the hub's tunnel.

A start asks the instrument's gateway to pair the agent and set the level
(`GatewayAgent.lend`), then starts ``fibsem-agent`` here with the token it
answers. The instrument decides: its gateway refuses a level above its own
limit, and its operator's Stop and level still win. The hub is never in the
path of a command: the agent talks to the App Server, not through the hub.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fibsem import paths
from fibsem.ai.agent.levels import AgentLevel
from fibsem.ai.agent.manager import (
    MANAGED,
    RECORD,
    AgentRunning,
    AgentSessionManager,
    read_json,
)
from fibsem.gateway.agent import API_KEY, END_WAIT_S
from fibsem.gateway.gateway import GatewayRefused
from fibsem.gateway.hub import Hub
from fibsem.keystore import stored_secret
from fibsem.util import timestamps

__all__ = ["HubAgents", "hub_sessions_dir"]

log = logging.getLogger(__name__)

# What the page shows of a session's start (`manager.json`'s ``about``).
_ABOUT_KEYS = ("started", "playbook", "goal", "level", "who", "budget_usd")


def hub_sessions_dir() -> Path:
    """Where the hub's sessions are kept: apart from the sessions of this
    PC's own agent, which the gateway's manager takes back by folder."""
    return paths.user_dir() / "agent" / "hub-sessions"


class HubAgents:
    """Start, end and show the agent the hub runs for each instrument."""

    def __init__(
        self,
        hub: Hub,
        root: Optional[Path] = None,
        command: Optional[List[str]] = None,
        end_wait_s: float = END_WAIT_S,
        environ: Optional[Dict[str, str]] = None,
        stored: Callable[[str], Optional[str]] = stored_secret,
    ) -> None:
        self._hub = hub
        self._environ = os.environ if environ is None else environ
        self._stored = stored
        self._lock = threading.Lock()
        # The level each running agent was started at, for a resume. Not
        # kept across a hub restart: a resume then pairs at Watch.
        self._levels: Dict[str, str] = {}
        self._manager = AgentSessionManager(
            root=root or hub_sessions_dir(),
            command=command,
            end_wait_s=end_wait_s,
            reachable=self._reachable,
            env_for=self._again,
        )
        hub.session = self.session

    def poll(self) -> None:
        """Notice agents that ended, and resume the ones that may."""
        self._manager.poll()

    # -- reading ---------------------------------------------------------------

    def session(self, name: str) -> Optional[Dict[str, Any]]:
        """The agent this hub runs for *name*, or None."""
        agent = self._manager.agent(name)
        if agent is None or agent.state == "ended":
            return None
        about = read_json(agent.folder / MANAGED).get("about") or {}
        record = read_json(agent.folder / RECORD)
        return {
            "session_id": agent.session_id,
            "state": agent.state,
            **{key: about[key] for key in _ABOUT_KEYS if about.get(key) is not None},
            **(
                {"experiment_id": record["experiment_id"]}
                if record.get("experiment_id")
                else {}
            ),
        }

    # -- acting ----------------------------------------------------------------

    def start(
        self,
        name: str,
        playbook: str,
        goal: Optional[str],
        level: str,
        who: str,
        budget_usd: Optional[float] = None,
        replace: bool = False,
    ) -> Dict[str, Any]:
        """Pair an agent at *name*'s gateway, then start ``fibsem-agent`` here
        pointed at its App Server. A failure unpairs it. Raises
        `GatewayRefused`."""
        with self._lock:
            instrument = self._hub.instrument(name)
            current = self._manager.agent(name)
            if current is not None and current.state != "ended":
                raise GatewayRefused(
                    409, "agent_running", "Its agent is already running: end it first."
                )
            key = self._api_key()
            if not key:
                raise GatewayRefused(
                    409,
                    "no_api_key",
                    f"No API key: run fibsem-cli key set {API_KEY} on this PC.",
                )
            if not instrument.ssh:
                raise GatewayRefused(
                    409,
                    "no_tunnel",
                    f"{name} has no ssh destination, so this PC can't reach its "
                    "app: add one to start an agent for it.",
                )
            try:
                url = self._hub.app_url(instrument)
            except OSError as down:
                raise GatewayRefused(503, "unreachable", str(down)) from None
            lent = self._hub.call(
                name,
                "POST",
                "/gateway/agent/token",
                {"level": level, "replace": replace},
            )
            level = lent["level"]
            args = ["--playbook", playbook, "--url", str(url)]
            if budget_usd is not None:
                args += ["--max-budget-usd", f"{float(budget_usd):g}"]
            about = {
                "started": timestamps.now().isoformat(),
                "playbook": playbook,
                "goal": goal or None,
                "level": level,
                "who": who,
                "via": "hub",
                "budget_usd": budget_usd,
                "instrument": name,
            }
            try:
                self._manager.start(
                    name,
                    str(uuid.uuid4()),
                    args,
                    goal=goal or None,
                    env={"FIBSEM_SERVER_TOKEN": lent["token"], API_KEY: key},
                    about=about,
                )
            except (AgentRunning, OSError) as error:
                self._unpair(name)
                raise GatewayRefused(409, "agent_not_started", str(error)) from None
            self._levels[name] = level
            log.info(
                "The hub started the agent for %s (%s, %s).", name, playbook, level
            )
            return self.session(name) or {}

    def end(self, name: str, who: str) -> None:
        """End *name*'s agent and unpair it at its gateway."""
        self._hub.instrument(name)
        ended = self._manager.end(name)
        self._levels.pop(name, None)
        if ended is not None:
            self._unpair(name)
            log.info("The hub ended the agent for %s (%s).", name, who)

    def level(self, name: str, level: str) -> Dict[str, Any]:
        """Set the level at *name*'s gateway, which may refuse one above its
        limit: ``{level}``."""
        answer = self._hub.call(name, "PUT", "/gateway/agent/level", {"level": level})
        if name in self._levels:
            self._levels[name] = answer["level"]
        return answer

    # -- the manager's calls -----------------------------------------------------

    def _api_key(self) -> Optional[str]:
        return self._environ.get(API_KEY) or self._stored(API_KEY)

    def _unpair(self, name: str) -> None:
        try:
            self._hub.call(name, "DELETE", "/gateway/agent/token")
        except GatewayRefused as refusal:
            log.warning("The agent for %s wasn't unpaired there: %s", name, refusal)

    def _reachable(self, name: str) -> bool:
        """Whether *name*'s App Server answers through its tunnel."""
        try:
            url = self._hub.app_url(self._hub.instrument(name))
        except (GatewayRefused, OSError):
            return False
        return url is not None and self._hub.answers(url)

    def _again(self, name: str) -> Dict[str, str]:
        """A resume's environment: a fresh pairing at the level it had, Watch
        when this hub no longer knows it, and nothing when its gateway
        refuses (the agent then can't sign in, and ends)."""
        level = self._levels.get(name, AgentLevel.WATCH.value)
        try:
            lent = self._hub.call(
                name, "POST", "/gateway/agent/token", {"level": level, "replace": False}
            )
        except GatewayRefused as refusal:
            log.warning("The agent for %s wasn't paired again: %s", name, refusal)
            return {}
        env = {"FIBSEM_SERVER_TOKEN": lent["token"]}
        key = self._api_key()
        if key:
            env[API_KEY] = key
        return env
