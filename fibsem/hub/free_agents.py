"""The agents this hub PC runs with no instrument.

Such an agent is the same ``fibsem-agent`` an instrument's agent is, started
on the ``plain`` playbook (``instrument: none``) with a goal: a person gives
it work and talks to it on its Agent page. It acts on no instrument, so it
takes no instrument's slot and no level is set for it. Several run at once,
up to a limit set on this PC (``install.hub.free_agents``).

Each session has its own Agent Server (`fibsem.server.agent.host`), held by
the Hub and served under ``/hub/agent-server/<session id>/``: the agent
hears from it and keeps its chat, plan and record through it, and the Hub's
page shows the same Agent page over it. Everything stays in the session's
folder on this hub PC (``~/.fibsem/hub/free-sessions/``).

The agent's token at its Agent Server is made here at each start, kept in
memory and handed to the agent through its environment, as a gateway hands a
local agent its own: never in a file, never shown. The people who may use the
Hub (this PC's operator, and the browsers paired here) are the operator
there.

A Hub stopped on purpose ends these agents first. One that outlived a Hub
that crashed is stopped by the next Hub and carries its session on with a
new token.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fibsem import config
from fibsem.ai.agent import playbooks, sources
from fibsem.ai.agent.manager import (
    MANAGED,
    RECORD,
    AgentRunning,
    AgentSessionManager,
    folder_name,
    folders,
    read_json,
    write_json,
)
from fibsem.ai.agent.report import MAX_GOAL
from fibsem.client.transport import HOST_TOKEN_ENV
from fibsem.gateway.gateway import GatewayRefused
from fibsem.hub.agents import API_KEY
from fibsem.hub.history import AGENT, HubHistory
from fibsem.hub.hub import TOKENS_DIR
from fibsem.server.agent.host import build_agent_host
from fibsem.server.auth import AuthConfig
from fibsem.util import timestamps

__all__ = [
    "AT",
    "LIMIT_MAX",
    "PLAYBOOK",
    "SESSIONS_DIR",
    "AgentServers",
    "FreeAgents",
    "default_budget_usd",
    "limit",
    "set_limit",
]

log = logging.getLogger(__name__)

# The playbook such an agent starts on: no instrument, no servers of its own.
PLAYBOOK = "plain"
# Each session's folder, apart from the instruments' agents' sessions.
SESSIONS_DIR = TOKENS_DIR / "free-sessions"
# Where each session's Agent Server is served: ``AT/<session id>/...``.
AT = "/hub/agent-server"
# The Agent Server's own folder (its events) in the session's folder.
HOST_FOLDER = "host"
LIMIT_MAX = 20
POLL_S = 1.0


def limit() -> int:
    """How many agents with no instrument run here at once."""
    try:
        found = int(config.load_user_preferences().install.hub.free_agents)
    except Exception:  # noqa: BLE001 - unreadable preferences are the default
        return config.HubPreferences.free_agents
    return max(0, min(found, LIMIT_MAX))


def default_budget_usd() -> Optional[float]:
    """What the `PLAYBOOK` lets one run spend when a start names no budget."""
    checkout = sources.find_root(None, playbooks.DEFAULT)
    if checkout is None:
        return None
    return playbooks.load(PLAYBOOK, checkout).session.max_budget_usd


def set_limit(value: Any) -> int:
    """Save the limit. Raises `GatewayRefused` for anything but 0 to
    `LIMIT_MAX`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise GatewayRefused(422, "bad_value", "The limit is a whole number.")
    if not 0 <= value <= LIMIT_MAX:
        raise GatewayRefused(
            422, "bad_value", f"The limit is a number from 0 to {LIMIT_MAX}."
        )

    def change(preferences) -> None:
        preferences.install.hub.free_agents = value

    config.update_user_preferences(change)
    return limit()


@dataclass
class _Free:
    """One session with no instrument, as this Hub holds it."""

    session_id: str
    # The agent's token at its Agent Server, in memory only.
    token: str
    # The session's Agent Server: a FastAPI app, served under `AT`.
    server: Any
    about: Dict[str, Any]
    args: List[str] = field(default_factory=list)
    # Left running by an earlier Hub: stopped and started again with this
    # Hub's token on the next round.
    taken_back: bool = False


class FreeAgents:
    """Start, keep and end the agents this PC runs with no instrument.

    *auth* is the Hub's own (its operator token and its paired browsers);
    *url* gives the Hub's address, where the agents reach their Agent
    Servers; *api_key* the key they use, None when there is none.
    """

    def __init__(
        self,
        auth: Any,
        url: Callable[[], str],
        root: Optional[Path] = None,
        command: Optional[List[str]] = None,
        api_key: Callable[[], Optional[str]] = lambda: None,
        history: Optional[HubHistory] = None,
        limit: Callable[[], int] = limit,
        clock: Callable[[], float] = time.time,
        manager: Optional[AgentSessionManager] = None,
    ) -> None:
        self._auth = auth
        self._url = url
        self._root = root or SESSIONS_DIR
        self._api_key = api_key
        self._history = history
        self._limit = limit
        self._clock = clock
        self._lock = threading.RLock()
        self._held: Dict[str, _Free] = {}
        # The sessions that ended while this Hub ran, and why. Each keeps its
        # Agent Server, so its page still reads.
        self._ended: Dict[str, Dict[str, Any]] = {}
        self._servers: Dict[str, Any] = {}
        self._manager = manager or AgentSessionManager(
            root=self._root,
            command=command,
            clock=clock,
            env_for=lambda session_id: self._env(session_id),
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._adopt()

    @property
    def root(self) -> Path:
        """Where this hub PC keeps these agents' session folders."""
        return self._root

    def _note(self, what: str, who: str, ok: bool = True) -> None:
        if self._history is not None:
            self._history.write(AGENT, what, who, None, ok)

    # -- each session's Agent Server -------------------------------------------

    def _open(self, session_id: str, folder: Path) -> Any:
        """A new Agent Server for *session_id*, over its *folder*, and the
        agent's token there. An earlier one of the session is closed first."""
        self._close(session_id)
        token = secrets.token_urlsafe(32)
        auth = AuthConfig(
            token=token,
            operator_token=self._auth.token,
            pairing=self._auth.pairing,
        )
        server = build_agent_host(
            auth, folder / HOST_FOLDER, name="Agent Hub", sessions=self._root
        )
        self._servers[session_id] = server
        return token, server

    def _close(self, session_id: str) -> None:
        server = self._servers.pop(session_id, None)
        if server is not None:
            try:
                server.state.host.close()
            except Exception:  # noqa: BLE001 - closing never stops the rest
                log.debug("Closing an Agent Server failed.", exc_info=True)

    def server(self, session_id: str) -> Optional[Any]:
        """Session *session_id*'s Agent Server, None when this Hub holds none."""
        with self._lock:
            return self._servers.get(session_id)

    def _address(self, session_id: str) -> str:
        return f"{self._url().rstrip('/')}{AT}/{session_id}"

    def _env(self, session_id: str) -> Dict[str, str]:
        with self._lock:
            held = self._held.get(session_id)
        env = {HOST_TOKEN_ENV: held.token} if held is not None else {}
        key = self._api_key()
        if key:
            env[API_KEY] = key
        return env

    # -- what the page shows -----------------------------------------------------

    def state(self) -> Dict[str, Any]:
        """``{limit, agents}``: the agents with no instrument running here,
        newest first, then those that ended while this Hub ran."""
        with self._lock:
            rows = []
            for session_id, held in self._held.items():
                agent = self._manager.agent(session_id)
                rows.append(
                    {
                        **held.about,
                        "session_id": session_id,
                        "state": "waiting"
                        if agent is not None and agent.state == "waiting"
                        else "running",
                        "why": None,
                        "page": True,
                    }
                )
            rows += [
                {**about, "page": session_id in self._servers}
                for session_id, about in self._ended.items()
            ]
        rows.sort(key=lambda row: str(row.get("started") or ""), reverse=True)
        rows.sort(key=lambda row: row["state"] == "ended")
        return {"limit": self._limit(), "agents": rows}

    def running(self) -> List[str]:
        with self._lock:
            return list(self._held)

    # -- acting --------------------------------------------------------------------

    def _may_start(self) -> None:
        if not self._api_key():
            raise GatewayRefused(
                409,
                "no_api_key",
                "No API key on this hub PC: set it in the Hub's Settings, or "
                f"run fibsem-cli key set {API_KEY} here.",
            )
        most = self._limit()
        if len(self._held) >= most:
            raise GatewayRefused(
                409,
                "too_many",
                f"{len(self._held)} agents with no instrument already run on "
                f"this hub PC, and the limit is {most}: end one first, or "
                "raise the limit in the Hub's Settings.",
            )

    def start(
        self,
        goal: Optional[str],
        who: str,
        budget_usd: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Start an agent with no instrument on this PC. Raises
        `GatewayRefused`."""
        goal = (goal or "").strip() or None
        if goal and len(goal) > MAX_GOAL:
            raise GatewayRefused(
                422,
                "goal_too_long",
                f"The goal is {len(goal)} characters: keep it to {MAX_GOAL} "
                "(the details can go to the agent in a message).",
            )
        with self._lock:
            self._may_start()
            session_id = str(uuid.uuid4())
            folder = self._root / folder_name(session_id, self._clock())
            token, server = self._open(session_id, folder)
            about = {
                "started": timestamps.now().isoformat(),
                "playbook": PLAYBOOK,
                "goal": goal,
                "who": who,
                "budget_usd": budget_usd,
            }
            args = ["--playbook", PLAYBOOK, "--host", self._address(session_id)]
            if budget_usd is not None:
                args += ["--max-budget-usd", f"{float(budget_usd):g}"]
            self._held[session_id] = _Free(session_id, token, server, about, args)
            try:
                self._manager.start(
                    session_id,
                    session_id,
                    args,
                    goal=goal,
                    env=self._env(session_id),
                    about=about,
                )
            except (AgentRunning, OSError) as error:
                del self._held[session_id]
                self._close(session_id)
                raise GatewayRefused(409, "agent_not_started", str(error)) from None
            log.info("The hub started an agent with no instrument for %s", who)
        self._note("Started an agent with no instrument on this hub PC", who)
        return self._row(session_id)

    def resume(
        self, session_id: str, who: str, budget_usd: Optional[float] = None
    ) -> Dict[str, Any]:
        """Carry on a session with no instrument that ended on this hub PC,
        in its own folder. Raises `GatewayRefused`."""
        session_id = str(session_id or "")
        folder = self._manager.folder(session_id) if session_id else None
        record = read_json(folder / RECORD) if folder else {}
        if not record.get("session_id"):
            raise GatewayRefused(
                404,
                "no_session",
                f"No session {session_id} with no instrument on this hub PC.",
            )
        managed = read_json(folder / MANAGED)
        started = managed.get("about") or {}
        with self._lock:
            if session_id in self._held:
                raise GatewayRefused(
                    409, "agent_running", "That session's agent already runs."
                )
            self._may_start()
            token, server = self._open(session_id, folder)
            about = {
                "started": timestamps.now().isoformat(),
                "playbook": PLAYBOOK,
                "goal": record.get("goal") or started.get("goal"),
                "who": who,
                "budget_usd": budget_usd,
                "resumed": True,
            }
            args = ["--playbook", PLAYBOOK, "--host", self._address(session_id)]
            for key in ("model", "effort"):
                if record.get(key):
                    args += [f"--{key}", str(record[key])]
            if budget_usd is not None:
                args += ["--max-budget-usd", f"{float(budget_usd):g}"]
            self._held[session_id] = _Free(session_id, token, server, about, args)
            self._ended.pop(session_id, None)
            try:
                self._manager.resume(
                    session_id,
                    session_id,
                    args,
                    env=self._env(session_id),
                    about=about,
                )
            except (AgentRunning, OSError) as error:
                del self._held[session_id]
                self._close(session_id)
                raise GatewayRefused(409, "agent_not_started", str(error)) from None
            log.info(
                "The hub resumed session %s (no instrument) for %s", session_id, who
            )
        self._note("Resumed a session with no instrument on this hub PC", who)
        return self._row(session_id)

    def end(self, session_id: str, who: str) -> Dict[str, Any]:
        """End the agent of session *session_id*."""
        with self._lock:
            if session_id not in self._held:
                raise GatewayRefused(
                    409, "no_agent", "No such agent runs on this hub PC."
                )
        self._finish(session_id, f"ended by {who}", who=who)
        return self._row(session_id)

    def _row(self, session_id: str) -> Dict[str, Any]:
        rows = self.state()["agents"]
        return next((row for row in rows if row["session_id"] == session_id), {})

    def _finish(self, session_id: str, why: str, who: str = "the Hub") -> None:
        """Stop the session's agent and forget its token. Its Agent Server
        stays, so its page still reads."""
        self._manager.end(session_id)
        with self._lock:
            held = self._held.pop(session_id, None)
            if held is None:
                return
            # Nobody holds the token any more: a new one, that nobody has.
            held.server.state.auth.token = secrets.token_urlsafe(32)
            self._ended[session_id] = {
                **held.about,
                "session_id": session_id,
                "state": "ended",
                "why": why,
                "ended": timestamps.now().isoformat(),
            }
        log.info("The hub's agent with no instrument (%s) ended: %s", session_id, why)
        self._note(f"An agent with no instrument on this hub PC ended: {why}", who)

    # -- keeping -------------------------------------------------------------------

    def _adopt(self) -> None:
        """Take back the agents an earlier run of the Hub left running (it
        crashed or was restarted). Each one's token went with that Hub: it
        gets a new Agent Server here, and on the next round it is stopped and
        carries its session on with the new token."""
        for folder in folders(self._root):
            managed = read_json(folder / MANAGED)
            session_id = managed.get("microscope")
            agent = self._manager.agent(session_id) if session_id else None
            if agent is None or agent.state == "ended" or agent.folder != folder:
                continue
            token, server = self._open(session_id, folder)
            args = list(managed.get("args") or [])
            if "--host" in args[:-1]:
                # A new Hub may listen on another port.
                args[args.index("--host") + 1] = self._address(session_id)
                managed["args"] = list(args)
                write_json(folder / MANAGED, managed)
            self._held[session_id] = _Free(
                session_id,
                token,
                server,
                dict(managed.get("about") or {}),
                args,
                taken_back=True,
            )
            log.info(
                "The Hub took back its agent with no instrument (session %s) "
                "after a restart.",
                session_id,
            )
            self._note(
                "Took an agent with no instrument back after the Hub restarted",
                "the Hub",
            )

    def _carry_on(self, session_id: str, held: _Free) -> None:
        """Stop an agent taken back after a restart; it starts again on its
        session with this Hub's token on the manager's next round."""
        held.taken_back = False
        agent = self._manager.agent(session_id)
        parked = self._manager.park(session_id)
        if parked is None or parked is not agent or parked.state == "ended":
            return  # ended instead: the next round says so
        parked.env = self._env(session_id)
        self._manager.unpark(session_id)

    def poll(self) -> None:
        """Notice the agents that ended, and carry on the ones taken back."""
        with self._lock:
            held = dict(self._held)
        for session_id, one in held.items():
            try:
                if one.taken_back:
                    self._carry_on(session_id, one)
                agent = self._manager.agent(session_id)
                if agent is None or agent.state == "ended":
                    why = (agent.ended if agent is not None else None) or "it ended"
                    self._finish(session_id, why)
            except Exception:  # noqa: BLE001 - one agent never costs the others
                log.debug("Keeping an agent with no instrument failed.", exc_info=True)
        self._manager.poll()

    def start_keeping(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="hub-free-agents", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop keeping, end every agent with no instrument and close their
        Agent Servers."""
        self._stop.set()
        for session_id in self.running():
            self._finish(session_id, "the hub was stopped")
        with self._lock:
            for session_id in list(self._servers):
                self._close(session_id)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:  # noqa: BLE001 - the keeping goes on
                log.debug(
                    "Keeping the agents with no instrument failed.", exc_info=True
                )
            self._stop.wait(POLL_S)


def _answer(status: int, error_type: str, message: str):
    body = json.dumps(
        {"detail": {"error_type": error_type, "message": message}}
    ).encode("utf-8")
    start = {
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
    }
    return start, {"type": "http.response.body", "body": body}


class AgentServers:
    """ASGI: each session's Agent Server under ``AT/<session id>/``. A call
    goes to that session's own app, which checks its token itself: the
    agent's, or the Hub's operator. A session this Hub holds no Agent Server
    for answers 404."""

    def __init__(self, agents: FreeAgents) -> None:
        self._agents = agents

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return
        path = scope.get("path", "")
        # Mounted at AT: the path is whole, or what comes after AT.
        if path.startswith(AT + "/") or path == AT:
            path = path[len(AT) :]
        session_id, _, route = path.lstrip("/").partition("/")
        server = self._agents.server(session_id) if session_id else None
        if server is None:
            start, body = _answer(
                404,
                "no_session",
                "This Hub runs no agent with that session now. A session that "
                "ended before the Hub last started is resumed in Sessions.",
            )
            await send(start)
            await send(body)
            return
        child = dict(scope)
        child["path"] = "/" + route
        child["raw_path"] = ("/" + route).encode("utf-8")
        child["root_path"] = ""
        await server(child, receive, send)
