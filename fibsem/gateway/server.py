"""The gateway's routes, on its own port, localhost only.

``GET /gateway``, ``/gateway/health`` and ``/gateway/history`` for anyone
paired but an agent (an agent's token is refused on the gateway's own routes); ``POST /gateway/start`` and ``/gateway/stop`` for the operator, or a
hub the operator granted "may start and stop apps" (`SWITCH_MODES`); forcing
past a running workflow, and ``/gateway/claim`` (a desktop launch on this
PC), for the operator only. ``/gateway/microscope/...`` connects and
disconnects the running host's microscope, ``/gateway/app/agent-page`` links the operator to
its Agent page, ``/gateway/show`` brings the gateway's window to the front,
``/gateway/quit`` stops (or restarts) the gateway for the
operator, and ``/gateway/agent`` reads, starts and ends this
PC's agent (`fibsem.gateway.agent`), resuming an ended session too, with
the same rules as starting the app. ``/gateway/defaults`` is what the page
fills in for the next start (the operator saves it), and ``/gateway/secrets``
says which keys this PC has, never what they are.
With *passthrough*, the App Server's ``/app``, ``/microscope``, ``/services``
and ``/capabilities`` are served on the gateway's port too
(`fibsem.gateway.passthrough`), its token checks as ever.
The gateway alone pairs, for every server on this PC: the pairing routes
(`fibsem.server.pairing_routes`) with the operator, agent and hub roles, an
agent on this PC asking to connect (``POST /connect``, with ``gateway.json``'s
token) and ``/gateway/connections``, where the operator allows or denies it.
It serves the web app (``/ui``); the page is ``/ui/gateway``.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.requests import HTTPConnection
from fastapi.responses import RedirectResponse

from fibsem.ai.agent.playbooks import DEFAULT as DEFAULT_PLAYBOOK
from fibsem.ai.agent.report import MAX_GOAL
from fibsem.gateway import settings as gateway_settings
from fibsem.gateway.agent import (
    SESSIONS_MAX,
    SESSIONS_PAGE,
    AgentState,
    GatewayAgent,
    SessionsPage,
)
from fibsem.gateway.defaults import GatewayDefaults, read_defaults, save_defaults
from fibsem.gateway.device_server import NOT_SET_UP, DeviceServerProcess
from fibsem.gateway.device_servers import DeviceServers, EntryRefused, PairRefused
from fibsem.gateway.gateway import (
    HISTORY_MAX,
    HISTORY_SHOWN,
    Gateway,
    GatewayRefused,
    GatewayState,
    History,
)
from fibsem.gateway.health import ServersHealth
from fibsem.gateway.host import HostMicroscope
from fibsem.gateway.passthrough import PassThrough
from fibsem.gateway.peers import GATEWAY, START_DEVICE_SERVER, PeerAsks
from fibsem.server.auth import _bearer_token, _matches
from fibsem.server.bodies import refuse_invalid_requests
from fibsem.server.discovery import health_answer
from fibsem.server.hosting import agent_connections
from fibsem.server.operator_token import load_operator_token
from fibsem.server.pairing import (
    AGENT,
    HUB,
    OPERATOR,
    SWITCH_MODES,
    PairedComputer,
    PairedTokens,
    Pairing,
)
from fibsem.server.pairing_routes import build_pairing_router
from fibsem.server.web_app import web_app_router

logger = logging.getLogger(__name__)

__all__ = ["GatewayAuth", "build_gateway_app"]


@dataclass
class GatewayAuth:
    """The install's operator token and this PC's paired computers: the one
    ``paired.json`` every server here checks tokens against. The gateway alone
    pairs: browsers and other computers as the operator, agents, and hubs; and
    an agent on this PC asks to connect with *connect_token* (in
    ``gateway.json``), which the operator allows (*connections*). Shaped as the
    App Server's `AuthConfig` where the pairing routes look."""

    token: str
    pairing: Pairing
    connections: Any = None
    connect_token: Optional[str] = None

    def is_operator(self, token: Optional[str]) -> bool:
        return self.operator(token) is not None

    def operator(self, token: Optional[str]) -> Optional[str]:
        """Who the operator is: "operator" for this install's token, the
        paired computer's name for an operator pairing; None otherwise."""
        if _matches(token, self.token):
            return "operator"
        computer = self.pairing.tokens.match(token)
        if computer is not None and computer.role == OPERATOR:
            return computer.name
        return None

    def hub(self, token: Optional[str]) -> Optional[PairedComputer]:
        computer = self.pairing.tokens.match(token)
        return computer if computer is not None and computer.role == HUB else None

    def peer(self, token: Optional[str]) -> Optional[PairedComputer]:
        """A SEM PC's gateway paired through the gateways (`fibsem.gateway.peers`):
        it may start and stop this PC's Device Server, and nothing else."""
        computer = self.pairing.tokens.match(token)
        return computer if computer is not None and computer.role == GATEWAY else None

    def is_agent(self, token: Optional[str]) -> bool:
        return False

    def is_connect(self, token: Optional[str]) -> bool:
        """Whether *token* is ``gateway.json``'s, which only asks to connect."""
        return self.connect_token is not None and _matches(token, self.connect_token)

    @classmethod
    def from_disk(
        cls, token_file: Optional[Path] = None, paired_file: Optional[Path] = None
    ) -> "GatewayAuth":
        """The install's tokens, and asks to connect with a fresh connect
        token, allowed at once from this PC while the operator's setting says
        so (``install.app_server.allow_local_agents``)."""
        tokens = PairedTokens(paired_file)
        return cls(
            token=load_operator_token(token_file),
            pairing=Pairing(tokens, roles=(OPERATOR, AGENT, HUB)),
            connections=agent_connections(tokens, lambda: None),
            connect_token=secrets.token_urlsafe(32),
        )


@dataclass
class Caller:
    who: str
    operator: bool
    may_switch: bool
    # This install's own operator token: the operator at this PC (the
    # gateway's window, or the page it opens), never a paired computer.
    local: bool = False


def _settings(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A start's app settings, ``{name: value}``; refused when not a dict."""
    settings = body.get("settings")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise _refused(422, "bad_setting", "settings is {name: value}.")
    return settings


def _ask_view(ask, connections, own: Optional[str] = None) -> dict:
    """An ask to connect as the Allow card shows it: who asks, from where, and
    what allowing it means. *own*: the agent the gateway started, running,
    which allowing ends."""
    replaced = connections.tokens.agent()
    replaces = own or (replaced.name if replaced is not None else None)
    words = (
        "It can read the session, and acts only at the agent level set on the "
        "Agent page. It stays allowed until you unpair it in Pairing."
    )
    if own:
        words += f" It replaces {own}, which is running: allowing ends it."
    elif replaces:
        words += f" It replaces {replaces}, which is unpaired."
    return {**ask.to_dict(), "replaces": replaces, "if_allowed": words}


# The version of the gateway's own routes (/gateway, /agent, pairing), which a
# hub checks: a change an older hub can't work with raises the minor number, one
# it can ignore the patch. The App Server's routes it passes on have their own.
GATEWAY_API_VERSION = "0.1.0"


def _refused(status: int, error_type: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error_type": error_type, "message": message})


def build_gateway_app(
    gateway: Gateway,
    auth: GatewayAuth,
    agent: Optional[GatewayAgent] = None,
    microscope: Optional[HostMicroscope] = None,
    on_quit: Optional[Callable[[bool], None]] = None,
    on_show: Optional[Callable[[], bool]] = None,
    defaults_file: Optional[Path] = None,
    passthrough: Optional[PassThrough] = None,
    device_server: Optional[DeviceServerProcess] = None,
    remote_servers: Optional[DeviceServers] = None,
    peers: Optional[PeerAsks] = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if passthrough is not None:
            await passthrough.close()

    app = FastAPI(
        title="fibsem gateway", version=GATEWAY_API_VERSION, lifespan=lifespan
    )
    refuse_invalid_requests(app)
    app.state.auth = auth

    def caller(request: HTTPConnection) -> Caller:
        token = _bearer_token(request)
        who = auth.operator(token)
        if who is not None:
            local = _matches(token, auth.token)
            return Caller(who=who, operator=True, may_switch=True, local=local)
        hub = auth.hub(token)
        if hub is not None:
            return Caller(
                who=hub.name, operator=False, may_switch=SWITCH_MODES in hub.grants
            )
        raise _refused(401, "unauthorized", "Pair with the gateway first.")

    def switcher(request: HTTPConnection) -> Caller:
        found = caller(request)
        if not found.may_switch:
            raise _refused(
                403,
                "no_grant",
                "This hub may read the gateway but not start or stop apps: "
                "the operator grants that on the gateway's page.",
            )
        return found

    def _act(found: Caller, body: Dict[str, Any], act):
        force = bool(body.get("force", False))
        if force and not found.operator:
            raise _refused(403, "operator_only", "Only the operator forces a stop.")
        via = str(body.get("via") or "") if found.operator else "hub"
        try:
            return act(force, via or "page")
        except GatewayRefused as refusal:
            raise _refused(refusal.status, refusal.error_type, str(refusal)) from None

    @app.get("/")
    def root():
        return RedirectResponse("/ui/gateway")

    @app.get("/health")
    def health():
        """Open to anyone: the program, its version and its routes' version,
        the same shape as the App Server's and the Agent Server's."""
        return health_answer("fibsem-gateway", GATEWAY_API_VERSION)

    @app.get("/gateway", response_model=GatewayState)
    def get_gateway(found: Caller = Depends(caller)):
        """Off or running, the app and its process, another host the
        gateway didn't start, and the last exit."""
        return gateway.state()

    @app.get("/gateway/health", response_model=ServersHealth)
    def get_health(found: Caller = Depends(caller)):
        """How the App Server and its microscope are, as last asked."""
        return gateway.health.state()

    @app.get("/gateway/history", response_model=History)
    def get_history(
        limit: int = Query(HISTORY_SHOWN, ge=1, le=HISTORY_MAX),
        found: Caller = Depends(caller),
    ):
        """The latest *limit* starts and stops first, each with who made it
        and from where; ``more`` says whether older ones are kept."""
        return gateway.history(limit)

    @app.post("/gateway/start", response_model=GatewayState)
    def start(body: Dict[str, Any], found: Caller = Depends(switcher)):
        """``{app, window, experiment | last, create, force, connect,
        configuration, settings}``:
        stop the running app and start this one, with its window at this PC
        (the default) or without, connecting *configuration* (the default one)
        as it starts when *connect*."""
        return _act(
            found,
            body,
            lambda force, via: gateway.start_app(
                who=found.who,
                via=via,
                app=body.get("app"),
                window=bool(body.get("window", True)),
                experiment=body.get("experiment"),
                last=bool(body.get("last", False)),
                force=force,
                connect=None if body.get("connect") is None else bool(body["connect"]),
                configuration=body.get("configuration"),
                create=bool(body.get("create", False)),
                settings=_settings(body),
            ),
        )

    @app.post("/gateway/stop", response_model=GatewayState)
    def stop(body: Optional[Dict[str, Any]] = None, found: Caller = Depends(switcher)):
        """Back to off: ``{force}`` stops past a running workflow, and an app
        with its window closes it (both the operator's only)."""
        body = body or {}
        return _act(
            found,
            body,
            lambda force, via: gateway.stop(
                found.who, via, force, window=found.operator
            ),
        )

    @app.post("/gateway/claim", response_model=GatewayState)
    def claim(body: Dict[str, Any], found: Caller = Depends(caller)):
        """``{pid, app}``: a desktop launch on this PC takes the app's place."""
        if not found.operator:
            raise _refused(403, "operator_only", "Only a launch at the PC claims.")
        try:
            pid = int(body["pid"])
        except (KeyError, TypeError, ValueError):
            raise _refused(422, "bad_pid", "Give the window's pid.") from None
        return _act(
            found,
            body,
            lambda force, via: gateway.claim(
                pid,
                str(body.get("app", "")),
                found.who,
                str(body.get("via") or "desktop launch"),
            ),
        )

    if microscope is not None:

        @app.get("/gateway/microscope/connection")
        def get_connection(found: Caller = Depends(caller)) -> dict:
            """The running host's ``/microscope/connection``."""
            return _act(found, {}, lambda force, via: microscope.connection())

        @app.post("/gateway/microscope/connect", status_code=202)
        def connect(
            body: Optional[Dict[str, Any]] = None, found: Caller = Depends(switcher)
        ) -> dict:
            """``{configuration}``: the host connects it (the default without)."""
            name = (body or {}).get("configuration")
            return _act(found, {}, lambda force, via: microscope.connect(name))

        @app.post("/gateway/microscope/disconnect")
        def disconnect(found: Caller = Depends(switcher)) -> dict:
            """The host lets the microscope go; refused while a run is on."""
            return _act(found, {}, lambda force, via: microscope.disconnect())

        @app.get("/gateway/app/agent-page")
        def agent_page(found: Caller = Depends(caller)) -> dict:
            """``{url}``: the running host's Agent page, signed in as the
            operator the way the window's link is. The operator only."""
            if not found.operator:
                raise _refused(
                    403, "operator_only", "Only the operator opens the Agent page."
                )
            return _act(found, {}, lambda force, via: {"url": microscope.agent_page()})

    if agent is not None:

        @app.get("/gateway/agent", response_model=AgentState)
        def get_agent(found: Caller = Depends(caller)):
            """The running agent, or none, and the latest sessions."""
            return agent.state()

        @app.get("/gateway/agent/sessions", response_model=SessionsPage)
        def get_sessions(
            limit: int = Query(SESSIONS_PAGE, ge=1, le=SESSIONS_MAX),
            found: Caller = Depends(caller),
        ):
            """This PC's newest *limit* agent sessions, newest first, each
            as `/gateway/agent` lists one; ``more`` says whether older ones
            are kept."""
            return agent.sessions(limit)

        @app.get("/gateway/agent/playbooks")
        def get_playbooks(found: Caller = Depends(caller)) -> List[dict]:
            """The playbooks on this PC, as ``fibsem-agent --list-playbooks``."""
            return _act(found, {}, lambda force, via: agent.playbooks())

        @app.get("/gateway/agent/playbooks/{name}")
        def get_playbook(name: str, found: Caller = Depends(caller)) -> dict:
            """What *name* runs with, as ``fibsem-agent --show-config``."""
            return _act(found, {}, lambda force, via: agent.configuration(name))

        def _budget(body: Dict[str, Any]) -> Optional[float]:
            budget = body.get("budget_usd")
            try:
                budget = float(budget) if budget not in (None, "") else None
            except (TypeError, ValueError):
                raise _refused(422, "bad_budget", "The budget is a number of dollars.")
            if budget is not None and budget <= 0:
                raise _refused(422, "bad_budget", "The budget is above $0.")
            return budget

        @app.post("/gateway/agent/start", response_model=AgentState)
        def start_agent(body: Dict[str, Any], found: Caller = Depends(switcher)):
            """``{playbook, goal, level, budget_usd, replace}``: pair the agent,
            set the level, start it. ``replace`` disconnects an agent paired
            some other way."""
            budget = _budget(body)
            playbook = str(body.get("playbook") or DEFAULT_PLAYBOOK)
            return _act(
                found,
                body,
                lambda force, via: agent.start(
                    playbook,
                    str(body.get("goal") or "").strip() or None,
                    str(body.get("level") or "watch"),
                    who=found.who,
                    via=via,
                    budget_usd=budget,
                    replace=bool(body.get("replace", False)),
                ),
            )

        @app.post("/gateway/agent/resume", response_model=AgentState)
        def resume_agent(body: Dict[str, Any], found: Caller = Depends(switcher)):
            """``{session_id, level, budget_usd, replace}``: carry on an ended
            session as its record says, with Start's checks."""
            budget = _budget(body)
            session_id = str(body.get("session_id") or "")
            if not session_id:
                raise _refused(422, "no_session", "Say which session to resume.")
            return _act(
                found,
                body,
                lambda force, via: agent.resume(
                    session_id,
                    str(body.get("level") or "watch"),
                    who=found.who,
                    via=via,
                    budget_usd=budget,
                    replace=bool(body.get("replace", False)),
                ),
            )

        @app.post("/gateway/agent/end", response_model=AgentState)
        def end_agent(
            body: Optional[Dict[str, Any]] = None, found: Caller = Depends(switcher)
        ):
            """``{stop_app, force}``: end the agent; with ``stop_app``, then the app."""
            body = body or {}
            return _act(
                found,
                body,
                lambda force, via: agent.end(
                    found.who,
                    via,
                    bool(body.get("stop_app", False)),
                    force,
                    window=found.operator,
                ),
            )

        def hub_caller(found: Caller) -> str:
            if found.operator:
                raise _refused(
                    403, "hub_only", "Only a paired Hub asks for the hub agent's slot."
                )
            return found.who

        def _hub(act):
            try:
                return act()
            except GatewayRefused as refusal:
                raise _refused(
                    refusal.status, refusal.error_type, str(refusal)
                ) from None

        @app.post("/gateway/agent/slot")
        def take_slot(body: Dict[str, Any], found: Caller = Depends(switcher)):
            """For an agent a paired Hub runs on its own PC.
            ``{session_id, playbook, goal, level, budget_usd, replace}``: the
            instrument's checks, the agent's token (in this answer only), the
            level. ``{session_id, rejoin: true, token}``: the same session
            back after its grace ran out; with no token, the same paired Hub
            after it was restarted. Answers ``{token, lease_s, grace_s,
            state}``."""
            hub = hub_caller(found)
            if body.get("rejoin"):
                token = body.get("token")
                return _hub(
                    lambda: agent.hub.rejoin(
                        hub, body.get("session_id"), str(token) if token else None
                    )
                )
            budget = _budget(body)
            goal = str(body.get("goal") or "").strip() or None
            if goal and len(goal) > MAX_GOAL:
                raise _refused(
                    422, "goal_too_long", f"Keep the goal to {MAX_GOAL} characters."
                )
            return _hub(
                lambda: agent.hub.take(
                    hub,
                    body.get("session_id"),
                    str(body.get("playbook") or DEFAULT_PLAYBOOK),
                    goal,
                    str(body.get("level") or "watch"),
                    budget_usd=budget,
                    replace=bool(body.get("replace", False)),
                )
            )

        @app.post("/gateway/agent/slot/renew")
        def renew_slot(body: Dict[str, Any], found: Caller = Depends(switcher)):
            """``{session_id}``, every ``lease_s / 3``: ``{lease_s, grace_s}``,
            or 409 once the session no longer holds the slot."""
            hub = hub_caller(found)
            return _hub(lambda: agent.hub.renew(hub, str(body.get("session_id") or "")))

        @app.post("/gateway/agent/slot/release")
        def release_slot(body: Dict[str, Any], found: Caller = Depends(switcher)):
            """``{session_id}``: its token unpaired, Watch, the slot free."""
            hub = hub_caller(found)
            agent.hub.release(hub, str(body.get("session_id") or ""))
            return {"state": agent.state()}

    if on_quit is not None:

        @app.post("/gateway/quit")
        def quit_gateway(
            body: Optional[Dict[str, Any]] = None, found: Caller = Depends(caller)
        ) -> dict:
            """``{restart}``: stop the gateway as closing its window does; with
            ``restart``, it then starts again. The operator only."""
            if not found.operator:
                raise _refused(
                    403, "operator_only", "Only the operator stops the gateway."
                )
            restart = bool((body or {}).get("restart", False))
            on_quit(restart)
            return {"stopping": True, "restart": restart}

    if device_server is not None:

        @app.get("/gateway/device-server")
        def get_device_server(found: Caller = Depends(caller)) -> dict:
            """This PC's Device Server (from ``device-server.yaml``): stopped,
            starting, running or not set up, its devices and its last log
            lines."""
            return asdict(device_server.state())

        def starter(request: HTTPConnection) -> Caller:
            """The operator, or a SEM PC's gateway paired to start and stop
            the Device Server (`fibsem.gateway.peers`); no one else."""
            peer = auth.peer(_bearer_token(request))
            if peer is not None and START_DEVICE_SERVER in peer.grants:
                return Caller(who=peer.name, operator=False, may_switch=False)
            found = caller(request)
            if not found.operator:
                raise _refused(
                    403, "operator_only", "Only the operator starts and stops it."
                )
            return found

        def device_server_act(found: Caller, act: str) -> dict:
            via = "page" if found.operator else "gateway"
            gateway.record_access(found.who, via, f"{act} the Device Server")
            done = device_server.start if act == "start" else device_server.stop
            return asdict(done())

        @app.post("/gateway/device-server/start")
        def start_device_server(found: Caller = Depends(starter)) -> dict:
            """Start it; a running one is left as it is. The operator, or a
            paired SEM PC's gateway."""
            return device_server_act(found, "start")

        @app.post("/gateway/device-server/stop")
        def stop_device_server(found: Caller = Depends(starter)) -> dict:
            """Stop it: it switches its devices off as it ends. The operator,
            or a paired SEM PC's gateway."""
            return device_server_act(found, "stop")

    if device_server is not None and peers is not None:

        @app.post("/gateway/peers/ask")
        def ask_to_pair(request: Request, body: Optional[Dict[str, Any]] = None):
            """Unauthenticated: a SEM PC's gateway asks to pair with this PC's
            Device Server, ``{name, id?, wait_s?}``. Nothing comes back until
            the operator here allows it: ``{state: waiting, id}``, then
            ``allowed`` with the two tokens, to that id only, or ``denied``.
            Only on a PC that serves devices."""
            if device_server.state().status == NOT_SET_UP:
                raise _refused(404, "no_devices", "This PC serves no devices.")
            body = body or {}
            try:
                wait_s = float(body.get("wait_s", 20.0))
            except (TypeError, ValueError):
                wait_s = 20.0
            origin = request.client.host if request.client else "unknown"
            return peers.ask(
                str(body.get("name") or ""),
                "this PC's SSH tunnel" if origin in ("127.0.0.1", "::1") else origin,
                ask_id=str(body.get("id") or "") or None,
                wait_s=wait_s,
            )

        @app.get("/gateway/peers/asks")
        def get_pair_asks(found: Caller = Depends(caller)) -> List[dict]:
            """The gateway asking to pair, for the Allow card: none or one."""
            if not found.operator:
                return []
            return [ask.to_dict() for ask in peers.asking()]

        def answer_ask(found: Caller, ask: str, allow: bool) -> dict:
            if not found.operator:
                raise _refused(
                    403, "operator_only", "Only the operator allows a pairing."
                )
            try:
                if allow:
                    computer = peers.allow(ask)
                    name = computer.name
                else:
                    name = peers.deny(ask).name
            except LookupError as error:
                raise _refused(404, "no_ask", str(error)) from None
            what = "allowed" if allow else "denied"
            gateway.record_access(found.who, "page", f"{what} {name} (gateway)")
            return {"answered": what, "name": name}

        @app.post("/gateway/peers/asks/{ask}/allow")
        def allow_pair_ask(ask: str, found: Caller = Depends(caller)) -> dict:
            """Pair the asking gateway: with the Device Server, and here to
            start and stop it only. The operator only."""
            return answer_ask(found, ask, True)

        @app.post("/gateway/peers/asks/{ask}/deny")
        def deny_pair_ask(ask: str, found: Caller = Depends(caller)) -> dict:
            return answer_ask(found, ask, False)

    if remote_servers is not None:

        def edit_entry(found: Caller, what: str, edit) -> GatewayState:
            if not found.operator:
                raise _refused(
                    403, "operator_only", "Only the operator adds Device Servers."
                )
            try:
                path = edit()
            except EntryRefused as refused:
                raise _refused(refused.status, "bad_entry", str(refused)) from None
            name = path.name if path is not None else "the configuration"
            gateway.record_access(found.who, "page", f"{what} in {name}")
            return gateway.state()

        @app.post("/gateway/device-servers", response_model=GatewayState)
        def add_device_server(
            body: Dict[str, Any], found: Caller = Depends(caller)
        ) -> GatewayState:
            """``{device, address, port, tunnel}``: point *device* at a Device
            Server in the configuration the app connects with (one entry's
            ``remote:``, and ``tunnel: user@host`` for an SSH tunnel). It
            counts from the app's next connect. The operator only."""
            device = str(body.get("device") or "")
            return edit_entry(
                found,
                f"added a Device Server for {device}",
                lambda: remote_servers.add(
                    device,
                    str(body.get("address") or ""),
                    body.get("port"),
                    body.get("tunnel"),
                ),
            )

        def remote_act(found: Caller, key: str, act: str):
            if not found.operator:
                raise _refused(
                    403, "operator_only", "Only the operator pairs, starts and stops."
                )
            try:
                if act == "pair":
                    remote_servers.pair(key)
                else:
                    remote_servers.switch(key, act)
            except PairRefused as refused:
                raise _refused(refused.status, "device_server", str(refused)) from None
            gateway.record_access(
                found.who, "page", f"{act} the Device Server at {key}"
            )
            return gateway.state()

        @app.post("/gateway/device-servers/{key}/pair", response_model=GatewayState)
        def pair_device_server(key: str, found: Caller = Depends(caller)):
            """Ask the Device Server's PC gateway to pair (through its SSH
            tunnel); the operator there allows it. The operator only."""
            return remote_act(found, key, "pair")

        @app.post("/gateway/device-servers/{key}/start", response_model=GatewayState)
        def start_remote_device_server(key: str, found: Caller = Depends(caller)):
            """Start it through its PC's gateway, once paired. The operator only."""
            return remote_act(found, key, "start")

        @app.post("/gateway/device-servers/{key}/stop", response_model=GatewayState)
        def stop_remote_device_server(key: str, found: Caller = Depends(caller)):
            """Stop it through its PC's gateway, once paired. The operator only."""
            return remote_act(found, key, "stop")

        @app.delete("/gateway/device-servers/{device}", response_model=GatewayState)
        def remove_device_server(
            device: str, found: Caller = Depends(caller)
        ) -> GatewayState:
            """Take *device* off its Device Server in that configuration. The
            operator only."""
            return edit_entry(
                found,
                f"removed the Device Server for {device}",
                lambda: remote_servers.remove(device),
            )

    @app.get("/gateway/defaults", response_model=GatewayDefaults)
    def get_defaults(found: Caller = Depends(caller)):
        """What the page fills in for the next start."""
        return read_defaults(defaults_file)

    @app.put("/gateway/defaults", response_model=GatewayDefaults)
    def put_defaults(body: Dict[str, Any], found: Caller = Depends(caller)):
        """``{configuration, connect, app, window, playbook, level,
        budget_usd, settings, apps}``: the operator only. Saved as
        *app*'s own choices too (`apps`), the other apps' kept."""
        if not found.operator:
            raise _refused(403, "operator_only", "Only the operator sets defaults.")
        try:
            defaults = GatewayDefaults.checked(body)
        except GatewayRefused as refusal:
            raise _refused(refusal.status, refusal.error_type, str(refusal)) from None
        defaults = defaults.merged(read_defaults(defaults_file))
        save_defaults(defaults, defaults_file)
        return defaults

    @app.get("/gateway/secrets")
    def get_secrets(found: Caller = Depends(caller)) -> List[dict]:
        """Each key fibsemOS reads, and each connector's token: whether it is
        set here and where from, and whether this caller may change it (the
        operator at this PC). The key itself never leaves this PC."""
        here = found.local
        return [{**row, "can_set": here} for row in gateway_settings.key_rows()]

    def at_this_pc(found: Caller) -> None:
        """Keys are set only by the operator at this PC: this install's own
        token, never a paired computer, a hub or an agent."""
        if not found.local:
            raise _refused(
                403,
                "at_this_pc_only",
                "Keys are set only at this PC, on the gateway's own page.",
            )

    @app.put("/gateway/secrets/{name}")
    def put_secret(name: str, body: Dict[str, Any], found: Caller = Depends(caller)):
        """``{value}``: store a key. Write-only: the answer says it is set,
        never what it is."""
        at_this_pc(found)
        try:
            row = gateway_settings.set_key(name, (body or {}).get("value"))
        except gateway_settings.KeyRefused as refused:
            raise _refused(refused.status, refused.error_type, str(refused)) from None
        gateway.record_access(found.who, "page", f"set key {name}")
        return {**row, "can_set": True}

    @app.delete("/gateway/secrets/{name}")
    def delete_secret(name: str, found: Caller = Depends(caller)):
        """Remove a stored key: this PC stops using it (turn it off where it
        was issued too)."""
        at_this_pc(found)
        try:
            row = gateway_settings.remove_key(name)
        except gateway_settings.KeyRefused as refused:
            raise _refused(refused.status, refused.error_type, str(refused)) from None
        gateway.record_access(found.who, "page", f"removed key {name}")
        return {**row, "can_set": True}

    def operator_only(found: Caller, what: str) -> None:
        if not found.operator:
            raise _refused(403, "operator_only", f"Only the operator {what}.")

    def _connector_refused(refused: gateway_settings.KeyRefused):
        return _refused(refused.status, refused.error_type, str(refused))

    @app.get("/gateway/connectors")
    def get_connectors(found: Caller = Depends(caller)) -> dict:
        """The lab's connectors (``mcp.yaml``) as the file has them, each
        with why it can't be used, if it can't; and whether this caller may
        change a command connector (the operator at this PC)."""
        operator_only(found, "reads the connectors")
        return {**gateway_settings.connectors(), "at_this_pc": found.local}

    @app.put("/gateway/connectors/{name}")
    def put_connector(name: str, body: Dict[str, Any], found: Caller = Depends(caller)):
        """``{entry, confirm?}``: add or replace a connector, checked as the
        agent reads it. A command connector: at this PC only, a new or
        changed command once *confirm* repeats it (428 until then)."""
        operator_only(found, "changes the connectors")
        body = body or {}
        try:
            saved = gateway_settings.save_connector(
                name, body.get("entry"), found.local, body.get("confirm")
            )
        except gateway_settings.KeyRefused as refused:
            raise _connector_refused(refused) from None
        gateway.record_access(found.who, "page", f"saved connector {name}")
        return saved

    @app.delete("/gateway/connectors/{name}")
    def delete_connector(name: str, found: Caller = Depends(caller)) -> dict:
        operator_only(found, "changes the connectors")
        try:
            gateway_settings.remove_connector(name, found.local)
        except gateway_settings.KeyRefused as refused:
            raise _connector_refused(refused) from None
        gateway.record_access(found.who, "page", f"removed connector {name}")
        return {"removed": name}

    @app.post("/gateway/connectors/{name}/test")
    def test_connector(name: str, found: Caller = Depends(caller)) -> dict:
        """``{tools}``: the tools it offers, asked as the agent asks."""
        operator_only(found, "tests the connectors")
        try:
            return gateway_settings.test_connector(name, found.local)
        except gateway_settings.KeyRefused as refused:
            raise _connector_refused(refused) from None

    @app.get("/gateway/this-pc")
    def get_this_pc(found: Caller = Depends(caller)) -> dict:
        """This PC's install settings, and whether this caller may change the
        ones kept to the operator at this PC."""
        operator_only(found, "reads this PC's settings")
        return {**gateway_settings.this_pc(), "at_this_pc": found.local}

    @app.put("/gateway/this-pc")
    def put_this_pc(body: Dict[str, Any], found: Caller = Depends(caller)) -> dict:
        """The fields to change: ``microscope``, ``allow_local_agents`` (at
        this PC only), ``port`` (from the next start), ``update_check``."""
        operator_only(found, "changes this PC's settings")
        try:
            saved = gateway_settings.save_this_pc(body, found.local)
        except gateway_settings.KeyRefused as refused:
            raise _connector_refused(refused) from None
        changed = ", ".join(f"{k} {v}" for k, v in sorted((body or {}).items()))
        gateway.record_access(found.who, "page", f"set {changed}")
        return {**saved, "at_this_pc": found.local}

    @app.post("/gateway/show")
    def show_window(found: Caller = Depends(caller)) -> dict:
        """``{shown}``: bring the gateway's window to the front (the desktop
        shortcut); false when it runs without one. The operator only."""
        if not found.operator:
            raise _refused(403, "operator_only", "Only the operator shows the window.")
        return {"shown": bool(on_show and on_show())}

    def _connections():
        if auth.connections is None:
            raise _refused(404, "connect_off", "This gateway doesn't take asks.")
        return auth.connections

    def _operator(found: Caller) -> None:
        if not found.operator:
            raise _refused(
                403, "operator_only", "Only the operator allows an agent in."
            )

    def _own() -> Optional[str]:
        return agent.own_running() if agent is not None else None

    if agent is not None and auth.connections is not None:
        # One agent per instrument: always allowing never replaces one in use.
        auth.connections.in_use = agent.in_use

        def always_allowed(computer, replaced) -> None:
            gateway.record_access(
                "always allow",
                "setting",
                f"Allowed {computer.name}"
                + (f", replacing {replaced}" if replaced else ""),
            )
            agent.allowed(computer.name, "always allow", replaced)

        auth.connections.on_allowed_always(always_allowed)

    @app.get("/gateway/connections")
    def get_connections(found: Caller = Depends(caller)) -> dict:
        """``{asking, always_allow_local}``: the agent on this PC asking to
        connect, as its card shows it, or none."""
        _operator(found)
        connections = _connections()
        asks = connections.asking()
        return {
            "asking": _ask_view(asks[0], connections, _own()) if asks else None,
            "always_allow_local": bool(connections.always_allow),
        }

    @app.post("/gateway/connections/{ask_id}/allow")
    def allow_connection(ask_id: str, found: Caller = Depends(caller)) -> dict:
        """Allow the agent asking to connect: a token of its own, in the one
        paired list with Unpair. One agent per instrument: it replaces the
        agent paired before, and ends the gateway's own agent if it runs."""
        _operator(found)
        connections = _connections()
        if not any(ask.id == ask_id for ask in connections.asking()):
            raise _refused(404, "not_found", "No agent is asking with that id.")
        own = _own()
        replaced = connections.tokens.agent()
        # The Allow first: one that fails (the ask expired) leaves the agent
        # running as it was. Then the agent the gateway started ends, as its
        # End does (the card said allowing ends it); its token is gone already.
        try:
            computer = connections.allow(ask_id)
        except LookupError as exc:
            raise _refused(404, "not_found", str(exc)) from None
        if own is not None:
            agent.end(found.who, "page")
        replacing = own or (replaced.name if replaced else None)
        allowed = f"Allowed {computer.name}" + (
            f", replacing {replacing}" if replacing else ""
        )
        logger.info("%s: %s", found.who, allowed)
        gateway.record_access(found.who, "page", allowed)
        if agent is not None:
            agent.allowed(computer.name, found.who, replacing)
        return {"id": ask_id, "allowed": True, "name": computer.name}

    @app.post("/gateway/connections/{ask_id}/deny")
    def deny_connection(
        ask_id: str,
        body: Optional[Dict[str, Any]] = None,
        found: Caller = Depends(caller),
    ) -> dict:
        """Refuse the agent asking to connect, with the operator's reason; its
        next ask waits a minute."""
        _operator(found)
        reason = str((body or {}).get("reason") or "")[:500]
        try:
            ask = _connections().deny(ask_id, reason)
        except LookupError as exc:
            raise _refused(404, "not_found", str(exc)) from None
        logger.info("%s denied %s", found.who, ask.name)
        gateway.record_access(
            found.who, "page", f"Denied {ask.name}" + (f": {reason}" if reason else "")
        )
        return {"id": ask_id, "allowed": False, "name": None}

    app.include_router(build_pairing_router(auth))
    app.include_router(web_app_router())
    if passthrough is not None:
        passthrough.on_answered = lambda *call: _record_library(gateway, auth, *call)
        # Last: the gateway's own routes are found first.
        app.include_router(passthrough.router())

    return app


_LIBRARY_KINDS = {"playbooks": "playbook", "skills": "skill", "agents": "subagent"}
_LIBRARY_VERBS = {"PUT": "saved", "POST": "made", "DELETE": "removed"}


def _record_library(
    gateway, auth: GatewayAuth, method: str, path: str, status: int, token: str
) -> None:
    """A playbook, skill or subagent saved, made or removed through the
    gateway (the Knowledge window) goes in History, with who did it."""
    parts = path.strip("/").split("/")
    verb = _LIBRARY_VERBS.get(method)
    if (
        verb is None
        or not 200 <= status < 300
        or len(parts) != 4
        or parts[:2] != ["app", "library"]
        or parts[2] not in _LIBRARY_KINDS
    ):
        return
    who = auth.operator(token)
    if who is None:
        hub = auth.hub(token)
        who = hub.name if hub is not None else "someone"
    gateway.record_access(who, "page", f"{verb} {_LIBRARY_KINDS[parts[2]]} {parts[3]}")
