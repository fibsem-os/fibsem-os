"""The hub's HTTP routes, on this PC only (127.0.0.1).

``GET /hub`` lists the instruments as they were last read, for anyone signed
in. The operator alone changes the list or pairs with an instrument: the
install's operator token, or a browser paired as the operator with a code from
``agent-hub pair``. The hub keeps its own paired computers
(``~/.fibsem/hub/paired.json``), apart from a gateway's on the same PC.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from starlette.requests import HTTPConnection

from fibsem import __version__, keystore
from fibsem.gateway import settings as gateway_settings
from fibsem.gateway.gateway import GatewayRefused
from fibsem.gateway.passthrough import PASSED, ROOT_PASSED
from fibsem.gateway.server import GatewayAuth
from fibsem.hub import agent_settings
from fibsem.hub import sessions as hub_sessions
from fibsem.hub.agents import API_KEY, HubAgents
from fibsem.hub.free_agents import AT as AGENT_SERVERS_AT
from fibsem.hub.free_agents import (
    AgentServers,
    FreeAgents,
    default_budget_usd,
    set_limit,
)
from fibsem.hub.history import AGENT, APP, INSTRUMENT, SETUP, HubHistory
from fibsem.hub.hub import TOKENS_DIR, Hub, HubState
from fibsem.hub.relay import AgentPageRelay
from fibsem.hub.simulator import NAME as SIMULATOR
from fibsem.hub.simulator import Simulator
from fibsem.server.agent.library import build_library_router
from fibsem.server.auth import _bearer_token, _matches
from fibsem.server.bodies import refuse_invalid_requests
from fibsem.server.operator_token import load_operator_token
from fibsem.server.pairing import OPERATOR, PairedTokens, Pairing
from fibsem.server.pairing_routes import build_pairing_router
from fibsem.server.web_app import web_app_router

__all__ = ["PAIRED_FILE", "build_hub_app", "hub_auth"]

PAIRED_FILE = TOKENS_DIR / "_paired.json"


def hub_auth(
    token_file: Optional[Path] = None, paired_file: Optional[Path] = None
) -> GatewayAuth:
    """Who is signed in at the hub: the install's operator token, and the
    browsers paired here as the operator."""
    return GatewayAuth(
        token=load_operator_token(token_file),
        pairing=Pairing(PairedTokens(paired_file or PAIRED_FILE), roles=(OPERATOR,)),
    )


# What the hub's page reads of an instrument's gateway, and what it asks of
# one: start and stop its app, and start, resume and end its agent. Nothing
# else of a gateway is reached through the hub.
_READS = re.compile(r"^gateway(/history|/agent(/sessions|/playbooks(/[^/]+)?)?)?$")
_ACTS = (
    "gateway/start",
    "gateway/stop",
    "gateway/agent/start",
    "gateway/agent/resume",
    "gateway/agent/end",
)


def _asked(route: str, body: Dict[str, Any]) -> str:
    """What one of `_ACTS` asks of an instrument's gateway, in words."""
    if route == "gateway/start":
        window = "with its window" if body.get("window", True) else "no window"
        return f"Started {body.get('app') or 'the app'} ({window})"
    if route == "gateway/stop":
        return "Stopped the app"
    if route == "gateway/agent/start":
        return (
            "Started an agent on the microscope PC "
            f"({body.get('playbook') or 'default'}, {body.get('level') or 'watch'})"
        )
    if route == "gateway/agent/resume":
        return (
            f"Resumed a session on the microscope PC ({body.get('level') or 'watch'})"
        )
    return "Ended the agent on the microscope PC"


def _refused(status: int, error_type: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error_type": error_type, "message": message})


def build_hub_app(
    hub: Hub,
    auth: GatewayAuth,
    on_quit: Optional[Callable[[], None]] = None,
    agents: Optional[HubAgents] = None,
    simulator: Optional[Simulator] = None,
    history: Optional[HubHistory] = None,
    on_show: Optional[Callable[[], bool]] = None,
    relay_transport: Any = None,
    free: Optional[FreeAgents] = None,
) -> FastAPI:
    app = FastAPI(title="fibsem hub")
    refuse_invalid_requests(app)
    app.state.auth = auth

    def note(kind: str, what: str, who: str, instrument: Optional[str] = None) -> None:
        if history is not None:
            history.write(kind, what, who, instrument)

    def noted(kind: str, what: str, who: str, instrument: Optional[str], act):
        """*act*'s answer, with a line in the history either way: what was
        done, or that it was refused and why."""
        try:
            answer = act()
        except HTTPException as refused:
            if history is not None:
                detail = refused.detail if isinstance(refused.detail, dict) else {}
                why = detail.get("message") or "refused"
                history.write(kind, f"{what}: refused. {why}", who, instrument, False)
            raise
        note(kind, what, who, instrument)
        return answer

    def operator(request: HTTPConnection) -> str:
        who = auth.operator(_bearer_token(request))
        if who is None:
            raise _refused(401, "unauthorized", "Sign in to the hub first.")
        return who

    def _act(act):
        try:
            return act()
        except GatewayRefused as refusal:
            raise _refused(refusal.status, refusal.error_type, str(refusal)) from None

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "program": "hub", "version": __version__}

    def _state(state: Optional[HubState] = None) -> HubState:
        state = state or hub.state()
        if simulator is not None:
            state.simulator = simulator.state()
        return state

    @app.get("/hub", response_model=HubState)
    def get_hub(who: str = Depends(operator)):
        """Each instrument in the list as it was last read, and whether this
        Hub runs its simulator."""
        return _state()

    @app.post("/hub/test")
    def test_address(body: Dict[str, Any], who: str = Depends(operator)) -> dict:
        """``{ssh | url}``: try reaching an instrument there before adding
        it. ``{ok, steps}``, each step ok, failed with what to do, or not
        reached."""
        return _act(
            lambda: hub.test(
                url=str(body.get("url") or "") or None,
                ssh=str(body.get("ssh") or "") or None,
            )
        )

    @app.post("/hub/instruments", response_model=HubState)
    def add_instrument(body: Dict[str, Any], who: str = Depends(operator)):
        """``{name, ssh | url}``: add an instrument, or change how the one
        with that name is reached: *ssh* is where a tunnel to it connects
        (``user@host``), *url* its gateway's address to use as it is."""
        name = str(body.get("name") or "").strip()
        known = any(i.name == name for i in hub.instruments())
        state = _act(
            lambda: hub.add(
                name,
                str(body.get("url") or "") or None,
                str(body.get("ssh") or "") or None,
            )
        )
        note(
            INSTRUMENT,
            "Changed how it is reached" if known else "Added the instrument",
            who,
            name,
        )
        return _state(state)

    @app.delete("/hub/instruments/{name}", response_model=HubState)
    def remove_instrument(name: str, who: str = Depends(operator)):
        state = _act(lambda: hub.remove(name))
        if simulator is not None and name == SIMULATOR:
            simulator.stop()
        note(INSTRUMENT, "Removed the instrument", who, name)
        return _state(state)

    @app.post("/hub/instruments/{name}/test")
    def test_instrument(name: str, who: str = Depends(operator)) -> dict:
        """Try reaching the instrument now, step by step: ``{ok, steps}``."""
        return _act(lambda: hub.test(name))

    if simulator is not None:

        @app.post("/hub/simulator/start", response_model=HubState)
        def start_simulator(who: str = Depends(operator)):
            """Start a simulated instrument on this hub PC: a gateway here
            with the Demo microscope, listed as "Simulator"."""
            noted(
                INSTRUMENT,
                "Started the simulator",
                who,
                SIMULATOR,
                lambda: _act(simulator.start),
            )
            return _state()

        @app.post("/hub/simulator/stop", response_model=HubState)
        def stop_simulator(who: str = Depends(operator)):
            """Stop the simulator. It stays in the list until removed."""
            simulator.stop()
            hub.check()
            note(INSTRUMENT, "Stopped the simulator", who, SIMULATOR)
            return _state()

    @app.post("/hub/instruments/{name}/pair", response_model=HubState)
    def pair_instrument(name: str, body: Dict[str, Any], who: str = Depends(operator)):
        """``{code}``: pair this hub with the instrument's gateway, using the
        code ``fibsem-gateway pair --hub`` shows there."""
        code = str(body.get("code") or "")
        if not code:
            raise _refused(422, "no_code", "Give the code its gateway shows.")
        return _state(
            noted(
                INSTRUMENT,
                "Paired the Hub with its gateway",
                who,
                name,
                lambda: _act(lambda: hub.pair(name, code)),
            )
        )

    def at_this_pc(request: HTTPConnection) -> bool:
        """Whether the caller holds this install's own token: the hub's page
        as ``agent-hub open`` opens it, never a paired browser."""
        return _matches(_bearer_token(request), auth.token)

    def this_pc_only(request: HTTPConnection) -> None:
        operator(request)
        if not at_this_pc(request):
            raise _refused(
                403,
                "at_this_pc_only",
                "The agent's setup is changed only at this PC, on the Hub's own page.",
            )

    # The playbooks, skills and subagents an agent on this PC runs with: the
    # same routes the Agent page's Knowledge window reads on a microscope PC.
    app.include_router(
        build_library_router(read=[Depends(operator)], write=[Depends(this_pc_only)])
    )

    def _key(name: str, request: HTTPConnection) -> None:
        if name in keystore.KEYS and name != API_KEY:
            raise _refused(
                404,
                "no_key",
                f"The hub keeps only {API_KEY} and its connectors' tokens.",
            )
        if not at_this_pc(request):
            raise _refused(
                403,
                "at_this_pc_only",
                "Keys are set only at this PC, on the Hub's own page.",
            )

    @app.get("/hub/secrets")
    def get_secrets(request: Request, who: str = Depends(operator)) -> list:
        """The keys the agents on this hub PC use (the API key, and each
        connector's token): whether each is set and where from, and whether
        this caller may change it. Never a key."""
        here = at_this_pc(request)
        return [
            {**row, "can_set": here}
            for row in gateway_settings.key_rows()
            if row["name"] == API_KEY or row["name"] not in keystore.KEYS
        ]

    def _settings(act):
        try:
            return act()
        except gateway_settings.KeyRefused as refused:
            raise _refused(refused.status, refused.error_type, str(refused)) from None

    @app.get("/hub/connectors")
    def get_connectors(request: Request, who: str = Depends(operator)) -> dict:
        """This hub PC's connectors (``mcp.yaml``), the ones an agent started
        here may use, as a gateway shows its own."""
        return {**gateway_settings.connectors(), "at_this_pc": at_this_pc(request)}

    @app.put("/hub/connectors/{name}", dependencies=[Depends(this_pc_only)])
    def put_connector(name: str, body: Dict[str, Any], who: str = Depends(operator)):
        body = body or {}
        answer = _settings(
            lambda: gateway_settings.save_connector(
                name, body.get("entry"), True, body.get("confirm")
            )
        )
        note(SETUP, f"Saved the connector {name}", who)
        return answer

    @app.delete("/hub/connectors/{name}", dependencies=[Depends(this_pc_only)])
    def delete_connector(name: str, who: str = Depends(operator)) -> dict:
        _settings(lambda: gateway_settings.remove_connector(name, True))
        note(SETUP, f"Removed the connector {name}", who)
        return {"removed": name}

    @app.post("/hub/connectors/{name}/test")
    def test_connector(
        name: str, request: Request, who: str = Depends(operator)
    ) -> dict:
        """``{tools}``: the tools it offers, asked as the agent asks."""
        return _settings(
            lambda: gateway_settings.test_connector(name, at_this_pc(request))
        )

    @app.put("/hub/secrets/{name}")
    def put_secret(
        name: str, body: Dict[str, Any], request: Request, who: str = Depends(operator)
    ):
        """``{value}``: store the key in this PC's key store. Write-only."""
        _key(name, request)
        try:
            row = gateway_settings.set_key(name, (body or {}).get("value"))
        except gateway_settings.KeyRefused as refused:
            raise _refused(refused.status, refused.error_type, str(refused)) from None
        # The key's name, never its value.
        note(SETUP, f"Set the key {name}", who)
        return {**row, "can_set": True}

    @app.delete("/hub/secrets/{name}")
    def delete_secret(name: str, request: Request, who: str = Depends(operator)):
        """Remove the stored key: this PC's agents stop using it."""
        _key(name, request)
        try:
            row = gateway_settings.remove_key(name)
        except gateway_settings.KeyRefused as refused:
            raise _refused(refused.status, refused.error_type, str(refused)) from None
        note(SETUP, f"Removed the key {name}", who)
        return {**row, "can_set": True}

    @app.get("/hub/history")
    def get_history(limit: int = 100, who: str = Depends(operator)) -> dict:
        """``{events, more}``: the latest *limit* lines of the Hub's history,
        newest first (`fibsem.hub.history`)."""
        if history is None:
            return {"events": [], "more": False}
        return history.read(limit)

    @app.get("/hub/agent-settings")
    def get_agent_settings(request: Request, who: str = Depends(operator)) -> dict:
        """What an agent started on this hub PC may be given beyond its own
        folder: the data folders, the allowed sites, the usable connectors,
        and whether the start form offers them (`fibsem.hub.agent_settings`)."""
        return {**agent_settings.read(), "at_this_pc": at_this_pc(request)}

    @app.put("/hub/agent-settings", dependencies=[Depends(this_pc_only)])
    def put_agent_settings(body: Dict[str, Any], who: str = Depends(operator)):
        """``{data_folders, allowed_sites, offered}``: changed only at this
        PC. A folder that isn't one, and a site that is a local address, are
        refused."""
        answer = noted(
            SETUP,
            "Changed what agents on this hub PC may be given ("
            + ", ".join(sorted(body or {}))
            + ")",
            who,
            None,
            lambda: _act(lambda: agent_settings.save(body)),
        )
        return {**answer, "at_this_pc": True}

    if history is not None:

        @app.middleware("http")
        async def library_saves(request: Request, call_next):
            """A playbook, skill or subagent saved or removed here, in the
            history: its path in the library, once the save went through."""
            response = await call_next(request)
            path = request.url.path
            if (
                path.startswith("/app/library/")
                and request.method in ("PUT", "POST", "DELETE", "PATCH")
                and response.status_code < 400
            ):
                verb = "Removed" if request.method == "DELETE" else "Saved"
                history.write(
                    SETUP, f"{verb} {path[len('/app/library/') :]}", "operator"
                )
            return response

    if agents is not None:

        @app.get("/hub/agents")
        def hub_agents(who: str = Depends(operator)) -> Dict[str, Any]:
            """The agents this hub PC runs, by instrument: the running one,
            else the last that ended here and why."""
            return agents.state()

        @app.post("/hub/instruments/{name}/hub-agent/start")
        def start_hub_agent(
            name: str, body: Dict[str, Any], who: str = Depends(operator)
        ):
            """``{playbook, goal, level, budget_usd, replace}``: run an agent
            for the instrument on this hub PC. Its gateway gives the slot, or
            refuses in its own words."""
            try:
                budget = float(body["budget_usd"])
            except (KeyError, TypeError, ValueError):
                budget = None
            return _act(
                lambda: agents.start(
                    name,
                    str(body.get("playbook") or ""),
                    str(body.get("goal") or "") or None,
                    str(body.get("level") or "watch"),
                    who,
                    budget,
                    bool(body.get("replace")),
                    agent_settings.launch(body.get("settings")),
                )
            )

        @app.post("/hub/instruments/{name}/hub-agent/resume")
        def resume_hub_agent(
            name: str, body: Dict[str, Any], who: str = Depends(operator)
        ):
            """``{session_id, level, budget_usd, replace}``: carry on a
            session that ended on this hub PC. Its gateway gives the slot as
            for a start, or refuses in its own words."""
            try:
                budget = float(body["budget_usd"])
            except (KeyError, TypeError, ValueError):
                budget = None
            return _act(
                lambda: agents.resume(
                    name,
                    str(body.get("session_id") or ""),
                    str(body.get("level") or "watch"),
                    who,
                    budget,
                    bool(body.get("replace")),
                    agent_settings.launch,
                )
            )

        @app.post("/hub/instruments/{name}/hub-agent/end")
        def end_hub_agent(name: str, who: str = Depends(operator)):
            """End the instrument's agent on this hub PC; its slot is given
            back to its gateway."""
            return _act(lambda: agents.end(name, who))

        @app.get("/hub/sessions")
        def get_sessions(limit: int = 50, who: str = Depends(operator)) -> dict:
            """``{sessions, totals, unread, more}``: the agent sessions on
            this hub PC and on every paired instrument, newest first, with
            what they spent by instrument and by day
            (`fibsem.hub.sessions`)."""
            return hub_sessions.list_sessions(
                hub, agents.root, limit, free.root if free is not None else None
            )

    if free is not None:

        def _budget(body: Dict[str, Any]) -> Optional[float]:
            try:
                return float(body["budget_usd"])
            except (KeyError, TypeError, ValueError):
                return None

        budget: Dict[str, Any] = {}

        def default_budget() -> Optional[float]:
            """What the plain playbook lets a run spend when the form names
            no budget; read once."""
            if "usd" not in budget:
                try:
                    budget["usd"] = default_budget_usd()
                except Exception:  # noqa: BLE001 - the form just shows none
                    budget["usd"] = None
            return budget["usd"]

        @app.get("/hub/free-agents")
        def free_agents(request: Request, who: str = Depends(operator)) -> dict:
            """``{limit, agents, at_this_pc}``: the agents with no instrument
            this hub PC runs, then those that ended while this Hub ran, and
            how many may run at once."""
            return {
                **free.state(),
                "at_this_pc": at_this_pc(request),
                "default_budget_usd": default_budget(),
            }

        @app.post("/hub/free-agents/start")
        def start_free_agent(body: Dict[str, Any], who: str = Depends(operator)):
            """``{goal, budget_usd}``: run an agent with no instrument on
            this hub PC."""
            body = body or {}
            return _act(
                lambda: free.start(
                    str(body.get("goal") or ""),
                    who,
                    _budget(body),
                    agent_settings.launch(body.get("settings")),
                )
            )

        @app.post("/hub/free-agents/resume")
        def resume_free_agent(body: Dict[str, Any], who: str = Depends(operator)):
            """``{session_id, budget_usd}``: carry on a session with no
            instrument that ended on this hub PC."""
            body = body or {}
            return _act(
                lambda: free.resume(
                    str(body.get("session_id") or ""),
                    who,
                    _budget(body),
                    agent_settings.launch,
                )
            )

        @app.post("/hub/free-agents/{session_id}/end")
        def end_free_agent(session_id: str, who: str = Depends(operator)):
            """End an agent with no instrument."""
            return _act(lambda: free.end(session_id, who))

        @app.put("/hub/free-agents/limit", dependencies=[Depends(this_pc_only)])
        def put_free_limit(body: Dict[str, Any], who: str = Depends(operator)):
            """``{limit}``: how many agents with no instrument run here at
            once. Changed only at this PC."""
            answer = _act(lambda: set_limit((body or {}).get("limit")))
            note(SETUP, f"Set the limit of agents with no instrument to {answer}", who)
            return {"limit": answer}

        # Each such session's own Agent Server: its agent and its Agent page
        # reach it here, and it checks their tokens itself.
        app.mount(AGENT_SERVERS_AT, AgentServers(free))

    # An instrument's Agent page, served here: its calls go on to that
    # instrument's gateway, which decides what a Hub may ask.
    relay = AgentPageRelay(
        hub, relay_transport, lambda name, did, who: note(AGENT, did, who, name)
    )

    def passes(prefix: str, exact: bool):
        if exact:

            async def pass_on_exact(
                name: str, request: Request, who: str = Depends(operator)
            ):
                return await relay.forward(name, prefix, request, who)

            return pass_on_exact

        async def pass_on(
            name: str, rest: str, request: Request, who: str = Depends(operator)
        ):
            return await relay.forward(name, f"{prefix}/{rest}", request, who)

        return pass_on

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]
    for prefix in PASSED:
        app.add_api_route(
            f"/hub/instruments/{{name}}/{prefix}/{{rest:path}}",
            passes(prefix, exact=False),
            methods=methods,
            include_in_schema=False,
        )
    for exact in PASSED + ROOT_PASSED:
        app.add_api_route(
            f"/hub/instruments/{{name}}/{exact}",
            passes(exact, exact=True),
            methods=methods,
            include_in_schema=False,
        )

    @app.get("/hub/instruments/{name}/{route:path}")
    def read_instrument(
        name: str, route: str, request: Request, who: str = Depends(operator)
    ):
        """An instrument's gateway's own answer: its state (``gateway``), its
        agent and sessions (``gateway/agent``, ``gateway/agent/sessions``),
        its playbooks (``gateway/agent/playbooks``) and its starts and stops
        (``gateway/history``)."""
        if not _READS.match(route):
            raise _refused(404, "not_found", "The hub reads no such thing.")
        asked = "/" + quote(route)
        if request.url.query:
            asked += f"?{request.url.query}"
        return _act(lambda: hub.ask(name, "GET", asked))

    @app.post("/hub/instruments/{name}/{route:path}")
    def ask_instrument(
        name: str,
        route: str,
        body: Optional[Dict[str, Any]] = None,
        who: str = Depends(operator),
    ):
        """Start or stop an instrument's app (its gateway's ``/gateway/start``
        and ``stop``), and start, resume or end its agent (``/gateway/agent/
        start``, ``resume`` and ``end``). Its gateway decides whether this
        hub may, and never lets one force a stop or close a window."""
        if route not in _ACTS:
            raise _refused(404, "not_found", "The hub asks no such thing.")
        body = body or {}
        kind = AGENT if "/agent/" in route else APP
        return noted(
            kind,
            _asked(route, body),
            who,
            name,
            lambda: _act(lambda: hub.ask(name, "POST", "/" + route, body)),
        )

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/ui/hub")

    if on_quit is not None:

        @app.post("/hub/quit")
        def quit_hub(who: str = Depends(operator)) -> dict:
            """Stop the hub: its tunnels close, and no instrument notices."""
            on_quit()
            return {"stopping": True}

    @app.post("/hub/show")
    def show_hub(who: str = Depends(operator)) -> dict:
        """Bring the Hub's window to the front, as the desktop shortcut does;
        ``shown`` is False when the Hub runs without one."""
        return {"shown": bool(on_show and on_show())}

    app.include_router(build_pairing_router(auth))
    app.include_router(web_app_router())
    return app
