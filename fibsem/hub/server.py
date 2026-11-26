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

from fibsem import __version__
from fibsem.gateway import settings as gateway_settings
from fibsem.gateway.gateway import GatewayRefused
from fibsem.gateway.server import GatewayAuth
from fibsem.hub.agents import API_KEY, HubAgents
from fibsem.hub.hub import TOKENS_DIR, Hub, HubState
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


# What the hub's page reads of an instrument's gateway, and the three things
# it asks of one: start, resume and end its agent. Nothing else of a gateway
# is reached through the hub.
_READS = re.compile(r"^gateway(/agent(/sessions|/playbooks(/[^/]+)?)?)?$")
_ACTS = ("gateway/agent/start", "gateway/agent/resume", "gateway/agent/end")


def _refused(status: int, error_type: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error_type": error_type, "message": message})


def build_hub_app(
    hub: Hub,
    auth: GatewayAuth,
    on_quit: Optional[Callable[[], None]] = None,
    agents: Optional[HubAgents] = None,
) -> FastAPI:
    app = FastAPI(title="fibsem hub")
    refuse_invalid_requests(app)
    app.state.auth = auth

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

    @app.get("/hub", response_model=HubState)
    def get_hub(who: str = Depends(operator)):
        """Each instrument in the list as it was last read."""
        return hub.state()

    @app.post("/hub/instruments", response_model=HubState)
    def add_instrument(body: Dict[str, Any], who: str = Depends(operator)):
        """``{name, ssh | url}``: add an instrument, or change how the one
        with that name is reached: *ssh* is where a tunnel to it connects
        (``user@host``), *url* its gateway's address to use as it is."""
        return _act(
            lambda: hub.add(
                str(body.get("name") or ""),
                str(body.get("url") or "") or None,
                str(body.get("ssh") or "") or None,
            )
        )

    @app.delete("/hub/instruments/{name}", response_model=HubState)
    def remove_instrument(name: str, who: str = Depends(operator)):
        return _act(lambda: hub.remove(name))

    @app.post("/hub/instruments/{name}/pair", response_model=HubState)
    def pair_instrument(name: str, body: Dict[str, Any], who: str = Depends(operator)):
        """``{code}``: pair this hub with the instrument's gateway, using the
        code ``fibsem-gateway pair --hub`` shows there."""
        code = str(body.get("code") or "")
        if not code:
            raise _refused(422, "no_code", "Give the code its gateway shows.")
        return _act(lambda: hub.pair(name, code))

    def at_this_pc(request: HTTPConnection) -> bool:
        """Whether the caller holds this install's own token: the hub's page
        as ``agent-hub open`` opens it, never a paired browser."""
        return _matches(_bearer_token(request), auth.token)

    def _key(name: str, request: HTTPConnection) -> None:
        if name != API_KEY:
            raise _refused(404, "no_key", f"The hub keeps only {API_KEY}.")
        if not at_this_pc(request):
            raise _refused(
                403,
                "at_this_pc_only",
                "Keys are set only at this PC, on the Hub's own page.",
            )

    @app.get("/hub/secrets")
    def get_secrets(request: Request, who: str = Depends(operator)) -> list:
        """The key the agents on this hub PC use: whether it is set and where
        from, and whether this caller may change it. Never the key."""
        here = at_this_pc(request)
        return [
            {**row, "can_set": here}
            for row in gateway_settings.key_rows()
            if row["name"] == API_KEY
        ]

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
        return {**row, "can_set": True}

    @app.delete("/hub/secrets/{name}")
    def delete_secret(name: str, request: Request, who: str = Depends(operator)):
        """Remove the stored key: this PC's agents stop using it."""
        _key(name, request)
        try:
            row = gateway_settings.remove_key(name)
        except gateway_settings.KeyRefused as refused:
            raise _refused(refused.status, refused.error_type, str(refused)) from None
        return {**row, "can_set": True}

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
            budget = body.get("budget_usd")
            return _act(
                lambda: agents.start(
                    name,
                    str(body.get("playbook") or ""),
                    str(body.get("goal") or "") or None,
                    str(body.get("level") or "watch"),
                    who,
                    float(budget) if isinstance(budget, (int, float)) else None,
                    bool(body.get("replace")),
                )
            )

        @app.post("/hub/instruments/{name}/hub-agent/end")
        def end_hub_agent(name: str, who: str = Depends(operator)):
            """End the instrument's agent on this hub PC; its slot is given
            back to its gateway."""
            return _act(lambda: agents.end(name, who))

    @app.get("/hub/instruments/{name}/{route:path}")
    def read_instrument(
        name: str, route: str, request: Request, who: str = Depends(operator)
    ):
        """An instrument's gateway's own answer: its state (``gateway``), its
        agent and sessions (``gateway/agent``, ``gateway/agent/sessions``) and
        its playbooks (``gateway/agent/playbooks``)."""
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
        """Start, resume or end an instrument's agent, as its gateway's
        ``/gateway/agent/start``, ``resume`` and ``end``. The agent runs on
        the instrument's PC, and its gateway decides whether this hub may."""
        if route not in _ACTS:
            raise _refused(404, "not_found", "The hub asks no such thing.")
        return _act(lambda: hub.ask(name, "POST", "/" + route, body or {}))

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/ui/hub")

    if on_quit is not None:

        @app.post("/hub/quit")
        def quit_hub(who: str = Depends(operator)) -> dict:
            """Stop the hub: its tunnels close, and no instrument notices."""
            on_quit()
            return {"stopping": True}

    app.include_router(build_pairing_router(auth))
    app.include_router(web_app_router())
    return app
