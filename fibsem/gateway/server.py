"""The gateway's routes, on its own port, localhost only.

``GET /gateway`` and ``/gateway/history`` for anyone paired; ``POST
/gateway/start`` and ``/gateway/stop`` for the operator, or a hub the operator
granted "may start and stop apps" (`SWITCH_MODES`); forcing past a running
workflow, and ``/gateway/claim`` (a desktop launch on this PC), for the
operator only. ``/gateway/microscope/...`` connects and disconnects the
running host's microscope, ``/gateway/app/agent-page`` links the operator to
its Agent page, and ``/gateway/agent`` reads, starts and ends this
PC's agent (`fibsem.gateway.agent`), with the same rules as starting the app.
Pairing is the App Server's (`fibsem.server.pairing_routes`), with the hub
role, and the page is the web app's ``/ui/gateway``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.requests import HTTPConnection
from fastapi.responses import RedirectResponse

from fibsem.ai.agent.profiles import DEFAULT_PROFILE
from fibsem.gateway.agent import AgentState, GatewayAgent
from fibsem.gateway.gateway import (
    Gateway,
    GatewayRefused,
    GatewayState,
    History,
)
from fibsem.gateway.host import HostMicroscope
from fibsem.server.auth import _bearer_token, _matches
from fibsem.server.operator_token import load_operator_token
from fibsem.server.pairing import (
    HUB,
    OPERATOR,
    SWITCH_MODES,
    PairedComputer,
    PairedTokens,
    Pairing,
)
from fibsem.server.pairing_routes import build_pairing_router
from fibsem.server.web_app import web_app_router

__all__ = ["GatewayAuth", "build_gateway_app"]


@dataclass
class GatewayAuth:
    """The install's operator token and this PC's paired computers, the App
    Server's file (hubs are listed with its browsers and agents). Shaped
    as the App Server's `AuthConfig` where the pairing routes look: no agents,
    no connect asks."""

    token: str
    pairing: Pairing
    dashboard_token: Optional[str] = None
    connections: Any = None

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

    def is_agent(self, token: Optional[str]) -> bool:
        return False

    def is_connect(self, token: Optional[str]) -> bool:
        return False

    @classmethod
    def from_disk(
        cls, token_file: Optional[Path] = None, paired_file: Optional[Path] = None
    ) -> "GatewayAuth":
        return cls(
            token=load_operator_token(token_file),
            pairing=Pairing(
                PairedTokens(paired_file),
                roles=(OPERATOR, HUB),
            ),
        )


@dataclass
class Caller:
    who: str
    operator: bool
    may_switch: bool


def _refused(status: int, error_type: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error_type": error_type, "message": message})


def build_gateway_app(
    gateway: Gateway,
    auth: GatewayAuth,
    agent: Optional[GatewayAgent] = None,
    microscope: Optional[HostMicroscope] = None,
) -> FastAPI:
    app = FastAPI(title="fibsem gateway")
    app.state.auth = auth

    def caller(request: HTTPConnection) -> Caller:
        token = _bearer_token(request)
        who = auth.operator(token)
        if who is not None:
            return Caller(who=who, operator=True, may_switch=True)
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
        return {"status": "ok"}

    @app.get("/gateway", response_model=GatewayState)
    def get_gateway(found: Caller = Depends(caller)):
        """Off or running, the app and its process, a holder the gateway
        didn't start, and the last exit."""
        return gateway.state()

    @app.get("/gateway/history", response_model=History)
    def get_history(found: Caller = Depends(caller)):
        """The latest starts and stops first, each with who made it and from
        where."""
        return gateway.history()

    @app.post("/gateway/start", response_model=GatewayState)
    def start(body: Dict[str, Any], found: Caller = Depends(switcher)):
        """``{app, window, experiment | last, force}``: stop the running app
        and start this one, with its window at this PC (the default) or
        without."""
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
            ),
        )

    @app.post("/gateway/stop", response_model=GatewayState)
    def stop(body: Optional[Dict[str, Any]] = None, found: Caller = Depends(switcher)):
        """Back to off: ``{force}`` stops past a running workflow (operator)."""
        body = body or {}
        return _act(found, body, lambda force, via: gateway.stop(found.who, via, force))

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

        @app.get("/gateway/agent/profiles")
        def get_profiles(found: Caller = Depends(caller)) -> List[dict]:
            """The profiles on this PC, as ``fibsem-agent --list-profiles``."""
            return _act(found, {}, lambda force, via: agent.profiles())

        @app.get("/gateway/agent/profiles/{name}")
        def get_profile(name: str, found: Caller = Depends(caller)) -> dict:
            """What *name* runs with, as ``fibsem-agent --show-config``."""
            return _act(found, {}, lambda force, via: agent.configuration(name))

        @app.post("/gateway/agent/start", response_model=AgentState)
        def start_agent(body: Dict[str, Any], found: Caller = Depends(switcher)):
            """``{profile, goal, level, budget_usd, replace}``: pair the agent,
            set the level, start it. ``replace`` disconnects an agent paired
            some other way."""
            budget = body.get("budget_usd")
            try:
                budget = float(budget) if budget not in (None, "") else None
            except (TypeError, ValueError):
                raise _refused(422, "bad_budget", "The budget is a number of dollars.")
            if budget is not None and budget <= 0:
                raise _refused(422, "bad_budget", "The budget is above $0.")
            profile = str(body.get("profile") or DEFAULT_PROFILE)
            return _act(
                found,
                body,
                lambda force, via: agent.start(
                    profile,
                    str(body.get("goal") or "").strip() or None,
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
                    found.who, via, bool(body.get("stop_app", False)), force
                ),
            )

    app.include_router(build_pairing_router(auth))
    app.include_router(web_app_router())
    return app
