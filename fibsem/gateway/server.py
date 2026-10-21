"""The gateway's routes, on its own port, localhost only.

``GET /gateway`` and ``/gateway/history`` for anyone paired; ``POST
/gateway/mode`` and ``/gateway/stop`` for the operator, or a hub the operator
granted "may start and stop apps" (`SWITCH_MODES`); forcing past a running
workflow, and ``/gateway/claim`` (a desktop launch on this PC), for the
operator only. Pairing is the App Server's (`fibsem.server.pairing_routes`),
with the hub role, and the page is the web app's ``/ui/gateway``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.requests import HTTPConnection
from fastapi.responses import RedirectResponse

from fibsem.gateway.gateway import (
    Gateway,
    GatewayRefused,
    GatewayState,
    History,
)
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


def build_gateway_app(gateway: Gateway, auth: GatewayAuth) -> FastAPI:
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
            raise _refused(403, "operator_only", "Only the operator forces a switch.")
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
        """The mode, its app and process, a holder the gateway didn't start,
        and the last exit."""
        return gateway.state()

    @app.get("/gateway/history", response_model=History)
    def get_history(found: Caller = Depends(caller)):
        """The latest switches first, each with who made it and from where."""
        return gateway.history()

    @app.post("/gateway/mode", response_model=GatewayState)
    def set_mode(body: Dict[str, Any], found: Caller = Depends(switcher)):
        """``{mode, app, experiment | last, force}``: stop the current mode and
        start this one."""
        return _act(
            found,
            body,
            lambda force, via: gateway.switch(
                str(body.get("mode", "")),
                who=found.who,
                via=via,
                app=body.get("app"),
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
        """``{pid, app}``: a desktop launch on this PC takes window mode."""
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

    app.include_router(build_pairing_router(auth))
    app.include_router(web_app_router())
    return app
