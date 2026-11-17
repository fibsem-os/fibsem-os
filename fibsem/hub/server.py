"""The hub's HTTP routes, on this PC only (127.0.0.1).

``GET /hub`` lists the instruments as they were last read, for anyone signed
in. The operator alone changes the list or pairs with an instrument: the
install's operator token, or a browser paired as the operator with a code from
``fibsem-hub pair``. The hub keeps its own paired computers
(``~/.fibsem/hub/paired.json``), apart from a gateway's on the same PC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from starlette.requests import HTTPConnection

from fibsem import __version__
from fibsem.gateway.gateway import GatewayRefused
from fibsem.gateway.server import GatewayAuth
from fibsem.hub.hub import TOKENS_DIR, Hub, HubState
from fibsem.server.auth import _bearer_token
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


def _refused(status: int, error_type: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error_type": error_type, "message": message})


def build_hub_app(
    hub: Hub,
    auth: GatewayAuth,
    on_quit: Optional[Callable[[], None]] = None,
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

    if on_quit is not None:

        @app.post("/hub/quit")
        def quit_hub(who: str = Depends(operator)) -> dict:
            """Stop the hub: its tunnels close, and no instrument notices."""
            on_quit()
            return {"stopping": True}

    app.include_router(build_pairing_router(auth))
    app.include_router(web_app_router())
    return app
