"""Pairing's routes: a computer pairs with the code (``/pair``), an agent on
this PC asks to connect with the discovery file's token (``/connect``), and the
operator opens pairing, sees the paired computers and unpairs one
(``/pairing``). See `fibsem.server.pairing` and `fibsem.server.connections`."""

from ipaddress import ip_address
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request

from fibsem.server.auth import AuthConfig, client_from_user_agent, require_operator
from fibsem.server.connections import ASK_WAIT_S
from fibsem.server.pairing import SWITCH_MODES, Pairing, PairingRefused

__all__ = ["build_pairing_router"]

# What a paired computer may be granted beyond its role.
GRANTS = (SWITCH_MODES,)


def _refused(status_code: int, error_type: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error_type": error_type, "message": message},
    )


def _same_origin(request: Request) -> None:
    """Refuse a page from another site: pairing answers a token, and a browser
    sends no CORS check for a simple POST."""
    origin = request.headers.get("origin")
    if origin is None:
        return
    if urlsplit(origin).netloc != request.headers.get("host"):
        raise _refused(403, "browser_refused", "Pairing refuses another site's page.")


def _loopback(request: Request) -> bool:
    """Whether the request came from this PC's loopback address: the peer's,
    never what the body says."""
    try:
        return ip_address(request.client.host).is_loopback
    except (AttributeError, ValueError):
        return False


def _pairing(auth: AuthConfig) -> Pairing:
    if auth.pairing is None:
        raise _refused(404, "pairing_off", "This server doesn't pair computers.")
    return auth.pairing


def _grants(body: Dict[str, Any]) -> Tuple[str, ...]:
    grants = body.get("grants", [])
    if not isinstance(grants, list) or not all(g in GRANTS for g in grants):
        raise _refused(422, "bad_grants", f"Grants are from {list(GRANTS)}.")
    return tuple(dict.fromkeys(grants))


def build_pairing_router(auth: AuthConfig) -> APIRouter:
    router = APIRouter()

    @router.get("/pair")
    def can_pair():
        """Unauthenticated: whether pairing is open, and whether a browser may
        ask for a code shown where the server runs (the headless server)."""
        pairing = _pairing(auth)
        return {
            "open": pairing.state()["open"],
            "request": pairing.announce is not None,
        }

    @router.post("/pair", dependencies=[Depends(_same_origin)])
    def pair(body: Dict[str, Any]):
        """Unauthenticated: pair with the code, answering this computer's
        token, with the role pairing was opened for."""
        try:
            computer, token = _pairing(auth).pair(
                str(body.get("code", "")),
                str(body.get("name", "")),
                str(body.get("browser", "")),
            )
        except PairingRefused as refusal:
            raise _refused(403, refusal.error_type, str(refusal)) from None
        return {
            "token": token,
            "id": computer.id,
            "name": computer.name,
            "role": computer.role,
            "grants": list(computer.grants),
        }

    @router.post("/pair/request", dependencies=[Depends(_same_origin)])
    def request_code():
        """Unauthenticated: ask for a code, shown only where the server runs."""
        pairing = _pairing(auth)
        if pairing.announce is None:
            raise _refused(
                404,
                "pairing_off",
                "Open pairing at the microscope PC: Tools → Host → Pair a computer.",
            )
        return {"opened": pairing.request()}

    @router.post("/connect")
    def connect(request: Request, body: Optional[Dict[str, Any]] = None):
        """With the discovery file's connect token: ask the operator to allow
        this agent, waiting up to ``wait_s`` (at most 25 s) for an answer.
        ``{"state": "allowed", "token", "name"}``, ``{"state": "denied",
        "message", "retry_in_s"}``, or ``{"state": "waiting", "id"}`` to ask
        again."""
        header = request.headers.get("Authorization", "")
        if not auth.is_connect(header.removeprefix("Bearer ")):
            raise _refused(
                401, "unauthorized", "Ask with the token in the discovery file."
            )
        if auth.connections is None:
            raise _refused(404, "connect_off", "This server doesn't take asks.")
        body = body or {}
        try:
            wait_s = min(ASK_WAIT_S, max(0.0, float(body.get("wait_s", ASK_WAIT_S))))
        except (TypeError, ValueError):
            wait_s = ASK_WAIT_S
        client = client_from_user_agent(request.headers.get("User-Agent"))
        return auth.connections.ask(
            client.replace("/", " ") if client else str(body.get("client", "")),
            str(body.get("origin", "")),
            str(body.get("user", "")),
            wait_s=wait_s,
            loopback=_loopback(request),
        )

    operator = APIRouter(prefix="/pairing", dependencies=[Depends(require_operator())])

    def _listing(pairing: Pairing) -> dict:
        connections = auth.connections
        return {
            **pairing.state(),
            "computers": [c.to_dict() for c in pairing.tokens.computers()],
            "always_allow_local": bool(connections and connections.always_allow),
        }

    @operator.get("")
    def get_pairing():
        """Pairing as the dialog shows it: open, its code and time left, and
        the paired computers."""
        return _listing(_pairing(auth))

    @operator.post("")
    def open_pairing(body: Optional[Dict[str, Any]] = None):
        """Open pairing with a new code: for the operator's app, or with
        ``{"role": "agent"}`` for an agent (``"hub"`` on a gateway, with
        ``"grants"``)."""
        pairing = _pairing(auth)
        body = body or {}
        role = body.get("role", "operator")
        if role not in pairing.roles:
            raise _refused(
                422, "bad_role", f"Pairing is for one of {list(pairing.roles)}."
            )
        pairing.open(role, _grants(body))
        return _listing(pairing)

    @operator.delete("")
    def close_pairing():
        pairing = _pairing(auth)
        pairing.close()
        return _listing(pairing)

    @operator.put("/always_allow_local")
    def always_allow_local(body: Dict[str, Any]):
        """``{"on": true}``: allow an agent logged in to this PC that asks to
        connect without the operator's click; saved for the next launch."""
        if auth.connections is None:
            raise _refused(404, "connect_off", "This server doesn't take asks.")
        auth.connections.set_always_allow(bool(body.get("on")))
        return _listing(_pairing(auth))

    @operator.patch("/computers/{computer_id}")
    def set_grants(computer_id: str, body: Dict[str, Any]):
        """Replace one paired computer's grants: ``{"grants": [...]}``."""
        pairing = _pairing(auth)
        if pairing.tokens.set_grants(computer_id, _grants(body)) is None:
            raise _refused(404, "not_found", "No paired computer has that id.")
        return _listing(pairing)

    @operator.delete("/computers/{computer_id}")
    def unpair(computer_id: str):
        """Unpair one computer: its token stops working at once."""
        pairing = _pairing(auth)
        if not pairing.tokens.remove(computer_id):
            raise _refused(404, "not_found", "No paired computer has that id.")
        return _listing(pairing)

    router.include_router(operator)
    return router
