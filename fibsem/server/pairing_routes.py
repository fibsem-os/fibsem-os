"""Pairing's routes: a computer pairs with the code (``/pair``), and the
operator opens pairing, sees the paired computers and unpairs one
(``/pairing``). See `fibsem.server.pairing`."""

from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request

from fibsem.server.auth import AuthConfig, require_operator
from fibsem.server.pairing import ROLES, Pairing, PairingRefused

__all__ = ["build_pairing_router"]


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


def _pairing(auth: AuthConfig) -> Pairing:
    if auth.pairing is None:
        raise _refused(404, "pairing_off", "This server doesn't pair computers.")
    return auth.pairing


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
                str(body.get("code", "")), str(body.get("name", ""))
            )
        except PairingRefused as refusal:
            raise _refused(403, refusal.error_type, str(refusal)) from None
        return {
            "token": token,
            "id": computer.id,
            "name": computer.name,
            "role": computer.role,
        }

    @router.post("/pair/request", dependencies=[Depends(_same_origin)])
    def request_code():
        """Unauthenticated: ask for a code, shown only where the server runs."""
        pairing = _pairing(auth)
        if pairing.announce is None:
            raise _refused(
                404,
                "pairing_off",
                "Open pairing at the microscope PC: Tools → App Server → Pair a computer.",
            )
        return {"opened": pairing.request()}

    operator = APIRouter(prefix="/pairing", dependencies=[Depends(require_operator())])

    def _listing(pairing: Pairing) -> dict:
        return {
            **pairing.state(),
            "computers": [c.to_dict() for c in pairing.tokens.computers()],
        }

    @operator.get("")
    def get_pairing():
        """Pairing as the dialog shows it: open, its code and time left, and
        the paired computers."""
        return _listing(_pairing(auth))

    @operator.post("")
    def open_pairing(body: Optional[Dict[str, Any]] = None):
        """Open pairing with a new code: for the operator's app, or with
        ``{"role": "agent"}`` for an agent."""
        pairing = _pairing(auth)
        role = (body or {}).get("role", "operator")
        if role not in ROLES:
            raise _refused(422, "bad_role", f"Pairing is for one of {list(ROLES)}.")
        pairing.open(role)
        return _listing(pairing)

    @operator.delete("")
    def close_pairing():
        pairing = _pairing(auth)
        pairing.close()
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
