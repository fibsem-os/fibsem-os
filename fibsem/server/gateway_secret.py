"""The secret a server the gateway starts is handed, so nothing on the PC skips
the gateway.

The gateway makes a new secret at each start of a server (the App Server, the
Agent Server) and hands it through the server's environment (`ENV`), never on
disk. The server takes it out of its environment at once, so the programs it
starts in turn (the agent, a code sandbox) never see it, and while this PC's
gateway runs it refuses every call that doesn't carry it (`HEADER`), whoever
makes it: the operator's token or an agent's are still checked as ever, after.
Its /health and the stops are never refused, and a websocket isn't either
until the gateway passes them on.

A server started some other way (a desktop window opened by hand, a test, a
live eval) has no secret from the gateway: the App Server's own one in its
discovery file then applies to agents' calls (`fibsem.server.auth`).
"""

from __future__ import annotations

import hmac
import os
import re
from typing import Optional

from fibsem.server.discovery import read_gateway_file

__all__ = ["ENV", "HEADER", "GatewayOnly", "is_stop", "take"]

ENV = "FIBSEM_GATEWAY_SECRET"
# What the gateway adds to every call it passes on or makes itself.
HEADER = "X-Fibsem-Gateway"

_STOP = re.compile(r"(^|_)(stop|halt|abort)(_|$)")
# Never refused for coming another way: liveness, and the page's own files.
_OPEN = ("/health",)

_taken: list = []


def is_stop(path: str) -> bool:
    """Whether *path* stops something: its last part is a stop, a halt or an
    abort (``/app/workflow/stop``, ``/app/agent/stop``, a device's ``stop``
    command)."""
    last = path.rstrip("/").rsplit("/", 1)[-1]
    return bool(_STOP.search(last))


def take() -> Optional[str]:
    """The secret this process was started with, taken out of its environment
    the first time it is asked for: None when the gateway didn't start it."""
    if not _taken:
        _taken.append(os.environ.pop(ENV, None) or None)
    return _taken[0]


def _matches(given: Optional[str], secret: str) -> bool:
    return given is not None and hmac.compare_digest(
        given.encode("utf-8"), secret.encode("utf-8")
    )


def _refusal(gateway: dict) -> bytes:
    import json

    return json.dumps(
        {
            "detail": {
                "error_type": "through_gateway",
                "message": "This PC's gateway started this server, so every call "
                f"comes through it ({gateway.get('url', 'port 8010')}).",
            }
        }
    ).encode("utf-8")


class GatewayOnly:
    """ASGI middleware: while the server holds a secret from the gateway
    (``app.state.auth.gateway_secret`` with ``from_gateway``) and this PC's
    gateway runs, an HTTP call without it is refused (401 ``through_gateway``)
    before any route sees it."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or self._passes(scope):
            await self.app(scope, receive, send)
            return
        gateway = read_gateway_file()
        if gateway is None:
            # No gateway: the operator still reaches a server it left behind.
            await self.app(scope, receive, send)
            return
        body = _refusal(gateway)
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    def _passes(scope) -> bool:
        auth = getattr(scope["app"].state, "auth", None)
        secret = getattr(auth, "gateway_secret", None)
        if secret is None or not getattr(auth, "from_gateway", False):
            return True
        path = scope.get("path", "")
        if path in _OPEN or is_stop(path):
            return True
        name = HEADER.lower().encode("latin-1")
        for key, value in scope.get("headers", ()):
            if key == name:
                return _matches(value.decode("latin-1"), secret)
        return False
