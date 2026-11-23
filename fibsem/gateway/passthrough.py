"""The gateway's one port: the App Server's routes, passed through.

A call to ``/app/...``, ``/microscope/...``, ``/services/...`` or
``/capabilities`` on the gateway goes to the App Server running on this PC and
its answer comes back unchanged: the status, the headers and the body,
streamed both ways, so a long-poll, an image or a file read arrives as it
would from the App Server itself. The caller's token goes on as it came, and
the App Server checks it as ever (the gateway reads the same paired file, so
nobody pairs again). With no App Server running, or one that refuses the
connection, the gateway answers 503 or 502 ``no_app``; with one that stops
answering mid-call, 504 ``app_lost``. A client reads both as the App Server
not answering (`fibsem.client.transport`), as it would reading it directly.

The routes are the gateway's own async handlers: a parked long-poll holds no
thread of the gateway's. A stop (the workflow's, a task's, the agent's, or a
device's stop or halt command) goes through a client of its own, so it never
waits behind the other traffic for a connection.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Iterable, Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

logger = logging.getLogger(__name__)

__all__ = ["APP_LOST", "NO_APP", "PASSED", "PassThrough", "is_stop"]

# The App Server's routes the gateway serves on its own port. Its pairing,
# web app, /health and /chat stay its own: the gateway has its own of each.
PASSED = ("app", "microscope", "services")
ROOT_PASSED = ("capabilities",)


# Not passed on either way: they describe one hop, not the call (RFC 9110).
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

# The longest an App Server answer may take: a long command answers at once
# with a run id, so a 120 s move or acquisition, or a parked long-poll (30 s),
# is the most a call waits.
READ_TIMEOUT_S = 300.0
CONNECT_TIMEOUT_S = 5.0

_STOP = re.compile(r"(^|_)(stop|halt|abort)(_|$)")


def is_stop(path: str) -> bool:
    """Whether *path* stops something: its last part is a stop, a halt or an
    abort (``/app/workflow/stop``, ``/app/agent/stop``, a device's ``stop``
    command)."""
    last = path.rstrip("/").rsplit("/", 1)[-1]
    return bool(_STOP.search(last))


# The error types a client reads as the App Server not answering
# (`fibsem.client.transport`): nothing reached it, or a call may have.
NO_APP = "no_app"
APP_LOST = "app_lost"


def _no_app(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"detail": {"error_type": NO_APP, "message": message}}, status_code=status
    )


def _headers(items: Iterable) -> list:
    return [(k, v) for k, v in items if k.lower() not in HOP_BY_HOP]


class PassThrough:
    """Calls on the gateway passed to the App Server at *upstream* (its url,
    or None while none runs)."""

    def __init__(
        self,
        upstream: Callable[[], Optional[str]],
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._upstream = upstream
        timeout = httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)
        # Stops have their own connections, never queued behind other calls.
        self._stops = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=CONNECT_TIMEOUT_S),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()
        await self._stops.aclose()

    def _url(self) -> Optional[str]:
        try:
            return self._upstream()
        except Exception:  # noqa: BLE001 - no file is no app
            logger.debug("could not find the App Server", exc_info=True)
            return None

    async def forward(self, request: Request):
        url = self._url()
        if url is None:
            return _no_app(
                503, "The app isn't running: start it on the gateway's page."
            )
        path = request.url.path
        client = self._stops if is_stop(path) else self._client
        target = httpx.URL(f"{url.rstrip('/')}{path}").copy_with(
            query=request.url.query.encode("utf-8") or None
        )
        sent = client.build_request(
            request.method,
            target,
            headers=_headers(
                (k.decode("latin-1"), v.decode("latin-1"))
                for k, v in request.headers.raw
            ),
            content=request.stream(),
        )
        try:
            answer = await client.send(sent, stream=True)
        except httpx.ConnectError as error:
            # Nothing reached the App Server: as if it weren't running.
            return _no_app(502, f"The app didn't answer: {error}")
        except httpx.HTTPError as error:
            # It may have been reached and acted: a client must not resend.
            logger.warning(
                {"msg": "pass-through failed", "path": path, "error": repr(error)}
            )
            return JSONResponse(
                {
                    "detail": {
                        "error_type": APP_LOST,
                        "message": f"The app stopped answering mid-call: {error!r}",
                    }
                },
                status_code=504,
            )
        response = StreamingResponse(
            answer.aiter_raw(),
            status_code=answer.status_code,
            background=BackgroundTask(answer.aclose),
        )
        # every header as sent, repeated ones too, and none of the hop's
        response.raw_headers = [
            (k.encode("latin-1"), v.encode("latin-1"))
            for k, v in _headers(answer.headers.multi_items())
        ]
        return response

    def router(self) -> APIRouter:
        router = APIRouter()
        methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
        for prefix in PASSED:
            router.add_api_route(
                f"/{prefix}/{{rest:path}}",
                self.forward,
                methods=methods,
                include_in_schema=False,
            )
        # the prefix itself too (``GET /microscope``): else a redirect to
        # its slashed form, which the App Server redirects back
        for name in PASSED + ROOT_PASSED:
            router.add_api_route(
                f"/{name}", self.forward, methods=methods, include_in_schema=False
            )
        return router
