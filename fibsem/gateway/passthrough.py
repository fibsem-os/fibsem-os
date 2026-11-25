"""The gateway's one port: the App Server's routes, passed through.

A call to ``/app/...``, ``/microscope/...``, ``/services/...`` or
``/capabilities`` on the gateway goes to the App Server running on this PC and
its answer comes back unchanged: the status, the headers and the body,
streamed both ways, so a long-poll, an image or a file read arrives as it
would from the App Server itself. The caller's token goes on as it came, and
the App Server checks it as ever (the gateway reads the same paired file, so
nobody pairs again). A call with ``gateway.json``'s connect token is answered here, 403
``connect_first``, as the App Server answers its own: the client asks to
connect (``POST /connect``, the gateway's), and the operator allows it. With
no App Server running, or one that refuses the
connection, the gateway answers 503 or 502 ``no_app``; with one that stops
answering mid-call, 504 ``app_lost``. A client reads both as the App Server
not answering (`fibsem.client.transport`), as it would reading it directly.

While no App Server runs, calls go to the agent host the gateway runs
(`fibsem.gateway.agent_host`), which serves the agent's own routes: its
events, session, plan, transcript, messages, level and Stop. A route only the
app serves (the microscope's, a workflow's) is answered 503 ``no_app`` then,
as with nothing running, not the host's 404.

The routes are the gateway's own async handlers: a parked long-poll holds no
thread of the gateway's. A stop (the workflow's, a task's, the agent's, or a
device's stop or halt command) goes through a client of its own, so it never
waits behind the other traffic for a connection.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Iterable, Optional, Tuple

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from fibsem.server import discovery

logger = logging.getLogger(__name__)

__all__ = ["APP_LOST", "CONNECT_FIRST", "NO_APP", "PASSED", "PassThrough", "is_stop"]

# The App Server's routes the gateway serves on its own port. Its /health stays
# its own; pairing and the web app are the gateway's alone.
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
        # The gateway's own, added below: never a caller's.
        "x-fibsem-gateway",
    }
)

# The longest an App Server answer may take when the call doesn't say: a long
# command answers at once with a run id, so a 120 s move or acquisition, or a
# parked long-poll (30 s), is the most such a call waits.
READ_TIMEOUT_S = 300.0
CONNECT_TIMEOUT_S = 5.0
# A call that says how long it waits (`fibsem.client.transport.WAIT_HEADER`),
# such as a long acquisition, is waited for that long and this much more, up
# to `WAIT_MAX_S`.
WAIT_HEADER = "x-fibsem-wait-s"
WAIT_MARGIN_S = 30.0
WAIT_MAX_S = 3 * 3600.0

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
# What the App Server answers a connect token with (`fibsem.server.auth`): the
# client asks to connect, then carries on with the token an Allow gives.
CONNECT_FIRST = "connect_first"


def _no_app(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"detail": {"error_type": NO_APP, "message": message}}, status_code=status
    )


def _timeout(said: Optional[str]):
    """The pass-through's timeout for a call that said it waits *said*
    seconds: that and `WAIT_MARGIN_S`, at least `READ_TIMEOUT_S` and at most
    `WAIT_MAX_S`; the usual for a call that said nothing readable."""
    try:
        wait = float(said) if said else None
    except ValueError:
        wait = None
    if wait is None or not wait > 0:
        return httpx.USE_CLIENT_DEFAULT
    read = min(max(wait + WAIT_MARGIN_S, READ_TIMEOUT_S), WAIT_MAX_S)
    return httpx.Timeout(read, connect=CONNECT_TIMEOUT_S)


def _headers(items: Iterable) -> list:
    return [(k, v) for k, v in items if k.lower() not in HOP_BY_HOP]


_kept_secret: Tuple[Optional[float], Optional[str]] = (None, None)


def app_server_secret() -> Optional[str]:
    """This PC's App Server's secret for the gateway, from its discovery file
    (`fibsem.server.auth`); None while none runs or it has none. Read again
    only when the file changes: it is asked on every call passed on."""
    global _kept_secret
    try:
        changed = discovery.DISCOVERY_FILE.stat().st_mtime
    except OSError:
        changed = None
    if changed is not None and _kept_secret[0] == changed:
        return _kept_secret[1]
    found = discovery.read_discovery_file()
    secret = found.get("gateway_secret") if found else None
    _kept_secret = (changed, secret)
    return secret


# What a FastAPI server answers for a route it doesn't have.
NOT_FOUND = {"detail": "Not Found"}
NOT_THE_HOSTS = (
    "The app isn't running, and this needs it: start it on the gateway's page."
)


class PassThrough:
    """Calls on the gateway passed to the App Server at *upstream* (its url,
    or None while none runs), else to the agent host at *host*."""

    def __init__(
        self,
        upstream: Callable[[], Optional[str]],
        transport: Optional[httpx.AsyncBaseTransport] = None,
        is_connect: Callable[[Optional[str]], bool] = lambda token: False,
        secret: Optional[Callable[[], Optional[str]]] = None,
        host: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self._upstream = upstream
        self._host = host
        # The App Server's secret (its discovery file's): every call passed on
        # carries it, so the App Server knows it came through the gateway.
        self._secret = secret or app_server_secret
        # gateway.json's token only asks to connect: answered here, never sent on
        self._is_connect = is_connect
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

    def _url(self) -> Tuple[Optional[str], bool]:
        """Where a call goes, and whether that is the agent host."""
        for find, is_host in ((self._upstream, False), (self._host, True)):
            if find is None:
                continue
            try:
                url = find()
            except Exception:  # noqa: BLE001 - no file is no server
                logger.debug("could not find the server", exc_info=True)
                url = None
            if url is not None:
                return url, is_host
        return None, False

    async def forward(self, request: Request):
        bearer = request.headers.get("authorization", "")
        if bearer.lower().startswith("bearer ") and self._is_connect(bearer[7:]):
            return JSONResponse(
                {
                    "detail": {
                        "error_type": CONNECT_FIRST,
                        "message": "This token only asks to connect (POST "
                        "/connect): the operator allows the agent on the "
                        "gateway's page.",
                    }
                },
                status_code=403,
            )
        url, to_host = self._url()
        if url is None:
            return _no_app(
                503, "The app isn't running: start it on the gateway's page."
            )
        path = request.url.path
        client = self._stops if is_stop(path) else self._client
        target = httpx.URL(f"{url.rstrip('/')}{path}").copy_with(
            query=request.url.query.encode("utf-8") or None
        )
        # A stop keeps its short client; any other call may say how long it waits.
        wait = (
            httpx.USE_CLIENT_DEFAULT
            if client is self._stops
            else _timeout(request.headers.get(WAIT_HEADER))
        )
        headers = _headers(
            (k.decode("latin-1"), v.decode("latin-1")) for k, v in request.headers.raw
        )
        try:
            # The App Server's own: never sent to the host.
            secret = None if to_host else self._secret()
        except Exception:  # noqa: BLE001 - no secret: the App Server says so
            logger.debug("could not read the App Server's secret", exc_info=True)
            secret = None
        if secret:
            headers.append(("X-Fibsem-Gateway", secret))
        sent = client.build_request(
            request.method,
            target,
            headers=headers,
            content=request.stream(),
            timeout=wait,
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
        if to_host and answer.status_code == 404:
            # A route the host doesn't have is the app's.
            body = await answer.aread()
            await answer.aclose()
            try:
                not_found = json.loads(body) == NOT_FOUND
            except ValueError:
                not_found = False
            if not_found:
                return _no_app(503, NOT_THE_HOSTS)
            return Response(
                body,
                status_code=404,
                headers=dict(_headers(answer.headers.multi_items())),
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
