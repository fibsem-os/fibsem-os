"""The fibsem MCP served by the gateway, at ``/mcp/fibsem`` on its port.

The sidecar (`sidecar.build_sidecar`) over streamable HTTP instead of stdio,
so an agent on another computer needs only an MCP client and the SSH tunnel,
not fibsem installed:

    claude mcp add --transport http fibsem http://localhost:8010/mcp/fibsem \\
        --header "Authorization: Bearer <token>"

The gateway starts this process the first time something calls ``/mcp``
(`fibsem.gateway.mcp`), passes the calls on and adds the secret it started it
with (`fibsem.server.gateway_secret`), so nothing on the PC reaches it any
other way.

Each caller is its own:

- **Token.** Each MCP session calls through the gateway's own port with the
  token its caller sent, so the gateway and the App Server check it as they
  check any agent's: the slot, the level, and a connect token's "connect
  first". This process holds no token of its own.
- **Options.** What the stdio sidecar takes as arguments come as query values
  on the address: ``instrument`` (an instrument kind), ``record=no``
  (``--no-record``) and ``always_load`` (tool names, comma separated). Each
  set of options is its own MCP server, with its own sessions.
- **Files.** Nothing is read or written on this PC for a caller
  (`sidecar.SERVED`): a fetched file comes back in the answer, and evidence
  comes as its bytes.

Waking the agent: ``GET /mcp/fibsem/watch`` streams the lines ``fibsem-mcp
watch`` prints (`fibsem.ai.agent.watch`), one a line, for as long as the call
stays open. A caller that runs commands in the background runs ``curl -sN``
on it, and a new watch carries on after the last event the caller's last one
read. A client that can't run anything in the background calls the
``app_wait`` tool instead, which answers the same lines.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import queue
import socket
import threading
import time
import weakref
from contextvars import ContextVar
from typing import Callable, Dict, List, Optional, Tuple

from fibsem.ai.agent import watch as watching
from fibsem.ai.agent.mcp import sidecar
from fibsem.client.transport import ServerUnreachable, Transport
from fibsem.server import discovery
from fibsem.server.gateway_secret import HEADER as SECRET_HEADER
from fibsem.server.gateway_secret import take

logger = logging.getLogger(__name__)

__all__ = [
    "DISCOVERY_FILE",
    "SERVED_PATH",
    "WATCH_PATH",
    "Callers",
    "build_app",
    "serve_for_gateway",
]

SERVED_PATH = "/mcp/fibsem"
WATCH_PATH = SERVED_PATH + "/watch"
# Where this process says it listens, for the gateway (`fibsem.gateway.mcp`).
DISCOVERY_FILE = discovery.DISCOVERY_DIR / "mcp-server.json"

# The longest app_wait waits for a line.
WAIT_MAX_S = 55.0
# An app_wait watch nobody has asked for lines for this long stops.
WAIT_IDLE_S = 15 * 60.0

INSTRUCTIONS = (
    "The fibsem MCP, served by the microscope PC's gateway. Nothing wakes you "
    "between turns unless you watch: run, in the BACKGROUND and never in the "
    "foreground, curl -sN -H 'Authorization: Bearer <your token>' "
    "<this server's address>/watch (for example "
    "http://localhost:8010/mcp/fibsem/watch). Each line it prints is a wake-up "
    "(a proposal, a message from the operator, a mill ending). When it exits, "
    "read its last line and start it again: it carries on where it stopped. "
    "If you can't run a command in the background, call app_wait whenever you "
    "would otherwise sit waiting: it answers the same lines."
)


def _bearer(headers) -> Optional[str]:
    value = headers.get("authorization", "") if headers is not None else ""
    if value.lower().startswith("bearer "):
        return value[7:].strip() or None
    return None


def _key(token: str) -> str:
    """A caller's name for its kept state, never its token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


class _Caller:
    """One MCP session's caller: its connection, through the gateway with its
    own token."""

    def __init__(self, transport, token: str) -> None:
        self.transport = transport
        self.token = token
        self.key = _key(token)


_calling: "ContextVar[Optional[_Caller]]" = ContextVar(
    "fibsem_mcp_served_caller", default=None
)


class Callers:
    """The client a served sidecar's tools call through: the connection of the
    caller whose message is being handled, made once for each MCP session with
    the token its caller sent (`bind`)."""

    # The sidecar names each caller's MCP client on its own connection.
    per_caller = True
    # Every session's connection is made at its first message: never waits.
    connected = True

    def __init__(self, url: str, connect: Optional[Callable[[str], object]] = None):
        self.url = url
        self._connect = connect or self.transport
        self._sessions: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def transport(self, token: str):
        """A connection to the gateway with *token*: that one server and
        nothing else, and a connect token's refusal is answered, never asked
        for here (the caller asks to connect itself)."""
        found = Transport(self.url, token, discover=False)
        found.asks = False
        return found

    def connect(self, token: str):
        """A new connection with *token*, for a watch of its own."""
        return self._connect(token)

    def bind(self, session, token: str):
        """Make *session*'s caller the one calls go through, from here in this
        context: the reset token for `unbind`."""
        with self._lock:
            caller = self._sessions.get(session)
            if caller is None or caller.token != token:
                caller = _Caller(self._connect(token), token)
                self._sessions[session] = caller
                close = getattr(caller.transport, "close", None)
                if close is not None:
                    weakref.finalize(session, close)
        return _calling.set(caller)

    @staticmethod
    def unbind(reset) -> None:
        _calling.reset(reset)

    @staticmethod
    def current() -> _Caller:
        caller = _calling.get()
        if caller is None:
            raise ServerUnreachable("fibsem-mcp: this call came with no caller's token")
        return caller

    def request(self, method, path, **kwargs):
        return self.current().transport.request(method, path, **kwargs)

    @property
    def named_user_agent(self) -> Optional[str]:
        return None

    def set_user_agent(self, user_agent: str) -> None:
        caller = _calling.get()
        if caller is not None and caller.transport.user_agent != user_agent:
            caller.transport.user_agent = user_agent

    def close(self) -> None:
        with self._lock:
            callers = list(self._sessions.values())
            self._sessions.clear()
        for caller in callers:
            try:
                caller.transport.close()
            except Exception:  # noqa: BLE001 - closing what is going anyway
                pass


def _binding(callers: Callers):
    """The MCP middleware that binds each message to its caller's connection."""

    async def bind_caller(ctx, call_next):
        request = getattr(ctx, "request", None)
        token = _bearer(getattr(request, "headers", None))
        if token is None:
            return await call_next(ctx)
        reset = callers.bind(ctx.session, token)
        try:
            return await call_next(ctx)
        finally:
            callers.unbind(reset)

    return bind_caller


# --- Waking the agent --------------------------------------------------------


class _Stopped(Exception):
    """A watch was stopped: its caller left."""


class _Stoppable:
    """A connection whose next request ends the watch once it is stopped."""

    def __init__(self, transport, stopped: threading.Event, idle=None) -> None:
        self._transport = transport
        self._stopped = stopped
        self._idle = idle

    def request(self, method, path, **kwargs):
        if self._idle is not None and self._idle():
            self._stopped.set()
        if self._stopped.is_set():
            raise _Stopped()
        return self._transport.request(method, path, **kwargs)


_ENDED = object()


class Watcher:
    """`watch.watch` in a thread of its own, its lines kept for `next` until
    it ends or is stopped. Each event it has read moves *cursors*' entry for
    *key*, so the caller's next watch carries on after it."""

    def __init__(
        self,
        transport,
        cursors: Dict[str, str],
        key: str,
        report_reads: bool = True,
        idle_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = watching.POLL_S,
    ) -> None:
        self._transport = transport
        self._cursors = cursors
        self._key = key
        self._report_reads = report_reads
        self._idle_s = idle_s
        self._clock = clock
        self._poll_s = poll_s
        self.used = clock()
        self._lines: "queue.Queue" = queue.Queue()
        self._stopped = threading.Event()
        self.ended = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mcp-watch", daemon=True)

    def start(self) -> "Watcher":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stopped.set()

    def _idle(self) -> bool:
        return self._idle_s is not None and self._clock() - self.used > self._idle_s

    def _sleep(self, seconds: float) -> None:
        if self._stopped.wait(seconds):
            raise _Stopped()

    def _moved(self, last: str) -> None:
        self._cursors[self._key] = last

    def _run(self) -> None:
        client = _Stoppable(self._transport, self._stopped, self._idle)
        try:
            watching.watch(
                client,
                out=self._lines.put,
                sleep_fn=self._sleep,
                poll_s=self._poll_s,
                after=self._cursors.get(self._key),
                on_cursor=self._moved,
                report_reads=self._report_reads,
            )
        except _Stopped:
            pass
        except Exception:  # noqa: BLE001 - a watch that breaks says so and ends
            logger.warning("A served watch ended", exc_info=True)
            self._lines.put(
                watching.Wake(
                    watching.SERVER_GONE,
                    "SERVER GONE: the watch broke; start it again.",
                )
            )
        finally:
            self.ended.set()
            self._lines.put(_ENDED)
            try:
                self._transport.close()
            except Exception:  # noqa: BLE001
                pass

    def next(self, wait_s: float):
        """The next line, None when none came within *wait_s*, or `_ENDED`."""
        try:
            return self._lines.get(timeout=wait_s)
        except queue.Empty:
            return None

    def waiting(self) -> List[object]:
        """Every line already kept, at once."""
        lines = []
        while True:
            try:
                lines.append(self._lines.get_nowait())
            except queue.Empty:
                return lines


class Waits:
    """The app_wait tool's watches, one for each caller, kept between calls."""

    def __init__(self, callers: Callers, cursors: Dict[str, str]) -> None:
        self._callers = callers
        self._cursors = cursors
        self._watches: Dict[str, Watcher] = {}
        self._lock = threading.Lock()

    def _watch_for(self, caller: _Caller) -> Watcher:
        with self._lock:
            found = self._watches.get(caller.key)
            if found is None or found.ended.is_set():
                found = Watcher(
                    self._callers.connect(caller.token),
                    self._cursors,
                    caller.key,
                    # Marked read when app_wait answers them, not when kept.
                    report_reads=False,
                    idle_s=WAIT_IDLE_S,
                ).start()
                self._watches[caller.key] = found
            found.used = time.monotonic()
            return found

    def wait(self, timeout_s: float = 25.0) -> str:
        caller = self._callers.current()
        found = self._watch_for(caller)
        wait_s = max(0.0, min(float(timeout_s), WAIT_MAX_S))
        first = found.next(wait_s)
        if first is None:
            return f"NOTHING NEW in {wait_s:g} s: call app_wait again to keep waiting."
        lines = [first, *found.waiting()]
        said = [line for line in lines if line is not _ENDED]
        ids = [i for line in said for i in getattr(line, "message_ids", ())]
        watching.report_read(caller.transport, ids)
        text = "\n".join(str(line) for line in said)
        if _ENDED in lines:
            text += ("\n" if text else "") + (
                "WATCH ENDED: the next app_wait starts a new one."
            )
        return text


def add_wait_tool(mcp, waits: Waits) -> None:
    """The ``app_wait`` tool, for a client that can't run a watch in the
    background."""
    from mcp.types import ToolAnnotations

    def app_wait(timeout_s: float = 25.0) -> str:
        return waits.wait(timeout_s)

    mcp.add_tool(
        sidecar._flag_failures(app_wait),
        name="app_wait",
        title="Wait for news",
        description=(
            "Wait up to timeout_s seconds (at most 55) for something worth "
            "acting on: a proposal, a message from the operator, a command "
            "ending, a level change. Answers the lines a watch prints, one per "
            "line, each event once; NOTHING NEW when none came. Only for a "
            "client that can't run the watch (<this address>/watch) in the "
            "background: nothing wakes you between calls, so call it whenever "
            "you would otherwise sit waiting."
        ),
        annotations=ToolAnnotations(
            title="Wait for news",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )


# --- The app -----------------------------------------------------------------


def options(query) -> Tuple[Optional[str], Optional[bool], Tuple[str, ...]]:
    """The session options a query carries: (instrument, record, always_load).
    Raises ValueError for a value the sidecar has no such option for."""
    instrument = (query.get("instrument") or "").strip() or None
    record_given = (query.get("record") or "").strip().lower()
    if record_given in ("", "yes"):
        record = None
    elif record_given == "no":
        record = False
    else:
        raise ValueError("record is no, or left out")
    always = tuple(
        sorted(
            {
                n.strip()
                for n in (query.get("always_load") or "").split(",")
                if n.strip()
            }
        )
    )
    return instrument, record, always


def _json(status: int, error_type: str, message: str) -> Tuple[int, bytes]:
    return status, json.dumps(
        {"detail": {"error_type": error_type, "message": message}}
    ).encode("utf-8")


async def _answer(send, status: int, body: bytes, media: str = "application/json"):
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", media.encode("latin-1"))],
        }
    )
    await send({"type": "http.response.body", "body": body})


def build_app(
    gateway_url: str,
    secret: Optional[str] = None,
    callers: Optional[Callers] = None,
    build: Callable[..., object] = sidecar.build_sidecar,
    admit: Optional[Callable] = None,
):
    """The ASGI app: ``/mcp/fibsem`` (MCP), ``/mcp/fibsem/watch`` and
    ``/health``. With *secret*, every call but /health must carry it
    (`SECRET_HEADER`), as the gateway adds it. The process serving it sets
    `sidecar.SERVED` (`serve_for_gateway`). *admit* (a test's) answers a
    caller's token as the gateway would: an awaitable of None, or a refusal
    (status, body)."""
    import anyio
    import httpx

    callers = callers or Callers(gateway_url)
    cursors: Dict[str, str] = {}
    waits = Waits(callers, cursors)
    servers: Dict[tuple, object] = {}
    state: dict = {"group": None, "lock": None}

    async def _run_manager(manager, *, task_status=anyio.TASK_STATUS_IGNORED):
        async with manager.run():
            task_status.started()
            await anyio.sleep_forever()

    def _build(key):
        instrument, record, always = key
        mcp = build(
            callers,
            None,
            instrument,
            record,
            set(always),
            instructions=INSTRUCTIONS,
            middleware=(_binding(callers),),
        )
        add_wait_tool(mcp, waits)
        mcp.streamable_http_app(streamable_http_path=SERVED_PATH, host="127.0.0.1")
        return mcp

    async def server_for(key):
        async with state["lock"]:
            found = servers.get(key)
            if found is None:
                found = await anyio.to_thread.run_sync(_build, key)
                await state["group"].start(_run_manager, found.session_manager)
                servers[key] = found
            return found

    async def gateway_says(token):
        """What the gateway answers *token* at /capabilities: a refusal as
        (status, body), else None."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                resp = await client.get(
                    f"{gateway_url.rstrip('/')}/capabilities",
                    headers={"Authorization": f"Bearer {token}"},
                )
            except httpx.HTTPError as error:
                return _json(502, "no_gateway", f"The gateway didn't answer: {error}")
        if resp.status_code in (401, 403):
            return resp.status_code, resp.content
        # Any other answer (no app yet, say): the tools say so when called.
        return None

    refusal_for = admit or gateway_says

    async def admitted(scope, send, token) -> bool:
        """Whether *token* opens the MCP: asked of the gateway itself, so a
        refused token is answered at once in the gateway's words."""
        refused = await refusal_for(token)
        if refused is None:
            return True
        await _answer(send, *refused)
        return False

    async def serve_mcp(scope, receive, send, token):
        from starlette.datastructures import Headers, QueryParams

        query = QueryParams(scope.get("query_string", b"").decode("latin-1"))
        try:
            key = options(query)
        except ValueError as error:
            await _answer(send, *_json(400, "invalid", str(error)))
            return
        headers = Headers(scope=scope)
        if scope["method"] == "POST" and "mcp-session-id" not in headers:
            if not await admitted(scope, send, token):
                return
        try:
            mcp = await server_for(key)
        except Exception as error:  # noqa: BLE001 - an instrument kind not known
            logger.warning("No served MCP for %s", key, exc_info=True)
            await _answer(
                send, *_json(400, "invalid", f"{type(error).__name__}: {error}")
            )
            return
        await mcp.session_manager.handle_request(scope, receive, send)

    async def serve_watch(scope, receive, send, token):
        if scope["method"] != "GET":
            await _answer(send, *_json(405, "invalid", "GET the watch"))
            return
        if not await admitted(scope, send, token):
            return
        found = Watcher(callers.connect(token), cursors, _key(token)).start()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        gone = anyio.Event()

        async def listen():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    gone.set()
                    return

        try:
            async with anyio.create_task_group() as group:
                group.start_soon(listen)
                while not gone.is_set():
                    line = await anyio.to_thread.run_sync(
                        found.next, 1.0, abandon_on_cancel=True
                    )
                    if line is None:
                        continue
                    if line is _ENDED:
                        break
                    await send(
                        {
                            "type": "http.response.body",
                            "body": (str(line) + "\n").encode("utf-8"),
                            "more_body": True,
                        }
                    )
                group.cancel_scope.cancel()
            await send({"type": "http.response.body", "body": b""})
        except OSError:
            pass  # the caller went away mid-line
        finally:
            found.stop()

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            message = await receive()
            if message["type"] != "lifespan.startup":
                return
            async with anyio.create_task_group() as group:
                state["group"], state["lock"] = group, anyio.Lock()
                await send({"type": "lifespan.startup.complete"})
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.shutdown":
                        group.cancel_scope.cancel()
                        break
            callers.close()
            await send({"type": "lifespan.shutdown.complete"})
            return
        if scope["type"] != "http":
            return
        path = scope["path"].rstrip("/")
        if path == "/health":
            await _answer(send, 200, b'{"ok": true}')
            return
        from starlette.datastructures import Headers

        headers = Headers(scope=scope)
        if secret is not None and not hmac.compare_digest(
            (headers.get(SECRET_HEADER) or "").encode("utf-8"), secret.encode("utf-8")
        ):
            await _answer(
                send,
                *_json(401, "through_gateway", "Call this through the gateway's port."),
            )
            return
        token = _bearer(headers)
        if token is None:
            await _answer(
                send,
                *_json(
                    401,
                    "unauthorized",
                    "Send your agent token (Authorization: Bearer <token>): pair "
                    "with the gateway first.",
                ),
            )
            return
        if path == WATCH_PATH:
            await serve_watch(scope, receive, send, token)
        elif path == SERVED_PATH:
            await serve_mcp(scope, receive, send, token)
        else:
            await _answer(send, *_json(404, "not_found", f"No {scope['path']} here."))

    return app


def _free_socket(bind: str) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind, 0))
    return sock


def serve_for_gateway(argv: List[str], stop: threading.Event) -> int:
    """This process as the gateway runs it (``fibsem.gateway.child``): serves
    until *stop* is set, on a free port it names in `DISCOVERY_FILE`."""
    import uvicorn

    parser = argparse.ArgumentParser(prog="fibsem-mcp-served")
    parser.add_argument("--gateway", required=True, help="the gateway's own url")
    args = parser.parse_args(argv)

    # Nothing on this PC is read or written for a caller.
    sidecar.SERVED = True
    sock = _free_socket("127.0.0.1")
    port = sock.getsockname()[1]
    app = build_app(args.gateway, secret=take())
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"
        )
    )

    def ending() -> None:
        stop.wait()
        server.should_exit = True

    threading.Thread(target=ending, name="mcp-served-stop", daemon=True).start()

    def say_where() -> None:
        # Named once it answers: the gateway passes calls on as soon as it reads it.
        while not server.started:
            if server.should_exit:
                return
            time.sleep(0.05)
        discovery.DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
        partial = DISCOVERY_FILE.with_name(DISCOVERY_FILE.name + ".partial")
        with open(partial, "w") as f:
            json.dump(
                {"url": f"http://127.0.0.1:{port}", **discovery.this_process()}, f
            )
        partial.replace(DISCOVERY_FILE)
        logger.info("The gateway's MCP listens on port %d", port)

    threading.Thread(target=say_where, name="mcp-served-where", daemon=True).start()
    try:
        server.run(sockets=[sock])
    finally:
        found = discovery.read_discovery_file(DISCOVERY_FILE)
        if found is not None and found.get("pid") == discovery.this_process()["pid"]:
            DISCOVERY_FILE.unlink(missing_ok=True)
    return 0
