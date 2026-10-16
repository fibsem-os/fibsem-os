"""Authentication and the agent level gate for the App Server (FIB-852, FIB-1035).

One bearer token per server instance, and one gate: the agent level the
operator sets from the Agent chip. Each route needs a level:

- ``watch``: reads, and adding notes. Every valid token at Watch or above.
- ``assist``: stopping what is running, and suggesting an answer: at Assist
  and Drive, with no request (FIB-1283).
- ``drive``: acting on the session and the microscope. At Assist the agent
  asks first, and an approved request lends Drive to its calls
  (:func:`lending_drive`).

Fail closed: a server always has a token (generated when the hosting does not
supply one), and starts at Watch unless the hosting says otherwise. Off
refuses everything, reads included.

A second, read-only **dashboard token** (FIB-1035) reaches the monitoring
routes -- /capabilities and GETs of the app's read routes -- and nothing else:
not the microscope, not a POST, whatever the level. It is carried in the
dashboard link and never written to the discovery file, so the dashboard and
the agent are told apart, and what governs the agent never cuts off the
dashboard.
"""

import hmac
import logging
import secrets
import threading
import time
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional

from fastapi import HTTPException, Request

from fibsem.acting import AGENT, current_actor
from fibsem.control import MicroscopeBusy
from fibsem.server.calls import current_call

# Re-exported: AgentLevel lives where the GUI can import it without FastAPI.
from fibsem.server.levels import AgentLevel

# What each level reaches, in order: a route needing a level is open at it and
# every level above.
_LEVEL_RANK = {
    AgentLevel.OFF: 0,
    AgentLevel.WATCH: 1,
    AgentLevel.ASSIST: 2,
    AgentLevel.DRIVE: 3,
}


@dataclass
class AuthConfig:
    """Token and agent level for one server instance.

    The token is fixed for the instance's lifetime; the level is the one
    mutable thing, through :meth:`set_level` and :meth:`stop_agent` -- the
    Agent chip's seam. Changing it live (rather than rebuilding the server)
    keeps the token stable, so connected agents gain or lose reach without
    being disconnected.
    """

    token: str
    # The dashboard's own token: monitoring reads only (require_monitoring).
    # None means no dashboard access, which is what a bare AuthConfig(token=...)
    # gets; generate() always makes one.
    dashboard_token: Optional[str] = None
    # The agent level (FIB-1035): the only gate.
    level: AgentLevel = AgentLevel.WATCH
    # Set by stop_agent, cleared by the next set_level: the operator braked, so a
    # refusal says so rather than only naming the level.
    stopped: bool = False
    # Monotonic time of the last request that presented the valid token; None
    # until one arrives. The connected agent's proof of life: its event
    # long-poll alone touches the server every half-minute, so staleness here
    # means nobody is on the other end (used to hand a parked question to the
    # operator without waiting out the full hand-over time).
    last_seen_monotonic: Optional[float] = None
    # The MCP client behind the sidecar, as the sidecar names it in its
    # User-Agent ("claude-code/2.1.3"); None until a sidecar request names one.
    # Self-reported, so a label for the operator, not an identity: every client
    # shares the one token.
    agent_client: Optional[str] = None
    # Called after each level change and on Stop agent (on_level_change).
    _listeners: List[Callable[[], Optional[Callable]]] = field(
        default_factory=list, repr=False, compare=False
    )

    def mark_seen(self, user_agent: Optional[str] = None) -> None:
        """Record that the token holder just made a request. Any thread —
        a float rebind is atomic, and readers only ever want 'roughly now'.

        A request from the sidecar also records which client it serves.
        Anything else (``fibsem-mcp watch``, curl) keeps the name already
        known: those are the same agent's other hands."""
        self.last_seen_monotonic = time.monotonic()
        client = client_from_user_agent(user_agent)
        if client is not None:
            self.agent_client = client

    def seconds_since_seen(self) -> Optional[float]:
        """Seconds since the token holder was last heard from, or None if never."""
        seen = self.last_seen_monotonic
        if seen is None:
            return None
        return max(0.0, time.monotonic() - seen)

    def set_level(self, level: AgentLevel) -> None:
        """Set the agent level, effective for the next request."""
        self._set_level(AgentLevel(level), stopped=False)

    def stop_agent(self) -> None:
        """The operator's Stop agent: down to Watch at once. The agent still
        sees, so it can say what happened; its next action is refused as
        stopped."""
        self._set_level(AgentLevel.WATCH, stopped=True)

    def on_level_change(self, callback: Callable[[AgentLevel, bool], None]) -> None:
        """Call ``callback(level, stopped)`` after the level changes, and on
        Stop agent. A bound method is held weakly: a server rebuilt on the same
        auth does not keep the old one's listener alive."""
        if hasattr(callback, "__self__"):
            self._listeners.append(weakref.WeakMethod(callback))
        else:
            self._listeners.append(lambda: callback)

    def _set_level(self, level: AgentLevel, stopped: bool) -> None:
        changed = level is not self.level or stopped
        # Atomic rebinds, read independently by requests.
        self.level = level
        self.stopped = stopped
        if not changed:
            return
        live = []
        for ref in self._listeners:
            callback = ref()
            if callback is None:
                continue
            live.append(ref)
            try:
                callback(level, stopped)
            except Exception:  # noqa: BLE001 - a listener never blocks the level
                logging.exception("Agent level listener failed")
        self._listeners[:] = live

    def reaches(self, needed: AgentLevel) -> bool:
        """Whether the current level opens a route that needs *needed*."""
        return _LEVEL_RANK[self.level] >= _LEVEL_RANK[needed]

    @classmethod
    def generate(
        cls, level: AgentLevel = AgentLevel.WATCH, token: Optional[str] = None
    ) -> "AuthConfig":
        return cls(
            token=token or secrets.token_urlsafe(32),
            dashboard_token=secrets.token_urlsafe(32),
            level=AgentLevel(level),
        )


def client_from_user_agent(user_agent: Optional[str]) -> Optional[str]:
    """The client the sidecar serves, from its User-Agent, or None.

    The sidecar sends ``fibsem-mcp/<version> <client>/<version>``, the client
    taken from the MCP initialize request's clientInfo; the product after its
    own is the client."""
    if not user_agent:
        return None
    products = user_agent.split()
    if len(products) < 2 or not products[0].startswith("fibsem-mcp/"):
        return None
    return products[1]


# Drive, lent to the calls an approved request makes (FIB-1283), for those calls
# only. A context variable: set by the request's worker around each call it
# makes into the app, so nothing arriving over the network can carry it.
_lent: "ContextVar[bool]" = ContextVar("fibsem_drive_lent", default=False)


@contextmanager
def lending_drive() -> Iterator[None]:
    """Lend Drive to the calls made inside this block: an action the operator
    approved runs as if the agent were at Drive. Off still refuses everything."""
    token = _lent.set(True)
    try:
        yield
    finally:
        _lent.reset(token)


def drive_is_lent() -> bool:
    """Whether this call is an approved request's (:func:`lending_drive`): the
    app's doing, not the agent's."""
    return _lent.get()


def _bearer_token(request: Request) -> Optional[str]:
    header = request.headers.get("Authorization")
    if header is None or not header.startswith("Bearer "):
        return None
    return header[len("Bearer ") :]


def _matches(token: Optional[str], expected: Optional[str]) -> bool:
    return (
        token is not None
        and expected is not None
        and hmac.compare_digest(token, expected)
    )


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail={
            "error_type": "unauthorized",
            "message": "Missing or invalid bearer token.",
        },
    )


def _dashboard_refused() -> HTTPException:
    return HTTPException(
        status_code=403,
        detail={
            "error_type": "dashboard_read_only",
            "message": "The dashboard token reads the monitoring routes only.",
        },
    )


def _level_detail(auth: AuthConfig, needed: AgentLevel) -> dict:
    if auth.stopped:
        return {
            "error_type": "agent_stopped",
            "level": AgentLevel.WATCH.value,
            "message": "The operator stopped the agent and set it to Watch: it "
            "can look and add notes, but not act, until they raise the level.",
        }
    if auth.level is AgentLevel.ASSIST and needed is AgentLevel.DRIVE:
        return {
            "error_type": "ask_first",
            "level": AgentLevel.ASSIST.value,
            "message": "At Assist the agent asks before it acts: put this call "
            "in a request (request_create, POST /app/agent/requests) with your "
            "reason, and the operator approves or rejects it in AutoLamella.",
        }
    return {
        "error_type": "level",
        "level": auth.level.value,
        "message": f"Not allowed while the agent level is "
        f"{auth.level.value.title()}: the operator sets the level from the "
        "Agent chip in AutoLamella.",
    }


def require_level(needed: AgentLevel):
    """Build a FastAPI dependency enforcing a valid token and an agent level of
    at least *needed*, and refusing everything while the level is Off.

    The agent token only: the dashboard token is refused here (403), so it
    reaches nothing beyond the routes guarded by :func:`require_monitoring`.
    """
    needed = AgentLevel(needed)

    def dependency(request: Request) -> None:
        auth: AuthConfig = request.app.state.auth
        token = _bearer_token(request)
        if not _matches(token, auth.token):
            if _matches(token, auth.dashboard_token):
                raise _dashboard_refused()
            raise _unauthorized()
        lent = _lent.get()
        # A valid token is proof of life whatever happens next -- a level
        # refusal below is still the agent talking. An approved request's call
        # is the app's doing, not the agent's.
        if not lent:
            auth.mark_seen(request.headers.get("User-Agent"))
        if auth.level is AgentLevel.OFF:
            raise HTTPException(
                status_code=403,
                detail={
                    "error_type": "agent_off",
                    "level": AgentLevel.OFF.value,
                    "message": "The agent is off: the operator has set the "
                    "agent level to Off in AutoLamella.",
                },
            )
        if not (auth.reaches(needed) or lent):
            raise HTTPException(status_code=403, detail=_level_detail(auth, needed))

    return dependency


def require_monitoring():
    """The monitoring routes' dependency: the agent token at Watch, or the
    dashboard token for a GET. The dashboard is not the agent, so it does not
    count as the agent being heard from."""
    as_agent = require_level(AgentLevel.WATCH)

    def dependency(request: Request) -> None:
        auth: AuthConfig = request.app.state.auth
        if _matches(_bearer_token(request), auth.dashboard_token):
            if request.method != "GET":
                # The read router carries POSTs too (the workflow stop, the
                # grid plan): the dashboard only looks.
                raise _dashboard_refused()
            return
        as_agent(request)

    return dependency


def action_name(request: Request) -> str:
    """What a request does, in words, for the record, the log and the toast
    (FIB-1359): "focus sweep", "move stage relative", "set electron imaging
    settings". Not the route: an operator reads it."""
    params = request.path_params
    for key in ("command", "query"):
        if key in params:
            name = params[key].replace("_", " ")
            # A one-word command says too little on its own: "milling run", not
            # "run" (FIB-1377).
            if " " not in name and "device" in params:
                return f"{params['device'].replace('_', ' ')} {name}"
            return name
    words = [
        part.replace("_", " ")
        for part in request.url.path.split("/")
        if part and part not in ("microscope", "beams", "devices", "services")
    ]
    if request.method == "PUT":
        return " ".join(["set", *words])
    return words[-1] if words else request.url.path


def described_action(request: Request) -> str:
    """The agent's call as the line an operator reads (FIB-1426): its tool and
    arguments described as a request's card says them, else :func:`action_name`."""
    call = current_call()
    if call is not None and call.action:
        return call.action
    return action_name(request)


def take_control(request: Request):
    """Take control of the microscope for this request (FIB-1270), or 409.

    The control a run, a script and the operator take too, so a server command
    never interleaves with them or with another request: concurrent requests on
    the threadpool would interleave multi-call vendor operations (the
    FIB-517/542/569 incident class). A structured refusal naming who is in
    control beats queueing: a caller blocked behind an hour-long mill should be
    told, not held. During a run's question, the run has lent control, so the
    request borrows it.
    """
    control = request.app.state.microscope.control
    taking = control.take(current_actor() or AGENT, described_action(request))
    try:
        entry = taking.__enter__()
    except MicroscopeBusy as busy:
        raise HTTPException(
            status_code=409,
            detail={
                "error_type": "microscope_busy",
                "message": str(busy),
                "in_control": {
                    "by": busy.in_control.by,
                    "action": busy.in_control.action,
                },
            },
        ) from None
    once = threading.Lock()

    def release() -> None:
        # The request and a run it started may both get here: give up once.
        if once.acquire(blocking=False):
            taking.__exit__(None, None, None)

    request.state.release_control = release
    request.state.control_entry = entry
    # Entered and exited by hand, not ``with``: an exception from the route is
    # thrown in here, and must reach FastAPI as it is.
    try:
        yield
    finally:
        if not getattr(request.state, "control_adopted", False):
            release()


def adopt_control(request: Request) -> Optional[Callable[[], None]]:
    """Keep the control of the microscope this request took past its answer,
    for a long command that runs on (`fibsem.devices.runs`). Returns what gives
    it up, which the run calls when it ends; None when the request took none.
    Its starter may act beside it while it runs (`Control.let_act_beside`)."""
    release = getattr(request.state, "release_control", None)
    if release is not None:
        request.state.control_adopted = True
        request.app.state.microscope.control.let_act_beside(request.state.control_entry)
    return release
