"""Authentication and scope gating for the fibsem server (FIB-852).

One bearer token per server instance, and these scopes:

- ``read``: granted to every valid token. Observation only.
- ``control``: app-level workflow control. Armed by the hosting (GUI toggle in
  the embedded app; unused in bench hosting until an app router exists).
- ``configure``: editing task parameters (protocol and per-item task configs).
  Its own rung, deliberately separate from ``control``: answering a question
  with geometry and editing protocols are different grants (FIB-864).
- ``hardware``: anything that commands or mutates the microscope. In the App
  Server, armed at Drive (FIB-1267); on the bench, by ``--arm-hardware``.
- ``stop``: stopping what is running (the workflow, a mill). Armed at Assist and
  Drive (FIB-1283): stopping never waits for approval.

Fail closed: a server always has a token (generated when the hosting does not
supply one), and only ``read`` is armed unless the hosting says otherwise.

A second, read-only **dashboard token** (FIB-1035) reaches the monitoring
routes -- /capabilities and GETs of the app's read routes -- and nothing else:
not the microscope, not a POST, whatever is armed. It is carried in the
dashboard link and never written to the discovery file, so the dashboard and
the agent are told apart, and what governs the agent never cuts off the
dashboard.
"""

import hmac
import logging
import secrets
import time
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, FrozenSet, Iterable, Iterator, List, Optional

from fastapi import HTTPException, Request

from fibsem.acting import AGENT, current_actor
from fibsem.control import MicroscopeBusy

# Re-exported: AgentLevel lives where the GUI can import it without FastAPI.
from fibsem.server.levels import AgentLevel


class Scope(str, Enum):
    READ = "read"
    CONTROL = "control"
    CONFIGURE = "configure"
    HARDWARE = "hardware"
    STOP = "stop"


LEVEL_SCOPES = {
    AgentLevel.OFF: frozenset(),
    AgentLevel.WATCH: frozenset(),
    # Assist acts only by asking (FIB-1283): every action is a request the
    # operator approves. Stopping needs no approval.
    AgentLevel.ASSIST: frozenset({Scope.STOP}),
    # Hardware routes are mounted only with the devices setting on, and take
    # control of the microscope, so a run or script refuses them (FIB-1267).
    AgentLevel.DRIVE: frozenset(
        {Scope.CONTROL, Scope.CONFIGURE, Scope.HARDWARE, Scope.STOP}
    ),
}


@dataclass
class AuthConfig:
    """Token + armed scopes for one server instance.

    ``read`` is implicitly armed for any valid token and need not be listed.

    The token is fixed for the instance's lifetime; the armed set is the one
    mutable thing, and only through :meth:`set_armed` — the arming dialog's
    seam. Mutating it live (rather than rebuilding the server) keeps the
    token stable, so connected agents gain or lose a scope without being
    disconnected. Arming is deliberately never persisted: every session
    starts read-only and arming is a fresh, deliberate act.
    """

    token: str
    armed: FrozenSet[Scope] = field(default_factory=frozenset)
    # The dashboard's own token: monitoring reads only (require_monitoring).
    # None means no dashboard access, which is what a bare AuthConfig(token=...)
    # gets; generate() always makes one.
    dashboard_token: Optional[str] = None
    # The agent level (FIB-1035), set by the App Server through set_level. None
    # is a server with no level model -- the bench and headless hostings --
    # where the armed scopes alone decide.
    level: Optional[AgentLevel] = None
    # Set by stop_agent, cleared by the next set_level: the operator braked, so a
    # refusal says so rather than only naming the level.
    stopped: bool = False
    # Monotonic time of the last request that presented the valid token; None
    # until one arrives. The connected agent's proof of life: its event
    # long-poll alone touches the server every half-minute, so staleness here
    # means nobody is on the other end (used to hand a parked question to the
    # operator without waiting out the full hand-over time).
    last_seen_monotonic: Optional[float] = None
    # Called after each level change and on Stop agent (on_level_change).
    _listeners: List[Callable[[], Optional[Callable]]] = field(
        default_factory=list, repr=False, compare=False
    )

    def mark_seen(self) -> None:
        """Record that the token holder just made a request. Any thread —
        a float rebind is atomic, and readers only ever want 'roughly now'."""
        self.last_seen_monotonic = time.monotonic()

    def seconds_since_seen(self) -> Optional[float]:
        """Seconds since the token holder was last heard from, or None if never."""
        seen = self.last_seen_monotonic
        if seen is None:
            return None
        return max(0.0, time.monotonic() - seen)

    def set_level(self, level: AgentLevel) -> None:
        """Set the agent level, effective for the next request.

        Rewrites the armed scopes to the level's, hardware included: the
        level alone decides (FIB-1267)."""
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
        self.armed = LEVEL_SCOPES[level]
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

    def set_armed(self, scope: Scope, armed: bool) -> None:
        """Arm or disarm one scope, effective for the next request."""
        if scope is Scope.READ:
            return  # read is unconditional; there is nothing to disarm
        current = set(self.armed)
        if armed:
            current.add(scope)
        else:
            current.discard(scope)
        # One atomic rebind (reads never see a half-edited set).
        self.armed = frozenset(current)

    @classmethod
    def generate(
        cls,
        arm_control: bool = False,
        arm_configure: bool = False,
        arm_hardware: bool = False,
        token: Optional[str] = None,
    ) -> "AuthConfig":
        armed = set()
        if arm_control:
            armed.add(Scope.CONTROL)
        if arm_configure:
            armed.add(Scope.CONFIGURE)
        if arm_hardware:
            armed.add(Scope.HARDWARE)
        return cls(
            token=token or secrets.token_urlsafe(32),
            armed=frozenset(armed),
            dashboard_token=secrets.token_urlsafe(32),
        )

    def is_armed(self, scope: Scope) -> bool:
        return scope is Scope.READ or scope in self.armed

    def armed_scopes(self) -> Iterable[Scope]:
        return (s for s in Scope if self.is_armed(s))


# Scopes lent to the calls an approved request makes (FIB-1283), for those calls
# only. A context variable: set by the request's worker around each call it
# makes into the app, so nothing arriving over the network can carry it.
_granted: "ContextVar[FrozenSet[Scope]]" = ContextVar(
    "fibsem_granted_scopes", default=frozenset()
)


@contextmanager
def granting(scopes: Iterable[Scope]) -> Iterator[None]:
    """Lend *scopes* to the calls made inside this block: an action the operator
    approved runs as if its scope were armed. Off still refuses everything."""
    token = _granted.set(frozenset(scopes))
    try:
        yield
    finally:
        _granted.reset(token)


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


def _not_armed_detail(auth: AuthConfig, scope: Scope) -> dict:
    detail = {"error_type": "scope_not_armed", "scope": scope.value}
    if auth.level is AgentLevel.ASSIST and not auth.stopped:
        detail["error_type"] = "ask_first"
        detail["level"] = AgentLevel.ASSIST.value
        detail["message"] = (
            "At Assist the agent asks before it acts: put this call in a request "
            "(request_actions, POST /app/agent/requests) with your reason, and "
            "the operator approves or rejects it in AutoLamella."
        )
    elif auth.stopped:
        detail["error_type"] = "agent_stopped"
        detail["level"] = AgentLevel.WATCH.value
        detail["message"] = (
            "The operator stopped the agent and set it to Watch: it can look and "
            "add notes, but not act, until they raise the level."
        )
    elif auth.level is None:
        detail["message"] = f"The '{scope.value}' scope is not armed on this server."
    else:
        detail["level"] = auth.level.value
        detail["message"] = (
            f"Not allowed while the agent level is {auth.level.value.title()}: "
            "the operator sets the level from the Agent chip in AutoLamella."
        )
    return detail


def require_scope(scope: Scope, *alternatives: Scope):
    """Build a FastAPI dependency enforcing a valid token + an armed scope (any
    one of ``scope`` and ``alternatives``), and refusing everything while the
    agent level is Off.

    The agent token only: the dashboard token is refused here (403), so it
    reaches nothing beyond the routes guarded by :func:`require_monitoring`.
    """

    def dependency(request: Request) -> None:
        auth: AuthConfig = request.app.state.auth
        token = _bearer_token(request)
        if not _matches(token, auth.token):
            if _matches(token, auth.dashboard_token):
                raise _dashboard_refused()
            raise _unauthorized()
        granted = _granted.get()
        # A valid token is proof of life whatever happens next — a scope
        # refusal below is still the agent talking. An approved request's call
        # is the app's doing, not the agent's.
        if not granted:
            auth.mark_seen()
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
        if not any(auth.is_armed(s) or s in granted for s in (scope, *alternatives)):
            raise HTTPException(
                status_code=403,
                detail=_not_armed_detail(auth, scope),
            )

    return dependency


def require_monitoring():
    """The monitoring routes' dependency: the agent token as ``read``, or the
    dashboard token for a GET. The dashboard is not the agent, so it does not
    count as the agent being heard from."""
    as_agent = require_scope(Scope.READ)

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
    taking = control.take(
        current_actor() or AGENT, f"{request.method} {request.url.path}"
    )
    try:
        taking.__enter__()
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
    # Entered and exited by hand, not ``with``: an exception from the route is
    # thrown in here, and must reach FastAPI as it is.
    try:
        yield
    finally:
        taking.__exit__(None, None, None)
