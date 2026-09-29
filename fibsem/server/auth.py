"""Authentication and scope gating for the fibsem server (FIB-852).

One bearer token per server instance, three scopes:

- ``read``: granted to every valid token. Observation only.
- ``control``: app-level workflow control. Armed by the hosting (GUI toggle in
  the embedded app; unused in bench hosting until an app router exists).
- ``configure``: editing task parameters (protocol and per-item task configs).
  Its own rung, deliberately separate from ``control``: answering a question
  with geometry and editing protocols are different grants (FIB-864).
- ``hardware``: anything that commands or mutates the microscope. Armed
  explicitly by the hosting (GUI toggle, or ``--arm-hardware`` on the CLI).

Fail closed: a server always has a token (generated when the hosting does not
supply one), and only ``read`` is armed unless the hosting says otherwise.
"""

import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Iterable, Optional

from fastapi import HTTPException, Request


class Scope(str, Enum):
    READ = "read"
    CONTROL = "control"
    CONFIGURE = "configure"
    HARDWARE = "hardware"


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
    # Monotonic time of the last request that presented the valid token; None
    # until one arrives. The connected agent's proof of life: its event
    # long-poll alone touches the server every half-minute, so staleness here
    # means nobody is on the other end (used to hand a parked question to the
    # operator without waiting out the full hand-over time).
    last_seen_monotonic: Optional[float] = None

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
        return cls(token=token or secrets.token_urlsafe(32), armed=frozenset(armed))

    def is_armed(self, scope: Scope) -> bool:
        return scope is Scope.READ or scope in self.armed

    def armed_scopes(self) -> Iterable[Scope]:
        return (s for s in Scope if self.is_armed(s))


def _bearer_token(request: Request) -> Optional[str]:
    header = request.headers.get("Authorization")
    if header is None or not header.startswith("Bearer "):
        return None
    return header[len("Bearer ") :]


def require_scope(scope: Scope):
    """Build a FastAPI dependency enforcing a valid token + an armed scope."""

    def dependency(request: Request) -> None:
        auth: AuthConfig = request.app.state.auth
        token = _bearer_token(request)
        if token is None or not hmac.compare_digest(token, auth.token):
            raise HTTPException(
                status_code=401,
                detail={
                    "error_type": "unauthorized",
                    "message": "Missing or invalid bearer token.",
                },
            )
        # A valid token is proof of life whatever happens next — a scope
        # refusal below is still the agent talking.
        auth.mark_seen()
        if not auth.is_armed(scope):
            raise HTTPException(
                status_code=403,
                detail={
                    "error_type": "scope_not_armed",
                    "scope": scope.value,
                    "message": f"The '{scope.value}' scope is not armed on this server.",
                },
            )

    return dependency


def command_slot(request: Request):
    """Serialize hardware commands: one in flight per microscope, 409 otherwise.

    Concurrent requests on the threadpool would interleave multi-call vendor
    operations (the FIB-517/542/569 incident class). A structured refusal beats
    queueing: a caller blocked behind an hour-long mill should be told, not held.
    """
    lock: threading.Lock = request.app.state.command_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail={
                "error_type": "busy",
                "message": "Another hardware command is in progress on this microscope.",
            },
        )
    try:
        yield
    finally:
        lock.release()
