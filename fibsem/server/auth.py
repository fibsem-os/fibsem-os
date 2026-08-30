"""Authentication and scope gating for the fibsem server (FIB-852).

One bearer token per server instance, three scopes:

- ``read``: granted to every valid token. Observation only.
- ``control``: app-level workflow control. Armed by the hosting (GUI toggle in
  the embedded app; unused in bench hosting until an app router exists).
- ``hardware``: anything that commands or mutates the microscope. Armed
  explicitly by the hosting (GUI toggle, or ``--arm-hardware`` on the CLI).

Fail closed: a server always has a token (generated when the hosting does not
supply one), and only ``read`` is armed unless the hosting says otherwise.
"""

import hmac
import secrets
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Iterable, Optional

from fastapi import HTTPException, Request


class Scope(str, Enum):
    READ = "read"
    CONTROL = "control"
    HARDWARE = "hardware"


@dataclass(frozen=True)
class AuthConfig:
    """Token + armed scopes for one server instance.

    ``read`` is implicitly armed for any valid token and need not be listed.
    """

    token: str
    armed: FrozenSet[Scope] = field(default_factory=frozenset)

    @classmethod
    def generate(
        cls,
        arm_control: bool = False,
        arm_hardware: bool = False,
        token: Optional[str] = None,
    ) -> "AuthConfig":
        armed = set()
        if arm_control:
            armed.add(Scope.CONTROL)
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
