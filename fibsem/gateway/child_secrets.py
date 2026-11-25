"""The secrets the gateway hands the servers it starts (`fibsem.server.gateway_secret`):
a new one at each start, through the server's environment, kept here in memory
by the server's process id and never written down. Every call the gateway
makes to such a server, or passes on to it, carries its secret.

A server the gateway didn't start (an App Server whose window was opened by
hand) has its own secret in its discovery file instead, for agents' calls.
"""

from __future__ import annotations

import os
import secrets
import threading
from typing import Dict, Optional

from fibsem.server import discovery
from fibsem.server.gateway_secret import ENV, HEADER

__all__ = ["environment", "for_server", "for_url", "forget", "headers", "keep", "make"]

_lock = threading.Lock()
_kept: Dict[int, str] = {}


def make() -> str:
    """A new secret, for one start of one server."""
    return secrets.token_urlsafe(32)


def environment(secret: str, base: Optional[dict] = None) -> dict:
    """The environment to start a server with: *base* (else this process's)
    and *secret*."""
    env = dict(os.environ if base is None else base)
    env[ENV] = secret
    return env


def keep(pid: int, secret: str) -> None:
    """Remember the secret the server running as *pid* was started with."""
    with _lock:
        _kept[pid] = secret


def forget(pid: Optional[int]) -> None:
    """The server running as *pid* ended: its secret goes with it."""
    if pid is None:
        return
    with _lock:
        _kept.pop(pid, None)


def for_server(found: Optional[dict]) -> Optional[str]:
    """The secret for the server a discovery file (*found*) describes: the
    one the gateway started it with, else the one it wrote itself, else None."""
    if not found:
        return None
    with _lock:
        handed = _kept.get(found.get("pid"))
    return handed or found.get("gateway_secret")


def for_url(url: str) -> Optional[str]:
    """The secret for the App Server or agent host at *url* on this PC."""
    want = url.rstrip("/")
    for path in (discovery.DISCOVERY_FILE, discovery.HOST_DISCOVERY_FILE):
        found = discovery.read_discovery_file(path)
        if found and str(found.get("url", "")).rstrip("/") == want:
            return for_server(found)
    return None


def headers(url: str) -> dict:
    """The header to add to a call to the server at *url*: its secret, or
    nothing when it has none."""
    secret = for_url(url)
    return {HEADER: secret} if secret else {}
