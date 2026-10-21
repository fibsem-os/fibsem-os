"""This computer's side of pairing (`fibsem.server.pairing`): send the code the
microscope PC shows, and keep the token it answers for later connections.

The token is kept per server, written ``0600``: an operator's in
``~/.fibsem/remote/<host>_<port>.json`` (`RemoteMicroscope` reads it), an
agent's in ``~/.fibsem/agent/`` (the sidecar reads it, through `resolve`), so
an agent never picks up the operator's. The url defaults to the SSH tunnel's
end, ``http://localhost:8001``.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import httpx

from fibsem.server.discovery import DISCOVERY_DIR

__all__ = ["DEFAULT_URL", "agent_token", "pair", "paired_token"]

DEFAULT_URL = "http://localhost:8001"
REMOTE_DIR = DISCOVERY_DIR / "remote"
AGENT_DIR = DISCOVERY_DIR / "agent"


def _file(url: str, folder: Optional[Path] = None) -> Path:
    split = urlsplit(url)
    port = split.port or (443 if split.scheme == "https" else 80)
    return Path(folder or REMOTE_DIR) / f"{split.hostname}_{port}.json"


def paired_token(url: str, folder: Optional[Path] = None) -> Optional[str]:
    """The operator token this computer was paired with at *url*, or None."""
    try:
        with open(_file(url, folder)) as f:
            token = json.load(f).get("token")
    except (OSError, ValueError, AttributeError):
        return None
    return token if isinstance(token, str) and token else None


def agent_token(url: str) -> Optional[str]:
    """The agent token this computer was paired with at *url*, or None."""
    return paired_token(url, AGENT_DIR)


def pair(
    code: str,
    url: str = DEFAULT_URL,
    name: Optional[str] = None,
    folder: Optional[Path] = None,
) -> "tuple[str, str]":
    """Pair this computer with the App Server at *url* using the *code* it
    shows, keep the token and return it with its role (``operator`` or
    ``agent``, as pairing was opened). *name* is what the paired list calls
    this computer: its host name by default. A refusal (closed, expired, wrong
    code) raises ``PermissionError`` with the server's words."""
    url = url.rstrip("/")
    response = httpx.post(
        f"{url}/pair",
        json={"code": str(code), "name": name or socket.gethostname()},
        timeout=10.0,
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", {})
            message = detail.get("message", response.text)
        except (ValueError, AttributeError):
            message = response.text
        raise PermissionError(f"Pairing refused: {message}")
    token = response.json()["token"]
    role = response.json().get("role", "operator")
    if folder is None:
        folder = AGENT_DIR if role == "agent" else REMOTE_DIR
    path = _file(url, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Created user-only, never readable by others even for a moment.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"url": url, "token": token}, f, indent=2)
    os.chmod(path, 0o600)
    return token, role
