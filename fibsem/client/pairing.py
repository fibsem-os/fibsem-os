"""This computer's side of pairing (`fibsem.server.pairing`): send the code the
microscope PC shows, and keep the token it answers for later connections.

An agent on the microscope PC pairs by asking instead (`ask_to_connect`, with
the discovery file's connect token), and the operator allows it in the app.

The token is kept per server, written ``0600``: an operator's in
``~/.fibsem/remote/<host>_<port>.json`` (`RemoteMicroscope` reads it), an
agent's in ``~/.fibsem/agent/`` (the sidecar reads it, through `resolve`), so
an agent never picks up the operator's. The url defaults to the SSH tunnel's
end, ``http://localhost:8001``.
"""

from __future__ import annotations

import getpass
import json
import os
import socket
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import httpx

from fibsem.server.discovery import DISCOVERY_DIR

__all__ = [
    "DEFAULT_URL",
    "agent_token",
    "ask_to_connect",
    "device_server_token",
    "forget_agent_token",
    "keep_agent_token",
    "pair",
    "paired_token",
]

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


def forget_agent_token(url: str) -> None:
    """Drop the agent token kept for *url*: the operator unpaired it."""
    try:
        _file(url, AGENT_DIR).unlink()
    except OSError:
        pass


def _keep(url: str, token: str, folder: Path) -> None:
    path = _file(url, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Created user-only, never readable by others even for a moment.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"url": url, "token": token}, f, indent=2)
    os.chmod(path, 0o600)


def keep_agent_token(url: str, token: str) -> None:
    """Keep the agent token *url* gave this computer, for `agent_token`."""
    _keep(url.rstrip("/"), token, AGENT_DIR)


def _client_name(user_agent: str) -> str:
    """The client as it names itself, from a User-Agent: the MCP client the
    sidecar serves ("claude-code 2.1.3"), else the program ("fibsem-client")."""
    products = user_agent.split()
    product = products[1] if len(products) > 1 else (products or ["an agent"])[0]
    return product.replace("/", " ")


def _origin(environ: Optional[dict] = None) -> str:
    """Where the asking program connects from: this PC, or an address over SSH."""
    environ = os.environ if environ is None else environ
    ssh = environ.get("SSH_CONNECTION", "").split()
    return f"{ssh[0]} over SSH" if ssh else "this PC"


def ask_to_connect(
    url: str,
    connect_token: str,
    user_agent: str,
    wait_s: float = 120.0,
    post=None,
) -> Optional[str]:
    """Ask the App Server at *url* to let this agent connect, waiting up to
    *wait_s* for the operator's Allow. The agent token it answers is kept
    (`agent_token` reads it) and returned; None while the operator hasn't
    answered. A Deny raises ``PermissionError`` with the operator's words."""
    url = url.rstrip("/")
    post = post or httpx.post
    body = {
        "client": _client_name(user_agent),
        "origin": _origin(),
        "user": getpass.getuser(),
    }
    deadline = time.monotonic() + wait_s
    while True:
        left = deadline - time.monotonic()
        response = post(
            f"{url}/connect",
            json={**body, "wait_s": max(0.0, min(25.0, left))},
            headers={
                "Authorization": f"Bearer {connect_token}",
                "User-Agent": user_agent,
            },
            timeout=40.0,
        )
        if response.status_code != 200:
            try:
                message = response.json()["detail"]["message"]
            except (ValueError, KeyError, TypeError):
                message = response.text
            raise PermissionError(f"Connecting refused: {message}")
        answer = response.json()
        if answer.get("state") == "allowed":
            keep_agent_token(url, answer["token"])
            return answer["token"]
        if answer.get("state") == "denied":
            raise PermissionError(answer.get("message", "The operator denied it."))
        if time.monotonic() >= deadline:
            return None


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
    _keep(url, token, folder)
    return token, role


def device_server_token(url: str) -> Optional[str]:
    """The token for the Device Server at *url*: the one this computer was
    paired with there (``fibsem-cli pair-device``), else the token of a Device
    Server on this computer (`fibsem.server.device_auth`). None when there is
    none."""
    from fibsem.server.device_auth import read_device_token

    paired = paired_token(url)
    if paired is not None:
        return paired
    return read_device_token()
