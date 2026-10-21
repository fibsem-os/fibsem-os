"""For tests of the headless commands (``fibsem-volume``,
``fibsem-autolamella-server``): the App Server confined to a test port with its
discovery file under tmp, and a wait until the command is serving."""

import json
import socket
import threading
import time
from pathlib import Path
from typing import Callable, List

import httpx

from fibsem.client.pairing import ask_to_connect
from fibsem.server import hosting


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def confine_host(monkeypatch, tmp_path: Path) -> Path:
    """Every AppServerHost on a free port, its discovery file (and so the
    operator token beside it) under *tmp_path*. The discovery file's path."""
    real_host, port = hosting.AppServerHost, _free_port()
    path = tmp_path / "app-server.json"

    def confined(session, app_context, **kwargs):
        return real_host(
            session,
            app_context,
            port=port,
            discovery_path=path,
            serve_microscope=kwargs.get("serve_microscope", False),
        )

    monkeypatch.setattr(hosting, "AppServerHost", confined)
    return path


def serving(discovery: Path, timeout: float = 60) -> dict:
    """The discovery file's contents, once the command is serving."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if discovery.is_file():
            return json.loads(discovery.read_text())
        time.sleep(0.1)
    raise AssertionError("the command never served")


def run_in_background(main: Callable, argv: List[str]):
    """Start ``main(argv, stop)`` on a thread. The stop event, the thread and a
    list that gets its exit code."""
    stop, result = threading.Event(), []
    worker = threading.Thread(target=lambda: result.append(main(argv, stop)))
    worker.start()
    return stop, worker, result


def allowed_agent(found: dict, operator_token: str, timeout: float = 30) -> str:
    """An agent on this PC asks to connect with the discovery file's token, and
    the operator allows it through their route, as the web app's card does.
    The agent's token."""
    url, tokens = found["url"], []
    asking = threading.Thread(
        target=lambda: tokens.append(
            ask_to_connect(url, found["connect_token"], "fibsem-test/1", timeout)
        )
    )
    asking.start()
    operator = {"Authorization": f"Bearer {operator_token}"}
    deadline = time.monotonic() + timeout
    with httpx.Client(base_url=url, headers=operator) as client:
        while time.monotonic() < deadline:
            ask = client.get("/app/agent").json().get("asking")
            if ask is not None:
                allowed = client.post(f"/app/agent/connections/{ask['id']}/allow")
                assert allowed.is_success, allowed.text
                break
            time.sleep(0.05)
    asking.join(timeout)
    assert tokens and tokens[0], "the agent was never allowed"
    return tokens[0]
