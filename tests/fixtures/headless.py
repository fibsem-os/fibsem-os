"""For tests of the headless commands (``fibsem-volume``,
``fibsem-autolamella-server``): the App Server confined to a test port with its
discovery file under tmp, and a wait until the command is serving."""

import json
import socket
import threading
import time
from pathlib import Path
from typing import Callable, List

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
