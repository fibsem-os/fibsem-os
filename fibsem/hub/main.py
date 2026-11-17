"""``fibsem-hub``: run the hub on this PC, and ask the one that runs.

``fibsem-hub`` (or ``run``) runs it until Ctrl-C. ``status`` lists the
instruments as the running hub last read them, ``pair`` shows a code that
signs a browser on another computer in, ``open`` opens the Lab Hub page signed
in, and ``stop`` stops it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional, Sequence

from fibsem.hub.agents import HubAgents
from fibsem.hub.hub import Hub
from fibsem.hub.server import build_hub_app, hub_auth
from fibsem.server.discovery import DISCOVERY_DIR
from fibsem.server.operator_token import read_operator_token
from fibsem.server.uvicorn_server import Server
from fibsem.util.process import pid_alive

logger = logging.getLogger(__name__)

HUB_PORT = 8020
# The running hub's ``{"url", "pid"}``.
RUNNING_FILE = DISCOVERY_DIR / "hub.json"


def read_running(path: Optional[Path] = None) -> Optional[dict]:
    """The running hub's ``{"url", "pid"}``, or None when none runs."""
    try:
        data = json.loads((path or RUNNING_FILE).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "url" not in data:
        return None
    return data if pid_alive(data.get("pid")) else None


def _ask(route: str, method: str = "GET", body: Optional[dict] = None):
    """The running hub's answer to *route*, as the operator; None when no hub
    runs here."""
    found = read_running()
    if found is None:
        return None
    request = urllib.request.Request(
        found["url"] + route,
        method=method,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {read_operator_token()}",
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fibsem-hub",
        description="Reach a lab's microscope PCs from this one. Run it on a PC "
        "away from the microscopes: each microscope PC runs fibsem-gateway.",
    )
    commands = parser.add_subparsers(dest="command")
    run_ = commands.add_parser("run", help="Run the hub (the default).")
    run_.add_argument("--port", type=int, default=HUB_PORT)
    commands.add_parser("status", help="List the instruments as last read.")
    commands.add_parser("pair", help="Show a code that signs a browser in.")
    commands.add_parser("stop", help="Stop the hub.")
    commands.add_parser("open", help="Open the Lab Hub page, signed in.")
    return parser


def run(port: int) -> int:
    import uvicorn

    running = read_running()
    if running is not None:
        print(f"A hub already runs here (pid {running['pid']}, {running['url']}).")
        return 1
    hub = Hub()
    agents = HubAgents(hub)
    stopping = threading.Event()
    server = Server(
        uvicorn.Config(
            build_hub_app(hub, hub_auth(), stopping.set, agents),
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    )
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, name="hub-http", daemon=True)
    thread.start()
    RUNNING_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUNNING_FILE.write_text(
        json.dumps({"url": f"http://127.0.0.1:{port}", "pid": os.getpid()})
    )
    hub.start()
    print(f"Hub at http://127.0.0.1:{port}/ui/hub")
    try:
        # in short waits: Ctrl-C does not break a bare wait on Windows
        while not stopping.wait(1.0):
            # notice agents that ended, and resume the ones that may
            try:
                agents.poll()
            except Exception:
                logger.exception("The hub could not check its agents")
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()
        server.should_exit = True
        RUNNING_FILE.unlink(missing_ok=True)
        thread.join(10)
    return 0


def _status() -> int:
    state = _ask("/hub")
    if state is None:
        print("No hub runs here. Start it with: fibsem-hub")
        return 1
    if not state["instruments"]:
        print("No instruments yet.")
    for row in state["instruments"]:
        if not row["paired"]:
            what = "not paired"
        elif not row["reachable"]:
            what = f"not reachable: {row['why']}" if row["why"] else "not reachable"
        else:
            gateway = row["gateway"] or {}
            what = gateway.get("app") or gateway.get("mode") or "reachable"
        print(f"{row['name']}: {what}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        if args.command in (None, "run"):
            return run(getattr(args, "port", HUB_PORT))
        if args.command == "status":
            return _status()
        if args.command == "pair":
            answer = _ask("/pairing", "POST", {"role": "operator"})
            if answer is None:
                print("No hub runs here. Start it with: fibsem-hub")
                return 1
            print(f"Pairing code: {answer['code']} (5 minutes).")
            return 0
        if args.command == "open":
            found = read_running()
            if found is None:
                print("No hub runs here. Start it with: fibsem-hub")
                return 1
            webbrowser.open(
                f"{found['url']}/ui/hub#token={read_operator_token() or ''}"
            )
            return 0
        if args.command == "stop":
            if _ask("/hub/quit", "POST", {}) is None:
                print("No hub runs here.")
                return 1
            print("The hub is stopping.")
            return 0
    except urllib.error.URLError as error:
        print(f"The hub didn't answer: {error}")
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
