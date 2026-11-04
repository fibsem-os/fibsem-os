"""``fibsem-gateway``: run the gateway (with its tray icon when Qt is there),
install it to start at logon, open pairing, show whether the app runs, or
launch the app with the agent (`fibsem.gateway.launch`) and put a desktop
shortcut for that on the Desktop.

Microscope connection flags (``--manufacturer``, ``--ip-address``,
``--config``) after ``run`` are passed to the app as it starts, which then
connects at once; without them the operator connects in the window or on the
gateway's page.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from typing import List, Optional

from fibsem.gateway.agent import GatewayAgent
from fibsem.gateway.gateway import (
    GATEWAY_FILE,
    GATEWAY_PORT,
    Gateway,
    GatewayRefused,
    read_gateway_file,
)
from fibsem.gateway.host import HostMicroscope
from fibsem.server.operator_token import read_operator_token

__all__ = ["call", "main", "shortcut_command", "start_gateway"]

SHORTCUT_NAME = "fibsemOS Gateway"


def call(method: str, route: str, body: Optional[dict] = None, timeout: float = 3.0):
    """Call the gateway on this PC as the operator: its answer, or None when no
    gateway runs. Raises `urllib.error.HTTPError` for a refusal. Standard
    library only, so a desktop launch stays cheap to import."""
    found = read_gateway_file()
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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parser() -> argparse.ArgumentParser:
    from fibsem.gateway.launch import launch_options

    parser = argparse.ArgumentParser(
        prog="fibsem-gateway",
        description="Start and stop the app on this microscope PC, here or for a hub.",
    )
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser("run", help="Run the gateway (the default).")
    run.add_argument("--port", type=int, default=GATEWAY_PORT)
    run.add_argument("--no-tray", action="store_true", help="No tray icon.")
    install = commands.add_parser(
        "install", help="Start the gateway at logon (Windows: a Startup shortcut)."
    )
    install.add_argument("connection", nargs=argparse.REMAINDER)
    commands.add_parser("uninstall", help="Stop starting it at logon.")
    pair = commands.add_parser("pair", help="Open pairing and show the code.")
    pair.add_argument("--hub", action="store_true", help="Pair a hub.")
    pair.add_argument(
        "--may-switch",
        action="store_true",
        help="The hub may start and stop apps.",
    )
    commands.add_parser("status", help="Show whether the app runs.")
    launch = commands.add_parser(
        "launch",
        parents=[launch_options()],
        help="Start the app, connect the microscope, start the agent at Watch "
        "and open its page.",
    )
    launch.add_argument("app", nargs="?", default="autolamella")
    commands.add_parser(
        "shortcut", help='Put an "AutoLamella with the agent" shortcut on the Desktop.'
    )
    return parser


def _connection(extra: List[str]) -> List[str]:
    """*extra* checked as connection flags, passed on as given."""
    from fibsem.server.headless import connection_parser

    argparse.ArgumentParser(parents=[connection_parser()]).parse_args(extra)
    return extra


def run(port: int, tray: bool, connection: List[str]) -> int:
    import uvicorn

    from fibsem.gateway.server import GatewayAuth, build_gateway_app
    from fibsem.server.uvicorn_server import Server

    running = read_gateway_file()
    if running is not None:
        print(f"A gateway already runs here (pid {running['pid']}, {running['url']}).")
        return 1
    gateway = Gateway(connection)
    auth = GatewayAuth.from_disk()
    agent = GatewayAgent(gateway, auth.pairing.tokens)
    server = Server(
        uvicorn.Config(
            build_gateway_app(gateway, auth, agent, HostMicroscope()),
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    )
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, name="gateway-http", daemon=True)
    thread.start()
    GATEWAY_FILE.parent.mkdir(parents=True, exist_ok=True)
    GATEWAY_FILE.write_text(
        json.dumps({"url": f"http://127.0.0.1:{port}", "pid": os.getpid()})
    )
    gateway.start()
    print(f"Gateway at http://127.0.0.1:{port}/ui/gateway")
    try:
        if not (tray and _run_tray(gateway, auth, port, agent)):
            forever = threading.Event()
            # in short waits: Ctrl-C does not break a bare wait on Windows
            while not forever.wait(0.5):
                pass
    except KeyboardInterrupt:
        pass
    finally:
        quit_gateway(gateway)
        gateway.close()
        server.should_exit = True
        GATEWAY_FILE.unlink(missing_ok=True)
    return 0


def quit_gateway(gateway: Gateway) -> None:
    """Stop an app without its window that isn't running a workflow, so
    nothing is left holding the microscope without a gateway; a window, or
    a running workflow, is left alone."""
    try:
        gateway.stop("operator", "quit")
    except GatewayRefused as refusal:
        logging.warning("Gateway quit: %s", refusal)


def _run_tray(gateway: Gateway, auth, port: int, agent=None) -> bool:
    """The tray's event loop until Quit; False when Qt or a tray isn't there."""
    try:
        from fibsem.gateway.ui.tray import run_tray
    except ImportError:
        return False
    return run_tray(gateway, auth, port, agent)


def _startup_folder() -> Path:
    return (
        Path(os.environ.get("APPDATA", Path.home()))
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )


def pythonw() -> Path:
    """This environment's console-less Python (Windows)."""
    exe = Path(sys.executable)
    return exe.with_name("pythonw" + exe.suffix)


def start_gateway() -> None:
    """Run the gateway, with its tray, apart from this process (as the logon
    shortcut does, without connection flags)."""
    if sys.platform == "win32":
        python = pythonw()
        flags = {
            "creationflags": subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
        }
    else:
        python = Path(sys.executable)
        flags = {"start_new_session": True}
    subprocess.Popen(
        [str(python), "-m", "fibsem.gateway", "run"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path.home()),
        **flags,
    )


def shortcut_command(path: Path, connection: List[str]) -> List[str]:
    """PowerShell that writes a shortcut at *path* running the gateway with no
    console (``pythonw``)."""
    arguments = subprocess.list2cmdline(["-m", "fibsem.gateway", "run", *connection])

    def quoted(text: str) -> str:
        return "'" + text.replace("'", "''") + "'"

    script = (
        f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut({quoted(str(path))}); "
        f"$s.TargetPath = {quoted(str(pythonw()))}; "
        f"$s.Arguments = {quoted(arguments)}; "
        f"$s.WorkingDirectory = {quoted(str(Path.home()))}; "
        f"$s.Description = {quoted('Runs apps on this microscope PC for a hub')}; "
        "$s.Save()"
    )
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]


def install(connection: List[str]) -> int:
    if sys.platform != "win32":
        command = " ".join(["fibsem-gateway", "run", *connection])
        print(f"Starting at logon is set up on Windows only. Run: {command}")
        return 0
    path = _startup_folder() / f"{SHORTCUT_NAME}.lnk"
    subprocess.run(shortcut_command(path, connection), check=True)
    print(f"The gateway starts at logon: {path}")
    return 0


def uninstall() -> int:
    path = _startup_folder() / f"{SHORTCUT_NAME}.lnk"
    if path.exists():
        path.unlink()
        print(f"Removed {path}")
    else:
        print("The gateway isn't set to start at logon.")
    return 0


def agent_shortcut() -> int:
    """The "AutoLamella with the agent" desktop shortcut, running
    ``fibsem-autolamella-agent``."""
    from dataclasses import replace

    from fibsem.tools.apps import AUTOLAMELLA
    from fibsem.tools.desktop_shortcut import create_desktop_shortcut

    app = replace(
        AUTOLAMELLA,
        name=f"{AUTOLAMELLA.name} with the agent",
        command="fibsem-autolamella-agent",
    )
    try:
        path = create_desktop_shortcut(overwrite=True, app=app)
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as error:
        print(f"The shortcut wasn't made: {error}")
        return 1
    print(f"Shortcut made: {path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["run", *argv]
    args, extra = _parser().parse_known_args(argv)
    if args.command == "run":
        return run(args.port, not args.no_tray, _connection(extra))
    if extra:
        _parser().error(f"unrecognized arguments: {' '.join(extra)}")
    if args.command == "install":
        return install(_connection(args.connection))
    if args.command == "uninstall":
        return uninstall()
    if args.command == "launch":
        from fibsem.gateway.launch import launch_app

        return launch_app(args.app, args.configuration, args.profile)
    if args.command == "shortcut":
        return agent_shortcut()
    if args.command == "pair":
        body = {"role": "hub" if args.hub else "operator"}
        if args.hub and args.may_switch:
            body["grants"] = ["switch_modes"]
        listing = call("POST", "/pairing", body)
        if listing is None:
            print("No gateway runs here.")
            return 1
        print(f"Pairing code: {listing['code']} (5 minutes, {listing['role']}).")
        return 0
    state = call("GET", "/gateway")
    if state is None:
        print("No gateway runs here.")
        return 1
    print(json.dumps(state, indent=2))
    return 0
