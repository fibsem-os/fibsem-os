"""The fibsem host: one process per microscope that starts an app's session and
its App Server, connects the microscope, and either serves until Ctrl-C or
opens the app's window on the same session. The part every app's command
shares (``fibsem-volume``, ``fibsem-autolamella-server``, and the window
commands, which hand in the window). Each app opens its own experiment."""

import argparse
import logging
import signal
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from fibsem import utils
from fibsem.server.agent_setup import setup_text
from fibsem.workflows.session import AppSession

__all__ = [
    "connect",
    "connection_parser",
    "names_a_microscope",
    "open_session",
    "serve_until_stopped",
]


def connection_parser() -> argparse.ArgumentParser:
    """The flags that name the microscope a headless command connects to, as a
    parent parser: ``--manufacturer``, ``--ip-address``, ``--config``,
    ``--debug``. A configuration file's manufacturer and address stand unless
    a flag names another; with no file, the default configuration's, as the
    app's own Connect uses (``--manufacturer Demo`` for the Demo microscope)."""
    from fibsem.drivers import registry

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--manufacturer",
        default=None,
        # normalise BEFORE the choices check, so historical spellings
        # (--manufacturer Thermo) keep working while --help shows canonical
        type=registry.normalize_name,
        choices=registry.registered_manufacturers(),
        help="Microscope manufacturer (default: the configuration's)",
    )
    p.add_argument(
        "--ip-address",
        default=None,
        dest="ip_address",
        help="Microscope IP address (default: the configuration's)",
    )
    p.add_argument(
        "--config",
        default=None,
        dest="config_path",
        type=Path,
        metavar="PATH",
        help="Path to microscope config YAML (default: the default configuration)",
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p


def names_a_microscope(args: argparse.Namespace) -> bool:
    """Whether a connection flag names the microscope (not only --debug)."""
    return any(
        value is not None
        for value in (args.manufacturer, args.ip_address, args.config_path)
    )


def connect(session: AppSession, args: argparse.Namespace) -> Any:
    """Connect *session* to the microscope the connection flags name
    (`connection_parser`)."""
    return session.connect(
        args.config_path,
        manufacturer=args.manufacturer,
        ip_address=args.ip_address,
        debug=args.debug,
    )


def open_session(
    args: argparse.Namespace, experiment=None, windowed: bool = False
) -> AppSession:
    """A session connected as *args* say, with *experiment* open (when given).
    Headless it always connects (the default configuration without a flag);
    *windowed*, only when a flag names a microscope (otherwise the window's
    Connect does)."""
    session = AppSession()
    if not windowed or names_a_microscope(args):
        connect(session, args)
    if experiment is not None:
        session.adopt(experiment)
    return session


def announce_code(code: str) -> None:
    """A code a browser asked for, shown only in this terminal."""
    print(f"A browser asked to pair. Pairing code: {code} (5 minutes).", flush=True)


def serve_until_stopped(
    session,
    serve: Callable[[Any], bool],
    prog: str,
    stop: Optional[threading.Event] = None,
) -> int:
    """``serve(session)``, print where to reach it, then wait for Ctrl-C (or
    *stop*, or a terminate signal: a service stop, a Windows close) and stop
    the run, the server and the recording. The exit code."""
    if not serve(session):
        print("error: the host did not start serving; see the log.")
        return 1
    host = session.app_server
    experiment = session.experiment
    what = experiment.name if experiment is not None else "no experiment"
    print(f"The fibsem host is serving {what} at {host.url} (agent at Watch).")
    # The plain address, never the token: scrollback and service logs keep it.
    print(f"Chat and level controls: {host.url}/ui/agent (Pair this browser)")
    pairing = host.auth.pairing
    if pairing is not None:
        # Seeing this terminal is the proof: the code shows only here.
        pairing.announce = announce_code
    print(setup_text(host.url) or "fibsem-mcp is not installed beside this Python.")
    stop = stop or threading.Event()
    _stop_on_terminate(stop)
    try:
        # in short waits: Ctrl-C does not break a bare wait on Windows
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    logging.info(f"{prog}: stopping")
    session.runner.stop()
    session.runner.join(30)
    session.stop_app_server()
    session.stop_recording()
    return 0


def _stop_on_terminate(stop: threading.Event) -> None:
    """Set *stop* on SIGTERM (and SIGBREAK, a Windows console close), so a
    terminate ends as Ctrl-C does. Only the main thread can install it."""
    if threading.current_thread() is not threading.main_thread():
        return
    for name in ("SIGTERM", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, lambda *_: stop.set())
