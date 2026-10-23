"""Serve an app session over the App Server with no window, until Ctrl-C: the
part every app's headless command shares (``fibsem-volume``,
``fibsem-autolamella-server``). Each app opens its own experiment."""

import argparse
import logging
import signal
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from fibsem import utils
from fibsem.server.agent_setup import setup_text
from fibsem.workflows.session import AppSession

__all__ = ["connect", "connection_parser", "open_session", "serve_until_stopped"]


def connection_parser() -> argparse.ArgumentParser:
    """The flags that name the microscope a headless command connects to, as a
    parent parser: ``--manufacturer``, ``--ip-address``, ``--config``,
    ``--debug``. A configuration file's manufacturer and address stand unless
    a flag names another; with neither, the Demo microscope."""
    from fibsem.drivers import registry

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--manufacturer",
        default=None,
        # normalise BEFORE the choices check, so historical spellings
        # (--manufacturer Thermo) keep working while --help shows canonical
        type=registry.normalize_name,
        choices=registry.registered_manufacturers(),
        help="Microscope manufacturer (default: the configuration's, or Demo)",
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
        help="Path to microscope config YAML",
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p


def connect(args: argparse.Namespace) -> Any:
    """The microscope the connection flags name (`connection_parser`)."""
    manufacturer = args.manufacturer
    if manufacturer is None and args.config_path is None:
        manufacturer = "Demo"
    microscope, _ = utils.setup_session(
        manufacturer=manufacturer,
        ip_address=args.ip_address,
        config_path=args.config_path,
        debug=args.debug,
    )
    return microscope


def open_session(microscope, experiment=None) -> AppSession:
    """A session on *microscope* with *experiment* open (when given) and its
    events recorded."""
    session = AppSession()
    session.microscope = microscope
    session.start_recording()
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
        print("error: the App Server did not start; see the log.")
        return 1
    host = session.app_server
    experiment = session.experiment
    what = experiment.name if experiment is not None else "no experiment"
    print(f"Serving {what} at {host.url} (agent at Watch).")
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
