"""Serve an app session over the App Server with no window, until Ctrl-C: the
part every app's headless command shares (``fibsem-volume``,
``fibsem-autolamella-server``). Each app opens its own experiment."""

import argparse
import logging
import threading
from typing import Any, Callable, Optional

from fibsem import utils
from fibsem.server.agent_setup import setup_text
from fibsem.server.hosting import chat_url
from fibsem.workflows.session import AppSession

__all__ = ["connect", "open_session", "serve_until_stopped"]


def connect(args: argparse.Namespace) -> Any:
    """The microscope the connection flags name (``fibsem.cli``'s, Demo by
    default)."""
    microscope, _ = utils.setup_session(
        manufacturer=args.manufacturer,
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


def serve_until_stopped(
    session,
    serve: Callable[[Any], bool],
    prog: str,
    stop: Optional[threading.Event] = None,
) -> int:
    """``serve(session)``, print where to reach it, then wait for Ctrl-C (or
    *stop*) and stop the run, the server and the recording. The exit code."""
    if not serve(session):
        print("error: the App Server did not start; see the log.")
        return 1
    host = session.app_server
    experiment = session.experiment
    what = experiment.name if experiment is not None else "no experiment"
    print(f"Serving {what} at {host.url} (agent at Watch).")
    print(f"Chat and level controls: {chat_url(host)}")
    print(setup_text(host.url) or "fibsem-mcp is not installed beside this Python.")
    stop = stop or threading.Event()
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
