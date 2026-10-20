"""``fibsem-volume``: open a volume experiment on a microscope and serve it over
the App Server, with no window. The agent level starts at Watch; the operator
raises it from the chat page."""

import argparse
import logging
import threading
from pathlib import Path
from typing import List, Optional

from fibsem import utils
from fibsem.applications.volume.experiment import VolumeExperiment
from fibsem.applications.volume.server import serve
from fibsem.cli import _build_connection_parser
from fibsem.server.agent_setup import setup_text
from fibsem.server.hosting import chat_url
from fibsem.workflows.experiment import EXPERIMENT_FILENAME
from fibsem.workflows.session import AppSession

__all__ = ["main", "open_session"]


def open_session(microscope, path: Path, create: bool = False) -> AppSession:
    """A session on *microscope* with the volume experiment at *path* open,
    made there first when *create*. Its events are recorded."""
    path = Path(path)
    if create:
        experiment = VolumeExperiment.create(path.parent, path.name)
    else:
        experiment = VolumeExperiment.load(path)
    session = AppSession()
    session.microscope = microscope
    session.start_recording()
    session.adopt(experiment)
    return session


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fibsem-volume",
        description="Serve a volume experiment over the App Server, with no window.",
        parents=[_build_connection_parser()],
    )
    parser.add_argument("experiment", type=Path, help="The volume experiment's folder.")
    parser.add_argument(
        "--create",
        action="store_true",
        help="Make a new volume experiment at that folder.",
    )
    parser.add_argument(
        "--serve-microscope",
        action="store_true",
        help="Also serve the /microscope routes to the agent.",
    )
    return parser


def main(
    argv: Optional[List[str]] = None, stop: Optional[threading.Event] = None
) -> int:
    """Serve until Ctrl-C (or *stop* is set), then stop the run and the server."""
    args = _parser().parse_args(argv)
    folder = args.experiment
    if args.create and folder.exists():
        print(f"error: {folder} exists; leave out --create to open it.")
        return 1
    if not args.create and not (folder / EXPERIMENT_FILENAME).is_file():
        print(f"error: no experiment at {folder}; add --create to make one.")
        return 1
    microscope, _ = utils.setup_session(
        manufacturer=args.manufacturer,
        ip_address=args.ip_address,
        config_path=args.config_path,
        debug=args.debug,
    )
    session = open_session(microscope, folder, create=args.create)
    session.serve_microscope = args.serve_microscope
    if not serve(session):
        print("error: the App Server did not start; see the log.")
        return 1
    host = session.app_server
    print(f"Serving {session.experiment.name} at {host.url} (agent at Watch).")
    print(f"Chat and level controls: {chat_url(host)}")
    print(setup_text(host.url) or "fibsem-mcp is not installed beside this Python.")
    stop = stop or threading.Event()
    try:
        # in short waits: Ctrl-C does not break a bare wait on Windows
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    logging.info("fibsem-volume: stopping")
    session.runner.stop()
    session.runner.join(30)
    session.stop_app_server()
    session.stop_recording()
    return 0
