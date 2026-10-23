"""``fibsem-volume``: open a volume experiment on a microscope and serve it over
the App Server, with no window. The agent level starts at Watch; the operator
raises it from the web app."""

import argparse
import threading
from pathlib import Path
from typing import List, Optional

from fibsem.applications.volume.experiment import VolumeExperiment
from fibsem.applications.volume.server import serve
from fibsem.server import headless
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
    return headless.open_session(microscope, experiment)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fibsem-volume",
        description="Serve a volume experiment over the App Server, with no window.",
        parents=[headless.connection_parser()],
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
    session = open_session(headless.connect(args), folder, create=args.create)
    session.serve_microscope = args.serve_microscope
    return headless.serve_until_stopped(session, serve, "fibsem-volume", stop)


if __name__ == "__main__":
    raise SystemExit(main())
