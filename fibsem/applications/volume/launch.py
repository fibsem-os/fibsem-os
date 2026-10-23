"""``fibsem-volume``: the Volume host. It opens a volume experiment on a
microscope and serves it over the App Server, with no window.
``fibsem-volume-ui`` is the same host with the window (it hands in *window*).
The agent level starts at Watch; the operator raises it from the chat page."""

import argparse
import threading
from pathlib import Path
from typing import Callable, List, Optional

import fibsem.config as fibsem_cfg
from fibsem.applications.volume.experiment import VolumeExperiment
from fibsem.applications.volume.server import serve
from fibsem.server import headless
from fibsem.workflows.experiment import EXPERIMENT_FILENAME
from fibsem.workflows.session import AppSession

__all__ = ["main", "open_session"]


def open_session(
    args: argparse.Namespace,
    path: Optional[Path],
    create: bool = False,
    windowed: bool = False,
) -> AppSession:
    """A session connected as *args* say (`headless.open_session`), with the
    volume experiment at *path* open (none for None), made there first when
    *create*."""
    experiment = None
    if path is not None:
        path = Path(path)
        if create:
            experiment = VolumeExperiment.create(path.parent, path.name)
        else:
            experiment = VolumeExperiment.load(path)
    return headless.open_session(args, experiment, windowed=windowed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fibsem-volume",
        description="Serve a volume experiment over the App Server, with no window.",
        parents=[headless.connection_parser()],
    )
    parser.add_argument(
        "experiment",
        type=Path,
        nargs="?",
        help="The volume experiment's folder; with the window, none open "
        "when left out.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Make a new volume experiment at that folder.",
    )
    parser.add_argument(
        "--serve-microscope",
        action="store_true",
        default=None,
        help="Also serve the /microscope routes to the agent (default: the "
        "App Server preference).",
    )
    return parser


def main(
    argv: Optional[List[str]] = None,
    stop: Optional[threading.Event] = None,
    window: Optional[Callable[[AppSession], int]] = None,
) -> int:
    """Serve until Ctrl-C (or *stop* is set), then stop the run and the
    server; with *window*, ``window(session)`` until it closes."""
    args = _parser().parse_args(argv)
    folder = args.experiment
    if folder is None and window is None:
        print("error: name the volume experiment's folder.")
        return 1
    if folder is not None:
        if args.create and folder.exists():
            print(f"error: {folder} exists; leave out --create to open it.")
            return 1
        if not args.create and not (folder / EXPERIMENT_FILENAME).is_file():
            print(f"error: no experiment at {folder}; add --create to make one.")
            return 1
    session = open_session(
        args, folder, create=args.create, windowed=window is not None
    )
    session.serve_microscope = (
        args.serve_microscope
        if args.serve_microscope is not None
        else fibsem_cfg.load_user_preferences().app_server.microscope
    )
    if window is not None:
        return window(session)
    return headless.serve_until_stopped(session, serve, "fibsem-volume", stop)


if __name__ == "__main__":
    raise SystemExit(main())
