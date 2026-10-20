"""``fibsem-autolamella-server``: open an AutoLamella experiment on a microscope
and serve it over the App Server, with no window. The agent level starts at
Watch; the operator raises it from the chat page. Set up in the window first
(positions, patterns, calibration), quit it, then serve the same experiment."""

import argparse
import threading
from pathlib import Path
from typing import List, Optional

import fibsem.config as fibsem_cfg
from fibsem.applications.autolamella import experiment_setup
from fibsem.applications.autolamella.server.context import AutoLamellaAppContext
from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
from fibsem.cli import _build_connection_parser
from fibsem.server import headless
from fibsem.workflows.experiment import EXPERIMENT_FILENAME

__all__ = ["main", "serve"]


def serve(session) -> bool:
    """Host the App Server over an AutoLamella *session* with no window.
    Whether it is serving (see ``AppSession.start_app_server``)."""
    return session.start_app_server(
        lambda buffer: AutoLamellaAppContext(session, event_buffer=buffer)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fibsem-autolamella-server",
        description="Serve an AutoLamella experiment over the App Server, "
        "with no window.",
        parents=[_build_connection_parser()],
    )
    parser.add_argument(
        "experiment",
        type=Path,
        nargs="?",
        help="The experiment's folder; the most recent one when left out.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Make a new experiment at that folder.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        help="The new experiment's protocol file; the default protocol when left out.",
    )
    parser.add_argument(
        "--serve-microscope",
        action="store_true",
        default=None,
        help="Also serve the /microscope routes to the agent (default: the "
        "App Server preference).",
    )
    return parser


def _experiment(args: argparse.Namespace):
    """The experiment to serve, or None and why not."""
    folder = args.experiment
    if args.create:
        if folder is None:
            return None, "--create needs the new experiment's folder."
        if folder.exists():
            return None, f"{folder} exists; leave out --create to open it."
        protocol = args.protocol or experiment_setup.default_protocol_path()
        try:
            task_protocol = AutoLamellaTaskProtocol.load(str(protocol))
        except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
            return None, f"could not load the protocol {protocol}: {exc}"
        return experiment_setup.create_experiment(
            str(folder.parent), folder.name, task_protocol
        ), None
    if args.protocol is not None:
        return None, "--protocol is for --create; an experiment keeps its own."
    if folder is None:
        last = fibsem_cfg.get_last_experiment_file()
        if last is None:
            return None, "no recent experiment; name one, or --create one."
        path = Path(last)
    else:
        path = folder / EXPERIMENT_FILENAME
        if not path.is_file():
            return None, f"no experiment at {folder}; add --create to make one."
    experiment, why = experiment_setup.load_for_adoption(path)
    if experiment is not None:
        experiment_setup.record_recent(experiment)
    return experiment, why


def main(
    argv: Optional[List[str]] = None, stop: Optional[threading.Event] = None
) -> int:
    """Serve until Ctrl-C (or *stop* is set), then stop the run and the server."""
    args = _parser().parse_args(argv)
    experiment, why = _experiment(args)
    if experiment is None:
        print(f"error: {why}")
        return 1
    session = headless.open_session(headless.connect(args), experiment)
    session.serve_microscope = (
        args.serve_microscope
        if args.serve_microscope is not None
        else fibsem_cfg.load_user_preferences().app_server.microscope
    )
    return headless.serve_until_stopped(
        session, serve, "fibsem-autolamella-server", stop
    )
