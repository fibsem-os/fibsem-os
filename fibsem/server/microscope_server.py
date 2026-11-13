"""``fibsem-microscope-server``: the hardware coordinator on its own. It connects
the microscope and serves ``/microscope`` and pairing, with no app, no
experiment and no agent, until Ctrl-C.

Run it on the support PC. Another computer pairs once (``--pair`` prints a
code) and connects with a configuration that says ``info.driver: remote``; its
own App Server runs the apps and keeps the experiment.

::

    fibsem-microscope-server --config microscope.yaml --pair
"""

import argparse
import threading
from typing import List, Optional

from fibsem.server import headless
from fibsem.workflows.session import AppSession

__all__ = ["main", "serve"]

PROG = "fibsem-microscope-server"


def serve(session) -> bool:
    """Host the microscope over *session*: no ``/app`` view, no agent."""
    session.serve_microscope = True
    session.serve_agent = False
    return session.start_app_server(lambda: None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Serve the microscope to another computer's App Server, with "
        "no app, experiment or agent.",
        parents=[headless.connection_parser()],
    )
    parser.add_argument(
        "--pair",
        action="store_true",
        help="Show a pairing code for another computer (5 minutes, works once).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="The server's port (default 8001).",
    )
    parser.add_argument(
        "--discovery",
        default=None,
        help="Where the server writes its discovery file (default "
        "~/.fibsem/app-server.json).",
    )
    return parser


def _announce(pair: bool):
    def announce(session) -> None:
        host = session.app_server
        microscope = session.microscope
        what = (
            f"the {microscope.system.info.model}"
            if microscope is not None
            else "no microscope yet"
        )
        print(f"The microscope server is serving {what} at {host.url}.")
        if pair:
            code = host.auth.pairing.open()
            print(
                f"Pairing code: {code} (5 minutes, works once). On the other "
                f"computer run: fibsem-cli pair <this computer's address>:"
                f"{host.url.rsplit(':', 1)[-1]} {code}",
            )
        print("Stop with Ctrl-C.", flush=True)

    return announce


def main(
    argv: Optional[List[str]] = None, stop: Optional[threading.Event] = None
) -> int:
    """Serve until Ctrl-C (or *stop* is set). The exit code."""
    args = _parser().parse_args(argv)
    session: AppSession = headless.open_session(args)
    session.app_server_port = args.port
    session.app_server_discovery = args.discovery
    return headless.serve_until_stopped(
        session, serve, PROG, stop, announce=_announce(args.pair)
    )


if __name__ == "__main__":
    raise SystemExit(main())
