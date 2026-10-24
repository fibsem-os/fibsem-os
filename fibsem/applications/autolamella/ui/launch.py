"""Starting the AutoLamella window: the command line, the update check, the app."""

from __future__ import annotations

import argparse
import sys
from typing import Callable, List, Optional

from PyQt5.QtCore import QTimer

from fibsem.applications.autolamella.ui.window import AutoLamellaWindow
from fibsem.tools.apps import AUTOLAMELLA
from fibsem.tools.desktop_launch import Splash
from fibsem.ui.icon import autolamella_icon
from fibsem.ui.qt.app import start_qt_app


def _start_update_check(on_done: Optional[Callable[[], None]] = None) -> None:
    """Ask PyPI about newer releases on a worker thread, once, at startup.

    Startup rather than on dialog open: the about dialog reads the cached result,
    so it stays instant and offline-safe by construction, and the latency lands
    where nothing is waiting on it. A no-op for source installs and when the user
    has opted out, so the common developer case makes no network call at all.

    ``on_done`` runs on the GUI thread once the check is over, however it ended --
    offline, refused or answered -- so the caller reads the cache rather than
    reasoning about which.
    """
    from fibsem import update_check
    from fibsem.ui.qt.threading import thread_worker

    if not update_check.is_enabled():
        return

    @thread_worker
    def _check() -> None:
        update_check.refresh()

    worker = _check()
    if on_done is not None:
        # Connected before start(): a check that fails at once (no route, no DNS)
        # could otherwise finish before anything is listening.
        worker.finished.connect(on_done)
    worker.start()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command-line flags accepted by the AutoLamella UI."""
    parser = argparse.ArgumentParser(
        prog="fibsem-autolamella-ui",
        description="Launch the AutoLamella user interface.",
    )
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Connect to the microscope with the default configuration as soon as the "
        "window is up, instead of waiting for the Connection tab.",
    )
    parser.add_argument(
        "--quickload",
        action="store_true",
        help="Connect as --quickstart does, then reopen the most recent experiment. "
        "Implies --quickstart -- the experiment tabs are built at connection time.",
    )
    return parser.parse_args(argv)


def run_ui(argv: Optional[List[str]] = None, splash: Optional[Splash] = None):
    """Run the AutoLamella UI.

    ``splash`` is the launcher's splash screen (fibsem.tools.desktop_launch), closed once the
    main window is up.
    """
    import faulthandler
    import signal

    args = _parse_args(argv)

    # Ctrl+C: PyQt never runs Python's SIGINT handler while blocked in the C++ event
    # loop — and if the GUI thread wedges, no Python runs at all — so restore the OS
    # default disposition: SIGINT then terminates the process at the kernel level,
    # making the app always killable with Ctrl+C. (A hard stop, not a graceful one;
    # use Cmd+Q / File→Exit for a clean shutdown when the loop is responsive.)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    # Diagnostics: dump every thread's Python stack on a fatal signal, and on demand
    # via `kill -USR1 <pid>` — the fastest way to see *where* a frozen GUI thread is
    # stuck (it does not terminate the process, so you can dump repeatedly).
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    app = start_qt_app(AUTOLAMELLA, autolamella_icon())
    window = AutoLamellaWindow()
    window.show()
    if splash is not None:
        splash.finish(window)
    _start_update_check(on_done=window.experiment_header.update_version_chip)
    if args.quickstart or args.quickload:
        # Once the event loop is running, not inline here: connecting blocks for
        # seconds against real hardware, and doing it before exec_() would hold up
        # the first paint — an empty window frame for the whole wait. A zero timer
        # fires as soon as the window system's queue, that first paint included,
        # has drained.
        QTimer.singleShot(0, lambda: window.quickstart(load_experiment=args.quickload))
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_ui()
