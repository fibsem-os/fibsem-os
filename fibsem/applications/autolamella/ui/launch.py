"""Starting the AutoLamella window: the command line, the update check, the app."""

from __future__ import annotations

import argparse
import sys
from typing import Callable, List, Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella import launch
from fibsem.applications.autolamella.ui.window import AutoLamellaWindow
from fibsem.tools.app_identity import set_app_identity
from fibsem.tools.apps import AUTOLAMELLA
from fibsem.tools.desktop_launch import Splash
from fibsem.ui.icon import autolamella_icon
from fibsem.ui.qt.gc import install_main_thread_gc


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


def run_ui(
    argv: Optional[List[str]] = None,
    splash: Optional[Splash] = None,
    session=None,
):
    """Run the AutoLamella UI on *session*, the host's (a new one when None).

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

    set_app_identity(AUTOLAMELLA.name, windows_app_id=AUTOLAMELLA.app_id)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(AUTOLAMELLA.name)
    app.setStyle("Fusion")
    app.setWindowIcon(autolamella_icon())
    # Cyclic garbage must only ever be collected on this thread: worker-thread GC
    # finalizes Qt/vispy objects off the GUI thread and hard-crashes Windows GL
    # drivers (access violation in glDrawArrays). See fibsem/ui/qt/gc.py.
    gc_collector = install_main_thread_gc(parent=app)  # noqa: F841 — kept alive for app lifetime
    window = AutoLamellaWindow(session=session)
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
    code = app.exec_()
    # The window has stopped the recording and let the microscope go.
    window.session.stop_app_server()
    sys.exit(code)


def host(argv: Optional[List[str]] = None, splash: Optional[Splash] = None) -> int:
    """``fibsem-autolamella-ui``: the AutoLamella host with the window. The
    host's flags connect the session first (none: the window's Connect does);
    the window's own (``--quickstart``) pass on to it."""
    return launch.main(
        argv, window=lambda session, rest: run_ui(rest, splash=splash, session=session)
    )


if __name__ == "__main__":
    run_ui()
