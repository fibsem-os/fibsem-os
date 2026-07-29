#!/usr/bin/env python
"""Standalone harness for the PyPI update-available indicator.

The update check is skipped entirely on source installs by design, so running
fibsemOS from a checkout shows nothing — which is exactly what makes this
feature awkward to try by hand. Every mode below fakes a wheel install (no git
revision, an older version) so the feature becomes reachable.

Note that ``fibsem.update_check`` binds ``get_revision`` into its own namespace
with a from-import, so it is that name which must be patched, not
``fibsem.versioning.get_revision``.

Usage:
    python scripts/test_update_indicator.py live        # real request to pypi.org
    python scripts/test_update_indicator.py dialog      # about dialog, update waiting
    python scripts/test_update_indicator.py button      # manual check button, clickable
    python scripts/test_update_indicator.py suppressed  # about dialog as-is on a checkout
    python scripts/test_update_indicator.py hang        # prove a wedged network cannot block
"""

import argparse
import socket
import sys
import time
from contextlib import ExitStack
from unittest.mock import patch

FAKE_INSTALLED_VERSION = "0.5.0"

# Long enough to be obviously a hang rather than a slow network, short enough
# that the harness itself does not look wedged.
HANG_BUDGET_SECONDS = 10


def _pretend_wheel_install(stack: ExitStack, version: str = FAKE_INSTALLED_VERSION):
    """Look like an old pip install, by an opted-in user.

    Both halves are needed: the check is skipped for source checkouts, and it is
    opt-in even for wheel installs, so a default preferences file disables it.
    """
    import fibsem
    from fibsem.config import ReportingPreferences, UserPreferences

    opted_in = UserPreferences(
        reporting=ReportingPreferences(update_check_enabled=True)
    )
    stack.enter_context(patch("fibsem.update_check.get_revision", lambda: None))
    stack.enter_context(patch.object(fibsem, "__version__", version))
    stack.enter_context(
        patch("fibsem.config.load_user_preferences", lambda: opted_in)
    )


def run_live() -> int:
    """Ask the real PyPI, pretending to be an old wheel install."""
    from fibsem import update_check

    with ExitStack() as stack:
        _pretend_wheel_install(stack)

        print(f"pretending to be fibsem {FAKE_INSTALLED_VERSION} installed from a wheel")
        print("enabled          :", update_check.is_enabled())

        started = time.perf_counter()
        status = update_check.refresh()
        elapsed_ms = (time.perf_counter() - started) * 1000

        print(f"refresh took     : {elapsed_ms:.0f} ms "
              f"(timeout is {update_check.REQUEST_TIMEOUT_SECONDS}s)")
        print("status           :", status)
        print("indicator shows  :", update_check.get_available_update() or "(nothing)")
    return 0


def run_dialog(mode: str) -> int:
    """Show the real about dialog.

    ``dialog``     an update is already known, so the indicator is showing
    ``button``     nothing known yet, so the manual check button is offered —
                   click it for a real request to PyPI
    ``suppressed`` the dialog exactly as it is here, on a source checkout
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    import fibsem.ui.widgets.about_dialog as about_dialog

    with ExitStack() as stack:
        if mode != "suppressed":
            _pretend_wheel_install(stack)
            # The dialog reads these directly, so they need patching here too.
            stack.enter_context(patch.object(about_dialog, "get_revision", lambda: None))
            stack.enter_context(patch.object(about_dialog, "get_branch", lambda: None))
        if mode == "dialog":
            stack.enter_context(
                patch("fibsem.update_check.get_available_update", lambda: "0.5.3")
            )
        about_dialog.AboutDialog(application="AutoLamella").exec_()
    return 0


def run_hang() -> int:
    """Point the check at a socket that accepts and then never replies.

    This is what a firewall blackholing packets looks like, and it is the reason
    the request carries an explicit timeout: without one it would block until
    the OS gave up, minutes later, with a worker thread stuck the whole time.
    """
    from fibsem import update_check

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    try:
        with ExitStack() as stack:
            _pretend_wheel_install(stack)
            stack.enter_context(
                patch.object(
                    update_check, "PYPI_URL", f"http://127.0.0.1:{port}/{{package}}"
                )
            )
            started = time.perf_counter()
            result = update_check.fetch_stable_versions()
            elapsed = time.perf_counter() - started
    finally:
        server.close()

    print(f"blackholed request returned {result!r} after {elapsed:.1f}s")
    ok = elapsed < HANG_BUDGET_SECONDS
    print(f"timeout is {update_check.REQUEST_TIMEOUT_SECONDS}s — "
          f"{'PASS' if ok else 'FAIL: it hung'}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="live",
        choices=["live", "dialog", "button", "suppressed", "hang"],
    )
    args = parser.parse_args()

    if args.mode == "live":
        return run_live()
    if args.mode in ("dialog", "button", "suppressed"):
        return run_dialog(args.mode)
    return run_hang()


if __name__ == "__main__":
    sys.exit(main())
