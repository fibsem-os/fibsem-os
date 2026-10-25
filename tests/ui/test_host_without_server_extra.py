"""Tools → Host on an install without the [server] extra says what to
install, rather than only that the host did not start."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")

from fibsem.ui.agent.app_server_dialog import AppServerDialog  # noqa: E402


def test_the_dialog_says_the_extra_is_missing(qapp):
    host = SimpleNamespace(running=False, start_error="the [server] extra")
    dialog = AppServerDialog(lambda: host, enabled=True)
    dialog._chk_enabled.setChecked(True)
    dialog.refresh()
    assert dialog._status_label.text() == "Not running: it needs the [server] extra."
    dialog.deleteLater()


def test_a_host_that_failed_otherwise_points_at_the_log(qapp):
    host = SimpleNamespace(running=False, start_error=None)
    dialog = AppServerDialog(lambda: host, enabled=True)
    dialog._chk_enabled.setChecked(True)
    dialog.refresh()
    assert dialog._status_label.text() == "Not running. It did not start; see the log."
    dialog.deleteLater()
