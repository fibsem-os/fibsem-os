"""A toast can carry a link (FIB-1376): the run-end toast's "Show summary".

The link is on the toast and on its entry in the notification bell, so it
stays a click away after the toast fades. Following it runs the callback once
and takes the toast down; a callback that raises is logged, never raised into Qt.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QPushButton, QWidget

from fibsem.ui.widgets.notifications import NotificationBell, ToastManager


@pytest.fixture()
def host(qapp):
    window = QWidget()
    window.setGeometry(0, 0, 800, 600)
    window.show()
    yield window
    window.close()


def _manager(host):
    manager = ToastManager(host)
    bell = NotificationBell(host)
    manager.set_notification_bell(bell)
    return manager, bell


def test_a_toast_without_a_link_shows_none(host):
    manager, _bell = _manager(host)
    manager.show_toast("Milling finished.")
    assert manager.toasts[-1].action_btn.isHidden()


def test_the_link_runs_its_callback_and_hides_the_toast(host):
    manager, _bell = _manager(host)
    calls = []
    manager.show_toast(
        "Run ended: 1 completed, 1 skipped",
        action=("Show summary", lambda: calls.append(1)),
    )
    toast = manager.toasts[-1]
    assert not toast.action_btn.isHidden()
    assert toast.action_btn.text() == "Show summary"

    toast.action_btn.click()
    assert calls == [1]
    assert not toast.hide_timer.isActive()  # hiding: the auto-hide is moot


def test_the_bell_keeps_the_link(host):
    manager, bell = _manager(host)
    calls = []
    manager.show_toast(
        "Run ended: 2 completed", action=("Show summary", lambda: calls.append(1))
    )
    links = [
        b for b in bell.popup.findChildren(QPushButton) if b.text() == "Show summary"
    ]
    assert len(links) == 1
    links[0].click()
    assert calls == [1]


def test_a_failing_link_is_logged_not_raised(host, caplog):
    manager, _bell = _manager(host)

    def boom():
        raise RuntimeError("no summary")

    manager.show_toast("Run ended: 1 failed", action=("Show summary", boom))
    manager.toasts[-1].action_btn.click()
    assert "toast action failed" in caplog.text
