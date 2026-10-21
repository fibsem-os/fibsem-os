"""Toasts for problems only: errors, warnings and a run's end still pop up; the
rest go to the notification bell without popping up."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QWidget

from fibsem.ui.widgets.notifications import NotificationBell, ToastManager


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def manager(qapp):
    host = QWidget()
    host.show()
    manager = ToastManager(host)
    manager.set_notification_bell(NotificationBell(host))
    manager.problems_only = True
    yield manager
    host.close()


def test_info_and_success_go_to_the_bell_without_popping_up(manager):
    manager.show_toast("Task Rough Milling complete for 01-loved-liger", "success")
    manager.show_toast("Agent edited Polishing", "info")

    assert manager.toasts == []
    assert manager.notification_bell.count == 2


def test_a_temporary_info_toast_goes_nowhere(manager):
    manager.show_toast("Agent: focus sweep", "info", temporary=True)

    assert manager.toasts == []
    assert manager.notification_bell.count == 0


@pytest.mark.parametrize("kind", ["warning", "error"])
def test_problems_still_pop_up(manager, kind):
    manager.show_toast("Someone else has control", kind, temporary=True)

    assert len(manager.toasts) == 1


def test_always_pops_up_whatever_its_kind(manager):
    manager.show_toast("Run ended: 4 done", "success", always=True)

    assert len(manager.toasts) == 1
    assert manager.notification_bell.count == 1


def test_off_pops_up_everything(manager):
    manager.problems_only = False

    manager.show_toast("Saved protocol", "info", temporary=True)

    assert len(manager.toasts) == 1


def test_the_service_carries_always_to_the_window(qapp):
    from fibsem.ui import notification_service

    received = []
    service = notification_service._get_service()
    service.toast.connect(lambda *args: received.append(args))
    service.toast_action.connect(lambda *args: received.append(args[:4]))

    notification_service.show_toast("Workflow finished.", always=True)
    notification_service.show(
        "Run ended", "success", action=("Show", print), always=True
    )
    notification_service.show("Task complete", "success")

    assert received == [
        ("Workflow finished.", "info", True, True),
        ("Run ended", "success", False, True),
        ("Task complete", "success", False, False),
    ]
