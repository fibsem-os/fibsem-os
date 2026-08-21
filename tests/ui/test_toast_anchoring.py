"""A toast belongs to the window the operator is working in, not to the host window.

Every toast used to be a child of the one window that owns the ToastManager (the
AutoLamella main window), wherever the work happened. Two consequences, one cosmetic
and one not: a notification about something done in the coincidence viewer appeared in
the corner of a different window, and on Windows a toast owned by the main window drags
the main window's z-order group in front of the (parentless) viewer -- reported from a
session as "when the coincidence milling stops the AutoLamella GUI comes to the front".

The anchor is the last *real* window of ours to hold focus, deliberately not
``QApplication.activeWindow()``: that is None whenever the app is unfocused, which is
the normal state while a run finishes and the operator watches the microscope software.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import sip
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget

from fibsem.ui.widgets.notifications import ToastManager


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def host(qapp):
    """Stand-in for the main window: owns the manager, and is the fallback anchor."""
    window = QWidget()
    window.setGeometry(0, 0, 800, 600)
    window.show()
    yield window
    window.close()


@pytest.fixture()
def viewer(qapp):
    """Stand-in for the coincidence viewer: a separate, parentless top-level."""
    window = QWidget()
    window.setGeometry(900, 100, 700, 500)
    window.show()
    yield window
    if not sip.isdeleted(window):  # a test may have destroyed it on purpose
        window.close()


def _bottom_right_of(window: QWidget):
    return window.mapToGlobal(window.rect().bottomRight())


def test_the_toast_belongs_to_the_window_the_work_happened_in(host, viewer):
    manager = ToastManager(host)
    manager._on_focus_changed(None, viewer)

    manager.show_toast("Recorded coincidence result for Lamella-01.")

    toast = manager.toasts[-1]
    assert toast.parent() is viewer, (
        "the toast is owned by the host window: on Windows that pulls the host in "
        "front of the window the operator is using"
    )
    corner = _bottom_right_of(viewer)
    assert toast.x() < corner.x() and toast.y() < corner.y()
    assert toast.x() > _bottom_right_of(host).x(), "positioned over the host window"


def test_it_falls_back_to_the_host_window_when_nothing_else_has_been_used(host):
    manager = ToastManager(host)

    manager.show_toast("Connecting to microscope...")

    assert manager.toasts[-1].parent() is host


def test_a_hidden_anchor_is_not_used(host, viewer):
    """The anchor is remembered, so it can be gone by the time the next toast fires."""
    manager = ToastManager(host)
    manager._on_focus_changed(None, viewer)
    viewer.close()

    manager.show_toast("Milling finished.")

    assert manager.toasts[-1].parent() is host


def test_a_destroyed_anchor_takes_its_toasts_but_not_the_manager(host, viewer):
    """The window can be *destroyed* rather than merely hidden, and Qt destroys its
    children with it -- so a toast still in the list is a wrapper whose C++ side is
    gone, and every method on it raises rather than returning False.

    ``sip.delete`` rather than ``deleteLater`` + ``processEvents``: processEvents does
    not deliver DeferredDelete, so that pairing closes the window without ever
    destroying it and this test would pass without exercising anything.
    """
    manager = ToastManager(host)
    manager._on_focus_changed(None, viewer)
    manager.show_toast("on the viewer")
    orphan = manager.toasts[-1]

    sip.delete(viewer)
    assert sip.isdeleted(orphan), "expected the toast to be destroyed with its window"

    manager.show_toast("after the window was destroyed")

    assert manager.toasts == [manager.toasts[-1]], "the dead toast was not pruned"
    assert manager.toasts[-1].parent() is host


def test_a_minimized_window_is_not_used_as_an_anchor(host, viewer):
    """Qt reports a minimized window as visible; a toast anchored to one would be
    placed at its restored geometry, i.e. over nothing."""
    manager = ToastManager(host)
    manager._on_focus_changed(None, viewer)
    viewer.showMinimized()

    manager.show_toast("Milling finished.")

    assert manager.toasts[-1].parent() is host


def test_transient_windows_never_become_the_anchor(host, viewer):
    """Toasts, popovers and tooltips are the app's own chrome, and dialogs are
    short-lived -- a toast anchored to one would be destroyed with it mid-fade."""
    manager = ToastManager(host)
    manager._on_focus_changed(None, viewer)

    for flags in (Qt.Tool, Qt.Popup, Qt.ToolTip, Qt.Dialog):
        transient = QWidget(None, flags)
        manager._on_focus_changed(None, transient)
        assert manager._anchor() is viewer, f"{flags} was taken as the anchor"


def test_each_window_stacks_its_own_toasts(host, viewer):
    """Two windows' notifications must not share a column -- when the windows overlap,
    a shared column overlaps too."""
    manager = ToastManager(host)
    manager.show_toast("first, on the host")
    manager._on_focus_changed(None, viewer)
    manager.show_toast("first, on the viewer")
    manager.show_toast("second, on the viewer")

    host_toasts = [t for t in manager.toasts if t.parent() is host]
    viewer_toasts = [t for t in manager.toasts if t.parent() is viewer]
    assert len(host_toasts) == 1 and len(viewer_toasts) == 2

    # the two on the viewer are stacked above one another, not on top of each other
    assert viewer_toasts[0].y() != viewer_toasts[1].y()
    # and the host's toast keeps its own corner, unshifted by the viewer's two
    only_host = ToastManager(host)
    only_host.show_toast("first, on the host")
    assert host_toasts[0].y() == only_host.toasts[-1].y()
