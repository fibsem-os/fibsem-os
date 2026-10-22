"""A toast still reaches a window built under a second QApplication.

Destroying the QApplication deletes the notification service with it, and the module
kept the dead wrapper: every toast after that raised "wrapped C/C++ object of type
_NotificationService has been deleted". pytest destroys the QApplication at the end of
a test file whenever the module-scoped ``qapp`` fixture holds the only reference, so
running test_a_milling_sessions_continue_decides_the_result.py and
test_run_milling_question.py together gave 17 errors. CI was green only because a full
run collects files that keep a QApplication at module level for the whole session.

The sequence runs in a fresh interpreter: in this one a QApplication may already be
pinned for the session, and it cannot be destroyed here without breaking every test
after this one.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PyQt5", reason="UI dependencies not installed")

SEQUENCE = textwrap.dedent(
    """
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QApplication, QWidget

    from fibsem.ui import notification_service


    class Window(QWidget):
        # Listens the way the main window does: connects when built,
        # disconnects when closed.
        def __init__(self):
            super().__init__()
            self.received = []
            notification_service._get_service().toast.connect(self._on_toast)

        def _on_toast(self, *args):
            self.received.append(args)

        def closeEvent(self, event):
            notification_service._get_service().toast.disconnect(self._on_toast)
            super().closeEvent(event)


    def open_use_and_close():
        window = Window()
        notification_service.show_toast("hello", "info")
        assert window.received == [("hello", "info", True, False)], window.received
        window.close()
        window.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


    app = QApplication([])
    open_use_and_close()
    open_use_and_close()  # closing a window leaves the service to the next one
    del app  # the end of a test file, when nothing else held the QApplication

    app = QApplication([])
    open_use_and_close()
    print("ok")
    """
)


def test_a_toast_reaches_a_window_built_under_a_second_qapplication():
    import fibsem

    # The fibsem this session imported, not whichever one is installed: the test runs
    # from a tmp directory, and in a worktree the installed one is another checkout.
    source = os.path.dirname(os.path.dirname(os.path.abspath(fibsem.__file__)))
    path = os.pathsep.join(p for p in (source, os.environ.get("PYTHONPATH")) if p)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=path)
    result = subprocess.run(
        [sys.executable, "-c", SEQUENCE],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
