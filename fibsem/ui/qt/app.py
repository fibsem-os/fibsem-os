"""The Qt application a window runs in."""

from __future__ import annotations

import sys
from typing import Any

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from fibsem.tools.app_identity import set_app_identity
from fibsem.ui.qt.gc import install_main_thread_gc

__all__ = ["start_qt_app"]


def start_qt_app(app: Any, icon: QIcon) -> QApplication:
    """The QApplication for *app* (a ``fibsem.tools.apps`` entry): its identity,
    name, style and icon, collecting garbage on the GUI thread only."""
    set_app_identity(app.name, windows_app_id=app.app_id)
    qapp = QApplication.instance() or QApplication(sys.argv)
    qapp.setApplicationName(app.name)
    qapp.setStyle("Fusion")
    qapp.setWindowIcon(icon)
    # Worker-thread GC finalizes Qt/vispy objects off the GUI thread and
    # hard-crashes Windows GL drivers (access violation in glDrawArrays). Held on
    # the application so it lives as long as it does. See fibsem/ui/qt/gc.py.
    qapp.main_thread_gc = install_main_thread_gc(parent=qapp)
    return qapp
