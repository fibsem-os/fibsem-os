"""The Qt application a window runs in."""

from __future__ import annotations

import sys
from typing import Any

from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication

from fibsem import diagnostics
from fibsem.tools.app_identity import set_app_identity
from fibsem.ui import notification_service
from fibsem.ui.qt.gc import install_main_thread_gc
from fibsem.ui.qt.style import set_app_style

__all__ = ["start_qt_app"]


def _tell_operator(record: diagnostics.CrashRecord) -> None:
    notification_service.show(
        f"Unexpected error ({record.error}). It was saved for a report.",
        "error",
    )


def start_qt_app(app: Any, icon: QIcon) -> QApplication:
    """The QApplication for *app* (a ``fibsem.tools.apps`` entry): its identity,
    name, style and icon, collecting garbage on the GUI thread only. An error
    nothing catches is recorded and told to the operator, on either binding:
    without a hook PyQt5 aborts the process when one escapes a slot."""
    diagnostics.install(app.name, on_error=_tell_operator)
    set_app_identity(app.name, windows_app_id=app.app_id)
    qapp = QApplication.instance() or QApplication(sys.argv)
    qapp.setApplicationName(app.name)
    set_app_style(qapp)
    qapp.setWindowIcon(icon)
    # Worker-thread GC finalizes Qt/vispy objects off the GUI thread and
    # hard-crashes Windows GL drivers (access violation in glDrawArrays). Held on
    # the application so it lives as long as it does. See fibsem/ui/qt/gc.py.
    qapp.main_thread_gc = install_main_thread_gc(parent=qapp)
    return qapp
