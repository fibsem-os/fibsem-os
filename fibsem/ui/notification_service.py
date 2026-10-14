# TODO: the following files still use napari.utils.notifications and need migrating:
#   fibsem/ui/FibsemSpotBurnWidget.py
#   fibsem/ui/FibsemUI.py
#   fibsem/ui/FibsemEmbeddedDetectionWidget.py
#   fibsem/ui/FibsemManipulatorWidget.py
#   fibsem/ui/FibsemLabellingUI.py          (standalone — needs listener registration)
#   fibsem/ui/FibsemSegmentationModelWidget.py

from __future__ import annotations

from typing import Callable, Optional, Tuple

from PyQt5 import sip
from PyQt5.QtCore import QObject, pyqtSignal


class _NotificationService(QObject):
    """App-wide toast signal bus. Thread-safe: cross-thread emits are queued to main thread."""

    toast = pyqtSignal(str, str, bool)  # (message, notification_type, temporary)
    # The same, with a link on the toast: (..., action_text, callback).
    toast_action = pyqtSignal(str, str, bool, str, object)


_service: Optional[_NotificationService] = None


def _get_service() -> _NotificationService:
    """The service, built again if the one held here has been deleted.

    Destroying the QApplication deletes the service with it, although it has no
    parent, and the module keeps the dead wrapper. A process that builds a second
    QApplication -- pytest does, once per test file, when the module-scoped ``qapp``
    fixture held the only reference to the first -- would otherwise fail every
    toast with "wrapped C/C++ object has been deleted". The listeners went with the
    old one; each window connects to the new one when it is built.
    """
    global _service
    if _service is None or sip.isdeleted(_service):
        _service = _NotificationService()
    return _service


def show(
    message: str,
    notification_type: str = "info",
    temporary: bool = False,
    action: Optional[Tuple[str, Callable[[], None]]] = None,
) -> None:
    """Emit a toast. Persists to notification history by default (temporary=False).

    ``action`` is a (text, callback) link shown on the toast and on its history
    entry, e.g. ("Show summary", open_summary). The callback runs on the GUI thread.

    Use show_toast() for transient validation messages that should not appear in history.
    """
    if action is not None:
        text, callback = action
        _get_service().toast_action.emit(
            message, notification_type, temporary, text, callback
        )
        return
    _get_service().toast.emit(message, notification_type, temporary)


def show_toast(message: str, notification_type: str = "info") -> None:
    """Emit a transient toast that does not go to notification history.

    Preferred for widget-level validation messages (no microscope connected, etc.).
    For workflow events that should persist in history, use show() with temporary=False.
    """
    _get_service().toast.emit(message, notification_type, True)
