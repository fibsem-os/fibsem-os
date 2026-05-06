# TODO: the following files still use napari.utils.notifications and need migrating:
#   fibsem/ui/FibsemSpotBurnWidget.py
#   fibsem/ui/FibsemUI.py
#   fibsem/ui/FibsemEmbeddedDetectionWidget.py
#   fibsem/ui/FibsemManipulatorWidget.py
#   fibsem/ui/FibsemMicroscopeConfigurationWidget.py
#   fibsem/ui/FibsemMicroscopeConfigurationWidgetBase.py
#   fibsem/ui/FibsemLabellingUI.py          (standalone — needs listener registration)
#   fibsem/ui/FibsemSegmentationModelWidget.py
#   fibsem/ui/FibsemModelTrainingWidget.py   (standalone — needs listener registration)
#   fibsem/ui/FibsemFeatureLabellingUI.py    (standalone — needs listener registration)
#   fibsem/correlation/app.py               (standalone — needs listener registration)
#   fibsem/correlation/ui/fm_import_wizard.py

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal


class _NotificationService(QObject):
    """App-wide toast signal bus. Thread-safe: cross-thread emits are queued to main thread."""

    toast = pyqtSignal(str, str, bool)  # (message, notification_type, temporary)


_service: Optional[_NotificationService] = None


def _get_service() -> _NotificationService:
    global _service
    if _service is None:
        _service = _NotificationService()
    return _service


def show(message: str, notification_type: str = "info", temporary: bool = False) -> None:
    """Emit a toast. Persists to notification history by default (temporary=False).

    Use show_toast() for transient validation messages that should not appear in history.
    """
    _get_service().toast.emit(message, notification_type, temporary)


def show_toast(message: str, notification_type: str = "info") -> None:
    """Emit a transient toast that does not go to notification history.

    Preferred for widget-level validation messages (no microscope connected, etc.).
    For workflow events that should persist in history, use show() with temporary=False.
    """
    _get_service().toast.emit(message, notification_type, True)
