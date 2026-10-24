"""Toasts caught in a list, for a test that checks what the operator is told."""

import pytest


def toasts_fixture():
    """A ``toasts`` fixture: toasts go to a list of ``(message, notification_type)``,
    not to the notification service.

    The service is a module-level QObject owned by whichever QApplication first
    created it; a later test module's teardown can delete it, and the next toast
    raises "wrapped C/C++ object has been deleted" from inside the widget under
    test.
    """

    @pytest.fixture(name="toasts")
    def _toasts(monkeypatch):
        from fibsem.ui import notification_service

        shown = []
        monkeypatch.setattr(
            notification_service,
            "show_toast",
            lambda message, notification_type="info", **_: shown.append(
                (message, notification_type)
            ),
        )
        return shown

    return _toasts
