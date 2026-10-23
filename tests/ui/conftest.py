import pytest

from tests.fixtures.main_window import main_window_fixture

# The AutoLamella window, not connected. A file that wants it connected, or on
# another configuration, declares its own with main_window_fixture.
main_ui = main_window_fixture()


@pytest.fixture
def toasts(monkeypatch):
    """Toasts go to a list of ``(message, notification_type)``, not to the
    notification service.

    The service is a module-level QObject owned by whichever QApplication first
    created it; a later test module's teardown can delete it, and the next toast
    raises "wrapped C/C++ object has been deleted" from inside the widget under
    test.
    """
    from fibsem.ui import notification_service

    shown = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda message, notification_type="info": shown.append(
            (message, notification_type)
        ),
    )
    return shown


@pytest.fixture
def no_quit(qapp):
    """Stub ``qapp.quit`` for the test: AutoLamella's closeEvent ends in
    app.quit(), which latches on the shared test QApplication."""
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        yield
    finally:
        qapp.quit = original_quit
