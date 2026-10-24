import pytest

from tests.fixtures.main_window import main_window_fixture, quit_stubbed
from tests.fixtures.toasts import toasts_fixture

# The AutoLamella window, not connected. A file that wants it connected, or on
# another configuration, declares its own with main_window_fixture.
main_ui = main_window_fixture()


toasts = toasts_fixture()


@pytest.fixture
def no_quit(qapp):
    """Stub ``qapp.quit`` for the test: AutoLamella's closeEvent ends in
    app.quit(), which latches on the shared test QApplication."""
    with quit_stubbed(qapp):
        yield
