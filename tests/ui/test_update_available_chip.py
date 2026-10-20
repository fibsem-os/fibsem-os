"""The header shows a newer release when the startup check found one, and nothing else.

Offline is the normal state on an air-gapped instrument PC, so a check that could
not reach PyPI has to look exactly like one that found nothing: no chip, no error.

The chip handler is borrowed onto a host carrying the real button, as the
connection chip's tests do. The startup check runs on a real worker thread.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("PyQt5")

from fibsem import update_check  # noqa: E402
from fibsem.applications.autolamella.ui.launch import (
    _start_update_check,  # noqa: E402  # noqa: E402
)
from tests.fixtures.experiment_header import HeaderHost  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_cached_status():
    update_check.reset()
    yield
    update_check.reset()


@pytest.fixture
def update_check_active(monkeypatch):
    """A wheel install whose user has opted in: the check actually runs."""
    from fibsem.config import ReportingPreferences, UserPreferences

    monkeypatch.setattr(update_check, "get_revision", lambda: None)
    prefs = UserPreferences(reporting=ReportingPreferences(update_check_enabled=True))
    monkeypatch.setattr("fibsem.config.load_user_preferences", lambda: prefs)


def _serve(monkeypatch, *versions):
    payload = json.dumps({"releases": {v: [] for v in versions}}).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return payload

    monkeypatch.setattr(
        update_check.urllib.request, "urlopen", lambda *a, **k: _Response()
    )


def _go_offline(monkeypatch):
    def _unreachable(*args, **kwargs):
        raise urllib.error.URLError("Network is unreachable")

    monkeypatch.setattr(update_check.urllib.request, "urlopen", _unreachable)


def _run_startup_check(qapp, host):
    """Start the real startup check and pump events until its callback has run."""
    done = []

    def _on_done():
        host.update_version_chip()
        done.append(True)

    _start_update_check(on_done=_on_done)
    deadline = time.monotonic() + 10
    while not done and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert done, "the startup check never reported back"


def test_chip_names_the_newer_release(qapp, monkeypatch, update_check_active):
    monkeypatch.setattr("fibsem.__version__", "0.6.0")
    _serve(monkeypatch, "0.5.3", "0.6.0", "0.6.1", "0.7.0rc1")
    host = HeaderHost()

    _run_startup_check(qapp, host)

    assert not host.btn_update_available.isHidden()
    # Stable releases only: an rc is not something to tell an instrument PC about.
    assert host.btn_update_available.text() == "0.6.1 available"


def test_offline_shows_nothing_and_does_not_raise(
    qapp, monkeypatch, update_check_active
):
    _go_offline(monkeypatch)
    host = HeaderHost()

    _run_startup_check(qapp, host)

    assert host.btn_update_available.isHidden()


def test_up_to_date_shows_nothing(qapp, monkeypatch, update_check_active):
    monkeypatch.setattr("fibsem.__version__", "0.6.1")
    _serve(monkeypatch, "0.6.0", "0.6.1")
    host = HeaderHost()

    _run_startup_check(qapp, host)

    assert host.btn_update_available.isHidden()


def test_check_off_starts_nothing(monkeypatch):
    # Default preferences: the check is opt-in, so no worker and no callback.
    monkeypatch.setattr(update_check, "is_enabled", lambda: False)
    called = []

    _start_update_check(on_done=lambda: called.append(True))

    assert called == []
