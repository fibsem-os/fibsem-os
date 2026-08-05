"""Tests for the PyPI update check.

Two properties matter more than the happy path: the network call must always
carry a timeout (a wedged firewall would otherwise stall a worker thread for the
rest of the session), and a failed check must be indistinguishable from no check
— being offline is the normal condition on an air-gapped instrument PC.
"""

import json
from contextlib import contextmanager

import pytest

from fibsem import update_check


@pytest.fixture(autouse=True)
def _clear_cached_status():
    update_check.reset()
    yield
    update_check.reset()


@pytest.fixture
def wheel_install(monkeypatch):
    """Look like a pip-installed wheel rather than a source checkout.

    Not sufficient on its own to make the check run — it is opt-in.
    """
    monkeypatch.setattr(update_check, "get_revision", lambda: None)


@pytest.fixture
def update_check_active(monkeypatch, wheel_install):
    """A wheel install whose user has opted in: the check actually runs."""
    from fibsem.config import ReportingPreferences, UserPreferences

    prefs = UserPreferences(
        reporting=ReportingPreferences(update_check_enabled=True)
    )
    monkeypatch.setattr("fibsem.config.load_user_preferences", lambda: prefs)


def _pypi_payload(*versions):
    return json.dumps({"releases": {v: [] for v in versions}}).encode("utf-8")


@contextmanager
def _fake_response(payload):
    class _Response:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    yield _Response()


def _stub_urlopen(monkeypatch, payload, seen=None):
    def _urlopen(request, **kwargs):
        if seen is not None:
            seen.update(kwargs)
            seen["url"] = request.full_url
        return _fake_response(payload).__enter__()

    monkeypatch.setattr(update_check.urllib.request, "urlopen", _urlopen)


# --- when the check should run at all --------------------------------------


def test_disabled_for_source_installs(monkeypatch):
    """A checkout 48 commits past a tag must not be told to pip install -U."""
    monkeypatch.setattr(update_check, "get_revision", lambda: "v0.5.1-48-g4cd11d9c")

    assert update_check.is_enabled() is False


def test_opt_in_so_disabled_by_default(wheel_install):
    """No preferences file must mean no outbound call."""
    assert update_check.is_enabled() is False


def test_enabled_once_the_user_opts_in(monkeypatch, wheel_install):
    from fibsem.config import ReportingPreferences, UserPreferences

    prefs = UserPreferences(
        reporting=ReportingPreferences(update_check_enabled=True)
    )
    monkeypatch.setattr("fibsem.config.load_user_preferences", lambda: prefs)

    assert update_check.is_enabled() is True


def test_opting_in_does_not_override_source_install_suppression(monkeypatch):
    from fibsem.config import ReportingPreferences, UserPreferences

    prefs = UserPreferences(
        reporting=ReportingPreferences(update_check_enabled=True)
    )
    monkeypatch.setattr("fibsem.config.load_user_preferences", lambda: prefs)
    monkeypatch.setattr(update_check, "get_revision", lambda: "v0.5.1-48-gabc")

    assert update_check.is_enabled() is False


def test_unreadable_preferences_fail_closed(monkeypatch, wheel_install):
    """A broken preferences file must not produce an unopted-into network call."""

    def _boom():
        raise OSError("preferences on fire")

    monkeypatch.setattr("fibsem.config.load_user_preferences", _boom)

    assert update_check.is_enabled() is False


def test_is_supported_is_about_the_install_not_the_preference(monkeypatch):
    monkeypatch.setattr(update_check, "get_revision", lambda: "v0.5.1-48-gabc")
    assert update_check.is_supported() is False

    monkeypatch.setattr(update_check, "get_revision", lambda: None)
    assert update_check.is_supported() is True


def test_force_bypasses_the_opt_in_preference(monkeypatch, wheel_install):
    """A user clicking 'check for updates' is consent for that one call."""
    from fibsem.config import ReportingPreferences, UserPreferences

    opted_out = UserPreferences(
        reporting=ReportingPreferences(update_check_enabled=False)
    )
    monkeypatch.setattr("fibsem.config.load_user_preferences", lambda: opted_out)
    monkeypatch.setattr("fibsem.__version__", "0.5.1", raising=False)
    _stub_urlopen(monkeypatch, _pypi_payload("0.5.3"))

    assert update_check.is_enabled() is False  # no *background* check
    assert update_check.refresh() is None  # ... and none happens
    assert update_check.refresh(force=True).latest == "0.5.3"  # but asking works


def test_force_does_not_override_source_install_suppression(monkeypatch):
    """Even an explicit click cannot make the comparison meaningful."""
    monkeypatch.setattr(update_check, "get_revision", lambda: "v0.5.1-48-gabc")

    def _boom(*args, **kwargs):
        raise AssertionError("must not reach the network")

    monkeypatch.setattr(update_check.urllib.request, "urlopen", _boom)

    assert update_check.refresh(force=True) is None


def test_disabled_check_makes_no_network_call(monkeypatch):
    monkeypatch.setattr(update_check, "get_revision", lambda: "v0.5.1-48-gabc")

    def _boom(*args, **kwargs):
        raise AssertionError("must not reach the network")

    monkeypatch.setattr(update_check.urllib.request, "urlopen", _boom)

    assert update_check.refresh() is None
    assert update_check.get_available_update() is None


# --- fetching ---------------------------------------------------------------


def test_timeout_is_always_passed(monkeypatch, update_check_active):
    """The whole reason this is not the old requests-based helper."""
    seen = {}
    _stub_urlopen(monkeypatch, _pypi_payload("0.5.1"), seen)

    update_check.fetch_stable_versions()

    assert seen["timeout"] == update_check.REQUEST_TIMEOUT_SECONDS
    assert seen["url"].startswith("https://pypi.org/pypi/fibsem/")


def test_prereleases_and_dev_versions_are_filtered(monkeypatch, update_check_active):
    _stub_urlopen(
        monkeypatch,
        _pypi_payload("0.5.0", "0.5.1", "0.5.1a0", "0.5.2.dev0", "0.6.0rc1"),
    )

    assert update_check.fetch_stable_versions() == ["0.5.1", "0.5.0"]


def test_unparseable_versions_are_skipped(monkeypatch, update_check_active):
    _stub_urlopen(monkeypatch, _pypi_payload("0.5.1", "not-a-version"))

    assert update_check.fetch_stable_versions() == ["0.5.1"]


@pytest.mark.parametrize(
    "exc",
    [OSError("network unreachable"), TimeoutError(), ValueError("bad json")],
    ids=["offline", "timeout", "malformed"],
)
def test_fetch_failures_return_empty(monkeypatch, update_check_active, exc):
    def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(update_check.urllib.request, "urlopen", _boom)

    assert update_check.fetch_stable_versions() == []


# --- refresh and the cached answer -----------------------------------------


def test_update_available(monkeypatch, update_check_active):
    monkeypatch.setattr("fibsem.__version__", "0.5.1", raising=False)
    _stub_urlopen(monkeypatch, _pypi_payload("0.5.1", "0.5.3"))

    status = update_check.refresh()

    assert status.update_available is True
    assert status.latest == "0.5.3"
    assert update_check.get_available_update() == "0.5.3"


def test_no_update_when_current(monkeypatch, update_check_active):
    monkeypatch.setattr("fibsem.__version__", "0.5.3", raising=False)
    _stub_urlopen(monkeypatch, _pypi_payload("0.5.1", "0.5.3"))

    status = update_check.refresh()

    assert status.update_available is False
    # Only the actionable case is surfaced.
    assert update_check.get_available_update() is None


def test_no_update_when_ahead_of_pypi(monkeypatch, update_check_active):
    monkeypatch.setattr("fibsem.__version__", "0.6.0", raising=False)
    _stub_urlopen(monkeypatch, _pypi_payload("0.5.3"))

    assert update_check.refresh().update_available is False
    assert update_check.get_available_update() is None


def test_failed_check_looks_like_no_check(monkeypatch, update_check_active):
    """Offline is normal on an air-gapped instrument; it is not an error state."""

    def _boom(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(update_check.urllib.request, "urlopen", _boom)

    assert update_check.refresh() is None
    assert update_check.get_status() is None
    assert update_check.get_available_update() is None


def test_unparseable_current_version_is_not_fatal(monkeypatch, update_check_active):
    monkeypatch.setattr("fibsem.__version__", "unknown", raising=False)
    _stub_urlopen(monkeypatch, _pypi_payload("0.5.3"))

    assert update_check.refresh() is None
    assert update_check.get_available_update() is None
