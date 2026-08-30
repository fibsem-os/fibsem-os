"""Tests for the sidecar's connection resolution and wait-for-server retry."""

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("mcp")

from fibsem.mcp import sidecar  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, responses):
        self._responses = responses
        self.base_url = "http://fake:1"
        self.closed = False

    def request(self, method, path, json=None):
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed = True


def test_try_resolve_returns_none_without_exiting(monkeypatch, tmp_path):
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    monkeypatch.setattr(
        "fibsem.server.discovery.DISCOVERY_FILE", tmp_path / "absent.json"
    )
    assert sidecar._try_resolve(None, None) is None
    assert sidecar._try_resolve("http://x", None) is None  # url without token
    assert sidecar._try_resolve("http://x", "t") == ("http://x", "t")


def test_retry_succeeds_after_server_appears():
    ok = _FakeResponse(200, {"scopes": {"read": True}})
    attempts = [
        _FakeClient([ConnectionError("refused")]),
        _FakeClient([_FakeResponse(401, {"detail": "unauthorized"})]),
        _FakeClient([ok]),
    ]
    sleeps = []
    client, caps = sidecar._connect_with_retry(
        lambda url, token: attempts.pop(0),
        "http://x",
        "t",
        wait_s=60,
        sleep_fn=sleeps.append,
    )
    assert caps == {"scopes": {"read": True}}
    assert len(sleeps) == 2  # slept between the three attempts
    assert not client.closed  # the successful client stays open


def test_retry_times_out_with_guidance():
    with pytest.raises(SystemExit) as excinfo:
        sidecar._connect_with_retry(
            lambda url, token: _FakeClient([ConnectionError("refused")]),
            "http://x",
            "t",
            wait_s=0,
            sleep_fn=lambda s: None,
        )
    assert "cannot reach the server" in str(excinfo.value)


def test_no_server_at_all_times_out_with_the_setup_message(monkeypatch, tmp_path):
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    monkeypatch.setattr(
        "fibsem.server.discovery.DISCOVERY_FILE", tmp_path / "absent.json"
    )
    with pytest.raises(SystemExit) as excinfo:
        sidecar._connect_with_retry(
            lambda url, token: _FakeClient([]),
            None,
            None,
            wait_s=0,
            sleep_fn=lambda s: None,
        )
    assert "no server found" in str(excinfo.value)
