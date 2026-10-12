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


def test_a_401_re_reads_the_discovery_file_and_retries(monkeypatch):
    """A restarted server's new token reaches a connected sidecar without a
    reconnect (FIB-1035 app test #5)."""
    resolutions = iter([("http://x:2", "new")])
    monkeypatch.setattr(sidecar, "_try_resolve", lambda url, token: next(resolutions))
    stale = _FakeClient([_FakeResponse(401, {"detail": "unauthorized"})])
    fresh = _FakeClient([_FakeResponse(200, {"ok": True})])
    made = []
    client = sidecar._ReconnectingClient(
        lambda url, token: made.append((url, token)) or fresh,
        None,
        None,
        stale,
        ("http://x:1", "old"),
    )
    resp = client.request("GET", "/app/status")
    assert resp.status_code == 200
    assert made == [("http://x:2", "new")]
    assert stale.closed


def test_a_401_with_nothing_new_to_find_is_returned_as_is(monkeypatch):
    monkeypatch.setattr(sidecar, "_try_resolve", lambda url, token: ("http://x", "t"))
    stale = _FakeClient([_FakeResponse(401, {"detail": "unauthorized"})])
    client = sidecar._ReconnectingClient(
        lambda url, token: pytest.fail("no new client"),
        "http://x",
        "t",
        stale,
        ("http://x", "t"),
    )
    assert client.request("GET", "/capabilities").status_code == 401


def test_a_dropped_connection_follows_a_moved_server(monkeypatch):
    monkeypatch.setattr(sidecar, "_try_resolve", lambda url, token: ("http://y", "t"))
    gone = _FakeClient([ConnectionError("refused")])
    moved = _FakeClient([_FakeResponse(200, {"ok": True})])
    client = sidecar._ReconnectingClient(
        lambda url, token: moved, None, None, gone, ("http://x", "t")
    )
    assert client.request("GET", "/app/status").status_code == 200
    assert client.base_url == "http://fake:1"  # delegated to the live client


def test_a_prompt_image_travels_as_mcp_image_content():
    """Not base64 inside the JSON, which overflowed the tool result (#6)."""
    import asyncio
    import base64
    import json

    jpeg = b"\xff\xd8\xffjpeg"
    payload = {
        "pending": True,
        "request": {
            "type": "PickPOI",
            "image": {"image_b64_jpeg": base64.b64encode(jpeg).decode(), "width": 4},
        },
    }
    fake = _FakeClient([_FakeResponse(200, payload)])
    capabilities = {"routers": {"app": True}, "scopes": {"read": True}}
    mcp = sidecar.build_sidecar(fake, capabilities)
    result = asyncio.run(mcp.call_tool("get_pending_prompt", {}))
    contents = result[0] if isinstance(result, tuple) else result
    contents = getattr(contents, "content", contents)
    texts = [c.text for c in contents if getattr(c, "type", "") == "text"]
    images = [c for c in contents if getattr(c, "type", "") == "image"]
    assert len(images) == 1 and base64.b64decode(images[0].data) == jpeg
    body = json.loads(texts[0])
    assert body["request"]["image"] == {"width": 4, "image_index": 0}
