"""Tests for the sidecar's connection resolution and wait-for-server retry."""

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("mcp")

from fibsem.ai.agent.mcp import sidecar  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeClient:
    """A transport's shape: ``request``, ``url``, ``user_agent``, ``close``."""

    def __init__(self, responses):
        self._responses = responses
        self.url = "http://fake:1"
        self.user_agent = None
        self.closed = False

    def request(self, method, path, json=None, timeout=None):
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed = True


def _refused():
    return sidecar.ServerUnreachable("refused")


def test_retry_succeeds_after_server_appears():
    ok = _FakeResponse(200, {"level": "watch"})
    attempts = [
        None,  # no server known yet
        _FakeClient([_refused()]),
        _FakeClient([_FakeResponse(401, {"detail": "unauthorized"})]),
        _FakeClient([ok]),
    ]
    sleeps = []
    client, caps = sidecar._connect_with_retry(
        "http://x",
        "t",
        wait_s=60,
        sleep_fn=sleeps.append,
        connect=lambda url, token: attempts.pop(0),
    )
    assert caps == {"level": "watch"}
    assert len(sleeps) == 3  # slept between the four attempts
    assert not client.closed  # the successful client stays open


def test_retry_times_out_with_guidance():
    with pytest.raises(sidecar.ServerUnreachable) as excinfo:
        sidecar._connect_with_retry(
            "http://x",
            "t",
            wait_s=0,
            sleep_fn=lambda s: None,
            connect=lambda url, token: _FakeClient([_refused()]),
        )
    assert "cannot reach the server" in str(excinfo.value)


def test_no_server_at_all_times_out_with_the_setup_message(monkeypatch, tmp_path):
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    monkeypatch.setattr(
        "fibsem.server.discovery.DISCOVERY_FILE", tmp_path / "absent.json"
    )
    monkeypatch.setattr(
        "fibsem.server.discovery.LEGACY_DISCOVERY_FILE", tmp_path / "absent.json"
    )
    with pytest.raises(sidecar.ServerUnreachable) as excinfo:
        sidecar._connect_with_retry(None, None, wait_s=0, sleep_fn=lambda s: None)
    assert "no server found" in str(excinfo.value)


def test_the_connection_names_the_sidecar():
    transport = sidecar._connect("http://127.0.0.1:9", "t")
    try:
        assert transport.user_agent == sidecar._user_agent()
    finally:
        transport.close()


def test_a_proposal_image_travels_as_mcp_image_content():
    """Not base64 inside the JSON, which overflowed the tool result (#6)."""
    import asyncio
    import base64
    import json

    jpeg = b"\xff\xd8\xffjpeg"
    payload = {
        "available": True,
        "proposals": [
            {
                "kind": "point_of_interest",
                "reference_image": {
                    "image_b64_jpeg": base64.b64encode(jpeg).decode(),
                    "width": 4,
                },
            }
        ],
    }
    fake = _FakeClient([_FakeResponse(200, payload)])
    capabilities = {"routers": {"app": True}, "level": "watch"}
    mcp = sidecar.build_sidecar(fake, capabilities)
    result = asyncio.run(mcp.call_tool("proposal_list", {}))
    contents = result[0] if isinstance(result, tuple) else result
    contents = getattr(contents, "content", contents)
    texts = [c.text for c in contents if getattr(c, "type", "") == "text"]
    images = [c for c in contents if getattr(c, "type", "") == "image"]
    assert len(images) == 1 and base64.b64decode(images[0].data) == jpeg
    body = json.loads(texts[0])
    assert body["proposals"][0]["reference_image"] == {"width": 4, "image_index": 0}


def test_the_user_agent_names_the_mcp_client():
    from types import SimpleNamespace

    from fibsem import __version__

    assert sidecar._user_agent() == f"fibsem-mcp/{__version__}"
    info = SimpleNamespace(name="Claude Desktop", version="1.2 (beta)")
    assert sidecar._user_agent(info).endswith(" Claude-Desktop/1.2-beta")


def test_the_client_is_named_once_initialize_says_who_it_is():
    import asyncio
    from types import SimpleNamespace

    live = _FakeClient([])
    client = sidecar._LazyClient(lambda: (live, {"level": "watch"}))
    mcp = sidecar.build_sidecar(client, {"routers": {}, "level": "watch"})
    name_client = mcp.middleware[-1]
    info = SimpleNamespace(name="claude-code", version="2.1.3")
    ctx = SimpleNamespace(
        session=SimpleNamespace(client_params=SimpleNamespace(client_info=info))
    )

    async def call_next(ctx):
        return "handled"

    assert asyncio.run(name_client(ctx, call_next)) == "handled"
    client._connected()
    assert live.user_agent.endswith(" claude-code/2.1.3")


def test_start_up_does_not_wait_for_the_server(monkeypatch):
    """The MCP client's handshake is answered whether or not a server answers
    (FIB-1416): the probe gives up at once, every tool is offered, and the
    first call is where the wait and its guidance happen."""
    refused = _FakeClient([_refused()])
    assert sidecar._probe(None, None, connect=lambda u, t: refused) is None
    assert refused.closed
    connects = []

    def connect():
        connects.append(True)
        raise sidecar.ServerUnreachable("fibsem-mcp: no server found.")

    client = sidecar._LazyClient(connect)
    mcp = sidecar.build_sidecar(client, None)
    assert connects == []  # nothing reached for until a tool is called
    names = {tool.name for tool in mcp._tool_manager.list_tools()}
    assert names == {t.name for t in sidecar.CATALOG}
    data, err = sidecar._call(client, "GET", "/app/workflow/status")
    assert data is None and err == "fibsem-mcp: no server found."
    # An error the agent can't take for a result (sim session 2026-10-19).
    assert isinstance(err, sidecar._Failure)
    assert connects == [True]


def test_the_first_call_connects_once_and_keeps_the_client(monkeypatch):
    live = _FakeClient([_FakeResponse(200, {"a": 1}), _FakeResponse(200, {"b": 2})])
    made = []

    def connect():
        made.append(True)
        return live, {"level": "watch"}

    client = sidecar._LazyClient(connect)
    client.set_user_agent("fibsem-mcp/0.6 claude-code/2.1.3")
    assert sidecar._call(client, "GET", "/a") == ({"a": 1}, None)
    assert sidecar._call(client, "GET", "/b") == ({"b": 2}, None)
    assert made == [True]
    assert live.user_agent == "fibsem-mcp/0.6 claude-code/2.1.3"


def test_a_server_that_answers_at_start_up_shapes_the_tools(monkeypatch):
    probe = _FakeClient([_FakeResponse(200, {"routers": {}, "level": "watch"})])
    capabilities = sidecar._probe(None, None, connect=lambda u, t: probe)
    assert capabilities == {"routers": {}, "level": "watch"}
    assert probe.closed


def test_the_connected_line_leaves_out_what_the_server_does_not_say():
    """``--check`` printed a raw None for a field the server left out (FIB-1429)."""
    assert sidecar._connected_lines("http://127.0.0.1:8001", {}) == [
        "fibsem-mcp: connected to http://127.0.0.1:8001"
    ]
    assert sidecar._connected_lines("http://h:1", {"level": "watch"}) == [
        "fibsem-mcp: connected to http://h:1 (agent level watch)"
    ]


def test_the_connected_line_warns_when_the_app_runs_other_code(tmp_path):
    """A sidecar registered bare imports the environment's fibsem, not the
    worktree on the app's PYTHONPATH: say so, and how to fix it (FIB-1429)."""
    from fibsem.server.discovery import fibsem_folder

    same = {"level": "assist", "fibsem": {"path": fibsem_folder()}}
    assert len(sidecar._connected_lines("http://h:1", same)) == 1

    worktree = tmp_path / "worktree"
    other = {"level": "assist", "fibsem": {"path": str(worktree / "fibsem")}}
    line, warning = sidecar._connected_lines("http://h:1", other)
    assert str(worktree / "fibsem") in warning and fibsem_folder() in warning
    assert f"PYTHONPATH={worktree}" in warning
