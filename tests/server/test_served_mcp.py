"""The fibsem MCP the gateway serves (`fibsem.ai.agent.mcp.served`): each caller
on its own token, the watch streamed and kept per caller, app_wait, and nothing
read or written on the microscope PC for a caller."""

import asyncio
import base64
import json
import os
import threading
import time

import httpx
import pytest
import uvicorn

pytest.importorskip("mcp")

from fibsem.ai.agent.mcp import served, sidecar  # noqa: E402
from fibsem.client.transport import Transport  # noqa: E402
from tests.waiting import free_port, wait_for  # noqa: E402

AGENT = "agent-token"
OTHER = "other-token"


@pytest.fixture(scope="module")
def app_server():
    """A real App Server with AutoLamella's context, on a port: its url and
    the context, whose event buffer the tests add to."""
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    from fibsem.ai.agent.levels import AgentLevel
    from fibsem.applications.autolamella.server.context import AutoLamellaAppContext
    from fibsem.server import AuthConfig, build_server
    from fibsem.workflows.session import AppSession

    session = AppSession()
    auth = AuthConfig.generate(level=AgentLevel.DRIVE, token=AGENT)
    context = AutoLamellaAppContext(session)
    app = build_server(
        session=session, app_context=context, auth=auth, operator=session.agent
    )
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"

    def up():
        try:
            return httpx.get(f"{url}/health", timeout=1).is_success
        except httpx.HTTPError:
            return False

    assert wait_for(up)
    yield url, context
    server.should_exit = True
    thread.join(5)
    session.disconnect()


def _transport(url, token):
    found = Transport(url, token, discover=False)
    found.asks = False
    return found


def test_the_options_come_from_the_query():
    assert served.options({}) == (None, None, ())
    assert served.options(
        {"instrument": "fibsem", "record": "no", "always_load": "b, a,,a"}
    ) == ("fibsem", False, ("a", "b"))
    with pytest.raises(ValueError):
        served.options({"record": "only"})


class _Session:
    """Stands for an MCP session: something a weak reference can hold."""


class _Answering:
    def __init__(self, token):
        self.token = token
        self.user_agent = "fibsem-client"
        self.closed = False

    def request(self, method, path, **kwargs):
        return self.token

    def close(self):
        self.closed = True


def test_each_session_calls_with_its_own_callers_token():
    callers = served.Callers("http://gateway", connect=_Answering)
    first, second = _Session(), _Session()

    reset = callers.bind(first, AGENT)
    assert callers.request("GET", "/capabilities") == AGENT
    callers.set_user_agent("fibsem-mcp claude-code/2")
    named = callers.current().transport
    callers.unbind(reset)

    reset = callers.bind(second, OTHER)
    assert callers.request("GET", "/capabilities") == OTHER
    assert callers.current().transport.user_agent == "fibsem-client"
    callers.unbind(reset)

    # The same session keeps its connection, and its client's name.
    reset = callers.bind(first, AGENT)
    assert callers.current().transport is named
    assert named.user_agent == "fibsem-mcp claude-code/2"
    callers.unbind(reset)

    # No caller bound: never another caller's connection.
    with pytest.raises(ConnectionError):
        callers.request("GET", "/capabilities")


def test_a_session_s_connection_closes_with_it():
    callers = served.Callers("http://gateway", connect=_Answering)
    session = _Session()
    callers.unbind(callers.bind(session, AGENT))
    transport = callers._sessions[session].transport
    del session
    assert wait_for(lambda: transport.closed)


def _lines(watcher, count, wait_s=10.0):
    got = []
    deadline = time.monotonic() + wait_s
    while len(got) < count and time.monotonic() < deadline:
        line = watcher.next(0.2)
        if line is not None:
            got.append(line)
    return got


def test_a_watch_says_the_lines_and_the_next_carries_on_after_them(app_server):
    url, context = app_server
    buffer = context._event_buffer
    cursors = {}
    key = served._key(AGENT)

    first = served.Watcher(_transport(url, AGENT), cursors, key, poll_s=0.3).start()
    assert wait_for(lambda: key in cursors or first._thread.is_alive())
    time.sleep(0.5)  # seeded at the tail
    buffer.append("request_decided", {"request_id": "ab12", "outcome": "approved"})
    assert [str(line) for line in _lines(first, 1)] == ["REQUEST ab12 approved"]
    assert wait_for(lambda: key in cursors)
    first.stop()
    assert first.ended.wait(5)

    # Said while no watch ran: the next one says it, and not the one before.
    buffer.append("request_decided", {"request_id": "cd34", "outcome": "approved"})
    second = served.Watcher(_transport(url, AGENT), cursors, key, poll_s=0.3).start()
    assert [str(line) for line in _lines(second, 1)] == ["REQUEST cd34 approved"]
    assert second.next(0.5) is None
    second.stop()
    assert second.ended.wait(5)


def test_app_wait_answers_what_came_and_says_when_nothing_did(app_server):
    url, context = app_server
    callers = served.Callers(url, connect=lambda token: _transport(url, token))
    waits = served.Waits(callers, {})
    session = _Session()
    reset = callers.bind(session, AGENT)
    try:
        assert waits.wait(0.5).startswith("NOTHING NEW in 0.5 s")
        context._event_buffer.append(
            "request_decided", {"request_id": "ef56", "outcome": "approved"}
        )
        assert wait_for(lambda: "REQUEST ef56 approved" in waits.wait(1.0))
        # Capped, so a client's long timeout can't outlast the call.
        assert waits.wait(0.0).startswith("NOTHING NEW in 0 s")
    finally:
        callers.unbind(reset)
        for found in waits._watches.values():
            found.stop()


def test_a_served_fetch_answers_the_file_itself():
    from mcp.server.mcpserver import Image
    from mcp.types import EmbeddedResource

    about, image = sidecar._fetched_content("exp", "a/b.png", b"\x89PNG", "image/png")
    assert json.loads(about) == {"experiment": "exp", "path": "a/b.png", "size": 4}
    assert isinstance(image, Image)

    _, text = sidecar._fetched_content("exp", "p.yaml", b"key: 1\n", "text/plain")
    assert text == "key: 1\n"

    _, blob = sidecar._fetched_content(
        "exp", "x.tif", b"\xff\xfe\x00", "application/octet-stream"
    )
    assert isinstance(blob, EmbeddedResource)
    assert base64.b64decode(blob.resource.blob) == b"\xff\xfe\x00"

    big = b"x" * (sidecar.FETCHED_MAX_BYTES + 1)
    assert sidecar._fetched_content("exp", "big.txt", big, "").startswith("refused")


def test_a_served_sidecar_never_reads_a_callers_file(monkeypatch, tmp_path):
    monkeypatch.setattr(sidecar, "SERVED", True)
    monkeypatch.setenv("FIBSEM_AGENT_WORKDIR", str(tmp_path))
    (tmp_path / "mark.png").write_bytes(b"png")

    refused = sidecar._evidence_body({"file": str(tmp_path / "mark.png")})

    assert "not read through the gateway's MCP" in refused
    # The bytes themselves, or an experiment image, still go.
    assert sidecar._evidence_body({"upload": "cG5n"}) == {"upload": "cG5n"}


class _Served:
    def __init__(self, url, port, server, thread):
        self.url = url
        self.port = port
        self.server = server
        self.thread = thread


@pytest.fixture
def served_mcp(app_server, monkeypatch):
    """The served MCP on a port, its callers' calls going to the App Server
    (standing in for the gateway's port), with a secret as the gateway adds."""
    url, _ = app_server
    monkeypatch.setattr(sidecar, "SERVED", True)

    async def admit(token):
        return None if token == AGENT else (401, b'{"detail": "unpaired"}')

    callers = served.Callers(url, connect=lambda token: _transport(url, token))
    app = served.build_app(url, secret="s3cret", callers=callers, admit=admit)
    sock = served._free_socket("127.0.0.1")
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    assert wait_for(lambda: server.started)
    yield _Served(f"http://127.0.0.1:{port}{served.SERVED_PATH}", port, server, thread)
    server.should_exit = True
    thread.join(5)


HEADERS = {
    "X-Fibsem-Gateway": "s3cret",
    "Accept": "application/json, text/event-stream",
}
INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}


def test_a_call_without_the_gateway_s_secret_or_a_token_is_refused(served_mcp):
    no_secret = httpx.post(
        served_mcp.url, json=INITIALIZE, headers={"Authorization": f"Bearer {AGENT}"}
    )
    assert no_secret.status_code == 401
    assert no_secret.json()["detail"]["error_type"] == "through_gateway"

    no_token = httpx.post(served_mcp.url, json=INITIALIZE, headers=HEADERS)
    assert no_token.status_code == 401
    assert no_token.json()["detail"]["error_type"] == "unauthorized"

    unpaired = httpx.post(
        served_mcp.url,
        json=INITIALIZE,
        headers={**HEADERS, "Authorization": "Bearer nobody"},
    )
    assert unpaired.status_code == 401
    assert unpaired.json() == {"detail": "unpaired"}

    bad_option = httpx.post(
        served_mcp.url + "?record=maybe",
        json=INITIALIZE,
        headers={**HEADERS, "Authorization": f"Bearer {AGENT}"},
    )
    assert bad_option.status_code == 400


def test_an_mcp_client_calls_the_tools_with_its_token(served_mcp):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def run():
        async with httpx.AsyncClient(
            headers={**HEADERS, "Authorization": f"Bearer {AGENT}"}, timeout=30
        ) as http:
            async with streamable_http_client(served_mcp.url, http_client=http) as (
                read,
                write,
                *_,
            ):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    names = {t.name for t in (await session.list_tools()).tools}
                    status = await session.call_tool("app_get_status", {})
                    waited = await session.call_tool("app_wait", {"timeout_s": 0.2})
                    return init, names, status, waited

    init, names, status, waited = asyncio.run(run())

    assert "/watch" in init.instructions
    assert {"app_wait", "app_get_status", "experiment_fetch_file"} <= names
    assert not status.is_error
    assert "workflow" in json.loads(status.content[0].text)
    assert waited.content[0].text.startswith("NOTHING NEW")


def test_the_watch_streams_lines_until_the_caller_leaves(served_mcp, app_server):
    _, context = app_server
    headers = {**HEADERS, "Authorization": f"Bearer {AGENT}"}
    got = []

    def read():
        with httpx.stream(
            "GET", served_mcp.url + "/watch", headers=headers, timeout=30
        ) as answer:
            got.append(answer.status_code)
            for line in answer.iter_lines():
                got.append(line)
                if "gh78" in line:
                    return

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    assert wait_for(lambda: got == [200])
    time.sleep(1.0)  # the watch seeds at the tail
    context._event_buffer.append(
        "request_decided", {"request_id": "gh78", "outcome": "approved"}
    )
    reader.join(15)
    assert got[-1] == "REQUEST gh78 approved"


def test_each_session_keeps_its_own_answers_and_replays_only_them():
    first, second = served.SessionEvents(kept=3), served.SessionEvents(kept=3)
    sent = []

    async def run():
        primed = await first.store_event("1", None)
        await first.store_event("1", {"answer": "first"})
        # Another session's stream of the same request id: never replayed here.
        await second.store_event("1", {"answer": "other"})
        stream = await first.replay_events_after(primed, _collect(sent))
        missing = await first.replay_events_after("nope", _collect(sent))
        return stream, missing

    stream, missing = asyncio.run(run())

    assert stream == "1"
    assert missing is None
    assert [event.message for event in sent] == [{"answer": "first"}]


def _collect(into):
    async def send(event):
        into.append(event)

    return send


def _sse(text):
    """The (id, data) of each event in an SSE body."""
    events, current = [], {}
    for line in text.splitlines():
        if not line:
            if current:
                events.append((current.get("id"), current.get("data")))
            current = {}
        elif ":" in line:
            field, _, value = line.partition(":")
            current[field] = value.strip()
    if current:
        events.append((current.get("id"), current.get("data")))
    return events


def test_an_answer_whose_stream_dropped_is_picked_up_again(served_mcp):
    headers = {
        **HEADERS,
        "Authorization": f"Bearer {AGENT}",
        "MCP-Protocol-Version": "2025-11-25",
    }
    opened = httpx.post(
        served_mcp.url,
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )
    session = {**headers, "Mcp-Session-Id": opened.headers["mcp-session-id"]}
    httpx.post(
        served_mcp.url,
        headers=session,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    call = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "app_get_status", "arguments": {}},
    }
    events = _sse(httpx.post(served_mcp.url, headers=session, json=call).text)
    primed = events[0][0]
    assert primed is not None, "no event ids: a dropped answer couldn't be resumed"

    # As if the stream had dropped after the first event: ask from there. The
    # replayed stream stays open for what comes next, so read until the answer.
    answer = None
    with httpx.stream(
        "GET",
        served_mcp.url,
        headers={**session, "Last-Event-ID": primed},
        timeout=10,
    ) as resumed:
        for line in resumed.iter_lines():
            if line.startswith("data:") and line[5:].strip():
                said = json.loads(line[5:])
                if said.get("id") == 2:
                    answer = said
                    break

    assert answer is not None and "result" in answer


def test_a_changing_call_with_no_answer_says_it_may_have_run():
    from fibsem.client.transport import ServerUnreachable

    lost = ServerUnreachable("POST /microscope/stage/move: no answer", timed_out=True)
    refused = ServerUnreachable("nothing answers", timed_out=False)

    assert sidecar._unreachable(lost, "POST").endswith(sidecar.MAY_HAVE_RUN)
    assert sidecar._unreachable(lost, "GET") == str(lost)
    assert sidecar._unreachable(refused, "POST") == str(refused)
