"""The gateway's MCP (`fibsem.gateway.mcp`): started on the first call to /mcp,
then every call passed on with its secret; a connect token answered here; none
without the mcp extra."""

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fibsem.gateway import child_secrets
from fibsem.gateway.mcp import NO_MCP, GatewayMcp
from fibsem.server.gateway_secret import HEADER

CONNECT = "connect-token"
PID = 424242


class _Process:
    """Stands for the MCP's child process: found once started."""

    def __init__(self):
        self.started = 0
        self.closed = False

    def found(self):
        return {"url": "http://mcp.test", "pid": PID} if self.started else None

    def ensure(self):
        self.started += 1

    def close(self):
        self.closed = True


def _gateway(installed=True):
    seen = []

    def answer(request):
        seen.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(b"{}"))

    process = _Process()
    mcp = GatewayMcp(
        "python",
        "http://127.0.0.1:8010",
        process=process,
        is_connect=lambda token: token == CONNECT,
        installed=lambda: installed,
        transport=httpx.MockTransport(answer),
        start_wait_s=1.0,
        poll_s=0.01,
    )
    app = FastAPI()
    app.include_router(mcp.router())
    return TestClient(app), mcp, process, seen


def test_nothing_starts_until_something_calls_the_mcp():
    _, mcp, process, _ = _gateway()

    mcp.ensure()

    assert process.started == 0
    assert not mcp.wanted


def test_the_first_call_starts_it_and_every_call_carries_its_secret():
    client, mcp, process, seen = _gateway()
    child_secrets.keep(PID, "the-secret")
    try:
        answer = client.post(
            "/mcp/fibsem?instrument=fibsem",
            json={},
            headers={"Authorization": "Bearer agent-token"},
        )
    finally:
        child_secrets.forget(PID)

    assert answer.status_code == 200
    assert process.started == 1
    assert mcp.wanted
    (sent,) = seen
    assert str(sent.url) == "http://mcp.test/mcp/fibsem?instrument=fibsem"
    assert sent.headers[HEADER] == "the-secret"
    assert sent.headers["authorization"] == "Bearer agent-token"
    # Kept running from here: a watcher starts it again if it ends.
    mcp.ensure()
    assert process.started == 2


def test_a_connect_token_is_answered_by_the_gateway():
    client, _, _, seen = _gateway()

    answer = client.get(
        "/mcp/fibsem/watch", headers={"Authorization": f"Bearer {CONNECT}"}
    )

    assert answer.status_code == 403
    assert answer.json()["detail"]["error_type"] == "connect_first"
    assert seen == []


def test_without_the_mcp_extra_it_says_what_to_install():
    client, _, process, _ = _gateway(installed=False)

    answer = client.post("/mcp/fibsem", json={})

    assert answer.status_code == 503
    assert answer.json()["detail"]["error_type"] == NO_MCP
    assert "fibsem[mcp]" in answer.json()["detail"]["message"]
    assert process.started == 0


def test_one_that_never_starts_says_so():
    client, mcp, process, _ = _gateway()
    process.found = lambda: None

    answer = client.post("/mcp/fibsem", json={})

    assert answer.status_code == 503
    assert "didn't start" in answer.json()["detail"]["message"]
