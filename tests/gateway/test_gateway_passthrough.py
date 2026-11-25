"""The gateway's one port: the App Server's routes through the gateway answer
exactly as the App Server does, the caller's token checked by the App Server as
ever; with no App Server the gateway says so; stops go their own way."""

import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from fibsem.gateway.gateway import Gateway
from fibsem.gateway.passthrough import CONNECT_FIRST, PassThrough, is_stop
from fibsem.gateway.server import GatewayAuth, build_gateway_app
from fibsem.server.pairing import HUB, OPERATOR, PairedTokens, Pairing
from tests.waiting import free_port, wait_for

AGENT = "agent-token"
OPERATOR_TOKEN = "operator-token"


@pytest.fixture(scope="module")
def app_server():
    """A real App Server over a session with no microscope, on a port."""
    import os

    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    from fibsem.ai.agent.levels import AgentLevel
    from fibsem.applications.autolamella.server.context import AutoLamellaAppContext
    from fibsem.server import AuthConfig, build_server
    from fibsem.workflows.session import AppSession

    session = AppSession()
    auth = AuthConfig.generate(level=AgentLevel.DRIVE, token=AGENT)
    auth.operator_token = OPERATOR_TOKEN
    app = build_server(
        session=session,
        app_context=AutoLamellaAppContext(session),
        auth=auth,
        operator=session.agent,
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
    yield url, session
    server.should_exit = True
    thread.join(5)
    session.disconnect()


@pytest.fixture
def gateways(tmp_path):
    """Gateways over an App Server url, each a client on one event loop, as
    uvicorn runs it."""
    made = []

    def make(upstream):
        client = TestClient(_gateway_app(tmp_path, upstream))
        client.__enter__()
        made.append(client)
        return client

    yield make
    for client in made:
        client.__exit__(None, None, None)


def _gateway_app(tmp_path, upstream, passthrough=None):
    gateway = Gateway(
        home=tmp_path,
        app_registry=lambda: {},
        last=lambda: None,
        recent=lambda: [],
        new_folder=lambda: str(tmp_path),
        configurations=lambda: [],
        discovery=lambda: None,
    )
    auth = GatewayAuth(
        token=OPERATOR_TOKEN,
        pairing=Pairing(PairedTokens(tmp_path / "paired.json"), roles=(OPERATOR, HUB)),
    )
    return build_gateway_app(
        gateway, auth, passthrough=passthrough or PassThrough(upstream)
    )


def _as(token):
    return {"Authorization": f"Bearer {token}"}


CALLS = [
    ("GET", "/capabilities", AGENT, None),
    ("GET", "/capabilities", None, None),
    ("GET", "/app/workflow/status", AGENT, None),
    ("GET", "/microscope", AGENT, None),
    ("GET", "/microscope/connection", OPERATOR_TOKEN, None),
    ("GET", "/microscope/connection", AGENT, None),
    ("GET", "/microscope/health", OPERATOR_TOKEN, None),
    ("GET", "/app/no/such/route", AGENT, None),
    ("POST", "/microscope/connect", OPERATOR_TOKEN, {"configuration": "nope"}),
    ("POST", "/app/experiment/notes", AGENT, {"text": ""}),
]


@pytest.mark.parametrize("method,path,token,body", CALLS)
def test_an_answer_through_the_gateway_is_the_app_servers(
    app_server, gateways, method, path, token, body
):
    url, _ = app_server
    gateway = gateways(lambda: url)
    headers = _as(token) if token else {}
    direct = httpx.request(method, f"{url}{path}", json=body, headers=headers)
    through = gateway.request(method, path, json=body, headers=headers)
    assert through.status_code == direct.status_code, through.text
    assert through.content == direct.content
    assert through.headers.get("content-type") == direct.headers.get("content-type")


def test_a_query_and_a_long_poll_pass_through(app_server, gateways):
    url, _ = app_server
    gateway = gateways(lambda: url)
    now = gateway.get("/app/events", headers=_as(AGENT)).json()["latest_seq"] or 0
    started = time.monotonic()
    answer = gateway.get(
        "/app/events", params={"since": now, "timeout": 1}, headers=_as(AGENT)
    )
    assert answer.status_code == 200, answer.text
    assert time.monotonic() - started >= 0.9, "the App Server held it, not the gateway"


def test_the_gateways_own_routes_stay_its_own(app_server, gateways):
    url, _ = app_server
    gateway = gateways(lambda: url)
    health = gateway.get("/health").json()
    assert (health["status"], health["program"]) == ("ok", "fibsem-gateway")
    assert gateway.get("/gateway", headers=_as(OPERATOR_TOKEN)).status_code == 200


def test_with_no_app_the_gateway_says_so(gateways):
    gateway = gateways(lambda: None)
    answer = gateway.get("/app/workflow/status", headers=_as(AGENT))
    assert answer.status_code == 503
    assert answer.json()["detail"]["error_type"] == "no_app"


def test_the_connect_token_is_never_passed_to_the_app_server(tmp_path):
    """The gateway file's token only asks to connect: the gateway answers it
    itself, without asking the App Server."""
    asked = []
    passthrough = PassThrough(
        lambda: asked.append(1) or None, is_connect=lambda token: token == "connect"
    )
    through = TestClient(_gateway_app(tmp_path, None, passthrough))
    answer = through.get("/capabilities", headers=_as("connect"))
    assert answer.status_code == 403
    assert answer.json()["detail"]["error_type"] == CONNECT_FIRST
    assert asked == []
    assert through.get("/capabilities", headers=_as(AGENT)).status_code == 503
    assert asked == [1]


def test_an_app_that_has_gone_is_a_502(gateways):
    gateway = gateways(lambda: f"http://127.0.0.1:{free_port()}")
    answer = gateway.get("/capabilities", headers=_as(AGENT))
    assert answer.status_code == 502
    assert answer.json()["detail"]["error_type"] == "no_app"


def test_stops_are_told_apart():
    for path in (
        "/app/workflow/stop",
        "/app/workflow/stop_task",
        "/app/agent/stop",
        "/microscope/devices/stage/commands/stop",
        "/microscope/devices/milling/commands/safety_halt",
    ):
        assert is_stop(path), path
    for path in ("/app/workflow/status", "/app/stopwatch", "/microscope/devices"):
        assert not is_stop(path), path


def test_a_client_through_the_gateway_reads_no_app_as_unreachable(gateways):
    from fibsem.client.transport import (
        GATEWAY_APP_LOST,
        GATEWAY_NO_APP,
        ServerUnreachable,
    )
    from fibsem.gateway.passthrough import APP_LOST, NO_APP

    assert (NO_APP, APP_LOST) == (GATEWAY_NO_APP, GATEWAY_APP_LOST)
    gateway = gateways(lambda: None)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(gateway.app, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: _answers(port))
        from fibsem.client.transport import Transport

        transport = Transport(f"http://127.0.0.1:{port}", AGENT, discover=False)
        try:
            with pytest.raises(ServerUnreachable) as raised:
                transport.request("GET", "/capabilities")
            assert not raised.value.timed_out
        finally:
            transport.close()
    finally:
        server.should_exit = True
        thread.join(5)


def _answers(port):
    try:
        return httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).is_success
    except httpx.HTTPError:
        return False


def test_a_call_that_says_how_long_it_waits_is_waited_for(tmp_path):
    """A long acquisition says how long it waits: the gateway waits that
    long and a margin, not its usual 300 s; a stop keeps its short wait."""
    from fibsem.client.transport import WAIT_HEADER, _read_s

    waited = {}

    def answer(request):
        waited[request.url.path] = request.extensions["timeout"]["read"]
        return httpx.Response(200, stream=httpx.ByteStream(b"{}"))

    passthrough = PassThrough(
        lambda: "http://app.test", transport=httpx.MockTransport(answer)
    )
    through = TestClient(_gateway_app(tmp_path, None, passthrough))
    through.get("/microscope/acquire", headers={**_as(AGENT), WAIT_HEADER: "1000"})
    through.get("/app/workflow/status", headers=_as(AGENT))
    through.get("/app/agent/stop", headers={**_as(AGENT), WAIT_HEADER: "1000"})
    through.get("/microscope/huge", headers={**_as(AGENT), WAIT_HEADER: "1e9"})
    assert waited == {
        "/microscope/acquire": 1030.0,
        "/app/workflow/status": 300.0,
        "/app/agent/stop": 30.0,
        "/microscope/huge": 3 * 3600.0,
    }
    assert _read_s(httpx.Timeout(10.0, read=400.0)) == 400.0
    assert _read_s(None) is None


def test_with_no_app_the_agent_hosts_routes_go_to_it(tmp_path):
    """While no App Server runs, the agent host answers its own routes; one
    only the app has is the gateway's no_app, not the host's 404, and the
    App Server's secret never goes to the host."""
    import json

    seen = []

    def reply(status, body):
        return httpx.Response(
            status,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps(body).encode()),
        )

    def host(request):
        seen.append((request.url.host, request.headers.get("x-fibsem-gateway")))
        if request.url.path == "/app/agent/plan":
            return reply(200, {"plan": "yes"})
        if request.url.path == "/app/agent/missing":
            return reply(404, {"detail": {"error_type": "no_plan", "message": "x"}})
        return reply(404, {"detail": "Not Found"})

    app = {"url": None}
    passthrough = PassThrough(
        lambda: app["url"],
        transport=httpx.MockTransport(host),
        secret=lambda: "app-secret",
        host=lambda: "http://host.test",
    )
    through = TestClient(_gateway_app(tmp_path, None, passthrough))
    assert through.get("/app/agent/plan", headers=_as(AGENT)).json() == {"plan": "yes"}
    answer = through.get("/microscope/connection", headers=_as(AGENT))
    assert answer.status_code == 503
    assert answer.json()["detail"]["error_type"] == "no_app"
    # The host's own refusals come through as it made them.
    answer = through.get("/app/agent/missing", headers=_as(AGENT))
    assert answer.status_code == 404
    assert answer.json()["detail"]["error_type"] == "no_plan"
    assert {secret for _, secret in seen} == {None}
    # With the app running again, its calls go to it.
    app["url"] = "http://app.test"
    through.get("/app/agent/plan", headers=_as(AGENT))
    assert seen[-1] == ("app.test", "app-secret")


def test_a_library_save_through_the_gateway_goes_in_history(
    app_server, tmp_path, monkeypatch
):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    url, _ = app_server
    app = _gateway_app(tmp_path, lambda: url)
    with TestClient(app) as gateway:
        r = gateway.post(
            "/app/library/playbooks/my-lamellae", headers=_as(OPERATOR_TOKEN)
        )
        assert r.status_code == 200, r.text
        history = gateway.get("/gateway/history", headers=_as(OPERATOR_TOKEN))
    assert "made playbook my-lamellae" in history.text


def test_each_server_gets_its_own_secret(tmp_path):
    """The App Server's secret goes only to it, the agent host's only to the
    host."""
    import json

    seen = []

    def server(request):
        seen.append((request.url.host, request.headers.get("x-fibsem-gateway")))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps({}).encode()),
        )

    app = {"url": None}
    passthrough = PassThrough(
        lambda: app["url"],
        transport=httpx.MockTransport(server),
        secret=lambda: "app-secret",
        host=lambda: "http://host.test",
        host_secret=lambda: "host-secret",
    )
    through = TestClient(_gateway_app(tmp_path, None, passthrough))
    through.get("/app/agent/plan", headers=_as(AGENT))
    app["url"] = "http://app.test"
    through.get("/app/agent/plan", headers=_as(AGENT))
    assert seen == [("host.test", "host-secret"), ("app.test", "app-secret")]
