"""``fibsem-gateway pair``: a code for the operator's browser, an agent or a
hub, from the gateway that runs here."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from fibsem.gateway import main as gateway_main


@pytest.fixture
def asked(monkeypatch):
    calls = []

    def call(method, route, body=None):
        calls.append((method, route, body))
        return {"code": "123456", "role": body["role"]}

    monkeypatch.setattr(gateway_main, "call", call)
    monkeypatch.setattr(gateway_main.diagnostics, "install", lambda name: None)
    return calls


@pytest.mark.parametrize(
    ("flags", "body"),
    [
        ([], {"role": "operator"}),
        (["--agent"], {"role": "agent"}),
        (["--hub"], {"role": "hub"}),
        (["--hub", "--may-switch"], {"role": "hub", "grants": ["switch_modes"]}),
    ],
)
def test_pair_opens_pairing_for_the_role_asked(asked, capsys, flags, body):
    assert gateway_main.main(["pair", *flags]) == 0
    assert asked == [("POST", "/pairing", body)]
    assert f"123456 (5 minutes, {body['role']})" in capsys.readouterr().out


def test_an_agent_and_a_hub_at_once_is_refused(asked):
    with pytest.raises(SystemExit):
        gateway_main.main(["pair", "--agent", "--hub"])
    assert asked == []


def test_the_gateway_file_is_this_users_alone(tmp_path):
    """The file a program here finds the gateway by: its address and the token
    that only asks to connect, readable by this PC's user only."""
    import json
    import os
    import stat

    from fibsem.gateway.gateway import write_gateway_file

    path = write_gateway_file("http://127.0.0.1:8010", "ask", tmp_path / "g.json")
    found = json.loads(path.read_text())
    started = found.pop("started")
    assert found.pop("program")
    assert found == {
        "url": "http://127.0.0.1:8010",
        "pid": os.getpid(),
        "connect_token": "ask",
    }
    from fibsem.util.process import same_process

    assert same_process(os.getpid(), started), "a reused pid isn't taken for it"
    assert not same_process(os.getpid(), started - 3600)
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["g.json"]


def test_a_taken_port_stops_the_gateway_before_it_writes_its_file(monkeypatch, capsys):
    """Another program on the port: the gateway says so and ends, and never
    writes gateway.json, which would send agents to that program."""
    import socket

    from fibsem.gateway import server as gateway_server

    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    port = taken.getsockname()[1]
    written = []

    class Stub:
        def __init__(self, *args, **kwargs):
            self.pairing = type("P", (), {"tokens": None})()
            self.connect_token = "c"
            self.is_connect = lambda token: False

        def start(self):
            pytest.fail("started")

        def close(self):
            pass

    monkeypatch.setattr(gateway_main, "read_gateway_file", lambda: None)
    monkeypatch.setattr(gateway_main, "Gateway", Stub)
    monkeypatch.setattr(gateway_main, "GatewayAgent", Stub)
    monkeypatch.setattr(gateway_main, "HostMicroscope", Stub)
    monkeypatch.setattr(gateway_server.GatewayAuth, "from_disk", Stub)
    monkeypatch.setattr(
        gateway_server, "build_gateway_app", lambda *a, **k: gateway_server.FastAPI()
    )
    monkeypatch.setattr(
        gateway_main, "write_gateway_file", lambda *a: written.append(a)
    )
    try:
        assert gateway_main.run(port, window=False, connection=[]) == 1
    finally:
        taken.close()
    assert written == []
    assert f"could not listen on port {port}" in capsys.readouterr().out


def test_the_port_comes_from_the_preferences_and_the_flag_wins(monkeypatch):
    """install.gateway.port sets where the gateway listens and where this PC
    looks for it; --port overrides it for one run."""
    from types import SimpleNamespace

    from fibsem import config
    from fibsem.server import discovery

    prefs = config.UserPreferences()
    prefs.install.gateway.port = 8020
    monkeypatch.setattr(config, "load_user_preferences", lambda: prefs)
    monkeypatch.setattr(discovery, "read_gateway_file", lambda: None)
    assert discovery.gateway_port() == 8020
    assert discovery.gateway_url() == "http://127.0.0.1:8020"
    ran = []
    monkeypatch.setattr(gateway_main, "run", lambda port, *a: ran.append(port) or 0)
    monkeypatch.setattr(gateway_main.diagnostics, "install", lambda name: None)
    monkeypatch.setattr(gateway_main, "refuse", lambda *a, **k: None, raising=False)
    gateway_main.main(["run", "--no-window"])
    gateway_main.main(["run", "--no-window", "--port", "8030"])
    assert ran == [8020, 8030]
    # A port out of range, or preferences that can't be read, is the default.
    prefs.install.gateway.port = 0
    assert discovery.gateway_port() == discovery.GATEWAY_PORT
    monkeypatch.setattr(config, "load_user_preferences", SimpleNamespace)
    assert discovery.gateway_port() == discovery.GATEWAY_PORT


def test_a_refused_token_names_the_gateway_while_one_runs(monkeypatch):
    """An agent calling the App Server with a token it doesn't know is told to
    connect through this PC's gateway, when one runs."""
    from fibsem.server import auth

    monkeypatch.setattr(auth, "read_gateway_file", lambda: None)
    assert auth._unauthorized().detail["message"] == "Missing or invalid bearer token."
    gateway = {"url": "http://127.0.0.1:8010", "pid": 1}
    monkeypatch.setattr(auth, "read_gateway_file", lambda: gateway)
    refused = auth._unauthorized()
    assert refused.status_code == 401
    assert refused.detail["error_type"] == "unauthorized"
    assert "through its gateway (http://127.0.0.1:8010)" in refused.detail["message"]


def test_while_a_gateway_runs_an_agent_call_must_come_through_it(monkeypatch, tmp_path):
    """The App Server's secret: the gateway adds it to each call it passes on;
    an agent's call without it is refused while a gateway runs. The App
    Server's own token (run alone: tests, evals) and the operator's pass."""
    from types import SimpleNamespace

    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    from fibsem.ai.agent.levels import AgentLevel
    from fibsem.server import auth
    from fibsem.server.discovery import write_discovery_file
    from fibsem.server.pairing import AGENT

    config = auth.AuthConfig(token="own", operator_token="operator")
    monkeypatch.setattr(
        config,
        "paired",
        lambda token: (
            SimpleNamespace(role=AGENT, name="claude-code")
            if token == "paired"
            else None
        ),
    )
    written = write_discovery_file("127.0.0.1", 8001, config, tmp_path / "app.json")
    secret = json.loads(written.read_text())["gateway_secret"]
    assert secret == config.gateway_secret
    app = FastAPI(dependencies=[Depends(auth.require_level(AgentLevel.WATCH))])
    app.state.auth = config
    app.get("/app/x")(lambda: {})
    client = TestClient(app)

    def call(token, through=False):
        headers = {"Authorization": f"Bearer {token}"}
        if through:
            headers[auth.GATEWAY_HEADER] = secret
        return client.get("/app/x", headers=headers)

    monkeypatch.setattr(auth, "read_gateway_file", lambda: None)
    assert call("paired").status_code == 200, "no gateway: as before"
    monkeypatch.setattr(auth, "read_gateway_file", lambda: {"url": "http://g:8010"})
    refused = call("paired")
    assert refused.status_code == 401
    assert refused.json()["detail"]["error_type"] == "through_gateway"
    assert "http://g:8010" in refused.json()["detail"]["message"]
    assert call("paired", through=True).status_code == 200
    assert call("own").status_code == 200
    assert call("operator").status_code == 200


def test_the_pass_through_adds_the_secret_and_drops_a_callers(tmp_path):
    from fibsem.gateway.passthrough import PassThrough
    from tests.gateway.test_gateway_passthrough import AGENT, _as, _gateway_app

    seen = []

    def answer(request):
        seen.append(request.headers.get_list("x-fibsem-gateway"))
        return httpx.Response(200, stream=httpx.ByteStream(b"{}"))

    passthrough = PassThrough(
        lambda: "http://app.test",
        transport=httpx.MockTransport(answer),
        secret=lambda: "s3cret",
    )
    through = TestClient(_gateway_app(tmp_path, None, passthrough))
    through.get("/app/x", headers={**_as(AGENT), "X-Fibsem-Gateway": "forged"})
    assert seen == [["s3cret"]]
