"""The agent a hub runs for an instrument: paired at the instrument's gateway,
started on the hub's PC, within the instrument's own limit."""

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fibsem.ai.agent.levels import AgentLevel
from fibsem.gateway.agent import API_KEY, HUB_AGENT_NAME, GatewayAgent
from fibsem.gateway.gateway import GatewayRefused
from fibsem.gateway.hub import Hub
from fibsem.gateway.hub_agents import HubAgents
from fibsem.gateway.server import GatewayAuth, build_gateway_app
from fibsem.server.pairing import HUB, SWITCH_MODES
from tests.gateway.test_gateway_agent import App, FakeGateway
from tests.gateway.test_hub import Ssh

ROOT = Path(__file__).resolve().parents[2]
STAND_IN = ROOT / "tests" / "agent" / "stand_in.py"
OPERATOR_TOKEN = "operator-token"
GATEWAY = "http://127.0.0.1:50000"
APP = "http://127.0.0.1:50001"


def _auth(tmp_path, name):
    token_file = tmp_path / f"{name}-token.json"
    token_file.write_text(json.dumps({"token": OPERATOR_TOKEN}))
    return GatewayAuth.from_disk(token_file, tmp_path / f"{name}-paired.json")


class Scope:
    """An instrument PC: its gateway app with a real `GatewayAgent`, the app it
    serves, and this PC's limit on a hub's level."""

    def __init__(self, tmp_path):
        self.app = App()
        self.limit = None
        self.auth = _auth(tmp_path, "scope")
        self.agent = GatewayAgent(
            FakeGateway(),
            self.auth.pairing.tokens,
            root=tmp_path / "scope-sessions",
            discovery=self.app.discovery,
            level=self.app.level,
            reachable=lambda url: self.app.up,
            last_heard=self.app.heard,
            environ={},
            stored=lambda name: None,
            microscope="scope-pc",
            hub_max_level=lambda: self.limit,
        )
        self.client = TestClient(
            build_gateway_app(FakeGateway(), self.auth, self.agent)
        )
        self.up = True

    def fetch(self, url, route, token, timeout, method="GET", body=None):
        if not self.up:
            raise OSError("connection refused")
        if url == APP:
            return (200 if self.app.up else 503), {}
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = self.client.request(method, route, json=body, headers=headers)
        return response.status_code, response.json()

    def pair(self, url, code, name):
        response = self.client.post("/pair", json={"code": code, "name": name})
        return response.json()["token"], response.json()["role"]

    def paired_agent(self):
        return self.auth.pairing.tokens.agent()


@pytest.fixture
def scope(tmp_path):
    return Scope(tmp_path)


@pytest.fixture
def made(tmp_path, scope, monkeypatch):
    monkeypatch.setenv("STAND_IN_ROOT", str(tmp_path / "hub-sessions"))
    monkeypatch.setenv("STAND_IN_PLAN", "wait")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    hub = Hub(
        tmp_path / "hub.yaml",
        tmp_path / "hub",
        fetch=scope.fetch,
        pair=scope.pair,
        tunnels=Ssh().tunnels,
    )
    hub.add("Scope 1", ssh="user@scope-1")
    built = []

    def make(grants=(SWITCH_MODES,), key="sk-test"):
        hub.pair("Scope 1", scope.auth.pairing.open(role=HUB, grants=grants))
        agents = HubAgents(
            hub,
            root=tmp_path / "hub-sessions",
            command=[sys.executable, str(STAND_IN)],
            end_wait_s=5.0,
            environ={API_KEY: key} if key else {},
            stored=lambda name: None,
        )
        built.append(agents)
        return hub, agents

    yield make
    for agents in built:
        for each in list(agents._manager._agents.values()):
            agents._manager.end(each.microscope)


def until(test, agents, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        agents.poll()
        if test():
            return True
        time.sleep(0.05)
    return False


def _args(tmp_path):
    (folder,) = (tmp_path / "hub-sessions").iterdir()
    return json.loads((folder / "session.json").read_text())["args"]


def test_start_pairs_at_the_instrument_and_runs_the_agent_on_the_hub(
    made, scope, tmp_path
):
    hub, agents = made()
    session = agents.start("Scope 1", "default", "mill 4", "assist", "operator", 5.0)
    assert session["state"] == "running" and session["level"] == "assist"
    assert session["playbook"] == "default" and session["goal"] == "mill 4"
    # Paired there as the hub's agent, at the level asked for.
    assert scope.paired_agent().name == HUB_AGENT_NAME
    assert scope.app.levels[-1][1] == "assist"
    # Started here, pointed at the app through the tunnel.
    assert until(lambda: (tmp_path / "hub-sessions").is_dir(), agents)
    assert until(
        lambda: any((tmp_path / "hub-sessions").glob("*/session.json")), agents
    )
    assert _args(tmp_path) == [
        "--playbook",
        "default",
        "--url",
        APP,
        "--max-budget-usd",
        "5",
    ]
    (row,) = hub.state().instruments
    assert row.session["session_id"] == session["session_id"]

    agents.end("Scope 1", "operator")
    assert scope.paired_agent() is None
    assert scope.app.levels[-1][1] == "watch"
    assert hub.state().instruments[0].session is None


def test_the_instruments_limit_refuses_a_higher_level_from_the_hub(made, scope):
    scope.limit = AgentLevel.ASSIST
    hub, agents = made()
    with pytest.raises(GatewayRefused) as refused:
        agents.start("Scope 1", "default", None, "drive", "operator")
    assert refused.value.error_type == "level_above_limit"
    assert "up to assist" in str(refused.value)
    assert scope.paired_agent() is None and agents.session("Scope 1") is None

    agents.start("Scope 1", "default", None, "assist", "operator")
    with pytest.raises(GatewayRefused):
        agents.level("Scope 1", "drive")
    assert agents.level("Scope 1", "watch") == {"level": "watch"}
    assert scope.app.levels[-1][1] == "watch"


def test_the_operator_at_the_instrument_is_not_held_to_the_hubs_limit(scope):
    scope.limit = AgentLevel.WATCH
    operator = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
    answer = scope.client.put(
        "/gateway/agent/level", json={"level": "drive"}, headers=operator
    )
    assert answer.status_code == 200 and scope.app.levels[-1][1] == "drive"


def test_a_hub_without_the_grant_cannot_pair_an_agent(made, scope):
    hub, agents = made(grants=())
    with pytest.raises(GatewayRefused) as refused:
        agents.start("Scope 1", "default", None, "watch", "operator")
    assert refused.value.error_type == "no_grant"
    assert scope.paired_agent() is None


def test_start_is_refused_before_anything_is_paired(made, scope):
    hub, agents = made(key=None)
    with pytest.raises(GatewayRefused) as refused:
        agents.start("Scope 1", "default", None, "watch", "operator")
    assert refused.value.error_type == "no_api_key"
    assert scope.paired_agent() is None

    hub, agents = made()
    scope.app.up = False
    with pytest.raises(GatewayRefused) as refused:
        agents.start("Scope 1", "default", None, "watch", "operator")
    assert refused.value.error_type == "no_app"

    scope.app.up = True
    agents.start("Scope 1", "default", None, "watch", "operator")
    with pytest.raises(GatewayRefused) as refused:
        agents.start("Scope 1", "default", None, "watch", "operator")
    assert refused.value.error_type == "agent_running"


def test_an_agent_someone_else_paired_is_not_replaced_without_saying_so(made, scope):
    hub, agents = made()
    scope.auth.pairing.tokens.add("claude-code", role="agent")
    with pytest.raises(GatewayRefused) as refused:
        agents.start("Scope 1", "default", None, "watch", "operator")
    assert refused.value.error_type == "agent_paired"
    agents.start("Scope 1", "default", None, "watch", "operator", replace=True)
    assert scope.paired_agent().name == HUB_AGENT_NAME


def test_an_instrument_reached_without_a_tunnel_gets_no_agent(made):
    hub, agents = made()
    hub.add("Scope 2", url="http://scope-2:8010")
    with pytest.raises(GatewayRefused) as refused:
        agents.start("Scope 2", "default", None, "watch", "operator")
    assert refused.value.error_type in ("not_paired", "no_tunnel")


def test_the_hubs_routes_start_end_and_set_the_level(made, scope, tmp_path):
    hub, agents = made()
    auth = _auth(tmp_path, "hub")
    client = TestClient(
        build_gateway_app(FakeGateway(), auth, hub=hub, hub_agents=agents)
    )
    operator = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
    body = {"playbook": "default", "goal": "mill 4", "level": "watch"}
    assert (
        client.post("/hub/instruments/Scope 1/agent/start", json=body).status_code
        == 401
    )
    started = client.post(
        "/hub/instruments/Scope 1/agent/start", json=body, headers=operator
    )
    assert started.status_code == 200
    assert started.json()["instruments"][0]["session"]["goal"] == "mill 4"

    level = client.put(
        "/hub/instruments/Scope 1/level", json={"level": "assist"}, headers=operator
    )
    assert level.json() == {"level": "assist"}
    bad = client.put(
        "/hub/instruments/Scope 1/level", json={"level": "max"}, headers=operator
    )
    assert bad.status_code == 422 and bad.json()["detail"]["error_type"] == "bad_level"

    ended = client.post("/hub/instruments/Scope 1/agent/end", headers=operator)
    assert ended.json()["instruments"][0]["session"] is None
    missing = client.post("/hub/instruments/Nope/agent/end", headers=operator)
    assert missing.status_code == 404


def test_the_hub_reads_the_level_the_instrument_caps_it_at(made, scope):
    hub, _ = made()
    hub.check()
    assert hub.state().instruments[0].level_cap is None
    scope.limit = AgentLevel.ASSIST
    hub.check()
    assert hub.state().instruments[0].level_cap == "assist"
