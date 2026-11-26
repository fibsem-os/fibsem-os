"""An agent on the hub PC: asking an instrument's gateway for its slot,
keeping it, joining again after the connection was away, and the endings."""

from types import SimpleNamespace

import pytest

from fibsem.gateway.gateway import GatewayRefused
from fibsem.hub.agents import HubAgents
from fibsem.hub.hub import Hub, Instrument, crosses_in_clear

URL = "https://scope-1:8010"


class FakeSlot:
    """The gateway's side of the slot routes, as agreed."""

    def __init__(self):
        self.calls = []
        self.down = False
        self.refuse = {}
        self.holder = None

    def send(self, url, method, route, token, body, timeout):
        self.calls.append((route, dict(body or {})))
        if self.down:
            raise OSError("connection refused")
        name = route.rsplit("/", 1)[-1]
        if name in self.refuse:
            reason = self.refuse[name]
            detail = {"error_type": reason, "message": f"Refused: {reason}."}
            return 409, {"detail": detail}
        if name == "slot":
            self.holder = body["session_id"]
            token = "new-token" if body.get("rejoin") else "agent-token"
            return 200, {"token": token, "lease_s": 30, "grace_s": 120}
        if name == "release":
            self.holder = None
        return 200, {"lease_s": 30, "state": "running"}


class FakeManager:
    def __init__(self):
        self.agents = {}
        self.started = []
        self.parked = []
        self.fail = False

    def start(self, name, session_id, args, goal=None, env=None, about=None):
        if self.fail:
            raise OSError("no such program")
        self.started.append((name, session_id, args, env))
        self.agents[name] = SimpleNamespace(state="running", ended=None)

    def end(self, name):
        if name in self.agents:
            self.agents[name].state = "ended"

    def park(self, name):
        self.parked.append(name)
        self.agents[name].state = "waiting"
        return self.agents[name]

    def unpark(self, name):
        # The real manager resumes it on its next poll, with its env.
        self.agents[name].state = "running"

    def agent(self, name):
        return self.agents.get(name)

    def poll(self):
        pass


@pytest.fixture
def made(tmp_path):
    slot, manager, now = FakeSlot(), FakeManager(), [1000.0]
    hub = Hub(
        file=tmp_path / "hub.yaml",
        tokens=tmp_path / "hub",
        fetch=lambda *a: (200, {}),
        pair=lambda url, code, name: ("hub-token", "hub"),
        send=slot.send,
    )
    hub.add("Scope 1", url=URL)
    hub.pair("Scope 1", "123456")
    agents = HubAgents(
        hub, api_key=lambda: "key", clock=lambda: now[0], manager=manager
    )
    return SimpleNamespace(
        hub=hub, agents=agents, slot=slot, manager=manager, now=now, tmp=tmp_path
    )


def _start(made, **more):
    return made.agents.start(
        "Scope 1", "fibsem-agent", "Mill", "suggest", "pat", **more
    )


def _routes(made):
    return [route for route, _ in made.slot.calls]


def test_start_takes_the_slot_and_gives_the_agent_the_gateways_token(made):
    state = _start(made)
    assert state["state"] == "running" and state["level"] == "suggest"
    route, body = made.slot.calls[0]
    assert route == "/gateway/agent/slot"
    assert body["session_id"] == state["session_id"] and body["level"] == "suggest"
    name, session_id, args, env = made.manager.started[0]
    assert args[:4] == ["--playbook", "fibsem-agent", "--url", URL]
    assert env == {"FIBSEM_SERVER_TOKEN": "agent-token", "ANTHROPIC_API_KEY": "key"}


def test_the_token_is_never_written_down(made):
    _start(made)
    for path in made.tmp.rglob("*"):
        if path.is_file():
            assert "agent-token" not in path.read_text(errors="replace")
    assert "agent-token" not in str(made.agents.state())


def test_plain_http_to_another_pc_is_refused_before_anything_is_asked(made):
    made.hub.add("Scope 2", url="http://scope-2:8010")
    with pytest.raises(GatewayRefused) as refused:
        made.agents.start("Scope 2", "fibsem-agent", None, "watch", "pat")
    assert refused.value.error_type == "plain_http"
    assert made.slot.calls == []


def test_what_crosses_in_clear():
    assert crosses_in_clear(Instrument("a", url="http://scope-1:8010"))
    assert not crosses_in_clear(Instrument("a", url="https://scope-1:8010"))
    assert not crosses_in_clear(Instrument("a", url="http://127.0.0.1:8010"))
    assert not crosses_in_clear(Instrument("a", url="http://localhost:8010"))
    assert not crosses_in_clear(Instrument("a", ssh="user@scope-1"))


def test_no_api_key_refuses(made):
    agents = HubAgents(made.hub, manager=made.manager)
    with pytest.raises(GatewayRefused) as refused:
        agents.start("Scope 1", "fibsem-agent", None, "watch", "pat")
    assert refused.value.error_type == "no_api_key" and made.slot.calls == []


def test_the_gateways_refusal_is_passed_on_and_nothing_starts(made):
    made.slot.refuse["slot"] = "other_holder"
    with pytest.raises(GatewayRefused) as refused:
        _start(made)
    assert refused.value.error_type == "other_holder"
    assert made.manager.started == [] and made.agents.state() == {}


def test_an_agent_that_does_not_start_gives_the_slot_back(made):
    made.manager.fail = True
    with pytest.raises(GatewayRefused) as refused:
        _start(made)
    assert refused.value.error_type == "agent_not_started"
    assert _routes(made)[-1] == "/gateway/agent/slot/release"
    assert made.slot.holder is None


def test_one_agent_for_an_instrument(made):
    _start(made)
    with pytest.raises(GatewayRefused) as refused:
        _start(made)
    assert refused.value.error_type == "agent_running"


def test_the_slot_is_renewed_at_a_third_of_its_lease(made):
    _start(made)
    made.now[0] += 9
    made.agents.poll()
    assert _routes(made) == ["/gateway/agent/slot"]
    made.now[0] += 2
    made.agents.poll()
    assert _routes(made)[-1] == "/gateway/agent/slot/renew"


def test_a_connection_that_is_away_keeps_the_agent_and_tries_again(made):
    _start(made)
    made.slot.down = True
    made.now[0] += 11
    made.agents.poll()
    assert made.agents.state()["Scope 1"]["state"] == "away"
    assert made.manager.agents["Scope 1"].state == "running"
    made.slot.down = False
    made.now[0] += 4
    made.agents.poll()
    assert made.agents.state()["Scope 1"]["state"] == "running"


def test_the_same_session_joins_again_after_the_slot_was_freed(made):
    state = _start(made)
    made.slot.refuse["renew"] = "lost_slot"
    made.now[0] += 11
    made.agents.poll()
    assert made.agents.state()["Scope 1"]["state"] == "away"
    made.agents.poll()
    route, body = made.slot.calls[-1]
    assert route == "/gateway/agent/slot"
    assert body == {
        "session_id": state["session_id"],
        "rejoin": True,
        "token": "agent-token",
    }
    assert made.agents.state()["Scope 1"]["state"] == "running"
    # The gateway made a new token: the agent is stopped and carries on
    # with it.
    assert made.manager.parked == ["Scope 1"]
    agent = made.manager.agents["Scope 1"]
    assert agent.state == "running"
    assert agent.env["FIBSEM_SERVER_TOKEN"] == "new-token"


@pytest.mark.parametrize(
    "reason", ["level_changed", "stopped", "ended", "other_holder"]
)
def test_a_refusal_to_join_again_ends_the_agent(made, reason):
    _start(made)
    made.slot.refuse["renew"] = "lost_slot"
    made.now[0] += 11
    made.agents.poll()
    made.slot.refuse["slot"] = reason
    made.agents.poll()
    state = made.agents.state()["Scope 1"]
    assert state["state"] == "ended" and reason in state["why"]
    assert made.manager.agents["Scope 1"].state == "ended"
    # The slot is someone else's now: never released by this hub.
    assert "/gateway/agent/slot/release" not in _routes(made)


def test_a_refusal_to_renew_ends_the_agent(made):
    _start(made)
    made.slot.refuse["renew"] = "stopped"
    made.now[0] += 11
    made.agents.poll()
    assert made.agents.state()["Scope 1"]["state"] == "ended"


def test_an_agent_that_ends_by_itself_gives_the_slot_back(made):
    _start(made)
    made.manager.agents["Scope 1"].state = "ended"
    made.manager.agents["Scope 1"].ended = "finished"
    made.agents.poll()
    assert made.agents.state()["Scope 1"]["why"] == "finished"
    assert made.slot.holder is None


def test_end_stops_the_agent_and_gives_the_slot_back(made):
    _start(made)
    state = made.agents.end("Scope 1", "pat")
    assert state["state"] == "ended" and state["why"] == "ended by pat"
    assert made.manager.agents["Scope 1"].state == "ended"
    assert made.slot.holder is None
    with pytest.raises(GatewayRefused):
        made.agents.end("Scope 1", "pat")


def test_the_routes_start_show_and_end_as_the_signed_in_operator(made, tmp_path):
    from fastapi.testclient import TestClient

    from fibsem.hub.server import build_hub_app, hub_auth

    token_file = tmp_path / "operator-token.json"
    token_file.write_text('{"token": "operator-token"}')
    app = build_hub_app(
        made.hub, hub_auth(token_file, tmp_path / "paired.json"), agents=made.agents
    )
    client = TestClient(app)
    body = {"playbook": "fibsem-agent", "goal": "Mill", "level": "suggest"}
    assert (
        client.post("/hub/instruments/Scope 1/hub-agent/start", json=body).status_code
        == 401
    )
    signed = {"Authorization": "Bearer operator-token"}
    started = client.post(
        "/hub/instruments/Scope 1/hub-agent/start", json=body, headers=signed
    )
    assert started.status_code == 200 and started.json()["state"] == "running"
    assert "token" not in started.text
    assert (
        client.get("/hub/agents", headers=signed).json()["Scope 1"]["level"]
        == "suggest"
    )
    ended = client.post("/hub/instruments/Scope 1/hub-agent/end", headers=signed)
    assert ended.json()["state"] == "ended"
    again = client.post("/hub/instruments/Scope 1/hub-agent/end", headers=signed)
    assert again.status_code == 409
    assert again.json()["detail"]["error_type"] == "no_agent"


def test_stopping_the_hub_ends_its_agents_and_gives_the_slots_back(made):
    _start(made)
    made.agents.stop()
    assert made.manager.agents["Scope 1"].state == "ended"
    assert made.slot.holder is None


def test_the_hubs_key_is_set_only_with_the_installs_own_token(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from fibsem import keystore
    from fibsem.hub.server import build_hub_app, hub_auth
    from fibsem.server.pairing import OPERATOR

    kept = {}
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(keystore, "stored_secret", lambda name: kept.get(name))
    monkeypatch.setattr(keystore, "set_secret", kept.__setitem__)
    monkeypatch.setattr(keystore, "delete_secret", lambda name: kept.pop(name, None))
    monkeypatch.setattr(keystore, "store_name", lambda: "test store")
    token_file = tmp_path / "operator-token.json"
    token_file.write_text('{"token": "operator-token"}')
    auth = hub_auth(token_file, tmp_path / "paired.json")
    _, paired = auth.pairing.tokens.add("laptop", role=OPERATOR)
    hub = Hub(file=tmp_path / "hub.yaml", tokens=tmp_path / "hub")
    client = TestClient(build_hub_app(hub, auth))
    here = {"Authorization": "Bearer operator-token"}
    away = {"Authorization": f"Bearer {paired}"}
    route = "/hub/secrets/ANTHROPIC_API_KEY"

    rows = client.get("/hub/secrets", headers=here).json()
    assert [row["name"] for row in rows] == ["ANTHROPIC_API_KEY"]
    assert rows[0]["set"] is False and rows[0]["can_set"] is True
    assert client.get("/hub/secrets", headers=away).json()[0]["can_set"] is False
    assert client.put(route, json={"value": "sk-1"}, headers=away).status_code == 403
    assert client.put(route, json={"value": "sk-1"}).status_code == 401

    saved = client.put(route, json={"value": "sk-1"}, headers=here)
    assert saved.status_code == 200 and saved.json()["set"] is True
    assert "sk-1" not in saved.text
    assert "sk-1" not in client.get("/hub/secrets", headers=here).text
    other = client.put(
        "/hub/secrets/FIBSEM_SLACK_TOKEN", json={"value": "x"}, headers=here
    )
    assert other.status_code == 404
    assert client.delete(route, headers=here).json()["set"] is False
