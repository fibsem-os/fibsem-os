"""The slot for an agent a paired Hub runs on its own PC: taking it (the
instrument's checks only, a token made here and given once), renewing it, the
grace, a rejoin by the same session and its refusals, End and release, and a
restarted gateway forgetting it. A fake App Server and a fake clock."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fibsem.gateway import hub_agent
from fibsem.gateway.agent import API_KEY, GatewayAgent
from fibsem.gateway.gateway import RUNNING, GatewayRefused, GatewayState
from fibsem.gateway.server import GatewayAuth, build_gateway_app
from fibsem.server.pairing import AGENT, HUB, SWITCH_MODES, PairedTokens

URL = "http://127.0.0.1:8001"
OPERATOR_TOKEN = "operator-token"
GRACE = 120.0


class FakeGateway:
    def __init__(self):
        self.mode = RUNNING
        self.before_stop = []
        self.watchers = []

    def state(self):
        return GatewayState(mode=self.mode, version="test")

    def stop(self, who, via, force=False, window=False):
        self.mode = "off"


class App:
    """The App Server: its level, as set and as read back."""

    def __init__(self):
        self.up = True
        self.level = "watch"
        self.levels = []

    def discovery(self):
        return {"url": URL, "pid": 1} if self.up else None

    def set_level(self, url, level):
        self.levels.append(level)
        self.level = level


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def made(tmp_path, monkeypatch):
    app = App()
    clock = Clock()

    def capabilities(url, method, path, *args, **kwargs):
        assert (url, method, path) == (URL, "GET", "/capabilities")
        return {"level": app.level}

    monkeypatch.setattr(hub_agent, "call_host", capabilities)

    def make(platform="linux", key=None):
        tokens = PairedTokens(tmp_path / "paired.json")
        agent = GatewayAgent(
            FakeGateway(),
            tokens,
            root=tmp_path / "sessions",
            discovery=app.discovery,
            level=app.set_level,
            reachable=lambda url: app.up,
            last_heard=lambda url: {"last_heard_s": None},
            playbooks=lambda: [{"name": "default"}],
            configuration=lambda name: {"mode": "code"},
            environ={API_KEY: key} if key else {},
            stored=lambda name: None,
            platform=platform,
            microscope="arctis-pc",
            hub_grace_s=lambda: GRACE,
        )
        agent.hub._clock = clock
        return agent, tokens

    return SimpleNamespace(make=make, app=app, clock=clock, tmp_path=tmp_path)


def _take(agent, session="s-1", level="drive", **kwargs):
    return agent.hub.take("hub-pc", session, "default", "Mill 01", level, **kwargs)


def _lapse(made, agent):
    made.clock.now += GRACE + 1
    agent.hub.follow()


def test_a_hub_takes_the_slot_with_a_token_made_here(made):
    # No API key on this PC and a code playbook on Windows: both the hub PC's.
    agent, tokens = made.make(platform="win32", key=None)
    answer = _take(agent)
    assert answer["lease_s"] == hub_agent.LEASE_S and answer["grace_s"] == GRACE
    paired = tokens.match(answer["token"])
    assert paired.role == AGENT and paired.name == "default via hub-pc"
    assert made.app.levels == ["drive"]
    slot = agent.slot.read()
    assert slot["came_in"] == "hub" and slot["session_id"] == "s-1"
    assert slot["level"] == "drive" and slot["state"] == "running"
    # Both pages read one answer: the hub's session is the running agent.
    running = agent.state().running
    assert running["session_id"] == "s-1" and running["hub"] == "hub-pc"
    assert running["via"] == "hub" and running["goal"] == "Mill 01"
    assert agent.in_use() == "default via hub-pc"
    # The token is never written down whole.
    assert answer["token"] not in json.dumps(agent.slot.read())
    assert answer["token"] not in (made.tmp_path / "paired.json").read_text()


def test_one_holder_at_a_time(made):
    agent, tokens = made.make()
    _take(agent)
    with pytest.raises(GatewayRefused) as refused:
        _take(agent, session="s-2")
    assert refused.value.error_type == hub_agent.BUSY


def test_an_agent_paired_another_way_needs_replace(made):
    agent, tokens = made.make()
    tokens.add("someone's agent", role=AGENT)
    with pytest.raises(GatewayRefused) as refused:
        _take(agent)
    assert refused.value.error_type == hub_agent.BUSY
    answer = _take(agent, replace=True)
    assert tokens.agent().name == "default via hub-pc" and answer["token"]


def test_no_app_is_not_ready_and_a_bad_level_is_refused(made):
    agent, tokens = made.make()
    made.app.up = False
    with pytest.raises(GatewayRefused) as refused:
        _take(agent)
    assert refused.value.error_type == hub_agent.NOT_READY
    made.app.up = True
    with pytest.raises(GatewayRefused) as refused:
        _take(agent, level="everything")
    assert refused.value.status == 422
    assert tokens.agent() is None and made.app.levels == []


def test_renewing_keeps_the_slot_past_the_grace(made):
    agent, tokens = made.make()
    answer = _take(agent)
    for _ in range(5):
        made.clock.now += GRACE / 2
        assert agent.hub.renew("hub-pc", "s-1")["lease_s"] == hub_agent.LEASE_S
        agent.hub.follow()
    assert tokens.match(answer["token"]) is not None
    assert agent.slot.read()["state"] == "running"
    # Another session or another hub doesn't hold it.
    for hub, session in (("hub-pc", "s-2"), ("other-hub", "s-1")):
        with pytest.raises(GatewayRefused) as refused:
            agent.hub.renew(hub, session)
        assert refused.value.error_type == hub_agent.LOST_SLOT


def test_within_the_grace_the_session_keeps_slot_and_level(made):
    agent, tokens = made.make()
    answer = _take(agent)
    made.clock.now += GRACE - 1
    agent.hub.follow()
    assert tokens.match(answer["token"]) is not None
    assert made.app.level == "drive"
    agent.hub.renew("hub-pc", "s-1")


def test_quiet_past_the_grace_sets_watch_and_frees_the_slot(made):
    agent, tokens = made.make()
    answer = _take(agent)
    _lapse(made, agent)
    assert tokens.match(answer["token"]) is None
    assert made.app.level == "watch"
    assert agent.slot.read()["state"] == "ended"
    assert agent.state().running is None and agent.in_use() is None
    with pytest.raises(GatewayRefused) as refused:
        agent.hub.renew("hub-pc", "s-1")
    assert refused.value.error_type == hub_agent.LOST_SLOT
    # The slot is free for another session, at the level it asks for.
    assert _take(agent, session="s-2", level="assist")["token"]


def test_the_same_session_comes_back_at_its_last_level(made):
    agent, tokens = made.make()
    first = _take(agent)
    # The operator lowered it while it ran: that is the level it comes back at.
    made.app.level = "assist"
    made.clock.now += hub_agent.LEVEL_READ_S
    agent.hub.follow()
    assert agent.slot.read()["level"] == "assist"
    _lapse(made, agent)
    again = agent.hub.rejoin("hub-pc", "s-1", first["token"])
    assert again["token"] != first["token"]
    assert tokens.match(first["token"]) is None
    assert tokens.match(again["token"]).name == "default via hub-pc"
    assert made.app.level == "assist"
    slot = agent.slot.read()
    assert slot["state"] == "running" and slot["session_id"] == "s-1"
    assert agent.hub.renew("hub-pc", "s-1")


def test_a_rejoin_needs_the_sessions_token(made):
    agent, tokens = made.make()
    first = _take(agent)
    _lapse(made, agent)
    for hub, session, token in (
        ("hub-pc", "s-1", ""),
        ("hub-pc", "s-1", "a guess"),
        ("hub-pc", "s-2", first["token"]),
        ("other-hub", "s-1", first["token"]),
    ):
        with pytest.raises(GatewayRefused) as refused:
            agent.hub.rejoin(hub, session, token)
        assert refused.value.error_type == hub_agent.LOST_SLOT
    assert made.app.level == "watch" and tokens.agent() is None
    # The right one still works after the wrong ones.
    assert agent.hub.rejoin("hub-pc", "s-1", first["token"])["token"]


def test_a_restarted_hub_comes_back_as_the_same_paired_hub_with_no_token(made):
    agent, tokens = made.make()
    first = _take(agent)
    _lapse(made, agent)
    # Not another Hub, and not another session, without the token either.
    for hub, session in (("other-hub", "s-1"), ("hub-pc", "s-2")):
        with pytest.raises(GatewayRefused) as refused:
            agent.hub.rejoin(hub, session, None)
        assert refused.value.error_type == hub_agent.LOST_SLOT
    again = agent.hub.rejoin("hub-pc", "s-1", None)
    assert again["token"] != first["token"]
    assert tokens.match(first["token"]) is None
    assert tokens.match(again["token"]).name == "default via hub-pc"
    assert "after its restart" in agent.slot.read()["why"]


def test_a_restarted_hub_is_held_to_the_same_checks(made):
    agent, tokens = made.make()
    _take(agent)
    _lapse(made, agent)
    made.app.level = "assist"
    with pytest.raises(GatewayRefused) as refused:
        agent.hub.rejoin("hub-pc", "s-1", None)
    assert refused.value.error_type == hub_agent.LEVEL_CHANGED
    assert tokens.agent() is None


def test_no_rejoin_after_the_pc_changed_the_level(made):
    agent, tokens = made.make()
    first = _take(agent)
    _lapse(made, agent)
    made.app.level = "assist"
    with pytest.raises(GatewayRefused) as refused:
        agent.hub.rejoin("hub-pc", "s-1", first["token"])
    assert refused.value.error_type == hub_agent.LEVEL_CHANGED
    # For good: setting Watch again doesn't bring it back.
    made.app.level = "watch"
    with pytest.raises(GatewayRefused) as refused:
        agent.hub.rejoin("hub-pc", "s-1", first["token"])
    assert refused.value.error_type == hub_agent.LEVEL_CHANGED
    assert tokens.agent() is None


def test_no_rejoin_after_stop(made):
    agent, tokens = made.make()
    first = _take(agent)
    made.app.level = "watch"  # Stop agent at the PC
    made.clock.now += hub_agent.LEVEL_READ_S
    agent.hub.follow()
    _lapse(made, agent)
    with pytest.raises(GatewayRefused) as refused:
        agent.hub.rejoin("hub-pc", "s-1", first["token"])
    assert refused.value.error_type == hub_agent.STOPPED_REASON


def test_no_rejoin_once_another_agent_took_the_slot(made):
    agent, tokens = made.make()
    first = _take(agent)
    _lapse(made, agent)
    agent.allowed("someone's agent", "operator", None)
    with pytest.raises(GatewayRefused) as refused:
        agent.hub.rejoin("hub-pc", "s-1", first["token"])
    assert refused.value.error_type == hub_agent.OTHER_HOLDER


def test_another_agent_taking_the_slot_at_the_pc_ends_the_hubs(made):
    agent, tokens = made.make()
    first = _take(agent)
    tokens.add("someone's agent", role=AGENT)
    agent.allowed("someone's agent", "operator", "default via hub-pc")
    with pytest.raises(GatewayRefused) as refused:
        agent.hub.renew("hub-pc", "s-1")
    assert refused.value.error_type == hub_agent.OTHER_HOLDER
    assert tokens.agent().name == "someone's agent"
    with pytest.raises(GatewayRefused):
        agent.hub.rejoin("hub-pc", "s-1", first["token"])


def test_end_at_the_pc_is_a_release(made):
    agent, tokens = made.make()
    first = _take(agent)
    agent.end("operator", "page")
    assert tokens.agent() is None and made.app.level == "watch"
    assert agent.slot.read()["state"] == "ended"
    for call in (
        lambda: agent.hub.renew("hub-pc", "s-1"),
        lambda: agent.hub.rejoin("hub-pc", "s-1", first["token"]),
    ):
        with pytest.raises(GatewayRefused) as refused:
            call()
        assert refused.value.error_type == hub_agent.ENDED


def test_unpairing_at_the_pc_ends_it(made):
    agent, tokens = made.make()
    _take(agent)
    tokens.remove(tokens.agent().id)
    agent.hub.follow()
    assert agent.slot.read()["state"] == "ended"
    with pytest.raises(GatewayRefused) as refused:
        agent.hub.renew("hub-pc", "s-1")
    assert refused.value.error_type == hub_agent.ENDED


def test_release(made):
    agent, tokens = made.make()
    first = _take(agent)
    agent.hub.release("other-hub", "s-1")  # not its session: nothing
    assert tokens.agent() is not None
    agent.hub.release("hub-pc", "s-1")
    assert tokens.agent() is None and made.app.level == "watch"
    assert agent.slot.read()["state"] == "ended"
    with pytest.raises(GatewayRefused):
        agent.hub.rejoin("hub-pc", "s-1", first["token"])


def test_a_restarted_gateway_ends_a_hub_slot_it_doesnt_know(made):
    agent, tokens = made.make()
    _take(agent)
    again, _ = made.make()
    assert again.slot.read()["state"] == "ended"
    assert tokens.agent() is None
    assert again.state().running is None


# -- the routes ----------------------------------------------------------------


@pytest.fixture
def client(made, tmp_path):
    token_file = tmp_path / "operator-token.json"
    token_file.write_text(json.dumps({"token": OPERATOR_TOKEN}))
    auth = GatewayAuth.from_disk(token_file, tmp_path / "gateway-paired.json")
    agent, _ = made.make()
    # The gateway's agent pairs into the gateway's own paired file.
    agent._tokens = agent.hub._tokens = auth.pairing.tokens
    client = TestClient(build_gateway_app(agent._gateway, auth, agent))
    return SimpleNamespace(client=client, auth=auth, agent=agent)


def _as(token):
    return {"Authorization": f"Bearer {token}"}


def test_the_routes_take_renew_rejoin_and_release(client, made):
    c, tokens = client.client, client.auth.pairing.tokens
    _, hub = tokens.add("hub-pc", HUB, grants=(SWITCH_MODES,))
    body = {"session_id": "s-1", "playbook": "default", "level": "assist"}
    taken = c.post("/gateway/agent/slot", json=body, headers=_as(hub))
    assert taken.status_code == 200, taken.text
    answer = taken.json()
    assert answer["lease_s"] == hub_agent.LEASE_S and answer["grace_s"] == GRACE
    assert answer["state"]["running"]["hub"] == "hub-pc"
    assert tokens.match(answer["token"]).role == AGENT
    # The gateway's page reads the same answer.
    page = c.get("/gateway/agent", headers=_as(OPERATOR_TOKEN)).json()
    assert page["running"]["session_id"] == "s-1"
    renew = {"session_id": "s-1"}
    assert (
        c.post("/gateway/agent/slot/renew", json=renew, headers=_as(hub)).status_code
        == 200
    )
    again = c.post("/gateway/agent/slot", json=body, headers=_as(hub))
    assert again.status_code == 409
    assert again.json()["detail"]["error_type"] == hub_agent.BUSY
    _lapse(made, client.agent)
    lost = c.post("/gateway/agent/slot/renew", json=renew, headers=_as(hub))
    assert lost.status_code == 409
    assert lost.json()["detail"]["error_type"] == hub_agent.LOST_SLOT
    back = c.post(
        "/gateway/agent/slot",
        json={"session_id": "s-1", "rejoin": True, "token": answer["token"]},
        headers=_as(hub),
    )
    assert back.status_code == 200, back.text
    assert back.json()["token"] != answer["token"]
    released = c.post("/gateway/agent/slot/release", json=renew, headers=_as(hub))
    assert released.status_code == 200 and released.json()["state"]["running"] is None
    assert tokens.agent() is None


def test_only_a_hub_with_the_grant_asks_for_the_slot(client):
    client, auth = client.client, client.auth
    _, reader = auth.pairing.tokens.add("reader", HUB)
    body = {"session_id": "s-1"}
    for token, status, error in (
        (OPERATOR_TOKEN, 403, "hub_only"),
        (reader, 403, "no_grant"),
        ("nobody", 401, "unauthorized"),
    ):
        answer = client.post("/gateway/agent/slot", json=body, headers=_as(token))
        assert answer.status_code == status
        assert answer.json()["detail"]["error_type"] == error


def test_a_quiet_spell_its_end_the_lapse_and_a_rejoin_are_logged(made, caplog):
    caplog.set_level("INFO", logger="fibsem.gateway.hub_agent")
    agent, _ = made.make()
    first = _take(agent)
    made.clock.now += hub_agent.LEASE_S + 5
    agent.hub.follow()
    agent.hub.follow()  # once, not every tick
    made.clock.now += 40
    agent.hub.renew("hub-pc", "s-1")
    _lapse(made, agent)
    with pytest.raises(GatewayRefused):
        agent.hub.rejoin("hub-pc", "s-1", "a guess")
    agent.hub.rejoin("hub-pc", "s-1", first["token"])
    lines = [r.getMessage() for r in caplog.records]
    assert sum("past one lease" in line for line in lines) == 1
    assert any("heard again after 75 s quiet" in line for line in lines)
    assert any("past the 120 s grace" in line for line in lines)
    assert any("rejoin of session s-1 refused (lost_slot)" in line for line in lines)
    assert any("rejoined after 121 s away" in line for line in lines)
    assert first["token"] not in caplog.text
