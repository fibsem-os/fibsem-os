"""The hub's page: what it reads of an instrument's gateway through the hub,
and starting, resuming and ending that instrument's agent."""

import json

import pytest
from fastapi.testclient import TestClient

from fibsem.gateway.agent import AgentState
from fibsem.gateway.gateway import OFF, GatewayRefused, GatewayState, History
from fibsem.gateway.server import GatewayAuth, build_gateway_app
from fibsem.hub.hub import Hub
from fibsem.hub.main import page_link
from fibsem.hub.server import build_hub_app, hub_auth
from fibsem.server.pairing import HUB, SWITCH_MODES

OPERATOR_TOKEN = "operator-token"
URL = "http://scope-1:8010"
OPERATOR = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}


class FakeGateway:
    def __init__(self):
        self.calls = []

    def state(self):
        return GatewayState(mode=OFF, version="1.2.3")

    def history(self):
        return History()

    def start_app(self, who, via, app=None, window=True, force=False, **more):
        self.calls.append(("start", who, via, app, window, force))
        return self.state()

    def stop(self, who, via, force=False, window=False):
        self.calls.append(("stop", who, via, force, window))
        return self.state()


class FakeAgent:
    """An instrument's agent, as its gateway's routes ask of it."""

    def __init__(self):
        self.calls = []

    def state(self):
        return AgentState(microscope="scope-1-pc")

    def in_use(self):
        return None

    def own_running(self):
        return None

    def sessions(self, limit):
        return {"sessions": [], "more": False}

    def playbooks(self):
        return [{"name": "fibsem-agent"}]

    def configuration(self, name):
        if name != "fibsem-agent":
            raise GatewayRefused(404, "no_playbook", f"No playbook named {name}.")
        return {"playbook": {"name": name}}

    def start(self, playbook, goal, level, who, via, budget_usd=None, replace=False):
        self.calls.append(("start", playbook, goal, level, who, via))
        return self.state()

    def resume(self, session_id, level, who, via, budget_usd=None, replace=False):
        self.calls.append(("resume", session_id, level, who, via))
        return self.state()

    def end(self, who, via, stop_app=False, force=False, window=False):
        self.calls.append(("end", who, via, stop_app, force))
        return self.state()


class Instrument:
    """One instrument's real gateway app, reached as the hub reaches it."""

    def __init__(self, tmp_path):
        token_file = tmp_path / "instrument.json"
        token_file.write_text(json.dumps({"token": "instrument-operator"}))
        self.auth = GatewayAuth.from_disk(token_file, tmp_path / "instrument-paired")
        self.agent = FakeAgent()
        self.gateway = FakeGateway()
        self.client = TestClient(build_gateway_app(self.gateway, self.auth, self.agent))
        self.up = True
        self.sent = []

    def fetch(self, url, route, token, timeout):
        return self.send(url, "GET", route, token, None, timeout)

    def send(self, url, method, route, token, body, timeout):
        if not self.up:
            raise OSError("connection refused")
        self.sent.append((url, method, route))
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = self.client.request(
            method, route, headers=headers, json=body if method != "GET" else None
        )
        return response.status_code, response.json()

    def pair(self, url, code, name):
        response = self.client.post("/pair", json={"code": code, "name": name})
        return response.json()["token"], response.json()["role"]


@pytest.fixture
def instrument(tmp_path):
    return Instrument(tmp_path)


@pytest.fixture
def hub(tmp_path, instrument):
    return Hub(
        tmp_path / "hub.yaml",
        tmp_path / "hub",
        fetch=instrument.fetch,
        pair=instrument.pair,
        send=instrument.send,
    )


@pytest.fixture
def page(tmp_path, hub):
    token_file = tmp_path / "operator-token.json"
    token_file.write_text(json.dumps({"token": OPERATOR_TOKEN}))
    auth = hub_auth(token_file, tmp_path / "hub-paired.json")
    return TestClient(build_hub_app(hub, auth), follow_redirects=False)


def _paired(hub, instrument, grants=(SWITCH_MODES,)):
    hub.add("Scope 1", URL)
    hub.pair("Scope 1", instrument.auth.pairing.open(role=HUB, grants=grants))


def test_the_page_reads_an_instruments_agent_through_the_hub(page, hub, instrument):
    _paired(hub, instrument)
    at = "/hub/instruments/Scope 1/gateway"
    assert page.get(f"{at}/agent").status_code == 401
    agent = page.get(f"{at}/agent", headers=OPERATOR)
    assert agent.status_code == 200 and agent.json()["microscope"] == "scope-1-pc"
    assert page.get(at, headers=OPERATOR).json()["version"] == "1.2.3"
    playbooks = page.get(f"{at}/agent/playbooks", headers=OPERATOR)
    assert playbooks.json() == [{"name": "fibsem-agent"}]
    one = page.get(f"{at}/agent/playbooks/fibsem-agent", headers=OPERATOR)
    assert one.json() == {"playbook": {"name": "fibsem-agent"}}
    sessions = page.get(f"{at}/agent/sessions?limit=5", headers=OPERATOR)
    assert sessions.status_code == 200
    assert instrument.sent[-1] == (URL, "GET", "/gateway/agent/sessions?limit=5")


def test_the_hub_starts_resumes_and_ends_an_instruments_agent(page, hub, instrument):
    _paired(hub, instrument)
    at = "/hub/instruments/Scope 1/gateway/agent"
    body = {"playbook": "fibsem-agent", "goal": "Screen grid 1", "level": "watch"}
    assert page.post(f"{at}/start", json=body).status_code == 401
    assert page.post(f"{at}/start", json=body, headers=OPERATOR).status_code == 200
    resume = {"session_id": "abc", "level": "watch"}
    assert page.post(f"{at}/resume", json=resume, headers=OPERATOR).status_code == 200
    assert page.post(f"{at}/end", headers=OPERATOR).status_code == 200
    # The gateway names this hub as who asked, never the operator.
    who = instrument.agent.calls[0][4]
    assert instrument.agent.calls == [
        ("start", "fibsem-agent", "Screen grid 1", "watch", who, "hub"),
        ("resume", "abc", "watch", who, "hub"),
        ("end", who, "hub", False, False),
    ]
    assert who != "operator"


def test_a_hub_without_the_grant_is_refused_in_the_gateways_words(
    page, hub, instrument
):
    _paired(hub, instrument, grants=())
    at = "/hub/instruments/Scope 1/gateway/agent"
    refused = page.post(f"{at}/start", json={}, headers=OPERATOR)
    assert refused.status_code == 403
    assert refused.json()["detail"]["error_type"] == "no_grant"
    assert not instrument.agent.calls
    # Reading needs no grant.
    assert page.get(at, headers=OPERATOR).status_code == 200


def test_a_hub_never_forces_an_end(page, hub, instrument):
    _paired(hub, instrument)
    at = "/hub/instruments/Scope 1/gateway/agent"
    forced = page.post(f"{at}/end", json={"force": True}, headers=OPERATOR)
    assert forced.status_code == 403
    assert forced.json()["detail"]["error_type"] == "operator_only"


def test_the_hub_starts_and_stops_an_instruments_app(page, hub, instrument):
    _paired(hub, instrument)
    at = "/hub/instruments/Scope 1/gateway"
    body = {"app": "autolamella", "window": False}
    assert page.post(f"{at}/start", json=body).status_code == 401
    assert page.post(f"{at}/start", json=body, headers=OPERATOR).status_code == 200
    assert page.post(f"{at}/stop", json={}, headers=OPERATOR).status_code == 200
    # Its gateway names this hub as who asked, and a hub never closes a window.
    who = instrument.gateway.calls[0][1]
    assert instrument.gateway.calls == [
        ("start", who, "hub", "autolamella", False, False),
        ("stop", who, "hub", False, False),
    ]
    assert who != "operator"


def test_a_hub_never_forces_an_app_to_stop_and_needs_the_grant(page, hub, instrument):
    _paired(hub, instrument)
    at = "/hub/instruments/Scope 1/gateway"
    for route in ("start", "stop"):
        forced = page.post(f"{at}/{route}", json={"force": True}, headers=OPERATOR)
        assert forced.status_code == 403
        assert forced.json()["detail"]["error_type"] == "operator_only"
    _paired(hub, instrument, grants=())
    refused = page.post(f"{at}/start", json={"app": "autolamella"}, headers=OPERATOR)
    assert refused.status_code == 403
    assert refused.json()["detail"]["error_type"] == "no_grant"
    assert not instrument.gateway.calls


def test_nothing_else_of_a_gateway_is_reached_through_the_hub(page, hub, instrument):
    _paired(hub, instrument)
    at = "/hub/instruments/Scope 1"
    before = len(instrument.sent)
    for method, route in [
        ("GET", "gateway/secrets"),
        ("GET", "gateway/history"),
        ("GET", "pairing"),
        ("GET", "app/events"),
        ("GET", "gateway/agent/playbooks/a/b"),
        ("POST", "gateway/quit"),
        ("POST", "gateway/claim"),
        ("POST", "gateway/microscope/connect"),
        ("POST", "gateway/agent/playbooks"),
        ("POST", "pairing"),
    ]:
        answer = page.request(method, f"{at}/{route}", json={}, headers=OPERATOR)
        assert answer.status_code == 404, (method, route)
    assert len(instrument.sent) == before


def test_an_instrument_that_is_not_paired_or_not_there_says_so(page, hub, instrument):
    hub.add("Scope 1", URL)
    at = "/hub/instruments/Scope 1/gateway/agent"
    unpaired = page.get(at, headers=OPERATOR)
    assert unpaired.status_code == 409
    assert unpaired.json()["detail"]["error_type"] == "not_paired"
    assert (
        page.get("/hub/instruments/Nobody/gateway/agent", headers=OPERATOR).status_code
        == 404
    )
    hub.pair("Scope 1", instrument.auth.pairing.open(role=HUB))
    instrument.up = False
    down = page.get(at, headers=OPERATOR)
    assert down.status_code == 503
    assert down.json()["detail"]["error_type"] == "unreachable"


def test_a_gateway_that_forgot_the_hub_does_not_sign_the_page_out(
    page, hub, instrument
):
    _paired(hub, instrument)
    for computer in instrument.auth.pairing.tokens.computers():
        instrument.auth.pairing.tokens.remove(computer.id)
    forgot = page.get("/hub/instruments/Scope 1/gateway/agent", headers=OPERATOR)
    # A 401 would make the page forget its own sign in at the hub.
    assert forgot.status_code == 409
    assert "pair again" in forgot.json()["detail"]["message"]


def test_the_hub_serves_its_page(page):
    assert page.get("/").headers["location"] == "/ui/hub"
    assert page.get("/ui/hub").status_code == 200


def test_the_page_link_keeps_the_token_out_of_what_is_sent(tmp_path, monkeypatch):
    monkeypatch.setattr("fibsem.hub.main.read_operator_token", lambda: "secret")
    link = page_link("http://127.0.0.1:8020")
    assert link == "http://127.0.0.1:8020/ui/hub#token=secret"


def test_the_hub_says_whether_it_was_written_for_an_instruments_gateway(
    hub, instrument, monkeypatch
):
    _paired(hub, instrument)
    hub.check()
    (row,) = hub.state().instruments
    assert row.api_version and row.same_version is True
    # A gateway one minor version on: an older hub can't work with it.
    major, minor, _ = row.api_version.split(".")
    monkeypatch.setattr(
        "fibsem.hub.hub.GATEWAY_API_VERSION", f"{major}.{int(minor) + 1}.0"
    )
    hub.check()
    (row,) = hub.state().instruments
    assert row.same_version is False and row.reachable is True
