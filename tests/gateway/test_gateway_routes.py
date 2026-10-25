"""Who may do what at the gateway's routes: the operator, a hub with and
without "may start and stop apps", and nobody; pairing a hub with its grant;
connecting the host's microscope; and the desktop launch's handshake with the
gateway."""

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.gateway import main as gateway_main  # noqa: E402
from fibsem.gateway.gateway import (  # noqa: E402
    OFF,
    RUNNING,
    GatewayRefused,
    GatewayState,
)
from fibsem.gateway.host import HostMicroscope  # noqa: E402
from fibsem.gateway.server import GatewayAuth, build_gateway_app  # noqa: E402
from fibsem.server.pairing import HUB, SWITCH_MODES, PairedTokens  # noqa: E402
from fibsem.tools import desktop_launch  # noqa: E402
from fibsem.tools.apps import AUTOLAMELLA  # noqa: E402

OPERATOR_TOKEN = "operator-token"


class FakeGateway:
    """Records what the routes ask for."""

    def __init__(self):
        self.calls = []
        self.refuse = None

    def state(self):
        return GatewayState(mode=OFF, version="test")

    def history(self):
        from fibsem.gateway.gateway import History

        return History()

    def _do(self, *call):
        self.calls.append(call)
        if self.refuse is not None:
            raise self.refuse
        return GatewayState(mode=call[0], version="test", who=call[1])

    def start_app(
        self, who, via, app=None, window=True, experiment=None, last=False, force=False
    ):
        return self._do(RUNNING, who, via, app, window, experiment, last, force)

    def stop(self, who, via, force=False):
        return self._do(OFF, who, via, force)

    def claim(self, pid, app, who, via):
        return self._do(RUNNING, who, via, pid, app)


@pytest.fixture
def setup(tmp_path):
    token_file = tmp_path / "operator-token.json"
    token_file.write_text(json.dumps({"token": OPERATOR_TOKEN}))
    auth = GatewayAuth.from_disk(token_file, tmp_path / "gateway-paired.json")
    gateway = FakeGateway()
    client = TestClient(build_gateway_app(gateway, auth))
    return client, gateway, auth


def _as(token):
    return {"Authorization": f"Bearer {token}"}


def _pair_hub(client, grants):
    opened = client.post(
        "/pairing", json={"role": HUB, "grants": grants}, headers=_as(OPERATOR_TOKEN)
    ).json()
    paired = client.post("/pair", json={"code": opened["code"], "name": "mac-mini"})
    assert paired.status_code == 200
    return paired.json()


def test_the_operator_starts_and_forces(setup):
    client, gateway, _ = setup
    r = client.post(
        "/gateway/start",
        json={"app": "autolamella", "window": False, "last": True, "force": True},
        headers=_as(OPERATOR_TOKEN),
    )
    assert r.status_code == 200
    assert gateway.calls == [
        (RUNNING, "operator", "page", "autolamella", False, None, True, True)
    ]


def test_a_start_shows_the_window_unless_asked_not_to(setup):
    client, gateway, _ = setup
    client.post(
        "/gateway/start", json={"app": "autolamella"}, headers=_as(OPERATOR_TOKEN)
    )
    assert gateway.calls[0][4] is True


def test_nobody_unpaired_reads_or_starts(setup):
    client, gateway, _ = setup
    assert client.get("/gateway").status_code == 401
    assert client.get("/gateway", headers=_as("guess")).status_code == 401
    assert client.post("/gateway/stop", json={}).status_code == 401
    assert gateway.calls == []


def test_a_hub_without_the_grant_only_reads(setup):
    client, gateway, _ = setup
    hub = _pair_hub(client, [])
    assert hub["role"] == HUB and hub["grants"] == []
    assert client.get("/gateway", headers=_as(hub["token"])).status_code == 200
    r = client.post("/gateway/stop", json={}, headers=_as(hub["token"]))
    assert (r.status_code, r.json()["detail"]["error_type"]) == (403, "no_grant")
    assert gateway.calls == []


def test_a_hub_with_the_grant_starts_but_never_forces(setup):
    client, gateway, _ = setup
    hub = _pair_hub(client, [SWITCH_MODES])
    r = client.post("/gateway/stop", json={}, headers=_as(hub["token"]))
    assert r.status_code == 200 and r.json()["who"] == "mac-mini"
    assert gateway.calls == [(OFF, "mac-mini", "hub", False)]
    r = client.post("/gateway/stop", json={"force": True}, headers=_as(hub["token"]))
    assert r.json()["detail"]["error_type"] == "operator_only"
    assert (
        client.post(
            "/gateway/claim",
            json={"pid": 1, "app": "autolamella"},
            headers=_as(hub["token"]),
        ).status_code
        == 403
    )
    # A hub can't open pairing or change its own grants.
    assert client.get("/pairing", headers=_as(hub["token"])).status_code == 401


def test_the_operator_takes_the_grant_back(setup):
    client, gateway, auth = setup
    hub = _pair_hub(client, [SWITCH_MODES])
    listing = client.patch(
        f"/pairing/computers/{hub['id']}",
        json={"grants": []},
        headers=_as(OPERATOR_TOKEN),
    ).json()
    assert [c["grants"] for c in listing["computers"]] == [[]]
    assert (
        client.post("/gateway/stop", json={}, headers=_as(hub["token"])).status_code
        == 403
    )
    bad = client.patch(
        f"/pairing/computers/{hub['id']}",
        json={"grants": ["vent"]},
        headers=_as(OPERATOR_TOKEN),
    )
    assert bad.status_code == 422


def test_an_app_server_pairs_no_hubs(setup):
    from fibsem.server.pairing import Pairing

    pairing = Pairing(PairedTokens(None))
    with pytest.raises(ValueError):
        pairing.open(HUB)


def test_a_refusal_keeps_its_status(setup):
    client, gateway, _ = setup
    gateway.refuse = GatewayRefused(409, "run_in_progress", "A workflow is running.")
    r = client.post("/gateway/stop", json={}, headers=_as(OPERATOR_TOKEN))
    assert r.status_code == 409
    assert r.json()["detail"] == {
        "error_type": "run_in_progress",
        "message": "A workflow is running.",
    }


def test_grants_survive_a_restart_and_old_files_load(tmp_path):
    path = tmp_path / "paired.json"
    tokens = PairedTokens(path)
    computer, _ = tokens.add("mac-mini", HUB, grants=(SWITCH_MODES,))
    assert PairedTokens(path).computers()[0].grants == (SWITCH_MODES,)
    rows = json.loads(path.read_text())
    del rows["computers"][0]["grants"]
    path.write_text(json.dumps(rows))
    assert PairedTokens(path).computers()[0].grants == ()


class FakeCalls:
    def __init__(self, state):
        self.state = state
        self.calls = []

    def __call__(self, method, route, body=None, timeout=3.0):
        self.calls.append((method, route, body))
        if isinstance(self.state, Exception):
            raise self.state
        return self.state if route == "/gateway" else {}


@pytest.fixture
def handshake(monkeypatch):
    def use(state, take_over=True):
        calls = FakeCalls(state)
        monkeypatch.setattr(gateway_main, "call", calls)
        monkeypatch.setattr(desktop_launch, "_take_over", lambda app, state: take_over)
        monkeypatch.delenv("FIBSEM_GATEWAY", raising=False)
        return calls

    return use


def _routes(calls):
    return [(method, route) for method, route, _ in calls.calls]


def test_with_no_gateway_a_launch_opens_as_today(handshake):
    calls = handshake(None)
    assert desktop_launch.gateway_allows("autolamella", AUTOLAMELLA)
    assert _routes(calls) == [("GET", "/gateway")]


def test_a_gateway_that_doesnt_answer_changes_nothing(handshake):
    handshake(OSError("refused"))
    assert desktop_launch.gateway_allows("autolamella", AUTOLAMELLA)


def test_with_the_gateway_off_a_launch_claims_the_app(handshake):
    calls = handshake({"mode": OFF})
    assert desktop_launch.gateway_allows("autolamella", AUTOLAMELLA)
    assert _routes(calls)[-1] == ("POST", "/gateway/claim")
    assert calls.calls[-1][2]["app"] == "autolamella"


def test_a_remote_run_is_taken_over_or_left(handshake):
    calls = handshake(
        {"mode": RUNNING, "window": False, "experiment": "exp", "who": "mac-mini"},
        take_over=False,
    )
    assert not desktop_launch.gateway_allows("autolamella", AUTOLAMELLA)
    assert _routes(calls) == [("GET", "/gateway")]
    calls = handshake({"mode": RUNNING, "window": False}, take_over=True)
    assert desktop_launch.gateway_allows("autolamella", AUTOLAMELLA)
    assert _routes(calls) == [
        ("GET", "/gateway"),
        ("POST", "/gateway/stop"),
        ("POST", "/gateway/claim"),
    ]
    assert calls.calls[1][2] == {"force": True, "via": "desktop launch"}


def test_a_window_the_gateway_opened_doesnt_ask(handshake, monkeypatch):
    calls = handshake({"mode": RUNNING, "window": True})
    monkeypatch.setenv("FIBSEM_GATEWAY", "1")
    assert desktop_launch.gateway_allows("autolamella", AUTOLAMELLA)
    assert calls.calls == []


def test_a_second_launch_beside_an_open_window_opens_as_today(handshake):
    calls = handshake({"mode": RUNNING, "window": True})
    assert desktop_launch.gateway_allows("autolamella", AUTOLAMELLA)
    assert _routes(calls) == [("GET", "/gateway")]


class FakeHost:
    """The host's connection routes, as `HostMicroscope` calls them."""

    def __init__(self, up=True):
        self.up = up
        self.calls = []

    def discovery(self):
        return {"url": "http://127.0.0.1:8001", "pid": 1} if self.up else None

    def __call__(self, url, method, route, body=None):
        self.calls.append((url, method, route, body))
        if route == "/microscope/disconnect":
            raise GatewayRefused(409, "run_on", "A run is on.")
        return {"connected": route == "/microscope/connection"}


@pytest.fixture
def host_setup(tmp_path):
    token_file = tmp_path / "operator-token.json"
    token_file.write_text(json.dumps({"token": OPERATOR_TOKEN}))
    auth = GatewayAuth.from_disk(token_file, tmp_path / "gateway-paired.json")
    host = FakeHost()
    microscope = HostMicroscope(discovery=host.discovery, call=host)
    client = TestClient(build_gateway_app(FakeGateway(), auth, microscope=microscope))
    return client, host


def test_the_page_connects_the_hosts_microscope_through_its_routes(host_setup):
    client, host = host_setup
    r = client.get("/gateway/microscope/connection", headers=_as(OPERATOR_TOKEN))
    assert r.json() == {"connected": True}
    r = client.post(
        "/gateway/microscope/connect",
        json={"configuration": "arctis"},
        headers=_as(OPERATOR_TOKEN),
    )
    assert r.status_code == 202
    r = client.post("/gateway/microscope/disconnect", headers=_as(OPERATOR_TOKEN))
    assert (r.status_code, r.json()["detail"]["error_type"]) == (409, "run_on")
    assert [call[1:] for call in host.calls] == [
        ("GET", "/microscope/connection", None),
        ("POST", "/microscope/connect", {"configuration": "arctis"}),
        ("POST", "/microscope/disconnect", {}),
    ]


def test_connecting_needs_the_app_running_and_the_grant(host_setup):
    client, host = host_setup
    reader = _pair_hub(client, [])
    assert (
        client.get(
            "/gateway/microscope/connection", headers=_as(reader["token"])
        ).status_code
        == 200
    )
    r = client.post("/gateway/microscope/connect", headers=_as(reader["token"]))
    assert r.json()["detail"]["error_type"] == "no_grant"
    host.up = False
    r = client.post("/gateway/microscope/connect", headers=_as(OPERATOR_TOKEN))
    assert (r.status_code, r.json()["detail"]["error_type"]) == (409, "no_app")


def test_install_writes_a_startup_shortcut_running_pythonw(tmp_path):
    command = gateway_main.shortcut_command(
        tmp_path / "fibsemOS Gateway.lnk", ["--manufacturer", "Thermo"]
    )
    assert command[:2] == ["powershell", "-NoProfile"]
    script = command[-1]
    assert "pythonw" in script
    assert "-m fibsem.gateway run --manufacturer Thermo" in script
    assert str(tmp_path / "fibsemOS Gateway.lnk") in script


def test_the_app_server_and_the_gateway_share_one_paired_file(tmp_path):
    # Two processes, two stores, one file: neither loses the other's pairings.
    path = tmp_path / "paired.json"
    app_server, gateway = PairedTokens(path), PairedTokens(path)
    _, browser = app_server.add("Firefox on Windows")
    _, hub = gateway.add("mac-mini", HUB, grants=(SWITCH_MODES,))
    assert app_server.match(hub).role == HUB
    assert gateway.match(browser).name == "Firefox on Windows"
    assert {c.name for c in PairedTokens(path).computers()} == {
        "Firefox on Windows",
        "mac-mini",
    }
    app_server.remove(app_server.match(hub).id)
    assert gateway.match(hub) is None


class FakeAgent:
    """Records what the agent routes ask for."""

    def __init__(self):
        self.calls = []
        self.refuse = None

    def state(self):
        from fibsem.gateway.agent import AgentState

        return AgentState(microscope="arctis-pc")

    def profiles(self):
        return [{"name": "default"}]

    def configuration(self, name):
        if self.refuse is not None:
            raise self.refuse
        return {"profile": {"name": name}}

    def start(self, profile, goal, level, who, via, budget_usd=None, replace=False):
        self.calls.append(
            ("start", profile, goal, level, who, via, budget_usd, replace)
        )
        return self.state()

    def end(self, who, via, stop_app=False, force=False):
        self.calls.append(("end", who, via, stop_app, force))
        return self.state()


@pytest.fixture
def with_agent(setup, tmp_path):
    _, gateway, auth = setup
    agent = FakeAgent()
    return TestClient(build_gateway_app(gateway, auth, agent)), agent


def test_the_operator_starts_and_ends_the_agent(with_agent):
    client, agent = with_agent
    r = client.post(
        "/gateway/agent/start",
        json={"profile": "overnight", "goal": " Mill 01 ", "level": "assist"},
        headers=_as(OPERATOR_TOKEN),
    )
    assert r.status_code == 200 and r.json()["microscope"] == "arctis-pc"
    r = client.post(
        "/gateway/agent/end", json={"stop_app": True}, headers=_as(OPERATOR_TOKEN)
    )
    assert r.status_code == 200
    assert agent.calls == [
        ("start", "overnight", "Mill 01", "assist", "operator", "page", None, False),
        ("end", "operator", "page", True, False),
    ]


def test_a_start_with_no_profile_takes_the_default_and_checks_the_budget(with_agent):
    client, agent = with_agent
    for budget in ("lots", -1):
        r = client.post(
            "/gateway/agent/start",
            json={"budget_usd": budget},
            headers=_as(OPERATOR_TOKEN),
        )
        assert r.json()["detail"]["error_type"] == "bad_budget"
    client.post(
        "/gateway/agent/start",
        json={"budget_usd": "7.5", "replace": True},
        headers=_as(OPERATOR_TOKEN),
    )
    assert agent.calls == [
        ("start", "default", None, "watch", "operator", "page", 7.5, True)
    ]


def test_a_hub_needs_the_grant_to_start_the_agent(with_agent):
    client, agent = with_agent
    reader = _pair_hub(client, [])
    assert client.get("/gateway/agent", headers=_as(reader["token"])).status_code == 200
    r = client.post("/gateway/agent/start", json={}, headers=_as(reader["token"]))
    assert r.status_code == 403
    switcher = _pair_hub(client, [SWITCH_MODES])
    r = client.post("/gateway/agent/end", json={}, headers=_as(switcher["token"]))
    assert r.status_code == 200
    assert agent.calls == [("end", "mac-mini", "hub", False, False)]
    r = client.post(
        "/gateway/agent/end", json={"force": True}, headers=_as(switcher["token"])
    )
    assert r.json()["detail"]["error_type"] == "operator_only"


def test_the_profiles_routes(with_agent):
    client, agent = with_agent
    assert client.get("/gateway/agent/profiles").status_code == 401
    r = client.get("/gateway/agent/profiles", headers=_as(OPERATOR_TOKEN))
    assert r.json() == [{"name": "default"}]
    r = client.get("/gateway/agent/profiles/default", headers=_as(OPERATOR_TOKEN))
    assert r.json() == {"profile": {"name": "default"}}
    agent.refuse = GatewayRefused(422, "bad_profile", "No profile 'x'")
    r = client.get("/gateway/agent/profiles/x", headers=_as(OPERATOR_TOKEN))
    assert (r.status_code, r.json()["detail"]["error_type"]) == (422, "bad_profile")


def test_without_an_agent_there_are_no_agent_routes(setup):
    client, _, _ = setup
    assert client.get("/gateway/agent", headers=_as(OPERATOR_TOKEN)).status_code == 404
