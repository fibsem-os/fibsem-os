"""Who may do what at the gateway's routes: the operator, a hub with and
without "may start and stop apps", and nobody; pairing a hub with its grant;
and the desktop launch's handshake with the gateway."""

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.gateway import main as gateway_main  # noqa: E402
from fibsem.gateway.gateway import (  # noqa: E402
    HEADLESS,
    OFF,
    WINDOW,
    GatewayRefused,
    GatewayState,
)
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

    def switch(
        self,
        mode,
        who,
        via,
        app=None,
        experiment=None,
        last=False,
        force=False,
        configuration=None,
    ):
        return self._do(mode, who, via, app, experiment, last, force, configuration)

    def stop(self, who, via, force=False):
        return self._do(OFF, who, via, force)

    def claim(self, pid, app, who, via):
        return self._do(WINDOW, who, via, pid, app)


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


def test_the_operator_switches_and_forces(setup):
    client, gateway, _ = setup
    r = client.post(
        "/gateway/mode",
        json={
            "mode": HEADLESS,
            "app": "autolamella",
            "last": True,
            "force": True,
            "configuration": "arctis",
        },
        headers=_as(OPERATOR_TOKEN),
    )
    assert r.status_code == 200
    assert gateway.calls == [
        (HEADLESS, "operator", "page", "autolamella", None, True, True, "arctis")
    ]


def test_nobody_unpaired_reads_or_switches(setup):
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


def test_a_hub_with_the_grant_switches_but_never_forces(setup):
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


def test_with_the_gateway_off_a_launch_claims_window_mode(handshake):
    calls = handshake({"mode": OFF})
    assert desktop_launch.gateway_allows("autolamella", AUTOLAMELLA)
    assert _routes(calls)[-1] == ("POST", "/gateway/claim")
    assert calls.calls[-1][2]["app"] == "autolamella"


def test_a_remote_run_is_taken_over_or_left(handshake):
    calls = handshake(
        {"mode": HEADLESS, "experiment": "exp", "who": "mac-mini"}, take_over=False
    )
    assert not desktop_launch.gateway_allows("autolamella", AUTOLAMELLA)
    assert _routes(calls) == [("GET", "/gateway")]
    calls = handshake({"mode": HEADLESS}, take_over=True)
    assert desktop_launch.gateway_allows("autolamella", AUTOLAMELLA)
    assert _routes(calls) == [
        ("GET", "/gateway"),
        ("POST", "/gateway/stop"),
        ("POST", "/gateway/claim"),
    ]
    assert calls.calls[1][2] == {"force": True, "via": "desktop launch"}


def test_a_window_the_gateway_opened_doesnt_ask(handshake, monkeypatch):
    calls = handshake({"mode": WINDOW})
    monkeypatch.setenv("FIBSEM_GATEWAY", WINDOW)
    assert desktop_launch.gateway_allows("autolamella", AUTOLAMELLA)
    assert calls.calls == []


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
