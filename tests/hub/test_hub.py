"""The hub: the instrument list, pairing with each instrument's gateway,
reading their health, and who may change the list."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fibsem.gateway.gateway import OFF, GatewayRefused, GatewayState, History
from fibsem.gateway.server import GatewayAuth, build_gateway_app
from fibsem.hub.hub import Hub, Instrument
from fibsem.hub.server import build_hub_app, hub_auth
from fibsem.hub.tunnels import Tunnels, command
from fibsem.server.pairing import HUB

OPERATOR_TOKEN = "operator-token"
URL = "http://scope-1:8010"


class FakeGateway:
    def state(self):
        return GatewayState(mode=OFF, version="1.2.3")

    def history(self):
        return History()


def _auth(tmp_path, name="operator-token.json"):
    token_file = tmp_path / name
    token_file.write_text(json.dumps({"token": OPERATOR_TOKEN}))
    return GatewayAuth.from_disk(token_file, tmp_path / f"paired-{name}")


class Instruments:
    """Instrument gateways by address, each a real gateway app: the hub's
    reads and its pairing go to their routes. One taken out of *up* doesn't
    answer."""

    def __init__(self, tmp_path):
        self.auth = _auth(tmp_path, "instrument.json")
        self.client = TestClient(build_gateway_app(FakeGateway(), self.auth))
        self.up = {URL}

    def fetch(self, url, route, token, timeout):
        if url not in self.up:
            raise OSError("connection refused")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = self.client.get(route, headers=headers)
        return response.status_code, response.json()

    def pair(self, url, code, name):
        if url not in self.up:
            raise OSError("connection refused")
        response = self.client.post("/pair", json={"code": code, "name": name})
        if response.status_code != 200:
            raise PermissionError(response.json()["detail"]["message"])
        return response.json()["token"], response.json()["role"]


@pytest.fixture
def instruments(tmp_path):
    return Instruments(tmp_path)


class FakeSsh:
    """An ssh that holds its tunnel open until ended, or ends at once saying
    *fails*."""

    def __init__(self, command, fails=None):
        self.command = command
        self.code = None if fails is None else 255
        self.stderr = SimpleNamespace(read=lambda: fails or "")

    def poll(self):
        return self.code

    def terminate(self):
        self.code = -15

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


class Ssh:
    """Every ssh the tunnels started, and the time they read."""

    def __init__(self, first_port=50_000):
        self.started = []
        self.fails = None
        self.now = 0.0
        ports = iter(range(first_port, 60_000))
        self.tunnels = Tunnels(
            spawn=self.spawn,
            port=lambda: next(ports),
            accepts=self.accepts,
            which=lambda name: "/usr/bin/ssh",
            clock=lambda: self.now,
            sleep=lambda s: None,
        )

    def spawn(self, command):
        self.started.append(FakeSsh(command, self.fails))
        return self.started[-1]

    def accepts(self, port):
        """Whether a live ssh forwards *port*."""
        return any(
            process.poll() is None and f"127.0.0.1:{port}:127.0.0.1" in forward
            for process in self.started
            for forward in process.command
        )


@pytest.fixture
def ssh():
    return Ssh()


@pytest.fixture
def hub(tmp_path, instruments, ssh):
    clock = iter(range(100, 10_000))
    return Hub(
        tmp_path / "hub.yaml",
        tmp_path / "hub",
        fetch=instruments.fetch,
        pair=instruments.pair,
        clock=lambda: float(next(clock)),
        tunnels=ssh.tunnels,
    )


def test_a_hub_with_no_list_reads_nothing(tmp_path):
    def fetch(*args):
        raise AssertionError("nothing is read without a list")

    hub = Hub(tmp_path / "hub.yaml", tmp_path / "hub", fetch=fetch)
    hub.check()
    state = hub.state()
    assert state.instruments == []


def test_the_list_is_kept_in_the_hubs_file(hub, tmp_path):
    hub.add("Scope 1", URL + "/", ssh="user@scope-1")
    hub.add("Scope 2", "http://scope-2:8010")
    again = Hub(tmp_path / "hub.yaml", tmp_path / "hub")
    assert again.instruments() == [
        Instrument("Scope 1", URL, "user@scope-1"),
        Instrument("Scope 2", "http://scope-2:8010"),
    ]
    # The same name again changes its address: one row.
    hub.add("Scope 2", "http://scope-2:8011")
    assert [i.url for i in hub.instruments()][1] == "http://scope-2:8011"


@pytest.mark.parametrize(
    "name, url, error",
    [
        ("", URL, "bad_name"),
        ("../x", URL, "bad_name"),
        ("Scope", "scope-1:8010", "bad_url"),
        ("Scope", "ftp://scope-1", "bad_url"),
        ("Scope", None, "no_address"),
    ],
)
def test_a_bad_name_or_address_is_refused(hub, name, url, error):
    with pytest.raises(GatewayRefused) as refused:
        hub.add(name, url)
    assert refused.value.error_type == error
    assert hub.instruments() == []


def test_an_unpaired_instrument_is_reachable_but_not_read(hub):
    hub.add("Scope 1", URL)
    hub.check()
    (row,) = hub.state().instruments
    assert row.reachable is True and row.paired is False
    assert row.gateway is None and "Pair this hub" in row.why


def test_pairing_keeps_a_token_and_the_instrument_is_read(hub, instruments, tmp_path):
    hub.add("Scope 1", URL)
    code = instruments.auth.pairing.open(role=HUB)
    (row,) = hub.pair("Scope 1", code).instruments
    assert row.paired and row.reachable and row.why is None
    assert row.gateway == {"mode": OFF, "version": "1.2.3", "switching": False}
    assert row.agent is None
    kept = tmp_path / "hub" / "scope-1.json"
    assert json.loads(kept.read_text())["role"] == HUB
    assert instruments.auth.hub(hub.token("Scope 1")) is not None


def test_a_wrong_code_is_refused_in_the_gateways_words(hub):
    hub.add("Scope 1", URL)
    with pytest.raises(GatewayRefused) as refused:
        hub.pair("Scope 1", "000000")
    assert refused.value.error_type == "pairing_refused"
    assert hub.token("Scope 1") is None


def test_an_instrument_that_stops_answering_says_since_when(hub, instruments):
    hub.add("Scope 1", URL)
    hub.check()
    up = hub.state().instruments[0]
    hub.check()
    assert hub.state().instruments[0].since == up.since, "still up: the same moment"

    instruments.up.clear()
    hub.check()
    down = hub.state().instruments[0]
    assert down.reachable is False and "didn't answer" in down.why
    assert down.since > up.since
    hub.check()
    assert hub.state().instruments[0].since == down.since


def test_one_dead_instrument_does_not_hide_the_others(hub):
    hub.add("Scope 1", URL)
    hub.add("Scope 2", "http://scope-2:8010")
    hub.check()
    one, two = hub.state().instruments
    assert one.reachable is True and two.reachable is False


def test_the_gateways_health_is_shown_for_each_instrument(
    hub, instruments, monkeypatch
):
    from fibsem.gateway.health import Health, ServersHealth

    health = ServersHealth(
        app=Health("ok", checked_at=5.0, answer_s=0.01, since=1.0),
        microscope=Health("halted", "Safety Halt is on.", since=2.0),
        heard={"devices": {"stage": {"last_ok": 4.0}}},
    )
    monkeypatch.setattr(
        FakeGateway,
        "state",
        lambda self: GatewayState(mode=OFF, version="1.2.3", health=health),
    )
    hub.add("Scope 1", URL)
    (row,) = hub.pair("Scope 1", instruments.auth.pairing.open(role=HUB)).instruments
    assert row.health == {
        "app": {"status": "ok", "since": 1.0},
        "microscope": {
            "status": "halted",
            "detail": "Safety Halt is on.",
            "since": 2.0,
        },
    }


def test_a_gateway_that_gives_no_health_shows_none(hub, instruments):
    hub.add("Scope 1", URL)
    (row,) = hub.pair("Scope 1", instruments.auth.pairing.open(role=HUB)).instruments
    assert row.health is None


def test_a_gateway_that_forgot_the_hub_says_to_pair_again(hub, instruments):
    hub.add("Scope 1", URL)
    hub.pair("Scope 1", instruments.auth.pairing.open(role=HUB))
    for computer in instruments.auth.pairing.tokens.computers():
        instruments.auth.pairing.tokens.remove(computer.id)
    hub.check()
    (row,) = hub.state().instruments
    assert row.reachable is True and row.gateway is None
    assert "pair again" in row.why


def test_removing_an_instrument_drops_its_token(hub, instruments, tmp_path):
    hub.add("Scope 1", URL)
    hub.pair("Scope 1", instruments.auth.pairing.open(role=HUB))
    assert hub.remove("Scope 1").instruments == []
    assert not (tmp_path / "hub" / "scope-1.json").exists()
    with pytest.raises(GatewayRefused):
        hub.remove("Scope 1")


@pytest.fixture
def served(tmp_path, hub):
    """The hub's own routes, with its own paired computers."""
    token_file = tmp_path / "operator-token.json"
    token_file.write_text(json.dumps({"token": OPERATOR_TOKEN}))
    auth = hub_auth(token_file, tmp_path / "hub-paired.json")
    stopped = []
    client = TestClient(build_hub_app(hub, auth, lambda: stopped.append(True)))
    return client, auth, stopped


def test_only_someone_signed_in_at_the_hub_reads_or_changes_the_list(served):
    client, auth, stopped = served
    operator = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
    assert client.get("/hub").status_code == 401
    assert client.post("/hub/instruments", json={}).status_code == 401
    assert client.get("/hub", headers=operator).json() == {"instruments": []}

    body = {"name": "Scope 1", "url": URL}
    added = client.post("/hub/instruments", json=body, headers=operator)
    assert added.status_code == 200
    assert added.json()["instruments"][0]["name"] == "Scope 1"

    # A browser on another computer signs in with a code from the hub.
    code = auth.pairing.open()
    token = client.post("/pair", json={"code": code, "name": "desk"}).json()["token"]
    desk = {"Authorization": f"Bearer {token}"}
    assert client.get("/hub", headers=desk).status_code == 200

    bad = client.post("/hub/instruments", json={"name": "x"}, headers=operator)
    assert bad.status_code == 422 and bad.json()["detail"]["error_type"] == "no_address"
    no_code = client.post("/hub/instruments/Scope 1/pair", json={}, headers=operator)
    assert no_code.status_code == 422
    gone = client.delete("/hub/instruments/Scope 1", headers=operator)
    assert gone.json() == {"instruments": []}
    assert (
        client.delete("/hub/instruments/Scope 1", headers=operator).status_code == 404
    )


def test_a_hub_only_pairs_browsers_and_stops_for_the_operator(served):
    client, auth, stopped = served
    operator = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
    # Nothing pairs with a hub as a hub or an agent: it is not a door.
    refused = client.post("/pairing", json={"role": "hub"}, headers=operator)
    assert refused.status_code == 422
    assert client.post("/hub/quit").status_code == 401 and not stopped
    assert client.post("/hub/quit", headers=operator).json() == {"stopping": True}
    assert stopped == [True]


def test_a_gateway_serves_no_hub_routes(tmp_path):
    client = TestClient(build_gateway_app(FakeGateway(), _auth(tmp_path)))
    operator = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
    assert client.get("/hub", headers=operator).status_code == 404


# -- tunnels ---------------------------------------------------------------


def test_an_instrument_with_ssh_is_reached_through_a_tunnel_the_hub_opens(
    hub, instruments, ssh
):
    instruments.up = {"http://127.0.0.1:50000"}
    hub.add("Scope 1", ssh="user@scope-1")
    hub.check()
    (row,) = hub.state().instruments
    assert row.reachable is True
    assert row.tunnel == {"open": True, "gateway_port": 50000}
    (process,) = ssh.started
    assert process.command[-2:] == ["--", "user@scope-1"]
    assert "BatchMode=yes" in process.command
    assert "127.0.0.1:50000:127.0.0.1:8010" in process.command
    forwards = [part for part in process.command if part.startswith("127.0.0.1:")]
    assert forwards == ["127.0.0.1:50000:127.0.0.1:8010"], (
        "the gateway is the one way onto an instrument: nothing else is forwarded"
    )

    hub.check()
    assert len(ssh.started) == 1, "an open tunnel is kept, not opened again"


def test_two_instruments_never_share_a_port(hub, ssh):
    hub.add("Scope 1", ssh="user@scope-1")
    hub.add("Scope 2", ssh="user@scope-2")
    hub.check()
    ports = [row.tunnel["gateway_port"] for row in hub.state().instruments]
    assert len(set(ports)) == 2


def test_a_tunnel_that_drops_is_opened_again(hub, instruments, ssh):
    instruments.up = {"http://127.0.0.1:50000"}
    hub.add("Scope 1", ssh="user@scope-1")
    hub.check()
    ssh.started[0].code = 255  # the network went away
    hub.check()
    assert len(ssh.started) == 2
    (row,) = hub.state().instruments
    # On the same ports: an agent given the app's address still reaches it.
    assert row.reachable is True and row.tunnel["gateway_port"] == 50000


def test_a_new_hub_opens_each_tunnel_on_the_ports_it_had(hub, tmp_path, instruments):
    hub.add("Scope 1", ssh="user@scope-1")
    hub.check()
    before = hub.state().instruments[0].tunnel
    hub.stop()

    again = Ssh(first_port=55_000)
    later = Hub(
        tmp_path / "hub.yaml",
        tmp_path / "hub",
        fetch=instruments.fetch,
        pair=instruments.pair,
        tunnels=again.tunnels,
    )
    later.check()
    assert later.state().instruments[0].tunnel == before


def test_ports_kept_while_the_app_was_forwarded_too_still_place_the_tunnel(
    tmp_path, instruments
):
    (tmp_path / "hub").mkdir()
    (tmp_path / "hub" / "_ports.json").write_text(
        json.dumps({"Scope 1": [51234, 51235]})
    )
    hub = Hub(
        tmp_path / "hub.yaml",
        tmp_path / "hub",
        fetch=instruments.fetch,
        pair=instruments.pair,
        tunnels=Ssh().tunnels,
    )
    hub.add("Scope 1", ssh="user@scope-1")
    hub.check()
    assert hub.state().instruments[0].tunnel["gateway_port"] == 51234
    assert json.loads((tmp_path / "hub" / "_ports.json").read_text()) == {
        "Scope 1": 51234
    }


def test_a_tunnel_that_cannot_open_says_why_and_waits_before_trying_again(hub, ssh):
    ssh.fails = "Warning: something\nuser@scope-1: Permission denied (publickey).\n"
    hub.add("Scope 1", ssh="user@scope-1")
    hub.check()
    (row,) = hub.state().instruments
    assert row.reachable is False and row.tunnel["open"] is False
    assert row.why == (
        "ssh to user@scope-1 failed: user@scope-1: Permission denied (publickey)."
    )
    hub.check()
    assert len(ssh.started) == 1, "not hammered: tried again after a wait"
    ssh.now += 60
    ssh.fails = None
    hub.check()
    assert len(ssh.started) == 2 and hub.state().instruments[0].tunnel["open"]


def test_removing_an_instrument_or_stopping_the_hub_closes_its_tunnel(hub, ssh):
    hub.add("Scope 1", ssh="user@scope-1")
    hub.add("Scope 2", ssh="user@scope-2")
    hub.check()
    by_destination = {p.command[-1]: p for p in ssh.started}
    hub.remove("Scope 1")
    assert by_destination["user@scope-1"].poll() is not None
    assert by_destination["user@scope-2"].poll() is None
    hub.stop()
    assert by_destination["user@scope-2"].poll() is not None


@pytest.mark.parametrize("destination", ["-oProxyCommand=x", "user@host x", "", "a;b"])
def test_an_ssh_destination_that_could_be_an_option_is_refused(hub, ssh, destination):
    with pytest.raises(GatewayRefused):
        hub.add("Scope 1", ssh=destination)
    assert ssh.started == []


def test_without_ssh_installed_the_tunnel_says_so():
    tunnels = Tunnels(which=lambda name: None, spawn=lambda command: 1 / 0)
    with pytest.raises(OSError, match="ssh isn't installed"):
        tunnels.ensure("Scope 1", "user@scope-1")


def test_the_tunnel_command_only_forwards(ssh):
    tunnel = ssh.tunnels.ensure("Scope 1", "user@scope-1")
    assert command(tunnel)[:2] == ["ssh", "-N"]
