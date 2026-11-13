"""The hub role: the instrument list, pairing with each instrument's gateway,
reading their health, and who may change the list."""

import json

import pytest
from fastapi.testclient import TestClient

from fibsem.gateway.gateway import OFF, GatewayRefused, GatewayState, History
from fibsem.gateway.hub import Hub, Instrument
from fibsem.gateway.server import GatewayAuth, build_gateway_app
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


@pytest.fixture
def hub(tmp_path, instruments):
    clock = iter(range(100, 10_000))
    return Hub(
        tmp_path / "hub.yaml",
        tmp_path / "hub",
        fetch=instruments.fetch,
        pair=instruments.pair,
        clock=lambda: float(next(clock)),
    )


def test_a_gateway_with_no_list_is_not_a_hub_and_reads_nothing(tmp_path):
    def fetch(*args):
        raise AssertionError("nothing is read without a list")

    hub = Hub(tmp_path / "hub.yaml", tmp_path / "hub", fetch=fetch)
    hub.check()
    state = hub.state()
    assert state.hub is False and state.instruments == []


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
    assert hub.state().hub is True


@pytest.mark.parametrize(
    "name, url, error",
    [
        ("", URL, "bad_name"),
        ("../x", URL, "bad_name"),
        ("Scope", "scope-1:8010", "bad_url"),
        ("Scope", "ftp://scope-1", "bad_url"),
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
    """The hub's own gateway, with its routes."""
    auth = _auth(tmp_path)
    client = TestClient(build_gateway_app(FakeGateway(), auth, hub=hub))
    return client, auth


def test_the_operator_changes_the_list_and_a_paired_hub_only_reads_it(served):
    client, auth = served
    operator = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
    assert client.get("/hub").status_code == 401
    assert client.get("/hub", headers=operator).json() == {
        "hub": False,
        "instruments": [],
    }

    body = {"name": "Scope 1", "url": URL}
    added = client.post("/hub/instruments", json=body, headers=operator)
    assert added.status_code == 200 and added.json()["hub"] is True
    assert added.json()["instruments"][0]["name"] == "Scope 1"

    code = auth.pairing.open(role=HUB)
    token = client.post("/pair", json={"code": code, "name": "other"}).json()["token"]
    other = {"Authorization": f"Bearer {token}"}
    assert client.get("/hub", headers=other).status_code == 200
    refused = client.post("/hub/instruments", json=body, headers=other)
    assert refused.status_code == 403
    assert client.delete("/hub/instruments/Scope 1", headers=other).status_code == 403

    bad = client.post("/hub/instruments", json={"name": "x"}, headers=operator)
    assert bad.status_code == 422 and bad.json()["detail"]["error_type"] == "bad_url"
    no_code = client.post("/hub/instruments/Scope 1/pair", json={}, headers=operator)
    assert no_code.status_code == 422
    gone = client.delete("/hub/instruments/Scope 1", headers=operator)
    assert gone.json() == {"hub": False, "instruments": []}
    assert (
        client.delete("/hub/instruments/Scope 1", headers=operator).status_code == 404
    )
