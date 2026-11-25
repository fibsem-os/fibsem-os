"""Pairing through the gateways: the SEM PC's gateway asks the device PC's
gateway to pair, the operator there allows or denies it, the two tokens go
only to whoever holds that ask's id, the paired gateway may start and stop
the Device Server and nothing else, and the SEM PC keeps both tokens."""

import json
import sys

import pytest
from fastapi.testclient import TestClient

from fibsem.gateway import peers
from fibsem.gateway.device_server import NOT_SET_UP, DeviceServerProcess
from fibsem.gateway.device_servers import ASKING, DeviceServers, PairRefused
from fibsem.gateway.peers import GATEWAY, MARK, START_DEVICE_SERVER, PeerAsks
from fibsem.gateway.server import GatewayAuth, build_gateway_app
from fibsem.server.pairing import OPERATOR, PairedTokens
from fibsem.util.tunnels import Tunnel
from tests.gateway.test_gateway_device_server import Proc
from tests.gateway.test_gateway_routes import OPERATOR_TOKEN, FakeGateway, _as


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def asks(tmp_path):
    clock = Clock()
    gateway_tokens = PairedTokens(tmp_path / "gateway-paired.json")
    device_tokens = PairedTokens(tmp_path / "device-server-paired.json")
    return PeerAsks(gateway_tokens, lambda: device_tokens, clock=clock), clock


def test_an_ask_waits_for_the_operator(asks):
    asking, _ = asks
    first = asking.ask("sem-pc", "this PC's SSH tunnel", wait_s=0)
    assert first["state"] == "waiting" and len(first["id"]) > 20
    shown = [a.to_dict() for a in asking.asking()]
    assert shown[0]["name"] == "sem-pc" and shown[0]["ask"] == first["id"][:8]
    assert "id" not in shown[0]  # the page never sees the id that collects
    again = asking.ask("sem-pc", "x", ask_id=first["id"], wait_s=0)
    assert again == {"state": "waiting", "id": first["id"]}


def test_one_ask_at_a_time(asks):
    asking, _ = asks
    asking.ask("sem-pc", "here", wait_s=0)
    assert asking.ask("other", "here", wait_s=0)["state"] == "busy"


def test_allow_hands_both_tokens_to_that_id_only(asks):
    asking, _ = asks
    ask_id = asking.ask("sem-pc", "here", wait_s=0)["id"]
    computer = asking.allow(ask_id[:8])
    assert computer.role == GATEWAY and computer.grants == (START_DEVICE_SERVER,)
    assert asking.ask("thief", "here", ask_id="guessed", wait_s=0)["state"] == (
        "lapsed"
    )
    answer = asking.ask("sem-pc", "here", ask_id=ask_id, wait_s=0)
    assert answer["state"] == "allowed" and "until" not in answer
    device = asking._device_tokens()
    assert device.match(answer["device_token"]).name == "sem-pc" + MARK
    assert device.match(answer["device_token"]).role == OPERATOR
    assert asking._gateway_tokens.match(answer["gateway_token"]).role == GATEWAY
    assert asking.asking() == []


def test_the_tokens_wait_only_so_long(asks):
    asking, clock = asks
    ask_id = asking.ask("sem-pc", "here", wait_s=0)["id"]
    asking.allow(ask_id[:8])
    clock.now += peers.HAND_S + 1
    assert asking.ask("sem-pc", "here", ask_id=ask_id, wait_s=0)["state"] == "lapsed"


def test_deny_and_lapse(asks):
    asking, clock = asks
    ask_id = asking.ask("sem-pc", "here", wait_s=0)["id"]
    asking.deny(ask_id[:8])
    assert asking.ask("sem-pc", "here", ask_id=ask_id, wait_s=0)["state"] == "denied"
    with pytest.raises(LookupError):
        asking.allow(ask_id[:8])
    asking.ask("sem-pc", "here", wait_s=0)
    clock.now += peers.ASK_LAPSE_S + 1
    assert asking.asking() == []
    with pytest.raises(LookupError):
        asking.allow("")


def test_unpairing_the_gateway_unpairs_its_device_server_token(asks):
    asking, _ = asks
    ask_id = asking.ask("sem-pc", "here", wait_s=0)["id"]
    computer = asking.allow(ask_id[:8])
    device = asking._device_tokens()
    device.add("by hand", OPERATOR)  # a --pair pairing stays
    assert asking.forget_orphans() == []
    asking._gateway_tokens.remove(computer.id)
    assert asking.forget_orphans() == ["sem-pc" + MARK]
    assert [c.name for c in device.computers()] == ["by hand"]


# -- the device PC's routes ----------------------------------------------------


def _device_server(tmp_path, set_up=True):
    if set_up:
        (tmp_path / "device-server.yaml").write_text("server:\n  port: 8799\n")
    started = []

    def popen(command, **kwargs):
        started.append(Proc(command, **kwargs))
        return started[-1]

    server = DeviceServerProcess(
        sys.executable,
        config=tmp_path / "device-server.yaml",
        home=tmp_path,
        popen=popen,
        get=lambda url, route, token: {"ok": True, "devices": {}},
        token=lambda: "device-token",
        clock=lambda: 1000.0,
    )
    return server, started


@pytest.fixture
def device_pc(tmp_path):
    token_file = tmp_path / "operator-token.json"
    token_file.write_text(json.dumps({"token": OPERATOR_TOKEN}))
    auth = GatewayAuth.from_disk(token_file, tmp_path / "gateway-paired.json")
    server, started = _device_server(tmp_path)
    device_tokens = PairedTokens(tmp_path / "device-server-paired.json")
    asking = PeerAsks(auth.pairing.tokens, lambda: device_tokens)
    gateway = FakeGateway()
    app = build_gateway_app(gateway, auth, device_server=server, peers=asking)
    return TestClient(app), started, gateway


def test_pairing_through_the_device_pcs_routes(device_pc):
    client, started, gateway = device_pc
    r = client.post("/gateway/peers/ask", json={"name": "sem-pc", "wait_s": 0})
    assert r.status_code == 200 and r.json()["state"] == "waiting"
    ask_id = r.json()["id"]
    assert client.get("/gateway/peers/asks").status_code == 401
    shown = client.get("/gateway/peers/asks", headers=_as(OPERATOR_TOKEN)).json()
    assert [a["name"] for a in shown] == ["sem-pc"]
    short = shown[0]["ask"]
    r = client.post(f"/gateway/peers/asks/{short}/allow", headers=_as(OPERATOR_TOKEN))
    assert r.json() == {"answered": "allowed", "name": "sem-pc"}
    answer = client.post(
        "/gateway/peers/ask", json={"name": "sem-pc", "id": ask_id, "wait_s": 0}
    ).json()
    assert answer["state"] == "allowed"
    peer = _as(answer["gateway_token"])
    # The paired gateway starts and stops the Device Server, and nothing else.
    r = client.post("/gateway/device-server/start", headers=peer)
    assert r.status_code == 200 and len(started) == 1
    assert client.get("/gateway", headers=peer).status_code == 401
    assert client.get("/gateway/device-server", headers=peer).status_code == 401
    assert ("sem-pc", "gateway", "start the Device Server") in gateway.access
    assert ("operator", "page", "allowed sem-pc (gateway)") in gateway.access


def test_only_the_operator_answers(device_pc):
    client, _, _ = device_pc
    client.post("/gateway/peers/ask", json={"name": "sem-pc", "wait_s": 0})
    assert client.post("/gateway/peers/asks/x/allow").status_code == 401
    r = client.post("/gateway/peers/asks/nomatch/deny", headers=_as(OPERATOR_TOKEN))
    assert r.status_code == 404


def test_a_pc_with_no_device_server_takes_no_asks(tmp_path):
    token_file = tmp_path / "operator-token.json"
    token_file.write_text(json.dumps({"token": OPERATOR_TOKEN}))
    auth = GatewayAuth.from_disk(token_file, tmp_path / "gateway-paired.json")
    server, _ = _device_server(tmp_path, set_up=False)
    assert server.state().status == NOT_SET_UP
    asking = PeerAsks(auth.pairing.tokens, lambda: PairedTokens(tmp_path / "d.json"))
    client = TestClient(
        build_gateway_app(FakeGateway(), auth, device_server=server, peers=asking)
    )
    r = client.post("/gateway/peers/ask", json={"name": "sem-pc"})
    assert r.status_code == 404 and asking.asking() == []


def test_the_device_server_gets_a_per_start_secret(tmp_path, monkeypatch):
    kept, envs = [], []
    monkeypatch.setattr(
        "fibsem.gateway.child_secrets.keep", lambda pid, s: kept.append((pid, s))
    )
    server, started = _device_server(tmp_path)
    popen = server._popen

    def seen(command, **kwargs):
        envs.append(kwargs.get("env") or {})
        return popen(command, **kwargs)

    server._popen = seen
    server.start()
    secret = envs[0].get("FIBSEM_GATEWAY_SECRET")
    assert secret and kept == [(started[0].pid, secret)]


# -- the SEM PC's side ---------------------------------------------------------

CONFIG = """\
info:
  manufacturer: Demo
hardware:
  devices:
    - name: fm
      driver: Odemis
      remote: 127.0.0.1:8765
      tunnel: meteor@meteor-pc
    - name: afm
      type: stage
      remote: 192.168.0.30:8766
"""


class Tunnels:
    def __init__(self):
        self.opened = []

    def get(self, name):
        return None

    def ensure(self, name, destination, target=None, port=None, reverse=()):
        self.opened.append((name, destination, target, port))
        return Tunnel(destination, port or 18010, target)

    def close(self, name):
        pass

    def close_all(self):
        pass


@pytest.fixture
def sem_pc(tmp_path, monkeypatch):
    monkeypatch.setattr(peers, "REMOTE_DIR", tmp_path / "remote")
    monkeypatch.setattr(peers, "GATEWAY_TOKENS", tmp_path / "remote" / "gateway")
    config = tmp_path / "sem.yaml"
    config.write_text(CONFIG)
    posted = []
    answers = []

    def post(url, route, body, token, timeout):
        posted.append((url, route, body, token))
        return answers.pop(0)

    tunnels = Tunnels()
    servers = DeviceServers(
        lambda: config,
        tunnels=tunnels,
        get=lambda url, token: (200, {"ok": True, "devices": {}}),
        token=lambda url: None,
        accepts=lambda port: True,
        post=post,
        name=lambda: "sem-pc",
    )
    return servers, tunnels, posted, answers, tmp_path


def test_pair_keeps_both_tokens_and_switch_uses_the_gateways(sem_pc):
    servers, tunnels, posted, answers, tmp_path = sem_pc
    answers += [
        (200, {"state": "waiting", "id": "the-id"}),
        (
            200,
            {"state": "allowed", "device_token": "dev-tok", "gateway_token": "gw-tok"},
        ),
    ]
    server = servers._server("127.0.0.1:8765")
    url = servers._gateway_url(server)
    assert tunnels.opened[-1] == (
        "127.0.0.1:8765 gateway",
        "meteor@meteor-pc",
        8010,
        None,
    )
    servers.check()
    assert servers.state()[0]["gateway_tunnel"] is False  # fakes are never open
    servers._pairings[server.key] = ASKING
    servers._pairing(server, url)
    assert [p[2]["id"] for p in posted] == [None, "the-id"]
    assert posted[0][2]["name"] == "sem-pc" and posted[0][3] is None
    assert servers._pairings[server.key] == "Paired."
    assert peers.kept(server.url) == "gw-tok"
    device_file = tmp_path / "remote" / "127.0.0.1_8765.json"
    assert json.loads(device_file.read_text())["token"] == "dev-tok"
    answers.append((200, {"status": "running"}))
    assert servers.switch(server.key, "start") == {"status": "running"}
    assert posted[-1][1:] == ("/gateway/device-server/start", None, "gw-tok")
    # Removing its last device forgets both tokens.
    servers.remove("fm")
    assert peers.kept(server.url) is None and not device_file.exists()


def test_a_denied_pairing_is_said(sem_pc):
    servers, _, _, answers, _ = sem_pc
    answers.append((200, {"state": "denied", "message": "The operator denied it."}))
    server = servers._server("127.0.0.1:8765")
    servers._pairing(server, "http://127.0.0.1:18010")
    assert servers._pairings[server.key] == "The operator denied it."
    assert peers.kept(server.url) is None


def test_pairing_needs_a_tunnel_and_switching_needs_pairing(sem_pc):
    servers, _, _, _, _ = sem_pc
    with pytest.raises(PairRefused, match="SSH tunnel"):
        servers.pair("192.168.0.30:8766")
    with pytest.raises(PairRefused, match="Pair with it first"):
        servers.switch("127.0.0.1:8765", "start")
    with pytest.raises(PairRefused) as missing:
        servers.pair("10.0.0.1:1")
    assert missing.value.status == 404


def test_a_gateway_that_unpaired_it_is_said(sem_pc):
    servers, _, _, answers, _ = sem_pc
    peers.keep("http://127.0.0.1:8765", "dev-tok", "gw-tok")
    answers.append((401, {"detail": "no"}))
    with pytest.raises(PairRefused, match="Pair again") as refused:
        servers.switch("127.0.0.1:8765", "stop")
    assert refused.value.status == 403
