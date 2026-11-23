"""An agent on this PC asks the gateway to connect with ``gateway.json``'s token,
and the operator allows or denies it: the rules, the gateway's routes, the App
Server checking the token the gateway paired, and the client's side."""

import json
import threading
import time

import httpx
import pytest

from fibsem.client import pairing as client_pairing
from fibsem.client.transport import Transport
from fibsem.server.connections import DENIED_S, Connections
from fibsem.server.pairing import AGENT, PairedTokens
from tests.server.test_operator_routes import (  # noqa: F401
    _bearer,
    microscope,
    server,
)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def connections(tmp_path, clock):
    return Connections(PairedTokens(tmp_path / "paired.json"), clock=clock)


def _asking(connections, answers, **ask):
    """Ask on a thread, as a client's long poll does; its answer lands in *answers*."""
    fields = {"client": "claude-code 2.1.3", "origin": "this PC", "user": "lab"}
    fields.update(ask)
    thread = threading.Thread(
        target=lambda: answers.append(connections.ask(wait_s=5, **fields))
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not connections.asking() and time.monotonic() < deadline:
        time.sleep(0.01)
    return thread


# --- the rules ------------------------------------------------------------------


def test_an_ask_waits_and_comes_back_waiting(connections):
    answer = connections.ask("claude-code 2.1.3", "this PC", "lab", wait_s=0)
    assert answer["state"] == "waiting"
    (ask,) = connections.asking()
    assert ask.id == answer["id"]
    assert ask.name == "claude-code on this PC"


def test_allow_pairs_it_as_an_agent_and_hands_every_asker_the_token(connections):
    answers = []
    first = _asking(connections, answers)
    # The agent's watch asks too: the same card.
    second = _asking(connections, answers, client="fibsem-client 0.6")
    assert len(connections.asking()) == 1
    computer = connections.allow(connections.asking()[0].id)
    first.join(5)
    second.join(5)
    assert computer.role == AGENT and computer.name == "claude-code on this PC"
    assert [a["state"] for a in answers] == ["allowed", "allowed"]
    assert answers[0]["token"] == answers[1]["token"]
    assert connections.tokens.match(answers[0]["token"]) == computer
    assert connections.asking() == []


def test_deny_refuses_and_the_next_ask_waits_a_minute(connections, clock):
    answers = []
    thread = _asking(connections, answers)
    connections.deny(connections.asking()[0].id, "not tonight")
    thread.join(5)
    assert answers[0]["state"] == "denied"
    assert "not tonight" in answers[0]["message"]
    assert connections.ask("x", "this PC", "lab", wait_s=0)["state"] == "denied"
    assert connections.tokens.computers() == []
    clock.now += DENIED_S
    assert connections.ask("x", "this PC", "lab", wait_s=0)["state"] == "waiting"


def test_an_agent_unpaired_since_its_allow_asks_afresh(connections):
    answers = []
    thread = _asking(connections, answers)
    computer = connections.allow(connections.asking()[0].id)
    thread.join(5)
    connections.tokens.remove(computer.id)
    assert connections.ask("x", "this PC", "lab", wait_s=0)["state"] == "waiting"


def test_deciding_what_nobody_asks_is_refused(connections):
    with pytest.raises(LookupError):
        connections.allow("nope")
    with pytest.raises(LookupError):
        connections.deny("nope")


# --- the routes: the gateway's ---------------------------------------------------


@pytest.fixture
def gateway(server, tmp_path):
    """This PC's gateway in front of *server*: the one place that pairs, on the
    same paired file and operator token, its connect token in ``gateway.json``."""
    from fibsem.gateway.gateway import Gateway, write_gateway_file
    from fibsem.gateway.passthrough import PassThrough
    from fibsem.gateway.server import GatewayAuth, build_gateway_app
    from tests.fixtures.app_server import serving

    auth = GatewayAuth.from_disk(
        tmp_path / "operator-token.json", tmp_path / "paired.json"
    )
    home = tmp_path / "gateway"
    home.mkdir()
    gateway = Gateway(
        home=home,
        app_registry=lambda: {},
        last=lambda: None,
        recent=lambda: [],
        new_folder=lambda: str(home),
        configurations=lambda: [],
        discovery=lambda: None,
    )
    app = build_gateway_app(
        gateway,
        auth,
        passthrough=PassThrough(
            lambda: server.session.app_server.url, is_connect=auth.is_connect
        ),
    )
    with serving(app) as url:
        found = write_gateway_file(url, auth.connect_token, tmp_path / "gateway.json")
        with httpx.Client(base_url=url, timeout=30) as client:
            client.auth_config = auth
            client.found = found
            client.operator = server.operator
            client.agent = server.agent
            client.server = server
            client.history = gateway.history
            yield client


def _found(gateway):
    return json.loads(gateway.found.read_text())


def test_the_app_server_takes_no_asks_and_pairs_nothing(server):
    found = json.loads(server.session.app_server._discovery_path.read_text())
    assert "token" not in found and "connect_token" not in found
    for method, path in (
        ("POST", "/connect"),
        ("GET", "/pair"),
        ("GET", "/pairing"),
        ("GET", "/ui/agent"),
        ("GET", "/chat"),
    ):
        answer = server.request(method, path, headers=server.operator)
        assert answer.status_code in (404, 405), (path, answer.status_code)


def test_the_gateway_file_only_asks_to_connect(gateway):
    found = _found(gateway)
    assert "token" not in found
    connect = _bearer(found["connect_token"])
    refused = gateway.get("/capabilities", headers=connect)
    assert refused.status_code == 403
    assert refused.json()["detail"]["error_type"] == "connect_first"
    assert gateway.get("/app/experiment", headers=connect).status_code == 403
    # The connect token is no operator either.
    assert gateway.get("/gateway/connections", headers=connect).status_code == 401
    # Only the connect token asks.
    assert gateway.post("/connect", headers=gateway.agent).status_code == 401


def _ask(gateway, wait_s=10, **headers):
    return gateway.post(
        "/connect",
        json={"origin": "this PC", "user": "lab", "wait_s": wait_s},
        headers={**_bearer(_found(gateway)["connect_token"]), **headers},
    ).json()


def _card(gateway):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        asking = gateway.get("/gateway/connections", headers=gateway.operator).json()
        if asking["asking"]:
            return asking["asking"]
        time.sleep(0.02)
    raise AssertionError("no ask came")


def test_the_operator_allows_the_asking_agent_and_it_acts_at_the_level(gateway):
    answers = []
    thread = threading.Thread(
        target=lambda: answers.append(
            _ask(gateway, **{"User-Agent": "fibsem-mcp/0.6 claude-code/2.1.3"})
        )
    )
    thread.start()
    asking = _card(gateway)
    assert asking["client"] == "claude-code 2.1.3"
    assert asking["name"] == "claude-code on this PC"
    assert "acts only at the agent level" in asking["if_allowed"]
    # The agent can't allow itself.
    agent_tried = gateway.post(
        f"/gateway/connections/{asking['id']}/allow", headers=gateway.agent
    )
    assert agent_tried.status_code == 401
    allowed = gateway.post(
        f"/gateway/connections/{asking['id']}/allow", headers=gateway.operator
    )
    assert allowed.json() == {
        "id": asking["id"],
        "allowed": True,
        "name": "claude-code on this PC",
    }
    thread.join(10)
    token = answers[0]["token"]
    # The App Server checks the token the gateway paired: at the level set.
    caps = gateway.get("/capabilities", headers=_bearer(token))
    assert caps.json()["level"] == "watch"
    direct = gateway.server.get("/capabilities", headers=_bearer(token))
    assert direct.json()["level"] == "watch"
    # Listed with Unpair, as a computer paired with a code is.
    listed = gateway.get("/pairing", headers=gateway.operator).json()["computers"]
    assert [(c["name"], c["role"]) for c in listed] == [
        ("claude-code on this PC", "agent")
    ]
    # Who let it in, on the gateway's History.
    (line,) = gateway.history().switches
    assert (line.kind, line.result, line.who) == (
        "agent",
        "Allowed claude-code on this PC",
        "operator",
    )


def test_the_operator_denies_with_a_reason(gateway):
    waiting = _ask(gateway, wait_s=0)
    assert waiting["state"] == "waiting"
    denied = gateway.post(
        f"/gateway/connections/{waiting['id']}/deny",
        json={"reason": "who is this?"},
        headers=gateway.operator,
    )
    assert denied.json() == {"id": waiting["id"], "allowed": False, "name": None}
    again = _ask(gateway, wait_s=0)
    assert again["state"] == "denied" and "who is this?" in again["message"]
    (line,) = gateway.history().switches
    assert (line.kind, line.result) == (
        "agent",
        "Denied an agent on this PC: who is this?",
    )
    missing = gateway.post("/gateway/connections/nope/deny", headers=gateway.operator)
    assert missing.status_code == 404


# --- the client ---------------------------------------------------------------------


def _allow_when_asked(gateway):
    """The operator, clicking Allow once the card shows an ask."""

    def allow():
        ask_id = _card(gateway)["id"]
        gateway.post(f"/gateway/connections/{ask_id}/allow", headers=gateway.operator)

    thread = threading.Thread(target=allow)
    thread.start()
    return thread


@pytest.fixture
def discovered(gateway, monkeypatch):
    """Clients find the gateway by its file, as on the microscope PC."""
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    monkeypatch.setattr("fibsem.server.discovery.GATEWAY_FILE", gateway.found)
    monkeypatch.setattr(
        "fibsem.server.discovery.DISCOVERY_FILE",
        gateway.server.session.app_server._discovery_path,
    )
    return gateway


def test_a_client_asks_keeps_the_token_and_later_connects_without_asking(discovered):
    gateway = discovered
    operator = _allow_when_asked(gateway)
    with Transport(user_agent="fibsem-mcp/0.6 claude-code/2.1.3") as transport:
        assert transport.url == str(gateway.base_url).rstrip("/")
        assert transport.fetch("GET", "/capabilities")["level"] == "watch"
    operator.join(20)
    kept = client_pairing.agent_token(transport.url)
    assert kept is not None
    # The next program connects with the kept token: no card.
    with Transport() as transport:
        assert transport.bearer == f"Bearer {kept}"
        assert transport.fetch("GET", "/capabilities")["level"] == "watch"
    asking = gateway.get("/gateway/connections", headers=gateway.operator).json()
    assert asking["asking"] is None


def test_a_client_waiting_too_long_is_told_to_try_again(discovered):
    with Transport(connect_wait_s=0) as transport:
        answer = transport.request("GET", "/capabilities")
    assert answer.status_code == 403
    assert answer.json()["detail"]["error_type"] == "connect_waiting"


def test_an_unpaired_agent_forgets_its_token_and_asks_again(discovered):
    gateway = discovered
    operator = _allow_when_asked(gateway)
    transport = Transport()
    transport.fetch("GET", "/capabilities")
    operator.join(20)
    computers = gateway.get("/pairing", headers=gateway.operator).json()["computers"]
    gateway.delete(f"/pairing/computers/{computers[0]['id']}", headers=gateway.operator)
    operator = _allow_when_asked(gateway)
    assert transport.fetch("GET", "/capabilities")["level"] == "watch"
    operator.join(20)
    transport.close()
    computers = gateway.get("/pairing", headers=gateway.operator).json()["computers"]
    assert len(computers) == 1


def test_with_no_gateway_an_app_server_alone_takes_the_launchers_token(
    server, monkeypatch
):
    monkeypatch.setattr(
        "fibsem.server.discovery.GATEWAY_FILE",
        server.session.app_server._discovery_path.parent / "none.json",
    )
    monkeypatch.setattr(
        "fibsem.server.discovery.DISCOVERY_FILE",
        server.session.app_server._discovery_path,
    )
    monkeypatch.setenv("FIBSEM_SERVER_TOKEN", server.config.token)
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    with Transport() as transport:
        assert transport.url == server.session.app_server.url
        assert transport.fetch("GET", "/capabilities")["level"] == "watch"


# --- always allow agents logged in to this PC ---------------------------------------


def test_always_allow_pairs_an_ask_from_this_pc_at_once(tmp_path, clock):
    saved, allowed = [], []
    connections = Connections(
        PairedTokens(tmp_path / "paired.json"),
        clock=clock,
        save_always_allow=saved.append,
    )
    connections.on_allowed_always(lambda computer, replaced: allowed.append(computer))
    connections.set_always_allow(True)
    assert saved == [True]
    answer = connections.ask(
        "claude-code 2.1.3", "10.0.0.5 over SSH", "lab", wait_s=0, loopback=True
    )
    assert answer["state"] == "allowed"
    assert connections.tokens.match(answer["token"]) == allowed[0]
    assert connections.asking() == []


def test_always_allow_asks_a_caller_from_another_address(connections):
    connections.set_always_allow(True)
    # What the body says of its origin is display text only.
    answer = connections.ask("x", "this PC", "lab", wait_s=0, loopback=False)
    assert answer["state"] == "waiting"
    assert connections.tokens.computers() == []


@pytest.mark.parametrize(
    "peer, loopback",
    [("127.0.0.1", True), ("::1", True), ("10.0.0.5", False), ("testclient", False)],
)
def test_loopback_is_the_peer_address(peer, loopback):
    from types import SimpleNamespace

    from fibsem.server.pairing_routes import _loopback

    assert _loopback(SimpleNamespace(client=SimpleNamespace(host=peer))) is loopback
    assert _loopback(SimpleNamespace(client=None)) is False


def test_the_operator_sets_always_allow_and_an_ask_from_this_pc_is_let_in(
    gateway, monkeypatch
):
    saved = []
    monkeypatch.setattr(
        gateway.auth_config.connections, "_save_always_allow", saved.append
    )
    agent_refused = gateway.put(
        "/pairing/always_allow_local", json={"on": True}, headers=gateway.agent
    )
    assert agent_refused.status_code in (401, 403)
    listing = gateway.put(
        "/pairing/always_allow_local", json={"on": True}, headers=gateway.operator
    ).json()
    assert listing["always_allow_local"] is True and saved == [True]
    # this test reaches the gateway on 127.0.0.1: an ask from this PC
    assert _ask(gateway, wait_s=0)["state"] == "allowed"


# --- one agent per microscope -------------------------------------------------------


def test_a_new_agent_replaces_the_allowed_one(tmp_path):
    tokens = PairedTokens(tmp_path / "paired.json")
    browser, _ = tokens.add("lab laptop")
    first, first_token = tokens.add("claude-code on this PC", AGENT)
    second, second_token = tokens.add("codex on mac-mini", AGENT)
    assert tokens.agent() == second
    assert [c.name for c in tokens.computers()] == ["lab laptop", "codex on mac-mini"]
    assert tokens.match(first_token) is None
    assert tokens.match(second_token) == second


def test_the_card_says_which_agent_allowing_replaces(gateway):
    gateway.auth_config.pairing.tokens.add("codex on mac-mini", AGENT)
    waiting = _ask(gateway, wait_s=0)
    asking = _card(gateway)
    assert asking["replaces"] == "codex on mac-mini"
    assert asking["if_allowed"].endswith(
        "It replaces codex on mac-mini, which is unpaired."
    )
    gateway.post(
        f"/gateway/connections/{waiting['id']}/allow", headers=gateway.operator
    )
    listed = gateway.get("/pairing", headers=gateway.operator).json()["computers"]
    assert [c["name"] for c in listed if c["role"] == "agent"] == [
        "an agent on this PC"
    ]
