"""An agent on this PC asks to connect with the discovery file's token, and the
operator allows or denies it: the rules, the routes, and the client's side."""

import json
import threading
import time

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fibsem.client import pairing as client_pairing  # noqa: E402
from fibsem.client.transport import Transport  # noqa: E402
from fibsem.server.connections import DENIED_S, Connections  # noqa: E402
from fibsem.server.pairing import AGENT, PairedTokens  # noqa: E402
from fibsem.workflows.actions import ACTION  # noqa: E402
from tests.server.test_operator_routes import (  # noqa: E402, F401
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


# --- the routes -------------------------------------------------------------------


def _found(server):
    return json.loads(server.session.app_server._discovery_path.read_text())


def test_the_discovery_file_only_asks_to_connect(server):
    found = _found(server)
    assert "token" not in found
    connect = _bearer(found["connect_token"])
    refused = server.get("/capabilities", headers=connect)
    assert refused.status_code == 403
    assert refused.json()["detail"]["error_type"] == "connect_first"
    assert server.get("/app/experiment", headers=connect).status_code == 403
    # The connect token is no operator either.
    assert server.get("/app/agent", headers=connect).status_code == 401
    # Only the connect token asks.
    assert server.post("/connect", headers=server.agent).status_code == 401


def test_the_operator_allows_the_asking_agent_and_it_acts_at_the_level(server):
    found = _found(server)
    answers = []
    thread = threading.Thread(
        target=lambda: answers.append(
            server.post(
                "/connect",
                json={"origin": "this PC", "user": "lab", "wait_s": 10},
                headers={
                    **_bearer(found["connect_token"]),
                    "User-Agent": "fibsem-mcp/0.6 claude-code/2.1.3",
                },
            ).json()
        )
    )
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        asking = server.get("/app/agent", headers=server.operator).json()["asking"]
        if asking:
            break
        time.sleep(0.02)
    assert asking["client"] == "claude-code 2.1.3"
    assert asking["name"] == "claude-code on this PC"
    assert "now Watch, so it can look but not act" in asking["if_allowed"]
    # The agent can't allow itself.
    agent_tried = server.post(
        f"/app/agent/connections/{asking['id']}/allow", headers=server.agent
    )
    assert agent_tried.status_code == 403
    allowed = server.post(
        f"/app/agent/connections/{asking['id']}/allow", headers=server.operator
    )
    assert allowed.json() == {
        "id": asking["id"],
        "allowed": True,
        "name": "claude-code on this PC",
    }
    thread.join(10)
    token = answers[0]["token"]
    caps = server.get("/capabilities", headers=_bearer(token))
    assert caps.json()["level"] == "watch"
    # Listed with Unpair, as a computer paired with a code is.
    listed = server.get("/pairing", headers=server.operator).json()["computers"]
    assert [(c["name"], c["role"]) for c in listed] == [
        ("claude-code on this PC", "agent")
    ]
    events = server.get("/app/events", headers=server.operator).json()["events"]
    said = [e["payload"]["what"] for e in events if e["kind"] == ACTION]
    assert "Allow claude-code on this PC to connect" in said


def test_the_operator_denies_with_a_reason(server):
    found = _found(server)
    connect = _bearer(found["connect_token"])
    waiting = server.post("/connect", json={"wait_s": 0}, headers=connect).json()
    assert waiting["state"] == "waiting"
    denied = server.post(
        f"/app/agent/connections/{waiting['id']}/deny",
        json={"reason": "who is this?"},
        headers=server.operator,
    )
    assert denied.json() == {"id": waiting["id"], "allowed": False, "name": None}
    again = server.post("/connect", json={"wait_s": 0}, headers=connect).json()
    assert again["state"] == "denied" and "who is this?" in again["message"]
    missing = server.post("/app/agent/connections/nope/deny", headers=server.operator)
    assert missing.status_code == 404


# --- the client ---------------------------------------------------------------------


def _allow_when_asked(server):
    """The operator, clicking Allow once the card shows an ask."""

    def allow():
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            asking = server.get("/app/agent", headers=server.operator).json()
            if asking.get("asking"):
                ask_id = asking["asking"]["id"]
                server.post(
                    f"/app/agent/connections/{ask_id}/allow", headers=server.operator
                )
                return
            time.sleep(0.02)

    thread = threading.Thread(target=allow)
    thread.start()
    return thread


@pytest.fixture
def discovered(server, monkeypatch):
    """Clients find the server by its discovery file, as on the microscope PC."""
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    path = server.session.app_server._discovery_path
    monkeypatch.setattr("fibsem.server.discovery.DISCOVERY_FILE", path)
    return server


def test_a_client_asks_keeps_the_token_and_later_connects_without_asking(discovered):
    server = discovered
    operator = _allow_when_asked(server)
    with Transport(user_agent="fibsem-mcp/0.6 claude-code/2.1.3") as transport:
        assert transport.fetch("GET", "/capabilities")["level"] == "watch"
    operator.join(20)
    kept = client_pairing.agent_token(server.session.app_server.url)
    assert kept is not None
    # The next program connects with the kept token: no card.
    with Transport() as transport:
        assert transport.bearer == f"Bearer {kept}"
        assert transport.fetch("GET", "/capabilities")["level"] == "watch"
    assert server.get("/app/agent", headers=server.operator).json()["asking"] is None


def test_a_client_waiting_too_long_is_told_to_try_again(discovered):
    with Transport(connect_wait_s=0) as transport:
        answer = transport.request("GET", "/capabilities")
    assert answer.status_code == 403
    assert answer.json()["detail"]["error_type"] == "connect_waiting"


def test_an_unpaired_agent_forgets_its_token_and_asks_again(discovered):
    server = discovered
    operator = _allow_when_asked(server)
    transport = Transport()
    transport.fetch("GET", "/capabilities")
    operator.join(20)
    computers = server.get("/pairing", headers=server.operator).json()["computers"]
    server.delete(f"/pairing/computers/{computers[0]['id']}", headers=server.operator)
    operator = _allow_when_asked(server)
    assert transport.fetch("GET", "/capabilities")["level"] == "watch"
    operator.join(20)
    transport.close()
    computers = server.get("/pairing", headers=server.operator).json()["computers"]
    assert len(computers) == 1


# --- always allow agents on this PC -----------------------------------------------


def test_always_allow_pairs_an_ask_from_this_pc_at_once(tmp_path, clock):
    saved, allowed = [], []
    connections = Connections(
        PairedTokens(tmp_path / "paired.json"),
        clock=clock,
        save_always_allow=saved.append,
    )
    connections.on_allowed_always(allowed.append)
    connections.set_always_allow(True)
    assert saved == [True]
    answer = connections.ask("claude-code 2.1.3", "this PC", "lab", wait_s=0)
    assert answer["state"] == "allowed"
    assert connections.tokens.match(answer["token"]) == allowed[0]
    assert connections.asking() == []


def test_always_allow_still_asks_over_ssh(connections):
    connections.set_always_allow(True)
    answer = connections.ask("x", "10.0.0.5 over SSH", "lab", wait_s=0)
    assert answer["state"] == "waiting"
    assert connections.tokens.computers() == []


def test_the_operator_sets_always_allow_and_it_goes_on_the_record(server, monkeypatch):
    saved = []
    monkeypatch.setattr(
        server.session.app_server.auth.connections, "_save_always_allow", saved.append
    )
    agent_refused = server.put(
        "/pairing/always_allow_local", json={"on": True}, headers=server.agent
    )
    assert agent_refused.status_code in (401, 403)
    listing = server.put(
        "/pairing/always_allow_local", json={"on": True}, headers=server.operator
    ).json()
    assert listing["always_allow_local"] is True and saved == [True]
    found = _found(server)
    answer = server.post(
        "/connect",
        json={"origin": "this PC", "user": "lab", "wait_s": 0},
        headers=_bearer(found["connect_token"]),
    ).json()
    assert answer["state"] == "allowed"
    events = server.get("/app/events", headers=server.operator).json()["events"]
    said = [e["payload"] for e in events if e["kind"] == ACTION]
    assert any(
        p["what"].endswith("to connect") and "Always allow" in p.get("reason", "")
        for p in said
    )
