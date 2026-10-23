"""Pairing a computer with a 4-digit code: the code's rules, the routes, and
`RemoteMicroscope.pair` over a real server.

Skipped when the [server] extra is not installed.
"""

import json
import os

import pytest

from tests.fixtures.app_server import serving

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.client import pairing as client_pairing  # noqa: E402
from fibsem.client.remote_microscope import RemoteMicroscope  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.pairing import (  # noqa: E402
    AGENT,
    MAX_TRIES,
    PairedTokens,
    Pairing,
    PairingRefused,
)

TOKEN = "agent-token"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def pairing(tmp_path, clock):
    return Pairing(PairedTokens(tmp_path / "paired.json"), clock=clock)


def _wrong(code):
    return f"{(int(code) + 1) % 10000:04d}"


# --- the code -----------------------------------------------------------------


def test_pairing_the_same_browser_again_replaces_its_row(pairing, tmp_path):
    """A browser that pairs again (its token lost or refused) takes its old
    row's place, rather than leaving rows of the same name (FIB-1517)."""
    first, old_token = pairing.pair(pairing.open(), "Chrome", browser="b1")
    pairing.pair(pairing.open(), "Firefox", browser="b2")
    again, token = pairing.pair(pairing.open(), "Chrome", browser="b1")
    assert [c.name for c in pairing.tokens.computers()] == ["Firefox", "Chrome"]
    assert pairing.tokens.match(old_token) is None
    assert pairing.tokens.match(token) == again
    assert "browser" not in again.to_dict()  # never shown
    # Kept across a restart; one with no id is never replaced.
    reloaded = PairedTokens(tmp_path / "paired.json")
    assert [c.browser for c in reloaded.computers()] == ["b2", "b1"]
    pairing.pair(pairing.open(), "script")
    pairing.pair(pairing.open(), "script")
    assert len(pairing.tokens.computers()) == 4


def test_the_right_code_pairs_once_and_closes_pairing(pairing):
    code = pairing.open()
    assert len(code) == 4 and code.isdigit()
    computer, token = pairing.pair(code, "lab laptop")
    assert computer.name == "lab laptop"
    assert pairing.tokens.match(token) == computer
    assert not pairing.state()["open"]
    with pytest.raises(PairingRefused) as refused:
        pairing.pair(code, "another")
    assert refused.value.error_type == "pairing_closed"


def test_wrong_codes_close_pairing_after_five(pairing):
    code = pairing.open()
    for _ in range(MAX_TRIES - 1):
        with pytest.raises(PairingRefused) as refused:
            pairing.pair(_wrong(code), "x")
        assert refused.value.error_type == "wrong_code"
    with pytest.raises(PairingRefused) as refused:
        pairing.pair(_wrong(code), "x")
    assert refused.value.error_type == "pairing_closed"
    # The right code no longer works either.
    with pytest.raises(PairingRefused):
        pairing.pair(code, "x")
    assert pairing.tokens.computers() == []


def test_a_code_expires_after_five_minutes(pairing, clock):
    code = pairing.open()
    clock.now += 299
    assert pairing.state()["open"]
    clock.now += 2
    assert pairing.state() == {
        "open": False,
        "role": None,
        "grants": [],
        "code": None,
        "expires_in_s": 0.0,
    }
    with pytest.raises(PairingRefused):
        pairing.pair(code, "x")


def test_tokens_are_kept_hashed_and_unpairing_revokes(tmp_path, pairing):
    computer, token = pairing.pair(pairing.open(), "laptop")
    saved = (tmp_path / "paired.json").read_text()
    assert token not in saved and computer.name in saved
    assert oct(os.stat(tmp_path / "paired.json").st_mode & 0o777) == "0o600"
    # Another start reads the same file.
    again = PairedTokens(tmp_path / "paired.json")
    assert again.match(token).id == computer.id
    assert again.remove(computer.id)
    assert again.match(token) is None


def test_a_browser_asks_for_a_code_only_where_it_is_announced(pairing, clock):
    assert not pairing.request()  # nowhere to show it
    said = []
    pairing.announce = said.append
    assert pairing.request()
    assert said == [pairing.state()["code"]]
    # An open code is never replaced by another ask.
    assert not pairing.request() and len(said) == 1
    for _ in range(MAX_TRIES):
        with pytest.raises(PairingRefused):
            pairing.pair(_wrong(said[0]), "x")
    # Guessing stays slow: asking again waits an hour.
    assert not pairing.request()
    clock.now += 3601
    assert pairing.request()


# --- the routes ---------------------------------------------------------------


@pytest.fixture
def auth(tmp_path):
    auth = AuthConfig.generate(level="watch", token=TOKEN)
    auth.operator_token = "install-operator"
    auth.pairing = Pairing(PairedTokens(tmp_path / "paired.json"))
    return auth


@pytest.fixture
def http(microscope, auth):
    return TestClient(build_server(microscope, auth=auth))


def _as(token):
    return {"Authorization": f"Bearer {token}"}


def test_a_paired_token_is_the_operators_until_unpaired(http, auth):
    opened = http.post("/pairing", headers=_as("install-operator")).json()
    assert opened["open"] and len(opened["code"]) == 4
    paired = http.post("/pair", json={"code": opened["code"], "name": "laptop"})
    assert paired.status_code == 200, paired.text
    token = paired.json()["token"]
    # The operator's: its routes, and the microscope at Off.
    auth.set_level(AgentLevel.OFF)
    assert http.get("/microscope/state", headers=_as(token)).status_code == 200
    listing = http.get("/pairing", headers=_as(token)).json()
    assert [c["name"] for c in listing["computers"]] == ["laptop"]
    assert not listing["open"]
    unpaired = http.delete(
        f"/pairing/computers/{paired.json()['id']}", headers=_as("install-operator")
    )
    assert unpaired.json()["computers"] == []
    assert http.get("/microscope/state", headers=_as(token)).status_code == 401


def test_the_agent_never_opens_pairing(http):
    refused = http.post("/pairing", headers=_as(TOKEN))
    assert refused.status_code == 403
    assert refused.json()["detail"]["error_type"] == "operator_only"


def test_a_wrong_code_and_another_sites_page_are_refused(http, auth):
    code = auth.pairing.open()
    wrong = http.post("/pair", json={"code": _wrong(code), "name": "x"})
    assert wrong.status_code == 403
    assert wrong.json()["detail"]["error_type"] == "wrong_code"
    foreign = http.post(
        "/pair",
        json={"code": code, "name": "x"},
        headers={"Origin": "http://evil.example"},
    )
    assert foreign.status_code == 403
    assert foreign.json()["detail"]["error_type"] == "browser_refused"
    # The chat page, served from the server itself, pairs.
    page = http.post(
        "/pair",
        json={"code": code, "name": "Firefox on Linux"},
        headers={"Origin": "http://testserver"},
    )
    assert page.status_code == 200


def test_a_browser_asks_the_headless_server_for_a_code(http, auth):
    assert http.get("/pair").json() == {"open": False, "request": False}
    assert http.post("/pair/request").status_code == 404
    said = []
    auth.pairing.announce = said.append
    assert http.get("/pair").json()["request"]
    assert http.post("/pair/request").json() == {"opened": True}
    paired = http.post("/pair", json={"code": said[0], "name": "browser"})
    assert paired.status_code == 200


# --- RemoteMicroscope.pair -----------------------------------------------------


@pytest.fixture
def url(microscope, auth):
    with serving(build_server(microscope, auth=auth)) as url:
        yield url


def test_a_remote_microscope_pairs_and_later_connects_with_no_code(
    url, auth, tmp_path, monkeypatch
):
    monkeypatch.setattr(client_pairing, "REMOTE_DIR", tmp_path / "remote")
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    remote = RemoteMicroscope.pair(auth.pairing.open(), url, name="laptop")
    try:
        assert remote.get_stage_position() is not None
    finally:
        remote.disconnect()
    (saved,) = (tmp_path / "remote").iterdir()
    assert json.loads(saved.read_text())["url"] == url
    assert oct(saved.stat().st_mode & 0o777) == "0o600"
    again = RemoteMicroscope.connect(url)
    try:
        assert auth.paired(again._token).name == "laptop"
    finally:
        again.disconnect()


def test_a_refused_code_says_why(url, auth, tmp_path, monkeypatch):
    monkeypatch.setattr(client_pairing, "REMOTE_DIR", tmp_path / "remote")
    with pytest.raises(PermissionError, match="isn't open"):
        client_pairing.pair("0000", url)
    assert not (tmp_path / "remote").exists()


def test_a_paired_computer_gives_its_control_lease_up_at_off(http, auth):
    code = auth.pairing.open()
    token = http.post("/pair", json={"code": code, "name": "laptop"}).json()["token"]
    lease = http.post("/microscope/control", json={}, headers=_as(token)).json()[
        "lease"
    ]
    auth.set_level(AgentLevel.OFF)
    released = http.delete(f"/microscope/control/{lease}", headers=_as(token))
    assert released.json() == {"released": True}


# --- agents -------------------------------------------------------------------


def test_a_paired_agent_is_gated_by_the_level_until_unpaired(http, auth):
    opened = http.post(
        "/pairing", json={"role": "agent"}, headers=_as("install-operator")
    ).json()
    assert opened["role"] == AGENT
    paired = http.post("/pair", json={"code": opened["code"], "name": "mac mini"})
    assert paired.json()["role"] == AGENT
    token = paired.json()["token"]
    assert http.get("/microscope/state", headers=_as(token)).status_code == 200
    assert auth.agent_computer == "mac mini"
    refused = http.get("/pairing", headers=_as(token))
    assert refused.json()["detail"]["error_type"] == "operator_only"
    auth.set_level(AgentLevel.OFF)
    off = http.get("/microscope/state", headers=_as(token))
    assert off.json()["detail"]["error_type"] == "agent_off"
    auth.set_level(AgentLevel.WATCH)
    http.delete(
        f"/pairing/computers/{paired.json()['id']}", headers=_as("install-operator")
    )
    assert http.get("/microscope/state", headers=_as(token)).status_code == 401
    # The App Server's own agent token names no computer.
    http.get("/microscope/state", headers=_as(TOKEN))
    assert auth.agent_computer is None


def test_pairing_is_for_the_operator_or_an_agent(http):
    refused = http.post(
        "/pairing", json={"role": "admin"}, headers=_as("install-operator")
    )
    assert refused.status_code == 422


def test_fibsem_mcp_pair_keeps_the_agent_token_apart(url, auth, tmp_path, monkeypatch):
    from fibsem.ai.agent.mcp.sidecar import _pair
    from fibsem.client.transport import resolve

    monkeypatch.setattr(client_pairing, "REMOTE_DIR", tmp_path / "remote")
    monkeypatch.setattr(client_pairing, "AGENT_DIR", tmp_path / "agent")
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    assert _pair([auth.pairing.open(AGENT), "--url", url, "--name", "mac mini"]) == 0
    _, token = resolve(url)
    assert auth.paired(token).role == AGENT
    # The operator's app finds no token of its own there.
    assert client_pairing.paired_token(url) is None
    assert _pair(["0000", "--url", url]) == 1
