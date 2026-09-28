"""The device server's token, scopes and pairing, over localhost: nothing without the
token, nothing written without the hardware scope, and a pairing code that works once."""

import os
import socket
import stat
import sys
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

import requests  # noqa: E402

from fibsem.devices.drivers.remote import (  # noqa: E402
    DeviceClient,
    RemoteDeviceRefused,
    connect_remote_beams,
)
from fibsem.server.auth import AuthConfig  # noqa: E402
from fibsem.server.devices import DeviceServer, demo_devices  # noqa: E402
from fibsem.server.pairing import (  # noqa: E402
    Pairing,
    check_connection,
    load_or_create_token,
    pair,
    read_token,
)
from fibsem.structures import BeamType  # noqa: E402


@pytest.fixture
def server():
    server = DeviceServer(demo_devices()).start()  # read-only, as by default
    yield server
    server.stop()


@pytest.fixture
def armed():
    server = DeviceServer(
        demo_devices(), auth=AuthConfig.generate(arm_hardware=True)
    ).start()
    yield server
    server.stop()


def _url(server: DeviceServer, path: str) -> str:
    return f"http://127.0.0.1:{server.port}/{path}"


# ── the token ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "health"),
        ("GET", "access"),
        ("GET", "devices"),
        ("GET", "devices/electron"),
        ("GET", "devices/electron/hfw"),
        ("GET", "devices/electron/hfw/metadata"),
        ("PUT", "devices/electron/hfw"),
        ("POST", "devices/electron/commands/acquire"),
    ],
)
def test_every_route_needs_the_token(armed, method, path):
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        response = requests.request(
            method, _url(armed, path), json={"value": 1e-4}, headers=headers
        )
        assert response.status_code == 401, (method, path)


def test_a_client_without_the_token_is_refused_by_name(armed):
    client = DeviceClient("127.0.0.1", armed.port, token="wrong")
    try:
        with pytest.raises(RemoteDeviceRefused, match="pair with it again"):
            connect_remote_beams("127.0.0.1", armed.port, client=client)
    finally:
        client.close()


def test_the_event_stream_needs_the_token(armed):
    from websockets.exceptions import InvalidStatus
    from websockets.sync.client import connect

    url = f"ws://127.0.0.1:{armed.port}/events"
    with pytest.raises(InvalidStatus):
        connect(url, open_timeout=2)
    with connect(
        url,
        open_timeout=2,
        additional_headers={"Authorization": f"Bearer {armed.auth.token}"},
    ):
        pass


# ── scopes ───────────────────────────────────────────────────────────


def test_a_server_is_read_only_unless_armed(server):
    client = DeviceClient("127.0.0.1", server.port, token=server.auth.token)
    beams = connect_remote_beams("127.0.0.1", server.port, client=client)
    try:
        sem = beams[BeamType.ELECTRON]
        before = sem.hfw.get_value()  # reads work
        with pytest.raises(RemoteDeviceRefused, match="hardware"):
            sem.hfw.set_value(before * 2)
        assert sem.hfw.get_value() == before  # and nothing was written
    finally:
        client.close()


def test_access_lists_the_armed_scopes(server, armed):
    def scopes(s: DeviceServer):
        headers = {"Authorization": f"Bearer {s.auth.token}"}
        return requests.get(_url(s, "access"), headers=headers).json()["scopes"]

    assert scopes(server) == ["read"]
    assert "hardware" in scopes(armed)


# ── the token file ─────────────────────────────────────────────────────


def test_the_token_file_is_made_once_and_kept(tmp_path):
    path = tmp_path / "sub" / "token"
    assert read_token(path) is None
    token = load_or_create_token(path)
    assert load_or_create_token(path) == token == read_token(path)
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


# ── pairing ───────────────────────────────────────────────────────────


def test_a_pairing_code_works_once(armed, tmp_path):
    token_file = tmp_path / "token"
    code = armed.app.state.pairing.open()

    pair("127.0.0.1", armed.port, code, token_file)

    assert read_token(token_file) == armed.auth.token
    with pytest.raises(PermissionError):
        pair("127.0.0.1", armed.port, code, token_file)


def test_pairing_needs_an_open_window(armed, tmp_path):
    with pytest.raises(PermissionError):
        pair("127.0.0.1", armed.port, "000000", tmp_path / "token")
    assert not (tmp_path / "token").exists()


def test_wrong_guesses_close_the_window():
    pairing = Pairing(attempts=3)
    code = pairing.open()
    wrong = f"{(int(code) + 1) % 10**6:06d}"
    assert not pairing.redeem(wrong)
    assert not pairing.redeem(wrong)
    assert pairing.is_open
    assert not pairing.redeem(wrong)
    assert not pairing.is_open
    assert not pairing.redeem(code)  # even the right code, now


def test_a_code_expires():
    pairing = Pairing(seconds=0.05)
    code = pairing.open()
    time.sleep(0.1)
    assert not pairing.redeem(code)


# ── the connection check ───────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_check_stops_at_an_unreachable_server():
    checks = check_connection("127.0.0.1", _free_port(), token="anything", timeout=1)
    assert [(c.name, c.ok) for c in checks] == [("reachable", False)]


@pytest.mark.parametrize("token", [None, "wrong"])
def test_check_stops_at_a_missing_or_wrong_token(armed, token):
    checks = check_connection("127.0.0.1", armed.port, token=token)
    assert [(c.name, c.ok) for c in checks] == [("reachable", True), ("token", False)]


def test_check_walks_a_good_connection(armed):
    checks = check_connection("127.0.0.1", armed.port, token=armed.auth.token)

    assert [(c.name, c.ok) for c in checks] == [
        ("reachable", True),
        ("token", True),
        ("access", True),
        ("devices", True),
    ]
    assert "hardware" in checks[2].detail
    assert "electron" in checks[3].detail and "ms" in checks[3].detail


def test_check_says_when_a_server_is_read_only(server):
    checks = check_connection("127.0.0.1", server.port, token=server.auth.token)
    assert "read only" in checks[2].detail
