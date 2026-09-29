"""A device server with a token, over localhost: nothing without it, on HTTP or the
event stream, and a client with the wrong one is told so once rather than retrying."""

import socket
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

import requests  # noqa: E402

from fibsem.devices.drivers.remote import (  # noqa: E402
    DEVICE_TOKEN_FILE_ENV,
    DeviceClient,
    RemoteDeviceRefused,
    connect_remote_beams,
    connect_remote_fm,
    read_device_token,
)
from fibsem.server.devices import (  # noqa: E402
    DeviceServer,
    demo_devices,
    demo_fm_devices,
    main,
)
from fibsem.structures import BeamType  # noqa: E402

TOKEN = "k3Jf9-shared-secret"
ROUTES = [
    ("GET", "health"),
    ("GET", "devices"),
    ("GET", "devices/electron"),
    ("GET", "devices/electron/hfw"),
    ("GET", "devices/electron/hfw/metadata"),
    ("PUT", "devices/electron/hfw"),
    ("POST", "devices/electron/commands/blank"),
]


@pytest.fixture
def server():
    local = {d.beam_type: d for d in demo_devices()}
    server = DeviceServer(local.values(), token=TOKEN).start()
    yield local, server
    server.stop()


def _url(server: DeviceServer, path: str) -> str:
    return f"http://127.0.0.1:{server.port}/{path}"


@pytest.mark.parametrize("method, path", ROUTES)
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": TOKEN},  # not a bearer header
        {"Authorization": "Bearer ümlaut".encode("utf-8")},  # not ASCII: 401, not 500
    ],
    ids=["none", "wrong", "no-bearer", "non-ascii"],
)
def test_every_route_refuses_without_the_token(server, method, path, headers):
    local, served = server
    before = local[BeamType.ELECTRON].hfw.get_value()
    response = requests.request(
        method, _url(served, path), json={"value": 1e-4}, headers=headers
    )
    assert response.status_code == 401
    assert local[BeamType.ELECTRON].hfw.get_value() == before  # nothing written


def test_the_api_description_is_not_served(server):
    _, served = server
    for path in ("docs", "redoc", "openapi.json"):
        assert requests.get(_url(served, path)).status_code == 404


def test_the_event_stream_refuses_the_handshake_without_the_token(server):
    from websockets.exceptions import InvalidStatus
    from websockets.sync.client import connect

    _, served = server
    url = f"ws://127.0.0.1:{served.port}/events"
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        with pytest.raises(InvalidStatus):
            connect(url, open_timeout=2, additional_headers=headers)
    with connect(
        url, open_timeout=2, additional_headers={"Authorization": f"Bearer {TOKEN}"}
    ):
        pass


def test_the_right_token_reads_writes_and_hears_events(server):
    local, served = server
    client = DeviceClient("127.0.0.1", served.port, heartbeat=0.5, token=TOKEN)
    beams = connect_remote_beams("127.0.0.1", served.port, client=client)
    try:
        sem = beams[BeamType.ELECTRON]
        seen = []
        sem.hfw.changed.connect(seen.append)
        sem.hfw.set_value(222e-6)
        assert local[BeamType.ELECTRON].hfw.get_value() == 222e-6
        local[BeamType.ELECTRON].hfw.set_value(333e-6)  # a change on the far side
        end = time.monotonic() + 2
        while 333e-6 not in seen and time.monotonic() < end:
            time.sleep(0.01)
        assert 333e-6 in seen
    finally:
        client.close()


def test_a_wrong_token_is_refused_by_name(server):
    _, served = server
    client = DeviceClient("127.0.0.1", served.port, token="wrong")
    try:
        with pytest.raises(RemoteDeviceRefused, match="token"):
            connect_remote_beams("127.0.0.1", served.port, client=client)
    finally:
        client.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_an_offline_client_with_the_wrong_token_stops_retrying(caplog):
    """The server comes up after the coordinator, with another token: the event
    stream is refused, and the client says so once and stops."""
    port = _free_port()
    client = DeviceClient("127.0.0.1", port, heartbeat=0.2, token="wrong")
    connect_remote_fm("127.0.0.1", port, client=client, offline=True)
    served = DeviceServer(demo_fm_devices(), port=port, token=TOKEN).start()
    try:
        end = time.monotonic() + 5
        while client._events.is_alive() and time.monotonic() < end:
            time.sleep(0.05)
        assert not client._events.is_alive()  # the retry loop ended
        assert not client.connected
        assert "refused this computer's token" in caplog.text
    finally:
        client.close()
        served.stop()


def test_the_token_file_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv(DEVICE_TOKEN_FILE_ENV, raising=False)
    assert read_device_token() is None

    path = tmp_path / "device-token"
    path.write_text(TOKEN + "\n")
    monkeypatch.setenv(DEVICE_TOKEN_FILE_ENV, str(path))
    assert read_device_token() == TOKEN

    path.write_text("")
    with pytest.raises(ValueError):
        read_device_token()
    monkeypatch.setenv(DEVICE_TOKEN_FILE_ENV, str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        read_device_token()


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.0.20", "fm-pc"])
def test_the_server_refuses_to_serve_the_network_without_a_token(host, monkeypatch):
    monkeypatch.delenv(DEVICE_TOKEN_FILE_ENV, raising=False)
    with pytest.raises(SystemExit):
        main(["--host", host])
