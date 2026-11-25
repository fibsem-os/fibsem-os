"""The Device Server says which version it runs, and the remote driver refuses
one whose routes are another minor version's."""

import logging

import pytest

import fibsem
from fibsem.devices.wire import DEVICE_API_VERSION, version_problem
from fibsem.drivers.remote.devices import (
    DeviceClient,
    RemoteDeviceVersionMismatch,
    connect_remote_beams,
)
from fibsem.server import devices as server_devices
from fibsem.server.devices import DeviceServer, demo_devices


@pytest.fixture
def server():
    server = DeviceServer(demo_devices()).start()
    yield server
    server.stop()


def test_health_names_the_program_and_its_versions(server):
    client = DeviceClient("127.0.0.1", server.port, events=False)
    try:
        health = client.health()
    finally:
        client.close()
    assert health["program"] == "fibsem-device-server"
    assert health["version"] == fibsem.__version__
    assert health["api_version"] == DEVICE_API_VERSION


def test_a_server_of_another_minor_is_refused(server, monkeypatch):
    monkeypatch.setattr(server_devices, "DEVICE_API_VERSION", "9.9.0")
    client = DeviceClient("127.0.0.1", server.port, events=False)
    try:
        with pytest.raises(RemoteDeviceVersionMismatch, match="9.9.0"):
            connect_remote_beams("127.0.0.1", server.port, client=client)
    finally:
        client.close()


def test_another_patch_connects(server, monkeypatch):
    major, minor, _ = DEVICE_API_VERSION.split(".")
    monkeypatch.setattr(server_devices, "DEVICE_API_VERSION", f"{major}.{minor}.99")
    client = DeviceClient("127.0.0.1", server.port, events=False)
    try:
        beams = connect_remote_beams("127.0.0.1", server.port, client=client)
        assert set(beams) and all(beam.online for beam in beams.values())
    finally:
        client.close()


def test_versions_compared():
    ours = {"api_version": DEVICE_API_VERSION, "version": fibsem.__version__}
    assert version_problem(ours, "pc:8765") == (False, "")

    refused, why = version_problem({"api_version": "9.0.0"}, "pc:8765")
    assert refused and "pc:8765" in why

    refused, why = version_problem({**ours, "version": "0.0.1"}, "pc:8765")
    assert not refused and "0.0.1" in why

    # a server too old to report one warns, so a METEOR PC not yet updated works
    refused, why = version_problem({"ok": True, "devices": {}}, "pc:8765")
    assert not refused and "older" in why


def test_an_older_server_only_warns(server, monkeypatch, caplog):
    def old(self):
        return {"ok": True, "devices": {}}

    monkeypatch.setattr(DeviceClient, "health", old)
    client = DeviceClient("127.0.0.1", server.port, events=False)
    try:
        with caplog.at_level(logging.WARNING):
            beams = connect_remote_beams("127.0.0.1", server.port, client=client)
        assert beams
        assert "doesn't report its version" in caplog.text
    finally:
        client.close()


def test_the_default_file_is_served(tmp_path, monkeypatch):
    path = tmp_path / "device-server.yaml"
    path.write_text(
        "info: {manufacturer: Demo}\n"
        "server: {port: 9011}\n"
        "hardware:\n  devices:\n    - {name: fm, type: fm}\n"
    )
    monkeypatch.setattr(server_devices, "DEFAULT_CONFIG", path)
    served = {}
    monkeypatch.setattr(
        server_devices,
        "_serve",
        lambda app, host, port, stop: served.update(port=port, stop=stop),
    )
    server_devices.main(["--token-file", str(tmp_path / "t.json")])
    assert served == {"port": 9011, "stop": None}


def test_the_gateway_stops_it():
    import socket
    import threading

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    stop = threading.Event()
    ended = []
    thread = threading.Thread(
        target=lambda: ended.append(
            server_devices.serve_for_gateway(["--port", str(port)], stop)
        )
    )
    thread.start()
    try:
        answered = None
        for _ in range(100):
            try:
                client = DeviceClient("127.0.0.1", port, events=False)
                try:
                    answered = client.health()["program"]
                finally:
                    client.close()
                break
            except Exception:
                threading.Event().wait(0.05)
        assert answered == "fibsem-device-server"
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not thread.is_alive() and ended == [0]
