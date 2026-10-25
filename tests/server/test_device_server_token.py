"""A Device Server needs a token on every route and stream (FIB-1264): this
install's, which a client on the same computer reads, or one paired by command.

Skipped when the [server] extra is not installed.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from fibsem import cli  # noqa: E402
from fibsem.client import pairing as client_pairing  # noqa: E402
from fibsem.drivers.remote.devices import (  # noqa: E402
    DeviceClient,
    RemoteDeviceNotPaired,
)
from fibsem.server.device_auth import DeviceAuth  # noqa: E402
from fibsem.server.devices import (  # noqa: E402
    DeviceServer,
    build_device_app,
    demo_devices,
)
from fibsem.server.pairing import PairedTokens, Pairing  # noqa: E402


@pytest.fixture(scope="module")
def beams():
    return demo_devices()


def test_no_token_reaches_nothing(beams, tmp_path):
    auth = DeviceAuth("install", Pairing(PairedTokens(tmp_path / "paired.json")))
    with TestClient(build_device_app(beams, auth)) as http:
        for path in ("/health", "/devices", "/devices/electron"):
            assert http.get(path).status_code == 401
        assert (
            http.get("/devices", headers={"Authorization": "Bearer x"}).status_code
            == 401
        )
        with pytest.raises(WebSocketDisconnect):
            with http.websocket_connect("/events") as socket:
                socket.receive_json()
        ok = http.get("/devices", headers={"Authorization": "Bearer install"})
        assert ok.status_code == 200


def test_a_client_on_the_same_computer_needs_no_steps(beams):
    server = DeviceServer(beams).start()
    client = DeviceClient("127.0.0.1", server.port)
    try:
        assert "electron" in client.describe()
    finally:
        client.close()
        server.stop()


def test_another_computer_pairs_by_command_once(beams, tmp_path, monkeypatch):
    auth = DeviceAuth("install", Pairing(PairedTokens(tmp_path / "paired.json")))
    server = DeviceServer(beams, auth=auth).start()
    url = f"http://127.0.0.1:{server.port}"
    # Another computer: no Device Server token of its own.
    monkeypatch.setattr(
        "fibsem.server.device_auth.DEVICE_TOKEN_FILE", tmp_path / "none.json"
    )
    try:
        with pytest.raises(RemoteDeviceNotPaired, match="fibsem-cli pair-device"):
            DeviceClient("127.0.0.1", server.port)
        with pytest.raises(SystemExit) as paired:
            cli.main(["pair-device", f"127.0.0.1:{server.port}", auth.pairing.open()])
        assert paired.value.code == 0
        # Every later connection uses the kept token.
        assert client_pairing.paired_token(url) is not None
        again = DeviceClient("127.0.0.1", server.port)
        assert "electron" in again.describe()
        again.close()
        (computer,) = auth.pairing.tokens.computers()
        auth.pairing.tokens.remove(computer.id)
        cut = DeviceClient("127.0.0.1", server.port)
        with pytest.raises(RemoteDeviceNotPaired, match=f"127.0.0.1:{server.port}"):
            cut.describe()
        cut.close()
    finally:
        server.stop()


def test_a_wrong_code_is_refused(beams, tmp_path, capsys):
    auth = DeviceAuth("install", Pairing(PairedTokens(tmp_path / "paired.json")))
    server = DeviceServer(beams, auth=auth).start()
    try:
        auth.pairing.open()
        with pytest.raises(SystemExit) as refused:
            cli.main(["pair-device", f"127.0.0.1:{server.port}", "0000"])
        assert refused.value.code == 1
        assert "Pairing refused" in capsys.readouterr().err
    finally:
        server.stop()


def test_the_pairing_routes_answer_on_a_device_server(beams, tmp_path):
    auth = DeviceAuth("install", Pairing(PairedTokens(tmp_path / "paired.json")))
    operator = {"Authorization": "Bearer install"}
    with TestClient(build_device_app(beams, auth)) as http:
        listing = http.get("/pairing", headers=operator)
        assert listing.status_code == 200
        assert listing.json()["always_allow_local"] is False
        # No agent asks to connect to a Device Server.
        asked = http.post("/connect", headers=operator)
        assert asked.status_code == 401
        switched = http.put(
            "/pairing/always_allow_local", json={"on": True}, headers=operator
        )
        assert switched.status_code == 404
        assert switched.json()["detail"]["error_type"] == "connect_off"
