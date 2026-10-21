"""A hardware fault crosses the device server as a fault, not as a crash.

A driver raises `DeviceFault` (or its own subclass of it) when the instrument refuses:
the server answers 502 with the usual `error_type`/`message` body and the device's name,
and a remote copy raises `DeviceFault` again. Any other exception is a 500 in the same
shape, with its traceback in the server's log rather than the answer.
"""

import logging

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.devices import (  # noqa: E402
    Device,
    DeviceFault,
    DeviceParameter,
    ResourceBusy,
    command,
)
from fibsem.server.devices import build_device_app  # noqa: E402


class ControllerError(DeviceFault):
    """A driver's own SDK error, as a plugin would declare it."""


class Unit(Device):
    gain = DeviceParameter(float)

    def read_gain(self):
        raise ControllerError("controller offline", device=self.name)

    @command
    def scan(self) -> None:
        raise ControllerError("dwell under the unit's minimum", device=self.name)

    @command
    def busy(self) -> None:
        raise ResourceBusy("scanner", "electron", "unit is live on electron")

    @command
    def crash(self) -> None:
        raise KeyError("a bug in the driver")


@pytest.fixture
def client():
    app = build_device_app([Unit("unit").connect()])
    with TestClient(app) as client:
        client.headers["Authorization"] = f"Bearer {app.state.auth.token}"
        yield client


def test_a_drivers_fault_is_a_502_naming_the_device(client):
    r = client.post("/devices/unit/commands/scan", json={"kwargs": {}})

    assert r.status_code == 502
    assert r.json()["detail"] == {
        "error_type": "ControllerError",
        "message": "dwell under the unit's minimum",
        "device": "unit",
    }


def test_a_fault_reading_a_parameter_is_a_502(client):
    assert client.get("/devices/unit/gain").status_code == 502
    assert client.get("/devices/unit/values").status_code == 502


def test_any_other_error_is_a_500_in_the_same_shape(client, caplog):
    with caplog.at_level(logging.ERROR):
        r = client.post("/devices/unit/commands/crash", json={"kwargs": {}})

    assert r.status_code == 500
    assert r.json()["detail"] == {
        "error_type": "KeyError",
        "message": "'a bug in the driver'",
    }
    assert "Traceback" not in r.text
    assert "KeyError" in caplog.text  # the traceback is the server's, in its log


def test_a_remote_copy_raises_the_fault_again():
    pytest.importorskip("websockets")
    from fibsem.drivers.remote.devices import DeviceClient
    from fibsem.server.devices import DeviceServer

    server = DeviceServer([Unit("unit").connect()]).start()
    client = DeviceClient("127.0.0.1", server.port, heartbeat=0.5)
    try:
        with pytest.raises(DeviceFault, match="dwell under") as raised:
            client.request("POST", "devices/unit/commands/scan", 5.0, json={})
        assert raised.value.device == "unit"
    finally:
        client.close()
        server.stop()


def test_a_resource_held_elsewhere_is_a_409_and_stays_busy_remotely(client):
    from fibsem.devices.errors import error_from

    r = client.post("/devices/unit/commands/busy", json={"kwargs": {}})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["holder"] == "electron"
    error = error_from(detail, r.status_code)
    assert isinstance(error, ResourceBusy)
    assert (error.resource, error.holder) == ("scanner", "electron")
    assert str(error) == "unit is live on electron"
