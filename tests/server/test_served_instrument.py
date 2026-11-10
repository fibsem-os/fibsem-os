"""The App Server serves instruments, not only a microscope.

A standalone cryo-FM's devices, built from a Device Server file with no
microscope, are served under ``/microscope/devices`` like a microscope's: listed,
read, written and commanded through the same gates, with the same blocklist. The
state routes still need a microscope.
"""

import pytest
from fastapi.testclient import TestClient

from fibsem.instrument import Instrument
from fibsem.microscope import FibsemMicroscope
from fibsem.server import AuthConfig, build_server
from fibsem.server.devices import device_instrument, served_fm_settings

TOKEN = "t"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
FM_PARTS = ["camera", "filter_set", "fm", "light_source", "objective"]


@pytest.fixture
def cryo_fm():
    instrument = device_instrument(served_fm_settings("fm"))
    yield instrument
    instrument.disconnect()


def _client(*args, **kwargs) -> TestClient:
    app = build_server(
        *args, auth=AuthConfig.generate(level="drive", token=TOKEN), **kwargs
    )
    client = TestClient(app)
    client.headers.update(AUTH)
    return client


def test_a_microscope_is_an_instrument_of_kind_fibsem(microscope):
    assert isinstance(microscope, Instrument)
    assert microscope.kind == FibsemMicroscope.kind == "fibsem"
    assert microscope.get_state().stage_position == (
        microscope.get_microscope_state().stage_position
    )


def test_devices_with_no_microscope_are_served(cryo_fm):
    client = _client(instruments=[cryo_fm])
    capabilities = client.get("/capabilities").json()
    assert capabilities["connected"] is False
    assert capabilities["instruments"] == ["devices"]

    assert sorted(client.get("/microscope/devices").json()) == FM_PARTS
    assert client.get("/microscope/services").json() == {}
    read = client.get("/microscope/devices/light_source/power")
    assert read.status_code == 200, read.text
    written = client.put("/microscope/devices/light_source/power", json={"value": 0.1})
    assert written.status_code == 200, written.text
    assert cryo_fm.devices["light_source"].power.get_value() == pytest.approx(0.1)
    # A command takes the instrument's control, and gives it back.
    retracted = client.post("/microscope/devices/objective/commands/retract", json={})
    assert retracted.status_code == 200, retracted.text
    assert cryo_fm.control.current is None
    assert client.get("/microscope/devices/electron/on").status_code == 404


def test_the_state_routes_still_need_a_microscope(cryo_fm):
    client = _client(instruments=[cryo_fm])
    for response in (
        client.get("/microscope"),
        client.get("/microscope/state"),
    ):
        assert response.status_code == 409
        assert response.json()["detail"]["error_type"] == "not_connected"


def test_the_blocklist_covers_devices_with_no_microscope(cryo_fm):
    client = _client(instruments=[cryo_fm])
    refused = client.put(
        "/microscope/devices/objective/limit_position", json={"value": 0.0}
    )
    assert refused.status_code == 403
    listing = client.get("/microscope/devices").json()
    assert listing["objective"]["parameters"]["limit_position"]["agent"] is False


def test_nothing_served_is_not_connected():
    client = _client(session=_NoMicroscope())
    response = client.get("/microscope/devices")
    assert response.status_code == 409
    assert response.json()["detail"]["error_type"] == "not_connected"
    assert client.get("/capabilities").json()["instruments"] == []


def test_a_microscope_and_other_devices_are_served_together(microscope, cryo_fm):
    client = _client(microscope, instruments=[cryo_fm])
    assert client.get("/capabilities").json()["instruments"] == ["fibsem", "devices"]
    devices = client.get("/microscope/devices").json()
    assert {"electron", "ion", "stage"} <= set(devices)
    assert "light_source" in devices
    # The microscope's control is the one a command takes.
    taken = client.post("/microscope/control", json={"action": "a check"})
    assert taken.status_code == 200, taken.text
    assert microscope.control.current is not None
    client.delete(f"/microscope/control/{taken.json()['lease']}")


class _NoMicroscope:
    """A session with nothing connected, as the app starts."""

    microscope = None
    connection = {"connected": False, "configurations": []}

    def record_event(self, kind, payload):
        pass
