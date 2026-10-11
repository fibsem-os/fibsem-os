"""The server's /devices routes (every device in microscope.devices) and its beam
settings routes, behind the server's token, scopes and command slot; and
FibsemClient over them.

Skipped when the [server] extra is not installed.
"""

import os

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.client import FibsemClient  # noqa: E402
from fibsem.structures import BeamType, FibsemImage, Point  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def microscope():
    previous = os.environ.get("FIBSEM_SIM_NO_DELAY")
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    yield microscope
    if previous is None:
        os.environ.pop("FIBSEM_SIM_NO_DELAY", None)
    else:
        os.environ["FIBSEM_SIM_NO_DELAY"] = previous


@pytest.fixture(scope="module")
def read_client(microscope):
    app = build_server(microscope, auth=AuthConfig(token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update(AUTH)
        yield client


@pytest.fixture(scope="module")
def armed_client(microscope):
    app = build_server(
        microscope, auth=AuthConfig.generate(arm_hardware=True, token=TOKEN)
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update(AUTH)
        yield client


class _Session:
    """requests' Session API over a TestClient, which refuses per-call timeouts."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, method):
        call = getattr(self._client, method)
        return lambda url, timeout=None, **kwargs: call(url, **kwargs)


@pytest.fixture
def fibsem_client(armed_client):
    """A FibsemClient over the in-process app."""
    client = FibsemClient.__new__(FibsemClient)
    client.base_url = ""
    client._session = _Session(armed_client)
    return client


def test_lists_every_device_the_microscope_built(read_client, microscope):
    resp = read_client.get("/devices")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == set(microscope.devices)
    assert "current" in body["ion"]["parameters"]
    assert "move_relative" in body["stage"]["commands"]


def test_device_routes_need_the_token(microscope):
    app = build_server(microscope, auth=AuthConfig(token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/devices").status_code == 401
        assert client.get("/devices/ion/current").status_code == 401


def test_read_is_read_scope(read_client, microscope):
    resp = read_client.get("/devices/ion/current")
    assert resp.status_code == 200
    assert resp.json()["value"] == microscope.get_beam_current(BeamType.ION)


def test_write_is_refused_until_hardware_is_armed(read_client):
    resp = read_client.put("/devices/electron/hfw", json={"value": 100e-6})
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "scope_not_armed"


def test_write_when_armed_reaches_the_device(armed_client, microscope):
    resp = armed_client.put("/devices/electron/hfw", json={"value": 120e-6})
    assert resp.status_code == 200
    assert microscope.get_field_of_view(BeamType.ELECTRON) == pytest.approx(120e-6)


def test_write_takes_the_command_slot(armed_client):
    lock = armed_client.app.state.command_lock
    assert lock.acquire(blocking=False)
    try:
        resp = armed_client.put("/devices/electron/hfw", json={"value": 100e-6})
        assert resp.status_code == 409
        assert resp.json()["detail"]["error_type"] == "busy"
    finally:
        lock.release()


def test_metadata_answers_choices(read_client, microscope):
    resp = read_client.get("/devices/ion/current/metadata")
    assert resp.status_code == 200
    choices = microscope.beams[BeamType.ION].parameters["current"].choices
    assert resp.json()["choices"] == list(choices)


def test_unknown_device_or_parameter_is_404(read_client):
    assert read_client.get("/devices/nope/current").status_code == 404
    assert read_client.get("/devices/ion/nope").status_code == 404


def test_command_is_hardware_scope(read_client):
    resp = read_client.post("/devices/ion/commands/blank", json={})
    assert resp.status_code == 403


def test_stop_commands_pass_the_command_slot_with_read_scope(read_client):
    # Like /stop_milling: a stop must reach a device whatever holds the slot.
    lock = read_client.app.state.command_lock
    assert lock.acquire(blocking=False)
    try:
        resp = read_client.post("/devices/electron/commands/stop_live", json={})
        assert resp.status_code == 200
    finally:
        lock.release()


def test_acquire_command_answers_a_tiff(fibsem_client):
    image = fibsem_client.call_command("electron", "acquire")
    assert isinstance(image, FibsemImage)
    assert image.data.ndim == 2


def test_beam_settings_get_and_put(armed_client):
    resp = armed_client.get("/beams/ion/beam_settings")
    assert resp.status_code == 200
    settings = resp.json()["beam_settings"]
    assert settings["beam_type"] == "ION"
    resp = armed_client.put(
        "/beams/ion/beam_settings", json={"beam_settings": settings}
    )
    assert resp.status_code == 200


def test_beam_settings_put_is_hardware_scope(read_client):
    settings = read_client.get("/beams/electron/beam_settings").json()
    resp = read_client.put("/beams/electron/beam_settings", json=settings)
    assert resp.status_code == 403


def test_unknown_beam_is_422(read_client):
    assert read_client.get("/beams/proton/beam_settings").status_code == 422


def test_client_wrappers_go_through_the_devices(fibsem_client, microscope):
    fibsem_client.set_field_of_view(150e-6, BeamType.ELECTRON)
    assert microscope.get_field_of_view(BeamType.ELECTRON) == pytest.approx(150e-6)
    assert fibsem_client.get_field_of_view(BeamType.ELECTRON) == pytest.approx(150e-6)
    assert fibsem_client.get_resolution(BeamType.ELECTRON) == tuple(
        microscope.get_resolution(BeamType.ELECTRON)
    )
    stigmation = fibsem_client.set_stigmation(Point(0.01, -0.01), BeamType.ION)
    assert isinstance(stigmation, Point)
    assert fibsem_client.get_stigmation(BeamType.ION).x == pytest.approx(0.01)
    assert fibsem_client.get_beam_current(BeamType.ION) == microscope.get_beam_current(
        BeamType.ION
    )


def test_client_settings_round_trip(fibsem_client):
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        settings = fibsem_client.get_beam_settings(beam_type)
        assert settings.beam_type is beam_type
        fibsem_client.set_beam_settings(settings)
        image_settings = fibsem_client.get_imaging_settings(beam_type)
        fibsem_client.set_imaging_settings(image_settings)
        detector = fibsem_client.get_detector_settings(beam_type)
        fibsem_client.set_detector_settings(detector, beam_type)
        system = fibsem_client.get_beam_system_settings(beam_type)
        fibsem_client.set_beam_system_settings(system)
    state = fibsem_client.get_microscope_state()
    fibsem_client.set_microscope_state(state)


def test_client_parameter_metadata_and_list(fibsem_client):
    assert "stage" in fibsem_client.list_devices()
    assert "choices" in fibsem_client.parameter_metadata("ion", "current")
