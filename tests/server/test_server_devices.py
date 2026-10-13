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
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    Point,
)

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


def test_write_takes_control(armed_client, microscope):
    with microscope.control.take("workflow", "a lamellae run"):
        resp = armed_client.put("/devices/electron/hfw", json={"value": 100e-6})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error_type"] == "microscope_busy"
    assert detail["in_control"]["by"] == "workflow"


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


@pytest.mark.parametrize("command", ["vent", "pump"])
def test_chamber_vacuum_commands_are_refused_when_armed(
    armed_client, microscope, command, monkeypatch
):
    # Pumping and venting are the operator's, at the microscope: refused
    # whatever is armed, before the chamber is reached.
    chamber = microscope.devices["chamber"]
    reached = []
    monkeypatch.setattr(chamber, "_vent", lambda: reached.append("vent"))
    monkeypatch.setattr(chamber, "_pump", lambda: reached.append("pump"))
    resp = armed_client.post(f"/devices/chamber/commands/{command}", json={})
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["error_type"] == "command_not_served"
    assert "operator" in detail["message"]
    assert reached == []


def test_chamber_vacuum_refusal_does_not_take_control(armed_client, microscope):
    # Refused as not served, not as busy: the answer doesn't depend on a run.
    with microscope.control.take("workflow", "a lamellae run"):
        resp = armed_client.post("/devices/chamber/commands/vent", json={})
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "command_not_served"


def test_chamber_reads_are_still_served(read_client):
    assert read_client.get("/devices/chamber/state").status_code == 200
    assert read_client.get("/devices/chamber/pressure").status_code == 200


def test_stop_commands_pass_with_read_scope_whoever_is_in_control(
    read_client, microscope
):
    # Like /stop_milling: a stop must reach a device whoever is in control.
    with microscope.control.take("workflow", "a lamellae run"):
        resp = read_client.post("/devices/electron/commands/stop_live", json={})
    assert resp.status_code == 200


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
    resp = read_client.get("/beams/proton/beam_settings")
    assert resp.status_code == 422
    assert resp.json()["detail"]["parameter"] == "beam_type"


def test_a_move_past_the_stage_limits_is_refused_not_a_500(armed_client, microscope):
    """The microscope routes refuse as the device routes do: a 4xx in the one
    refusal shape, not a 500 (FIB-1377)."""
    before = microscope.stage.position.get_value()
    target = FibsemStagePosition(x=before.x, y=1.0, z=before.z, r=before.r, t=before.t)
    resp = armed_client.post(
        "/move_stage_absolute", json={"position": target.to_dict()}
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["error_type"] == "StageLimitError"
    assert detail["message"]
    assert microscope.stage.position.get_value() == before  # nothing moved


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


@pytest.mark.parametrize(
    "path, body",
    [
        ("/acquire_image", {"beam_type": "ION"}),
        ("/acquire_image_preview", {"beam_type": "ION"}),
        ("/devices/ion/commands/acquire", {}),
    ],
)
def test_an_acquired_image_reaches_the_apps_views(armed_client, microscope, path, body):
    # FIB-1375: the app's FIB view listens on this signal, as for the workflow's
    # images, so an agent's frame replaces the one from before its move.
    shown = []
    microscope.fib_acquisition_signal.connect(shown.append)
    try:
        resp = armed_client.post(path, json=body)
    finally:
        microscope.fib_acquisition_signal.disconnect(shown.append)
    assert resp.status_code == 200
    assert len(shown) == 1
    assert shown[0].metadata.beam_type is BeamType.ION
