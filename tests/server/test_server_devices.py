"""The server's /microscope/devices routes (every device in microscope.devices),
behind the server's token, level and control of the microscope.

Skipped when the [server] extra is not installed.
"""

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemStagePosition,
)

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def read_client(microscope):
    app = build_server(microscope, auth=AuthConfig(token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update(AUTH)
        yield client


@pytest.fixture(scope="module")
def armed_client(microscope):
    app = build_server(microscope, auth=AuthConfig.generate(level="drive", token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update(AUTH)
        yield client


def test_lists_every_device_the_microscope_built(read_client, microscope):
    resp = read_client.get("/microscope/devices")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == set(microscope.devices)
    assert "current" in body["ion"]["parameters"]
    assert "move_relative" in body["stage"]["commands"]


def test_a_command_lists_its_arguments_as_json_schema(read_client):
    """An agent has no copy of the classes: the listing says what each argument is."""
    body = read_client.get("/microscope/devices").json()
    arguments = body["stage"]["commands"]["move_relative"]["arguments"]

    assert arguments["type"] == "object"
    (position,) = arguments["required"][:1]
    assert arguments["properties"][position] == {"$ref": "#/$defs/FibsemStagePosition"}
    assert arguments["$defs"]["FibsemStagePosition"]["type"] == "object"


def test_device_routes_need_the_token(microscope):
    app = build_server(microscope, auth=AuthConfig(token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/microscope/devices").status_code == 401
        assert client.get("/microscope/devices/ion/current").status_code == 401


def test_read_works_at_watch(read_client, microscope):
    resp = read_client.get("/microscope/devices/ion/current")
    assert resp.status_code == 200
    assert resp.json()["value"] == microscope.get_beam_current(BeamType.ION)


def test_write_is_refused_below_drive(read_client):
    resp = read_client.put("/microscope/devices/electron/hfw", json={"value": 100e-6})
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "level"


def test_write_when_armed_reaches_the_device(armed_client, microscope):
    resp = armed_client.put("/microscope/devices/electron/hfw", json={"value": 120e-6})
    assert resp.status_code == 200
    assert microscope.get_field_of_view(BeamType.ELECTRON) == pytest.approx(120e-6)


def test_write_takes_control(armed_client, microscope):
    with microscope.control.take("workflow", "a lamellae run"):
        resp = armed_client.put(
            "/microscope/devices/electron/hfw", json={"value": 100e-6}
        )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error_type"] == "microscope_busy"
    assert detail["in_control"]["by"] == "workflow"


def test_metadata_answers_choices(read_client, microscope):
    resp = read_client.get("/microscope/devices/ion/current/metadata")
    assert resp.status_code == 200
    choices = microscope.beams[BeamType.ION].parameters["current"].choices
    assert resp.json()["choices"] == list(choices)


def test_unknown_device_or_parameter_is_404(read_client):
    assert read_client.get("/microscope/devices/nope/current").status_code == 404
    assert read_client.get("/microscope/devices/ion/nope").status_code == 404


def test_command_needs_drive(read_client):
    resp = read_client.post("/microscope/devices/ion/commands/blank", json={})
    assert resp.status_code == 403


def test_the_chamber_is_read_but_never_pumped_or_vented(read_client, armed_client):
    # Pumping and venting are the operator's, at the microscope: the blocklist
    # refuses them whatever is armed. State and pressure are reads.
    chamber = read_client.get("/microscope/devices").json()["chamber"]
    assert {"state", "pressure"} <= set(chamber["parameters"])
    assert read_client.get("/microscope/devices/chamber/state").status_code == 200
    for command in ("vent", "pump"):
        resp = armed_client.post(
            f"/microscope/devices/chamber/commands/{command}", json={}
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["error_type"] == "not_allowed"


def test_a_stop_reaches_a_device_whoever_is_in_control(read_client, microscope):
    # A stop must reach a device whoever is in control; one the agent didn't
    # start needs Drive, and every stop a reason.
    stop = "/microscope/devices/electron/commands/stop_live"
    why = {"reason": "live view left running"}
    assert read_client.post(stop, json=why).status_code == 403
    app = build_server(microscope, auth=AuthConfig.generate(level="drive", token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        with microscope.control.take("workflow", "a lamellae run"):
            resp = client.post(stop, headers=AUTH, json=why)
    assert resp.status_code == 200


def test_a_move_past_the_stage_limits_is_refused_not_a_500(armed_client, microscope):
    """A stage move refuses as every device and service route does: a 4xx in the
    one refusal shape, not a 500 (FIB-1377)."""
    before = microscope.stage.position.get_value()
    target = FibsemStagePosition(x=before.x, y=1.0, z=before.z, r=before.r, t=before.t)
    resp = armed_client.post(
        "/microscope/services/stage_movement/commands/safe_absolute_stage_movement",
        json={"kwargs": {"stage_position": target.to_dict()}},
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["error_type"] == "StageLimitError"
    assert detail["message"]
    assert microscope.stage.position.get_value() == before  # nothing moved


@pytest.mark.parametrize(
    "path, body",
    [
        ("/microscope/devices/ion/commands/acquire", {}),
        ("/microscope/devices/ion/commands/acquire", {"answer": "preview"}),
    ],
)
def test_an_acquired_image_reaches_the_apps_views(armed_client, microscope, path, body):
    # FIB-1375: the app's views show what `imaging.show` is given, as for the
    # workflow's images, so an agent's frame replaces the one from before its move.
    shown = []
    microscope.imaging.image_shown.connect(shown.append)
    try:
        resp = armed_client.post(path, json=body)
    finally:
        microscope.imaging.image_shown.disconnect(shown.append)
    assert resp.status_code == 200
    assert len(shown) == 1
    assert shown[0].metadata.beam_type is BeamType.ION
