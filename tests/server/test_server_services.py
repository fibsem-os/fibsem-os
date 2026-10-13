"""The App Server's /services routes (FIB-1280): the microscope's services, served by
the device routes, behind the same token, scopes and control.

Skipped when the [server] extra is not installed.
"""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.devices.wire import to_wire  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import BeamType, FibsemStagePosition, MillingState  # noqa: E402

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


def test_microscope_lists_the_services_it_built(microscope):
    services = microscope.services
    assert services["milling"] is microscope.milling
    assert services["stage_movement"] is microscope.stage_movement
    assert services["spot_burn"] is microscope.spot_burn
    assert services["sample_handling"] is microscope.sample_handling
    assert services["imaging"] is microscope.imaging


def test_lists_every_service_with_its_commands(read_client, microscope):
    resp = read_client.get("/services")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == set(microscope.services)
    assert "progress" in body["milling"]["parameters"]
    assert "run" in body["milling"]["commands"]
    assert body["milling"]["commands"]["run"]["available"] is True
    assert "move_to_orientation" in body["stage_movement"]["commands"]
    assert body["milling"]["facts"] == {}


def test_service_routes_need_the_token(microscope):
    app = build_server(microscope, auth=AuthConfig(token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/services").status_code == 401
        assert client.get("/services/milling/state").status_code == 401


def test_read_is_read_scope(read_client):
    resp = read_client.get("/services/milling/state")
    assert resp.status_code == 200
    assert resp.json()["value"] == to_wire(MillingState.IDLE)


def test_command_is_refused_until_hardware_is_armed(read_client):
    resp = read_client.post("/services/milling/commands/estimate", json={})
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "scope_not_armed"


def test_command_when_armed_reaches_the_service(armed_client, microscope):
    microscope.milling.clear()
    resp = armed_client.post("/services/milling/commands/estimate", json={})
    assert resp.status_code == 200
    assert resp.json()["result"] == microscope.milling.estimate()


def test_command_takes_control(armed_client, microscope):
    with microscope.control.take("workflow", "a lamellae run"):
        resp = armed_client.post("/services/milling/commands/estimate", json={})
    assert resp.status_code == 409
    assert resp.json()["detail"]["error_type"] == "microscope_busy"


def test_queries_are_listed_apart_from_commands(read_client):
    stage_movement = read_client.get("/services/stage_movement").json()
    assert "project_stable_move" in stage_movement["queries"]
    assert "project_stable_move" not in stage_movement["commands"]
    assert "stable_move" not in stage_movement["queries"]


def test_query_is_read_scope_and_takes_no_control(read_client, microscope):
    base = FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3, r=0.0, t=0.0)
    body = {
        "kwargs": {
            "dx": 10e-6,
            "dy": -5e-6,
            "beam_type": to_wire(BeamType.ELECTRON),
            "base_position": to_wire(base),
        }
    }
    with microscope.control.take("workflow", "a lamellae run"):
        resp = read_client.post(
            "/services/stage_movement/queries/project_stable_move", json=body
        )
    assert resp.status_code == 200, resp.text
    expected = microscope.stage_movement.project_stable_move(
        dx=10e-6, dy=-5e-6, beam_type=BeamType.ELECTRON, base_position=base
    )
    assert resp.json()["result"] == to_wire(expected)


def test_unknown_query_or_a_command_asked_as_one_is_404(read_client):
    resp = read_client.post("/services/stage_movement/queries/nope", json={})
    assert resp.status_code == 404
    resp = read_client.post("/services/stage_movement/queries/stable_move", json={})
    assert resp.status_code == 404


def test_stop_needs_only_the_token_and_is_never_refused(read_client, microscope):
    with microscope.control.take("workflow", "a lamellae run"):
        resp = read_client.post("/services/milling/commands/stop", json={})
    assert resp.status_code == 200


def test_unknown_service_parameter_or_command_is_404(read_client, armed_client):
    resp = read_client.get("/services/nope/state")
    assert resp.status_code == 404
    assert "no service 'nope'" in resp.text
    assert read_client.get("/services/milling/nope").status_code == 404
    resp = armed_client.post("/services/milling/commands/nope", json={})
    assert resp.status_code == 404


def test_devices_and_services_do_not_mix(read_client):
    assert read_client.get("/devices/milling/state").status_code == 404
    assert read_client.get("/services/ion/current").status_code == 404


def test_an_agent_runs_a_focus_sweep(armed_client, microscope):
    from fibsem.autofunctions.autofocus import AutoFocusSettings, FocusSweepPass
    from fibsem.structures import BeamType

    settings = AutoFocusSettings(
        passes=[FocusSweepPass(search_range=20e-6, step_size=10e-6)],
        probe_resolution=(64, 64),
        use_autocontrast=False,
    )
    resp = armed_client.post(
        "/services/imaging/commands/focus_sweep",
        json={
            "kwargs": {
                "beam_type": to_wire(BeamType.ELECTRON),
                "hfw": 80e-6,
                "settings": settings.to_dict(),
            }
        },
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    assert result["n_iterations"] == 3
    beam = microscope.beams[BeamType.ELECTRON]
    assert beam.working_distance.get_value() == pytest.approx(
        result["working_distance"]
    )


def test_every_service_command_and_query_hint_resolves(microscope):
    """The routes decode arguments from these hints; one that names a type imported
    only for type checking would leave its arguments as raw JSON (FIB-1354)."""
    import typing

    for service in microscope.services.values():
        for name in [*service.commands, *getattr(service, "queries", {})]:
            typing.get_type_hints(getattr(service, name))  # raises if one doesn't


def test_hints_resolve_one_at_a_time():
    """A hint that can't resolve is left out; the others still decode."""
    from fibsem.devices.wire import decode_kwargs
    from fibsem.structures import Point

    def move(point: Point, stop_event: "OnlyForTypeChecking" = None):  # noqa: F821
        pass

    kwargs = decode_kwargs(move, {"point": {"x": 1.0, "y": 2.0}, "stop_event": None})
    assert kwargs["point"] == Point(1.0, 2.0)


def test_sequences_decode_each_element():
    from typing import List, Sequence, Tuple

    from fibsem.devices.wire import from_wire
    from fibsem.structures import Point

    points = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
    assert from_wire(Sequence[Point], points) == [Point(1.0, 2.0), Point(3.0, 4.0)]
    assert from_wire(List[Point], points) == [Point(1.0, 2.0), Point(3.0, 4.0)]
    assert from_wire(Tuple[Point, ...], points) == (Point(1.0, 2.0), Point(3.0, 4.0))
    assert from_wire(Tuple[BeamType, float], [1, 2.0]) == (BeamType.ELECTRON, 2.0)


def test_a_pattern_dict_names_its_shape():
    from fibsem.structures import FibsemPatternSettings, FibsemRectangleSettings

    rectangle = FibsemRectangleSettings(
        width=1e-6, height=2e-6, depth=1e-6, centre_x=0.0, centre_y=0.0
    )
    for type_name in ("Rectangle", "FibsemRectangleSettings"):
        data = {**rectangle.to_dict(), "type": type_name}
        assert FibsemPatternSettings.from_dict(data) == rectangle
    with pytest.raises(ValueError, match="Unknown pattern type: 'Square'"):
        FibsemPatternSettings.from_dict({**rectangle.to_dict(), "type": "Square"})


def test_an_agent_runs_a_spot_burn(armed_client, microscope):
    from fibsem.imaging.spot import SpotBurnSettings
    from fibsem.structures import Point

    settings = SpotBurnSettings(coordinates=[Point(0.5, 0.5)], exposure_time=0.01)
    resp = armed_client.post(
        "/services/spot_burn/commands/run",
        json={"kwargs": {"settings": settings.to_dict()}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"] == "finished"


def test_an_agent_prepares_milling_with_patterns(armed_client, microscope):
    from fibsem.structures import FibsemMillingSettings, FibsemRectangleSettings

    rectangle = FibsemRectangleSettings(
        width=1e-6, height=2e-6, depth=1e-6, centre_x=0.0, centre_y=0.0
    )
    try:
        resp = armed_client.post(
            "/services/milling/commands/prepare",
            json={
                "kwargs": {
                    "settings": FibsemMillingSettings().to_dict(),
                    "patterns": [{**rectangle.to_dict(), "type": "Rectangle"}],
                }
            },
        )
        assert resp.status_code == 200, resp.text
        assert microscope._connection.milling_system.patterns == [rectangle]
    finally:
        microscope.milling.clear()
        microscope.milling.restore()
