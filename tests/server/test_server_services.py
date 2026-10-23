"""The App Server's /services routes (FIB-1280): the microscope's services, served by
the device routes, behind the same token, scopes and control.

Skipped when the [server] extra is not installed.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.devices.wire import to_wire  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import BeamType, FibsemStagePosition, MillingState  # noqa: E402

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


def test_microscope_lists_the_services_it_built(microscope):
    services = microscope.services
    assert services["milling"] is microscope.milling
    assert services["stage_movement"] is microscope.stage_movement
    assert services["spot_burn"] is microscope.spot_burn
    assert services["sample_handling"] is microscope.sample_handling
    assert services["imaging"] is microscope.imaging


def test_lists_every_service_with_its_commands(read_client, microscope):
    resp = read_client.get("/microscope/services")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == set(microscope.services)
    assert "progress" in body["milling"]["parameters"]
    assert "run" in body["milling"]["commands"]
    assert body["milling"]["commands"]["run"]["available"] is True
    assert "move_to_orientation" in body["stage_movement"]["commands"]
    assert body["milling"]["facts"] == {}
    # What each returns, by its type's path, for a client to read it as.
    project = body["stage_movement"]["queries"]["project_stable_move"]
    assert project["returns"] == "fibsem.structures.FibsemStagePosition"


def test_service_routes_need_the_token(microscope):
    app = build_server(microscope, auth=AuthConfig(token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/microscope/services").status_code == 401
        assert client.get("/microscope/services/milling/state").status_code == 401


def test_read_works_at_watch(read_client):
    resp = read_client.get("/microscope/services/milling/state")
    assert resp.status_code == 200
    assert resp.json()["value"] == to_wire(MillingState.IDLE)


def test_command_is_refused_below_drive(read_client):
    resp = read_client.post("/microscope/services/milling/commands/clear", json={})
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "level"


def test_command_when_armed_reaches_the_service(armed_client, microscope):
    microscope.milling.clear()
    resp = armed_client.post("/microscope/services/milling/commands/clear", json={})
    assert resp.status_code == 200
    assert resp.json()["result"] is None


def test_command_takes_control(armed_client, microscope):
    with microscope.control.take("workflow", "a lamellae run"):
        resp = armed_client.post("/microscope/services/milling/commands/clear", json={})
    assert resp.status_code == 409
    assert resp.json()["detail"]["error_type"] == "microscope_busy"


def test_queries_are_listed_apart_from_commands(read_client):
    stage_movement = read_client.get("/microscope/services/stage_movement").json()
    assert "project_stable_move" in stage_movement["queries"]
    assert "project_stable_move" not in stage_movement["commands"]
    assert "stable_move" not in stage_movement["queries"]


def test_query_works_at_watch_and_takes_no_control(read_client, microscope):
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
            "/microscope/services/stage_movement/queries/project_stable_move", json=body
        )
    assert resp.status_code == 200, resp.text
    expected = microscope.stage_movement.project_stable_move(
        dx=10e-6, dy=-5e-6, beam_type=BeamType.ELECTRON, base_position=base
    )
    assert resp.json()["result"] == to_wire(expected)


def test_the_estimates_and_the_stage_poses_are_queries(read_client, microscope):
    from fibsem.imaging.spot import SpotBurnSettings
    from fibsem.structures import Point

    settings = SpotBurnSettings(
        coordinates=[Point(0.5, 0.5), Point(0.1, 0.1)], exposure_time=2.0
    )
    resp = read_client.post(
        "/microscope/services/spot_burn/queries/estimate",
        json={"kwargs": {"settings": to_wire(settings)}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"] == microscope.spot_burn.estimate(settings)

    geometry = {
        "rotation_reference": 0.0,
        "shuttle_pre_tilt": 35.0,
        "fib_column_tilt": 52.0,
    }
    resp = read_client.post(
        "/microscope/devices/stage/queries/pose_table", json={"kwargs": geometry}
    )
    assert resp.status_code == 200, resp.text
    assert set(resp.json()["result"]) == set(
        microscope.devices["stage"].poses(**geometry)
    )


def test_the_orientation_reads_are_queries(read_client, microscope):
    sem = microscope.get_orientation("SEM")
    at_sem = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=sem.r, t=sem.t)
    resp = read_client.post(
        "/microscope/services/stage_movement/queries/stage_orientation",
        json={"kwargs": {"position": to_wire(at_sem)}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"] == "SEM"
    resp = read_client.post(
        "/microscope/services/stage_movement/queries/current_milling_angle",
        json={"kwargs": {"position": to_wire(at_sem)}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"] == pytest.approx(
        microscope.get_current_milling_angle(at_sem)
    )


def test_the_device_reads_are_queries(read_client, microscope):
    sem = microscope.get_orientation("SEM")
    at_sem = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=sem.r, t=sem.t)
    position = {"stage_position": to_wire(at_sem)}

    def ask(query, **kwargs):
        resp = read_client.post(
            f"/microscope/services/stage_movement/queries/{query}",
            json={"kwargs": kwargs},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["result"]

    device = microscope.get_current_device(at_sem)
    assert ask("current_device", **position) == device
    assert ask("is_at_device", device=device, **position) is True
    state = microscope.get_device_imaging_state(device, at_sem)
    assert ask("device_imaging_state", device=device, **position) == to_wire(state)
    target = microscope.get_target_position(at_sem, target_orientation="FIB")
    answer = ask("target_position", target_orientation="FIB", **position)
    assert answer["x"] == pytest.approx(target.x)
    assert answer["r"] == pytest.approx(target.r)


def test_the_view_corrected_movements_are_queries(read_client, microscope):
    def ask(query, **kwargs):
        resp = read_client.post(
            f"/microscope/services/stage_movement/queries/{query}",
            json={"kwargs": kwargs},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["result"]

    ion = to_wire(BeamType.ION)
    move = microscope._y_corrected_stage_movement(10e-6, beam_type=BeamType.ION)
    answer = ask("y_corrected_movement", expected_y=10e-6, beam_type=ion)
    assert answer["y"] == pytest.approx(move.y)
    assert answer["z"] == pytest.approx(move.z)
    back = ask("inverse_y_corrected_movement", dy=move.y, dz=move.z, beam_type=ion)
    assert back == pytest.approx(10e-6)


def test_the_milling_angle_is_a_parameter_that_moves_milling(armed_client, microscope):
    before = microscope.system.stage.milling_angle
    try:
        resp = armed_client.put(
            "/microscope/services/stage_movement/milling_angle", json={"value": 22.0}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["value"] == pytest.approx(22.0)
        assert microscope.system.stage.milling_angle == pytest.approx(22.0)
        milling = microscope.get_orientation("MILLING")
        assert microscope.get_current_milling_angle(
            FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=milling.r, t=milling.t)
        ) == pytest.approx(22.0)
    finally:
        microscope.set_milling_angle(before)


def test_unknown_query_or_a_command_asked_as_one_is_404(read_client):
    resp = read_client.post("/microscope/services/stage_movement/queries/nope", json={})
    assert resp.status_code == 404
    resp = read_client.post(
        "/microscope/services/stage_movement/queries/stable_move", json={}
    )
    assert resp.status_code == 404


def test_a_stop_needs_drive_and_is_never_refused_for_control(
    read_client, microscope, monkeypatch
):
    stop = "/microscope/services/milling/commands/stop"
    # Something to stop (FIB-1398).
    monkeypatch.setattr(microscope.milling, "_running", True)
    why = {"reason": "the mill is off target"}
    assert read_client.post(stop, json=why).status_code == 403
    app = build_server(microscope, auth=AuthConfig.generate(level="drive", token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        with microscope.control.take("workflow", "a lamellae run"):
            resp = client.post(stop, headers=AUTH, json=why)
    assert resp.status_code == 200


def test_unknown_service_parameter_or_command_is_404(read_client, armed_client):
    resp = read_client.get("/microscope/services/nope/state")
    assert resp.status_code == 404
    assert "no service 'nope'" in resp.text
    assert read_client.get("/microscope/services/milling/nope").status_code == 404
    resp = armed_client.post("/microscope/services/milling/commands/nope", json={})
    assert resp.status_code == 404


def test_devices_and_services_do_not_mix(read_client):
    assert read_client.get("/microscope/devices/milling/state").status_code == 404
    assert read_client.get("/microscope/services/ion/current").status_code == 404


def test_an_agent_runs_a_focus_sweep(armed_client, microscope):
    from fibsem.autofunctions.autofocus import AutoFocusSettings, FocusSweepPass
    from fibsem.structures import BeamType

    settings = AutoFocusSettings(
        passes=[FocusSweepPass(search_range=20e-6, step_size=10e-6)],
        probe_resolution=(64, 64),
        use_autocontrast=False,
    )
    resp = armed_client.post(
        "/microscope/services/imaging/commands/focus_sweep",
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


def test_an_agent_runs_auto_brightness_contrast(armed_client, microscope):
    from fibsem.autofunctions.brightness_contrast import BrightnessContrastSettings
    from fibsem.structures import BeamType

    beam = microscope.beams[BeamType.ION]
    beam.detector_brightness.set_value(0.95)
    settings = BrightnessContrastSettings(probe_resolution=(384, 256))
    resp = armed_client.post(
        "/microscope/services/imaging/commands/auto_brightness_contrast",
        json={
            "kwargs": {
                "beam_type": to_wire(BeamType.ION),
                "settings": settings.to_dict(),
            }
        },
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    assert result["converged"] is True
    assert beam.detector_brightness.get_value() == pytest.approx(result["brightness"])


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


def test_a_stop_s_reason_reaches_a_command_that_takes_one():
    """Every caller sends a stop's reason beside the kwargs (`command_body`);
    the server gives it to a command that records it too."""
    from fibsem.devices.wire import command_body, command_kwargs

    def end(reason: str, by: str = ""):
        pass

    def stop():
        pass

    body = command_body({"by": "agent"}, "drift")
    assert body == {"kwargs": {"by": "agent"}, "reason": "drift"}
    assert command_kwargs(end, body) == {"by": "agent", "reason": "drift"}
    assert command_kwargs(stop, command_body(reason="drift")) == {}
    sent = command_body({"reason": "its own"}, "drift")
    assert command_kwargs(end, sent)["reason"] == "its own"
    assert command_body(preview=True) == {"kwargs": {}, "answer": "preview"}


def test_sequences_decode_each_element():
    from typing import List, Sequence, Tuple

    from fibsem.devices.wire import decode_argument
    from fibsem.structures import Point

    points = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
    expected = [Point(1.0, 2.0), Point(3.0, 4.0)]
    assert decode_argument("p", Sequence[Point], points) == expected
    assert decode_argument("p", List[Point], points) == expected
    assert decode_argument("p", Tuple[Point, ...], points) == tuple(expected)
    pair = decode_argument("p", Tuple[BeamType, float], [1, 2.0])
    assert pair == (BeamType.ELECTRON, 2.0)


def test_numbers_sent_as_text_decode_as_the_declared_type():
    """An agent's MCP client sends a value whose schema names no type as text."""
    from fibsem.devices.wire import ArgumentError, decode_argument

    def read(hint, value):
        return decode_argument("value", hint, value)

    assert read(float, "4.0") == 4.0 and isinstance(read(float, "4"), float)
    assert read(float, " 1.2e-6 ") == pytest.approx(1.2e-6)
    assert read(int, "4.0") == 4 and isinstance(read(int, "4.0"), int)
    assert read(int, 4.0) == 4 and isinstance(read(int, 4.0), int)
    assert read(bool, "true") is True and read(bool, "False") is False
    assert read(str, "4.0") == "4.0"
    assert read(BeamType, "1") is BeamType.ELECTRON
    assert read(BeamType, "ION") is BeamType.ION
    # Not a value of that type: refused, naming what was sent.
    for hint, value in [(int, "4.5"), (int, 4.5), (float, "fast"), (BeamType, "9")]:
        with pytest.raises(ArgumentError, match="value"):
            read(hint, value)


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
        "/microscope/services/spot_burn/commands/run",
        json={"kwargs": {"settings": settings.to_dict()}},
    )
    assert resp.status_code == 200, resp.text
    # A long command: answered at once with its run, read back when it ends.
    run = resp.json()["result"]
    ended = armed_client.get(
        "/microscope/services/spot_burn/run",
        params={"run_id": run["run_id"], "wait": 10},
    )
    assert ended.json()["run"]["result"] == "finished"


def test_an_agent_prepares_milling_with_patterns(armed_client, microscope):
    from fibsem.structures import FibsemMillingSettings, FibsemRectangleSettings

    rectangle = FibsemRectangleSettings(
        width=1e-6, height=2e-6, depth=1e-6, centre_x=0.0, centre_y=0.0
    )
    try:
        resp = armed_client.post(
            "/microscope/services/milling/commands/prepare",
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


def test_a_drivers_own_setting_sent_at_the_top_is_422_naming_its_block(armed_client):
    """Tescan's preset used to be a recipe field; it is in the driver's block now."""
    path = "/microscope/services/milling/commands/setup"
    resp = armed_client.post(
        path, json={"kwargs": {"settings": {"preset": "30 keV; 1 nA"}}}
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["parameter"] == "settings.preset"
    assert "did you mean settings.driver.Tescan.preset?" in detail["message"]
    # an unknown key inside the block is refused, naming what the block takes
    bad = {"driver": {"Tescan": {"ratee": 1e-8}}}
    resp = armed_client.post(path, json={"kwargs": {"settings": bad}})
    assert resp.status_code == 422, resp.text
    assert "Tescan.ratee is not a setting" in resp.json()["detail"]["message"]


def _projected_move(beam_type):
    base = FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3, r=0.0, t=0.0)
    return {
        "kwargs": {
            "dx": 10e-6,
            "dy": -5e-6,
            "beam_type": beam_type,
            "base_position": to_wire(base),
        }
    }


def test_an_enum_is_sent_by_name_or_value(read_client):
    """``list_services`` shows ``<BeamType.ELECTRON: 1>``, so an agent sends the
    name (FIB-1355)."""
    path = "/microscope/services/stage_movement/queries/project_stable_move"
    by_value = read_client.post(path, json=_projected_move(1))
    by_name = read_client.post(path, json=_projected_move("ELECTRON"))
    assert by_name.status_code == 200, by_name.text
    assert by_name.json() == by_value.json()


def test_an_argument_that_does_not_decode_is_422_naming_it(read_client):
    path = "/microscope/services/stage_movement/queries/project_stable_move"
    resp = read_client.post(path, json=_projected_move("PROTON"))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["parameter"] == "beam_type"
    assert detail["error_type"] == "ArgumentError"
    assert "PROTON" in detail["message"]


def test_a_cancelled_command_is_409_cancelled(armed_client, microscope, monkeypatch):
    from fibsem.cancellation import OperationCancelledError

    def cancelled_sweep() -> None:
        raise OperationCancelledError("AutoFocus cancelled by user.")

    monkeypatch.setattr(microscope.imaging, "focus_sweep", cancelled_sweep)
    resp = armed_client.post(
        "/microscope/services/imaging/commands/focus_sweep", json={}
    )
    assert resp.status_code == 409
    assert resp.json()["detail"] == {
        "error_type": "OperationCancelledError",
        "message": "AutoFocus cancelled by user.",
        "status": "cancelled",
    }


def test_a_key_its_structure_does_not_have_is_422_naming_where_it_belongs(
    read_client,
):
    """An overview's hfw at the top of its settings, not in image_settings, was
    dropped by from_dict and the default used without a word."""
    path = "/microscope/services/imaging/queries/estimate_overview"
    resp = read_client.post(
        path, json={"kwargs": {"settings": {"hfw": 400e-6, "nrows": 2}}}
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["error_type"] == "ArgumentError"
    assert detail["parameter"] == "settings.hfw"
    assert "did you mean settings.image_settings.hfw?" in detail["message"]


def test_an_unknown_key_is_refused_at_any_depth(read_client):
    path = "/microscope/services/imaging/queries/estimate_overview"
    passes = [{"search_range": 10e-6, "step_size": 1e-6, "steps": 10}]
    nested = {"autofocus_settings": {"passes": passes}}
    resp = read_client.post(path, json={"kwargs": {"settings": nested}})
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["parameter"] == (
        "settings.autofocus_settings.passes[0].steps"
    )
    misspelt = {"image_settings": {"dwell": 1e-6}}
    resp = read_client.post(path, json={"kwargs": {"settings": misspelt}})
    detail = resp.json()["detail"]
    assert detail["parameter"] == "settings.image_settings.dwell"
    assert "did you mean settings.image_settings.dwell_time?" in detail["message"]


def test_what_a_structure_writes_it_reads_back(read_client):
    from fibsem.structures import OverviewAcquisitionSettings

    path = "/microscope/services/imaging/queries/estimate_overview"
    settings = to_wire(OverviewAcquisitionSettings(nrows=2))
    resp = read_client.post(path, json={"kwargs": {"settings": settings}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"]["nrows"] == 2


def _tool(client, name, arguments):
    from fibsem.ai.agent.catalog import CATALOG, route_call

    spec = next(spec for spec in CATALOG if spec.name == name)
    method, path, body = route_call(spec, arguments)
    return path, client.request(method, path, json=body)


def test_beam_auto_focus_sweeps_and_answers_each_step(armed_client, microscope):
    settings = {
        "passes": [{"search_range": 20e-6, "step_size": 10e-6}],
        "probe_resolution": [64, 64],
        "use_autocontrast": False,
    }
    path, resp = _tool(
        armed_client, "beam_auto_focus", {"beam_type": "ELECTRON", "settings": settings}
    )
    assert path.endswith("/imaging/commands/focus_sweep")
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    steps = result["iterations"]
    assert len(steps) == 3
    assert all({"working_distance", "focus_score"} <= set(step) for step in steps)
    tried = [step["working_distance"] for step in steps]
    best = result["working_distance"]
    assert result["at_edge"] is (best in (min(tried), max(tried)))
    beam = microscope.beams[BeamType.ELECTRON]
    assert beam.working_distance.get_value() == pytest.approx(best)


def test_beam_auto_focus_asks_the_instrument_with_method_instrument(armed_client):
    path, resp = _tool(armed_client, "beam_auto_focus", {"method": "instrument"})
    assert path.endswith("/imaging/commands/auto_focus")
    assert resp.status_code == 200, resp.text
    with pytest.raises(ValueError, match="method sweep"):
        _tool(armed_client, "beam_auto_focus", {"method": "instrument", "hfw": 1e-4})


def test_a_sweep_that_cannot_set_the_working_distance_asks_the_instrument(
    armed_client, microscope, monkeypatch
):
    """Tescan's ion beam: nothing to sweep, so its own autofocus, and it says so."""
    monkeypatch.setattr(
        microscope, "is_working_distance_settable", lambda beam_type: False
    )
    _, resp = _tool(armed_client, "beam_auto_focus", {"beam_type": "ION"})
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    assert result["focused_with"] == "instrument"
    assert "ION beam's working distance can't be set" in result["reason"]
    beam = microscope.beams[BeamType.ION]
    assert result["working_distance"] == pytest.approx(beam.working_distance.cached)


def test_milling_answers_the_current_and_warns_on_one_the_beam_does_not_list(
    armed_client, microscope
):
    from fibsem.structures import FibsemMillingSettings, FibsemRectangleSettings

    listed = sorted(microscope.beams[BeamType.ION].current.choices)
    between = (listed[0] + listed[1]) / 2
    rectangle = FibsemRectangleSettings(
        width=1e-6, height=2e-6, depth=1e-6, centre_x=0.0, centre_y=0.0
    )
    try:
        for current, warned in ((listed[1], False), (between, True)):
            settings = FibsemMillingSettings(milling_current=current)
            resp = armed_client.post(
                "/microscope/services/milling/commands/prepare",
                json={
                    "kwargs": {
                        "settings": settings.to_dict(),
                        "patterns": [{**rectangle.to_dict(), "type": "Rectangle"}],
                    }
                },
            )
            assert resp.status_code == 200, resp.text
            prepared = resp.json()["result"]
            assert prepared["milling_current"] == pytest.approx(current)
            assert (prepared["warning"] is not None) is warned
            _, resp = _tool(armed_client, "milling_estimate_time", {})
            estimate = resp.json()["result"]
            assert estimate["seconds"] > 0
            assert estimate["milling_current"] == pytest.approx(current)
            assert estimate["warning"] == prepared["warning"]
    finally:
        microscope.milling.clear()
        microscope.milling.restore()


def test_a_post_with_no_arguments_sends_an_empty_object():
    """Round 10 F8: item_add with no arguments sent no body, which its route
    refused. A GET still sends none."""
    from fibsem.ai.agent.catalog import CATALOG, route_call

    tools = {spec.name: spec for spec in CATALOG}
    assert route_call(tools["item_add"], {}) == ("POST", "/app/experiment/items", {})
    assert route_call(tools["workflow_get_queue"], {})[2] is None
