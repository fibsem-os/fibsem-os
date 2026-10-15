"""Tests for the fibsem server factory: auth, scopes, control of the microscope
(FIB-1270), and the degree-normalized milling-angle boundary (FIB-852).

Skipped when the [server] extra is not installed (CI installs [test] only until
the FIB-849 leg lands).
"""

import os

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.server import AgentLevel, AuthConfig, build_server  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
POSITION = "/microscope/devices/stage/position"
MOVE_RELATIVE = "/microscope/services/stage_movement/commands/move_relative"
STOP_MILLING = "/microscope/services/milling/commands/stop"
ORIENTATION = "/microscope/services/stage_movement/commands/move_to_orientation"


def _pos(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0):
    # FibsemStagePosition.from_dict requires every axis key.
    return {
        "name": None,
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "t": t,
        "coordinate_system": "RAW",
    }


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
        yield client


@pytest.fixture(scope="module")
def armed_client(microscope):
    app = build_server(microscope, auth=AuthConfig.generate(level="drive", token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_health_is_open(read_client):
    resp = read_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_dashboard_page_is_served_open_like_health(read_client):
    # The page is static and holds no session data; its API calls carry the
    # bearer token, so serving the HTML itself is as safe as /health.
    resp = read_client.get("/dashboard")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "AutoLamella Dashboard" in resp.text


def test_event_log_page_is_served_open_like_the_dashboard(read_client):
    resp = read_client.get("/dashboard/log")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "AutoLamella Event Log" in resp.text


def test_missing_token_is_401(read_client):
    resp = read_client.get("/capabilities")
    assert resp.status_code == 401
    assert resp.json()["detail"]["error_type"] == "unauthorized"


def test_wrong_token_is_401(read_client):
    resp = read_client.get("/capabilities", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_capabilities_reports_the_level_and_routers(read_client):
    resp = read_client.get("/capabilities", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["routers"] == {"microscope": True, "app": False}
    assert body["level"] == "watch"
    assert "manufacturer" not in body  # the microscope's own: GET /microscope


def test_the_microscope_says_who_it_is(read_client):
    body = read_client.get("/microscope", headers=AUTH).json()
    assert body["manufacturer"] == "Demo"
    assert body["serial_number"] == "123456"
    assert "manipulator" in body["fitted"]
    assert "hardware" in body["system"]


def test_a_device_reads_every_value_in_one_call(read_client):
    body = read_client.get("/microscope/devices/electron/values", headers=AUTH).json()
    assert body["values"]["voltage"] > 0
    assert isinstance(body["unavailable"], list)


def test_read_route_with_token(read_client):
    resp = read_client.get(POSITION, headers=AUTH)
    assert resp.status_code == 200
    assert "x" in resp.json()["value"]


def test_hardware_route_refused_below_drive(read_client):
    resp = read_client.post(
        MOVE_RELATIVE, headers=AUTH, json={"kwargs": {"delta": _pos(0)}}
    )
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["error_type"] == "level"
    assert detail["level"] == "watch"


def test_stop_milling_needs_assist(read_client):
    # Watch and a watching agent stops nothing (FIB-1035).
    resp = read_client.post(STOP_MILLING, headers=AUTH)
    assert resp.status_code == 403


def test_valid_requests_stamp_last_seen(microscope):
    """The presence signal: any request bearing the valid token marks the
    agent as heard from — including a level refusal, which is still the agent
    talking. An invalid token marks nothing."""
    auth = AuthConfig(token=TOKEN)
    app = build_server(microscope, auth=auth)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert auth.seconds_since_seen() is None

        client.get(POSITION, headers=AUTH)
        age = auth.seconds_since_seen()
        assert age is not None and age < 5.0

        auth.last_seen_monotonic = None
        refused = client.post(
            MOVE_RELATIVE, headers=AUTH, json={"kwargs": {"delta": _pos()}}
        )
        assert refused.status_code == 403  # hardware not armed
        assert auth.seconds_since_seen() is not None

        auth.last_seen_monotonic = None
        client.get(POSITION, headers={"Authorization": "Bearer wrong"})
        assert auth.seconds_since_seen() is None


def test_hardware_route_works_when_armed(armed_client):
    resp = armed_client.get(POSITION, headers=AUTH)
    start = resp.json()["value"]
    resp = armed_client.post(
        MOVE_RELATIVE, headers=AUTH, json={"kwargs": {"delta": _pos(x=10e-6)}}
    )
    assert resp.status_code == 200
    assert resp.json()["result"]["x"] == pytest.approx(start["x"] + 10e-6, abs=1e-9)


def test_hardware_command_while_someone_is_in_control_is_409(armed_client, microscope):
    # FIB-1270: a run (or a script, or another command) in control refuses the
    # command at once, naming who, rather than interleaving with it.
    with microscope.control.take("workflow", "a lamellae run"):
        resp = armed_client.post(
            MOVE_RELATIVE, headers=AUTH, json={"kwargs": {"delta": _pos(x=1e-6)}}
        )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error_type"] == "microscope_busy"
    assert detail["in_control"] == {"by": "workflow", "action": "a lamellae run"}
    assert "workflow" in detail["message"]


def test_hardware_command_takes_control_as_the_agent(armed_client, microscope):
    seen = []
    microscope.control.changed.connect(seen.append)
    try:
        resp = armed_client.post(
            MOVE_RELATIVE, headers=AUTH, json={"kwargs": {"delta": _pos(x=1e-6)}}
        )
    finally:
        microscope.control.changed.disconnect(seen.append)
    assert resp.status_code == 200
    assert seen[0].by == "agent"
    assert seen[0].action == "move relative"
    assert seen[-1] is None  # released when the command ended
    assert microscope.control.current is None


@pytest.mark.parametrize(
    "method, path, params, expected",
    [
        (
            "POST",
            "/microscope/services/imaging/commands/focus_sweep",
            {"device": "imaging", "command": "focus_sweep"},
            "focus sweep",
        ),
        (
            "POST",
            "/microscope/services/milling/commands/run",
            {"device": "milling", "command": "run"},
            "milling run",
        ),
        (
            "POST",
            "/microscope/services/spot_burn/commands/run",
            {"device": "spot_burn", "command": "run"},
            "spot burn run",
        ),
        (
            "POST",
            MOVE_RELATIVE,
            {"device": "stage_movement", "command": "move_relative"},
            "move relative",
        ),
        (
            "PUT",
            "/microscope/devices/electron/voltage",
            {"device": "electron", "parameter": "voltage"},
            "set electron voltage",
        ),
    ],
)
def test_the_action_is_named_in_words_not_the_route(method, path, params, expected):
    """FIB-1359: Stop agent's record and toast read the action, not the route."""
    from starlette.requests import Request

    from fibsem.server.auth import action_name

    request = Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "path_params": params,
            "query_string": b"",
            "headers": [],
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )
    assert action_name(request) == expected


def test_hardware_command_borrows_control_a_run_has_lent(armed_client, microscope):
    # A run waiting on a question lends control: the command goes through.
    with microscope.control.take("workflow", "a lamellae run"):
        with microscope.control.lend():
            resp = armed_client.post(
                MOVE_RELATIVE, headers=AUTH, json={"kwargs": {"delta": _pos(x=1e-6)}}
            )
    assert resp.status_code == 200


def test_stop_milling_is_allowed_whoever_is_in_control(
    armed_client, microscope, monkeypatch
):
    # The stop must interrupt an in-flight command, so it never takes control.
    monkeypatch.setattr(microscope.milling, "_running", True)  # something to stop
    with microscope.control.take("workflow", "a lamellae run"):
        resp = armed_client.post(
            STOP_MILLING, headers=AUTH, json={"reason": "the run is milling past"}
        )
    assert resp.status_code == 200


def test_microscope_exception_becomes_structured_500(microscope, monkeypatch):
    app = build_server(microscope, auth=AuthConfig(token=TOKEN))
    monkeypatch.setattr(
        type(microscope),
        "get_microscope_state",
        lambda self: (_ for _ in ()).throw(RuntimeError("vendor fell over")),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/microscope/state", headers=AUTH)
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["error_type"] == "RuntimeError"
    assert "vendor fell over" in detail["message"]


def test_a_level_change_takes_effect_without_a_new_token(microscope):
    """The Agent chip's contract: change the live AuthConfig, next request obeys."""
    auth = AuthConfig.generate(token="live-level-token")
    app = build_server(microscope, auth=auth)
    headers = {"Authorization": "Bearer live-level-token"}
    with TestClient(app, raise_server_exceptions=False) as client:
        refused = client.post(
            "/microscope/devices/electron/commands/autocontrast", headers=headers
        )
        assert refused.status_code == 403

        auth.set_level(AgentLevel.DRIVE)
        allowed = client.post(
            "/microscope/devices/electron/commands/autocontrast", headers=headers
        )
        assert allowed.status_code == 200

        auth.set_level(AgentLevel.WATCH)
        refused_again = client.post(
            "/microscope/devices/electron/commands/autocontrast", headers=headers
        )
        assert refused_again.status_code == 403


def test_move_to_orientation_over_http(armed_client, microscope):
    """The remote replacement for the removed `/move_flat_to_beam`, through the
    stage movement service."""
    resp = armed_client.post(
        ORIENTATION, headers=AUTH, json={"kwargs": {"orientation": "FIB"}}
    )
    assert resp.status_code == 200
    assert microscope.get_stage_orientation() == "FIB"

    resp = armed_client.post(
        ORIENTATION, headers=AUTH, json={"kwargs": {"orientation": "SEM"}}
    )
    assert resp.status_code == 200
    assert microscope.get_stage_orientation() == "SEM"
    assert "r" in resp.json()["result"]
