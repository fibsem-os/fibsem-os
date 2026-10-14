"""The agent level (FIB-1035): Off refuses the agent everything, Watch reads and
notes, Drive acts on the session and, with the devices service on, on the
microscope (FIB-1267). The dashboard's token ignores the level."""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.acting import AGENT as AGENT_ACTOR  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.control import WORKFLOW  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.auth import AgentLevel  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402

AGENT = {"Authorization": "Bearer agent"}
STOP_MILLING = "/microscope/services/milling/commands/stop"
# Every agent stop carries a reason.
WHY = {"reason": "the trench is through the lamella"}
MOVE_RELATIVE = "/microscope/services/stage_movement/commands/move_relative"
# A one-micron stage nudge, in the route's body shape.
_NUDGE = {
    "kwargs": {
        "delta": {
            "name": None,
            "x": 1e-6,
            "y": 0.0,
            "z": 0.0,
            "r": 0.0,
            "t": 0.0,
            "coordinate_system": "RAW",
        }
    }
}
DASHBOARD = {"Authorization": "Bearer dashboard"}


class Host:
    experiment = None
    microscope = None
    _task_manager = None
    is_workflow_running = False


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


@pytest.fixture
def auth():
    auth = AuthConfig(token="agent", dashboard_token="dashboard")
    auth.set_level(AgentLevel.WATCH)
    return auth


@pytest.fixture
def client(tmp_path, microscope, auth):
    host = Host()
    exp = Experiment(path=tmp_path / "exp", name="level-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    host.experiment = exp
    host.microscope = microscope
    app = build_server(microscope, app_context=AutoLamellaAppContext(host), auth=auth)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _error(response) -> str:
    return response.json()["detail"]["error_type"]


def test_watch_reads_and_notes_but_acts_on_nothing(client):
    capabilities = client.get("/capabilities", headers=AGENT).json()
    assert capabilities["level"] == "watch"
    assert client.get("/app/workflow/status", headers=AGENT).status_code == 200
    note = client.post("/app/agent/notes", headers=AGENT, json={"text": "seen"})
    assert note.status_code == 200

    for path in ("/app/workflow/stop", STOP_MILLING, "/app/workflow/prompt/answer"):
        response = client.post(path, headers=AGENT, json={})
        assert response.status_code == 403, path
        assert _error(response) == "level"
        assert response.json()["detail"]["level"] == "watch"


def test_drive_acts_on_the_session_and_the_hardware(client, auth):
    auth.set_level(AgentLevel.DRIVE)
    capabilities = client.get("/capabilities", headers=AGENT).json()
    assert capabilities["level"] == "drive"
    assert "scopes" not in capabilities
    assert client.post("/app/workflow/stop", headers=AGENT, json=WHY).status_code == 200
    assert client.post(STOP_MILLING, headers=AGENT, json=WHY).status_code == 200
    moved = client.post(MOVE_RELATIVE, headers=AGENT, json=_NUDGE)
    assert moved.status_code == 200, moved.text


def test_drive_reads_the_chamber_but_cannot_vent_or_pump_it(client, auth):
    # Drive arms the hardware, but pumping and venting are the operator's.
    auth.set_level(AgentLevel.DRIVE)
    for command in ("vent", "pump"):
        resp = client.post(
            f"/microscope/devices/chamber/commands/{command}", headers=AGENT, json={}
        )
        assert resp.status_code == 403, command
        assert _error(resp) == "not_allowed"
    for parameter in ("state", "pressure"):
        read = client.get(f"/microscope/devices/chamber/{parameter}", headers=AGENT)
        assert read.status_code == 200, parameter


def test_watch_refuses_a_hardware_command_naming_the_level(client, auth):
    moved = client.post(MOVE_RELATIVE, headers=AGENT, json=_NUDGE)
    assert moved.status_code == 403
    assert _error(moved) == "level"
    assert moved.json()["detail"]["level"] == "watch"


def test_at_drive_a_run_in_control_refuses_a_hardware_command(client, auth, microscope):
    auth.set_level(AgentLevel.DRIVE)
    with microscope.control.take(WORKFLOW, "a lamella run"):
        moved = client.post(MOVE_RELATIVE, headers=AGENT, json=_NUDGE)
        # Stopping never takes control, so the run never refuses it.
        assert client.post(STOP_MILLING, headers=AGENT, json=WHY).status_code == 200
    assert moved.status_code == 409
    assert _error(moved) == "microscope_busy"
    assert moved.json()["detail"]["in_control"]["by"] == WORKFLOW


def test_with_the_microscope_off_drive_has_no_hardware_routes(microscope, auth):
    auth.set_level(AgentLevel.DRIVE)
    app = build_server(microscope, auth=auth, serve_microscope=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        moved = client.post(MOVE_RELATIVE, headers=AGENT, json=_NUDGE)
    assert moved.status_code == 404


def test_stop_agent_takes_hardware_with_the_rest(client, auth):
    auth.set_level(AgentLevel.DRIVE)
    auth.stop_agent()
    assert not auth.reaches(AgentLevel.DRIVE)
    moved = client.post(MOVE_RELATIVE, headers=AGENT, json=_NUDGE)
    assert moved.status_code == 403


def test_off_refuses_the_agent_everything_but_not_the_dashboard(client, auth):
    auth.set_level(AgentLevel.OFF)
    for method, path in (
        ("GET", "/capabilities"),
        ("GET", "/app/workflow/status"),
        ("GET", "/microscope/devices/stage/position"),
        ("POST", "/app/agent/notes"),
    ):
        response = client.request(method, path, headers=AGENT, json={"text": "x"})
        assert response.status_code == 403, path
        assert _error(response) == "agent_off"
    # The agent is still heard from: the operator can see it is knocking.
    assert auth.seconds_since_seen() is not None

    assert client.get("/app/workflow/status", headers=DASHBOARD).status_code == 200
    assert client.get("/capabilities", headers=DASHBOARD).status_code == 200


def test_back_from_off_to_watch(client, auth):
    auth.set_level(AgentLevel.OFF)
    auth.set_level(AgentLevel.WATCH)
    assert client.get("/app/workflow/status", headers=AGENT).status_code == 200


def test_each_level_reaches_itself_and_those_below():
    auth = AuthConfig.generate()
    assert auth.level is AgentLevel.WATCH
    reach = {
        AgentLevel.OFF: (),
        AgentLevel.WATCH: (AgentLevel.WATCH,),
        AgentLevel.ASSIST: (AgentLevel.WATCH, AgentLevel.ASSIST),
        AgentLevel.DRIVE: (AgentLevel.WATCH, AgentLevel.ASSIST, AgentLevel.DRIVE),
    }
    for level, reached in reach.items():
        auth.set_level(level)
        for needed in (AgentLevel.WATCH, AgentLevel.ASSIST, AgentLevel.DRIVE):
            assert auth.reaches(needed) is (needed in reached), (level, needed)


def test_assist_asks_first_to_stop_what_is_not_its_own(client, auth):
    """A wrong stop isn't free (Patrick, 2026-10-14): at Assist the agent asks
    the operator to stop the workflow or someone else's mill, as it asks to act."""
    auth.set_level(AgentLevel.ASSIST)
    for path in (STOP_MILLING, "/app/workflow/stop"):
        refused = client.post(path, headers=AGENT, json=WHY)
        assert refused.status_code == 403, path
        assert _error(refused) == "ask_first"
    moved = client.post(MOVE_RELATIVE, headers=AGENT, json=_NUDGE)
    assert moved.status_code == 403
    assert _error(moved) == "ask_first"


def test_the_agent_stops_its_own_action_at_any_level(client, auth, microscope):
    """While the agent's own command holds control (its mill, say), it can stop
    it at Watch; the workflow is never its own."""
    with microscope.control.take(AGENT_ACTOR, "milling run"):
        assert client.post(STOP_MILLING, headers=AGENT, json=WHY).status_code == 200
        refused = client.post("/app/workflow/stop", headers=AGENT, json=WHY)
    assert refused.status_code == 403
    with microscope.control.take(WORKFLOW, "a lamella run"):
        refused = client.post(STOP_MILLING, headers=AGENT, json=WHY)
    assert refused.status_code == 403
    assert _error(refused) == "level"


def test_a_stop_needs_a_reason_and_records_it(client, auth, microscope):
    auth.set_level(AgentLevel.DRIVE)
    for path in (STOP_MILLING, "/app/workflow/stop"):
        for body in (None, {}, {"reason": "  "}):
            refused = client.post(path, headers=AGENT, json=body)
            assert refused.status_code == 422, (path, body)
            assert _error(refused) == "missing_reason"
    recorded = []
    microscope.record_signal.connect(
        lambda kind, payload: recorded.append((kind, payload))
    )
    assert client.post(STOP_MILLING, headers=AGENT, json=WHY).status_code == 200
    assert client.post("/app/workflow/stop", headers=AGENT, json=WHY).status_code == 200
    stops = [payload for kind, payload in recorded if kind == "agent_stop"]
    assert stops == [
        {"action": "milling stop", "reason": WHY["reason"], "own": False},
        {"action": "workflow", "reason": WHY["reason"], "own": False},
    ]


def test_stop_agent_drops_to_watch_and_says_so(client, auth):
    auth.set_level(AgentLevel.DRIVE)
    auth.stop_agent()
    assert auth.level is AgentLevel.WATCH
    # It still sees, so it can report what happened.
    assert client.get("/app/workflow/status", headers=AGENT).status_code == 200
    refused = client.post("/app/workflow/stop", headers=AGENT, json=WHY)
    assert refused.status_code == 403
    assert _error(refused) == "agent_stopped"

    # Raising the level again is the operator's say-so: no longer "stopped".
    auth.set_level(AgentLevel.DRIVE)
    assert client.post("/app/workflow/stop", headers=AGENT, json=WHY).status_code == 200
    auth.set_level(AgentLevel.WATCH)
    refused = client.post("/app/workflow/stop", headers=AGENT, json=WHY)
    assert _error(refused) == "level"
