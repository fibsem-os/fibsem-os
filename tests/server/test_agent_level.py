"""The agent level (FIB-1035): Off refuses the agent everything, Watch reads and
notes, Drive acts on the session. The dashboard's token ignores the level."""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.auth import AgentLevel, Scope  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402

AGENT = {"Authorization": "Bearer agent"}
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
    assert client.get("/app/status", headers=AGENT).status_code == 200
    note = client.post("/app/agent/notes", headers=AGENT, json={"text": "seen"})
    assert note.status_code == 200

    for path in ("/app/workflow/stop", "/stop_milling", "/app/prompt/answer"):
        response = client.post(path, headers=AGENT, json={})
        assert response.status_code == 403, path
        assert _error(response) == "scope_not_armed"
        assert response.json()["detail"]["level"] == "watch"


def test_drive_acts_on_the_session_but_not_the_hardware(client, auth):
    auth.set_level(AgentLevel.DRIVE)
    capabilities = client.get("/capabilities", headers=AGENT).json()
    assert capabilities["level"] == "drive"
    assert capabilities["scopes"] == {
        "read": True,
        "control": True,
        "configure": True,
        "hardware": False,
    }
    assert client.post("/app/workflow/stop", headers=AGENT).status_code == 200
    assert client.post("/stop_milling", headers=AGENT).status_code == 200
    moved = client.post("/move_stage_relative", headers=AGENT, json={"dx": 1e-6})
    assert moved.status_code == 403


def test_off_refuses_the_agent_everything_but_not_the_dashboard(client, auth):
    auth.set_level(AgentLevel.OFF)
    for method, path in (
        ("GET", "/capabilities"),
        ("GET", "/app/status"),
        ("GET", "/stage_position"),
        ("POST", "/app/agent/notes"),
    ):
        response = client.request(method, path, headers=AGENT, json={"text": "x"})
        assert response.status_code == 403, path
        assert _error(response) == "agent_off"
    # The agent is still heard from: the operator can see it is knocking.
    assert auth.seconds_since_seen() is not None

    assert client.get("/app/status", headers=DASHBOARD).status_code == 200
    assert client.get("/capabilities", headers=DASHBOARD).status_code == 200


def test_back_from_off_to_watch(client, auth):
    auth.set_level(AgentLevel.OFF)
    auth.set_level(AgentLevel.WATCH)
    assert client.get("/app/status", headers=AGENT).status_code == 200


def test_a_level_never_arms_or_disarms_hardware():
    auth = AuthConfig.generate(arm_hardware=True)
    auth.set_level(AgentLevel.WATCH)
    assert auth.is_armed(Scope.HARDWARE)
    auth = AuthConfig.generate()
    auth.set_level(AgentLevel.DRIVE)
    assert not auth.is_armed(Scope.HARDWARE)


def test_assist_arms_nothing_until_approvals_exist():
    auth = AuthConfig.generate()
    auth.set_level(AgentLevel.ASSIST)
    assert not any(auth.is_armed(s) for s in (Scope.CONTROL, Scope.CONFIGURE))


def test_a_server_without_a_level_reports_none(microscope):
    app = build_server(microscope, auth=AuthConfig(token="agent"))
    with TestClient(app) as bare:
        assert bare.get("/capabilities", headers=AGENT).json()["level"] is None


def test_stop_agent_drops_to_watch_and_says_so(client, auth):
    auth.set_level(AgentLevel.DRIVE)
    auth.stop_agent()
    assert auth.level is AgentLevel.WATCH
    # It still sees, so it can report what happened.
    assert client.get("/app/status", headers=AGENT).status_code == 200
    refused = client.post("/app/workflow/stop", headers=AGENT)
    assert refused.status_code == 403
    assert _error(refused) == "agent_stopped"

    # Raising the level again is the operator's say-so: no longer "stopped".
    auth.set_level(AgentLevel.DRIVE)
    assert client.post("/app/workflow/stop", headers=AGENT).status_code == 200
    auth.set_level(AgentLevel.WATCH)
    refused = client.post("/app/workflow/stop", headers=AGENT)
    assert _error(refused) == "scope_not_armed"
