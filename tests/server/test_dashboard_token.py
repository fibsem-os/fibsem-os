"""The dashboard's own token (FIB-1035): monitoring reads, and nothing else."""

import pytest
from fastapi.testclient import TestClient
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.server import AuthConfig, build_server
from fibsem.server.discovery import write_discovery_file
from fibsem.structures.settings import MicroscopeState
from tests.run_state import stub_session

AGENT = "agent-token"
DASHBOARD = "dashboard-token"


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth():
    return AuthConfig(token=AGENT, dashboard_token=DASHBOARD, connect_token="c" * 43)


@pytest.fixture
def client(tmp_path, microscope, auth):
    host = stub_session()
    exp = Experiment(path=tmp_path / "exp", name="dashboard-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    host.experiment = exp
    host.microscope = microscope
    app = build_server(microscope, app_context=AutoLamellaAppContext(host), auth=auth)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


MONITORING = [
    "/capabilities",
    "/app/workflow/status",
    "/app/workflow/queue",
    "/app/events",
    "/app/events/history",  # the Events page's earlier records
    "/microscope/devices",  # what it has, not a device's values
]


@pytest.mark.parametrize("path", MONITORING)
def test_the_dashboard_token_reads_the_monitoring_routes(client, path):
    assert client.get(path, headers=_bearer(DASHBOARD)).status_code == 200
    assert client.get(path, headers=_bearer(AGENT)).status_code == 200


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", "/app/workflow/stop"),  # on the read router, but it acts
        ("POST", "/app/workflow/decide"),  # control
        ("POST", "/app/workflow/schedule"),  # configure
        ("GET", "/microscope/devices/stage/position"),  # the microscope
    ],
)
def test_the_dashboard_token_reaches_nothing_else(client, method, path):
    response = client.request(method, path, headers=_bearer(DASHBOARD), json={})
    assert response.status_code == 403
    assert response.json()["detail"]["error_type"] == "dashboard_read_only"


def test_an_unknown_token_is_still_unauthorized(client):
    assert (
        client.get("/app/workflow/status", headers=_bearer("nope")).status_code == 401
    )
    assert client.get("/microscope", headers=_bearer("nope")).status_code == 401


def test_the_dashboard_is_not_the_agent_being_heard_from(client, auth):
    client.get("/app/workflow/status", headers=_bearer(DASHBOARD))
    assert auth.seconds_since_seen() is None
    client.get("/app/workflow/status", headers=_bearer(AGENT))
    assert auth.seconds_since_seen() is not None


def test_a_bare_auth_config_gives_no_dashboard_access(microscope):
    app = build_server(microscope, auth=AuthConfig(token=AGENT))
    with TestClient(app) as bare:
        assert bare.get("/capabilities", headers=_bearer("None")).status_code == 401
    assert AuthConfig.generate().dashboard_token is not None


def test_the_discovery_file_carries_only_the_connect_token(tmp_path, auth):
    path = write_discovery_file("127.0.0.1", 8001, auth, path=tmp_path / "d.json")
    text = path.read_text()
    assert auth.connect_token in text
    assert AGENT not in text and DASHBOARD not in text


def test_the_sidecar_names_the_agent_and_other_clients_keep_it(client, auth):
    assert auth.agent_client is None
    sidecar_ua = "fibsem-mcp/0.6.0 claude-code/2.1.3"
    client.get(
        "/app/workflow/status", headers={**_bearer(AGENT), "User-Agent": sidecar_ua}
    )
    assert auth.agent_client == "claude-code/2.1.3"
    # fibsem-mcp watch and curl are the same agent's other hands.
    client.get(
        "/app/workflow/status", headers={**_bearer(AGENT), "User-Agent": "curl/8.5"}
    )
    assert auth.agent_client == "claude-code/2.1.3"
    # The dashboard is never the agent.
    client.get(
        "/app/workflow/status",
        headers={**_bearer(DASHBOARD), "User-Agent": "fibsem-mcp/0.6.0 x/1"},
    )
    assert auth.agent_client == "claude-code/2.1.3"
