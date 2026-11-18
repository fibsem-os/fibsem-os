"""Who each token is: the agent's, the operator's, and the discovery file's."""

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
OPERATOR = "operator-token"


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth():
    return AuthConfig(token=AGENT, operator_token=OPERATOR, connect_token="c" * 43)


@pytest.fixture
def client(tmp_path, microscope, auth):
    host = stub_session()
    exp = Experiment(path=tmp_path / "exp", name="tokens-exp")
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
def test_the_agent_and_the_operator_read_the_monitoring_routes(client, path):
    assert client.get(path, headers=_bearer(AGENT)).status_code == 200
    assert client.get(path, headers=_bearer(OPERATOR)).status_code == 200


def test_an_unknown_token_is_still_unauthorized(client):
    assert (
        client.get("/app/workflow/status", headers=_bearer("nope")).status_code == 401
    )
    assert client.get("/microscope", headers=_bearer("nope")).status_code == 401


def test_the_operator_is_not_the_agent_being_heard_from(client, auth):
    client.get("/app/workflow/status", headers=_bearer(OPERATOR))
    assert auth.seconds_since_seen() is None
    client.get("/app/workflow/status", headers=_bearer(AGENT))
    assert auth.seconds_since_seen() is not None


def test_the_discovery_file_carries_only_the_connect_token(tmp_path, auth):
    path = write_discovery_file("127.0.0.1", 8001, auth, path=tmp_path / "d.json")
    text = path.read_text()
    assert auth.connect_token in text
    assert AGENT not in text and OPERATOR not in text


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
    # The operator is never the agent.
    client.get(
        "/app/workflow/status",
        headers={**_bearer(OPERATOR), "User-Agent": "fibsem-mcp/0.6.0 x/1"},
    )
    assert auth.agent_client == "claude-code/2.1.3"
