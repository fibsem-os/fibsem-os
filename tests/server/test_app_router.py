"""The app router end to end: a real AgentContext over a real experiment,
mounted into the server, reached over HTTP and through MCP tools."""

import asyncio
import os

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.server import AgentContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


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
def host(tmp_path, microscope):
    host = Host()
    exp = Experiment(path=tmp_path / "exp", name="router-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    host.experiment = exp
    host.microscope = microscope
    return host


@pytest.fixture
def client(microscope, host):
    app = build_server(
        microscope, app_context=AgentContext(host), auth=AuthConfig(token=TOKEN)
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_capabilities_reports_the_app_router(client):
    body = client.get("/capabilities", headers=AUTH).json()
    assert body["routers"] == {"microscope": True, "app": True}


def test_without_app_context_the_routes_do_not_exist(microscope):
    app = build_server(microscope, auth=AuthConfig(token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as bare:
        assert bare.get("/capabilities", headers=AUTH).json()["routers"]["app"] is False
        assert bare.get("/app/status", headers=AUTH).status_code == 404


def test_app_routes_require_a_token(client):
    assert client.get("/app/status").status_code == 401


def test_status_and_queue_over_http(client):
    status = client.get("/app/status", headers=AUTH).json()
    assert status["experiment"]["name"] == "router-exp"
    assert status["workflow"]["running"] is False
    queue = client.get("/app/queue", headers=AUTH).json()
    assert queue["available"] is False  # no run yet


def test_task_outputs_round_trip(client, host):
    name = host.experiment.positions[0].name
    payload = client.get(f"/app/task_outputs/{name}", headers=AUTH).json()
    assert payload["available"] is True
    assert payload["item_name"] == name
    missing = client.get("/app/task_outputs/nope", headers=AUTH).json()
    assert missing["available"] is False


def test_summaries_and_protocol_over_http(client):
    for path in ("/app/experiment_summary", "/app/task_history", "/app/protocol"):
        body = client.get(path, headers=AUTH).json()
        assert body["available"] is True, path
    assert client.get("/app/run_summary", headers=AUTH).json()["available"] is False


def test_sidecar_grows_the_app_tools_from_capabilities(client):
    pytest.importorskip("mcp")
    from fibsem.mcp.sidecar import build_sidecar
    from fibsem.server.catalog import CATALOG

    client.headers["Authorization"] = f"Bearer {TOKEN}"
    capabilities = client.get("/capabilities").json()
    sidecar = build_sidecar(client, capabilities)
    listed = asyncio.run(sidecar.list_tools())
    names = {t.name for t in getattr(listed, "tools", listed)}
    assert {t.name for t in CATALOG if t.router == "app"} <= names

    result = asyncio.run(sidecar.call_tool("get_app_status", {}))
    if isinstance(result, tuple):
        result = result[0]
    contents = list(getattr(result, "content", result))
    text = "".join(getattr(c, "text", "") for c in contents)
    assert "router-exp" in text
