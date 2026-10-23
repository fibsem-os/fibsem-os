"""The App Server's devices service (FIB-1035): off, nothing that reaches the
instrument is registered; on, everything is."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.ai.agent.catalog import tools_for_capabilities  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

# One route of each kind that reaches the instrument: who it is, its state, a
# device read, a stop, a command, and the listings.
INSTRUMENT_ROUTES = [
    ("GET", "/microscope"),
    ("GET", "/microscope/state"),
    ("GET", "/microscope/devices/stage/position"),
    ("POST", "/microscope/services/milling/commands/stop"),
    ("POST", "/microscope/services/stage_movement/commands/move_relative"),
    ("GET", "/microscope/devices"),
    ("GET", "/microscope/services"),
]


def _client(microscope, **kwargs) -> TestClient:
    return TestClient(build_server(microscope, auth=AuthConfig(token=TOKEN), **kwargs))


def test_microscope_off_registers_no_instrument_route(microscope):
    client = _client(microscope, serve_microscope=False)
    for method, path in INSTRUMENT_ROUTES:
        response = client.request(method, path, headers=AUTH)
        assert response.status_code == 404, (method, path)
    assert client.get("/health").status_code == 200

    capabilities = client.get("/capabilities", headers=AUTH).json()
    assert capabilities["routers"]["microscope"] is False


def test_microscope_off_says_why(microscope):
    """FIB-1359: a bare "Not Found" read as a wrong path; the 404 names the
    operator's switch instead."""
    client = _client(microscope, serve_microscope=False)
    for method, path in INSTRUMENT_ROUTES + [
        ("POST", "/microscope/services/imaging/commands/focus_sweep"),
        ("PUT", "/microscope/devices/electron/voltage"),
    ]:
        response = client.request(method, path, headers=AUTH)
        assert response.status_code == 404, (method, path)
        detail = response.json()["detail"]
        assert detail["error_type"] == "microscope_off", (method, path)
        assert "Tools → App Server" in detail["message"]
    # A path that never existed is still a plain 404.
    missing = client.get("/no_such_route", headers=AUTH)
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Not Found"


def test_microscope_off_still_needs_the_token(microscope):
    client = _client(microscope, serve_microscope=False)
    assert client.get("/capabilities").status_code == 401
    assert client.get("/microscope/devices/stage/position").status_code == 401


def test_the_instrument_tools_are_offered_with_the_microscope_off(microscope):
    """The operator can turn the microscope routes on mid-session, so the agent
    has their tools from the start; until then each answers microscope_off."""
    client = _client(microscope, serve_microscope=False)
    capabilities = client.get("/capabilities", headers=AUTH).json()
    names = {t.name for t in tools_for_capabilities(capabilities)}
    assert {"stage_get_position", "stage_stable_move", "device_list"} <= names
    # An app tool still needs a server that hosts an application.
    assert "app_get_status" not in names


def test_get_capabilities_is_offered_with_the_microscope_off(microscope):
    """The agent's first read is its level, so it must exist whatever the
    services (FIB-1035 app test #1)."""
    client = _client(microscope, serve_microscope=False)
    capabilities = client.get("/capabilities", headers=AUTH).json()
    names = {t.name for t in tools_for_capabilities(capabilities)}
    assert "server_get_capabilities" in names


def test_the_instrument_is_served_by_default(microscope):
    client = _client(microscope)
    assert client.get("/microscope", headers=AUTH).status_code == 200
    assert client.get("/microscope/devices", headers=AUTH).status_code == 200
    capabilities = client.get("/capabilities", headers=AUTH).json()
    assert capabilities["routers"]["microscope"] is True
