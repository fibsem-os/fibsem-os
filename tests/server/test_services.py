"""The App Server's devices service (FIB-1035): off, nothing that reaches the
instrument is registered; on, or on the bench server, everything is."""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.auth import Scope  # noqa: E402
from fibsem.server.catalog import tools_for_capabilities  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

# One route of each kind that reaches the instrument: a microscope read, a
# hardware command, the always-allowed stop, and the device router.
INSTRUMENT_ROUTES = [
    ("GET", "/stage_position"),
    ("GET", "/microscope_state"),
    ("GET", "/beams/electron/imaging_settings"),
    ("POST", "/stop_milling"),
    ("POST", "/move_stage_relative"),
    ("GET", "/devices"),
    ("GET", "/services"),
]


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


def _client(microscope, **kwargs) -> TestClient:
    return TestClient(build_server(microscope, auth=AuthConfig(token=TOKEN), **kwargs))


def test_devices_off_registers_no_instrument_route(microscope):
    client = _client(microscope, devices=False)
    for method, path in INSTRUMENT_ROUTES:
        response = client.request(method, path, headers=AUTH)
        assert response.status_code == 404, (method, path)
    assert client.get("/health").status_code == 200

    capabilities = client.get("/capabilities", headers=AUTH).json()
    assert capabilities["routers"]["microscope"] is False
    assert capabilities["routers"]["devices"] is False


def test_devices_off_says_why(microscope):
    """FIB-1359: a bare "Not Found" read as a wrong path; the 404 names the
    operator's switch instead."""
    client = _client(microscope, devices=False)
    for method, path in INSTRUMENT_ROUTES + [
        ("POST", "/services/imaging/commands/focus_sweep"),
        ("PUT", "/devices/electron/voltage"),
    ]:
        response = client.request(method, path, headers=AUTH)
        assert response.status_code == 404, (method, path)
        detail = response.json()["detail"]
        assert detail["error_type"] == "devices_off", (method, path)
        assert "Tools → App Server" in detail["message"]
    # A path that never existed is still a plain 404.
    missing = client.get("/no_such_route", headers=AUTH)
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Not Found"


def test_devices_off_still_needs_the_token(microscope):
    client = _client(microscope, devices=False)
    assert client.get("/capabilities").status_code == 401
    assert client.get("/stage_position").status_code == 401


def test_the_agent_is_offered_no_instrument_tool_with_devices_off(microscope):
    client = _client(microscope, devices=False)
    capabilities = client.get("/capabilities", headers=AUTH).json()
    capabilities["scopes"] = {s.value: True for s in Scope}
    assert not [
        t.path for t in tools_for_capabilities(capabilities) if t.router == "microscope"
    ]


def test_get_capabilities_is_offered_with_devices_off(microscope):
    """The agent's first read is its level, so it must exist whatever the
    services (FIB-1035 app test #1)."""
    client = _client(microscope, devices=False)
    capabilities = client.get("/capabilities", headers=AUTH).json()
    names = {t.name for t in tools_for_capabilities(capabilities)}
    assert "get_capabilities" in names


def test_the_bench_server_serves_the_instrument_by_default(microscope):
    client = _client(microscope)
    assert client.get("/stage_position", headers=AUTH).status_code == 200
    assert client.get("/devices", headers=AUTH).status_code == 200
    capabilities = client.get("/capabilities", headers=AUTH).json()
    assert capabilities["routers"]["microscope"] is True
    assert capabilities["routers"]["devices"] is True
