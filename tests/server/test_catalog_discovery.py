"""Contract tests: the tool catalog maps onto real routes, and the discovery
file round-trips."""

import json
import os
import re

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fibsem import utils  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.catalog import CATALOG, NOT_TOOLS  # noqa: E402
from fibsem.server.discovery import (  # noqa: E402
    read_discovery_file,
    remove_discovery_file,
    write_discovery_file,
)


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    # Mounted with a real (empty-host) AutoLamellaAppContext so the contract test can
    # prove the app-router catalog entries dispatch too.
    from fibsem.applications.autolamella.server import AutoLamellaAppContext

    class _Host:
        experiment = None
        microscope = None
        _task_manager = None
        is_workflow_running = False

    return build_server(
        microscope,
        app_context=AutoLamellaAppContext(_Host()),
        auth=AuthConfig(token="t"),
    )


def test_every_catalog_entry_is_a_real_route(app):
    # Dispatch a real request per entry instead of introspecting app.routes:
    # route-object metadata shapes vary across FastAPI/Starlette versions (the
    # introspection form of this test passed locally and failed on CI's newer
    # FastAPI), but dispatch is version-proof. Any status except 404/405 proves
    # the (method, path) pair exists — 401 is the expected answer, since no
    # token is sent.
    from fastapi.testclient import TestClient

    client = TestClient(app, raise_server_exceptions=False)
    missing = [
        (t.name, t.method, t.path, resp.status_code)
        for t in CATALOG
        for resp in [client.request(t.method, t.path)]
        if resp.status_code in (404, 405)
    ]
    assert missing == []


def _route_key(method: str, path: str) -> str:
    return f"{method.upper()} {re.sub(r'{[^}]+}', '{}', path)}"


def test_every_route_is_a_tool_or_says_why_not(app):
    """The reverse of the test above: a route added without a tool fails here
    until someone decides, and NOT_TOOLS says why each one isn't a tool. The
    OpenAPI schema is FastAPI's own listing, the same across versions."""
    tools = {_route_key(t.method, t.path) for t in CATALOG}
    not_tools = {_route_key(*key.split(" ", 1)) for key in NOT_TOOLS}
    routes = {
        _route_key(method, path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
    }
    assert sorted(routes - tools - not_tools) == []
    # An entry that is a tool now, or names no route, is stale.
    assert sorted(not_tools & tools) == []
    assert sorted(not_tools - routes) == []
    assert all(reason.strip() for reason in NOT_TOOLS.values())


def test_catalog_names_unique_and_levels_valid():
    names = [t.name for t in CATALOG]
    assert len(names) == len(set(names))
    assert set(t.level for t in CATALOG) <= {"watch", "assist", "drive"}
    # A stop needs Drive (Patrick, 2026-10-14), so at Assist it goes in a
    # request; the agent's own mill it stops at any level, which the note says.
    by_name = {t.name: t for t in CATALOG}
    assert by_name["stop_milling"].level == "drive"
    assert "Assist or Drive, never at Watch" in by_name["stop_milling"].level_note
    assert by_name["stop_workflow"].level == "drive"
    # Only what drives the microscope takes control of it.
    assert all(t.level == "drive" for t in CATALOG if t.hardware)


def test_discovery_round_trip(tmp_path):
    auth = AuthConfig.generate(level="drive", token="secret")
    path = tmp_path / "agent-server.json"
    write_discovery_file("0.0.0.0", 8001, auth, path=path)
    data = read_discovery_file(path)
    assert data is not None
    assert data["url"] == "http://127.0.0.1:8001"  # 0.0.0.0 is not connectable
    assert data["token"] == "secret"
    assert "level" not in data  # live in /capabilities, never stale here
    if os.name != "nt":
        assert (path.stat().st_mode & 0o777) == 0o600
    remove_discovery_file(path)
    assert read_discovery_file(path) is None


def test_discovery_rejects_stale_or_malformed(tmp_path):
    path = tmp_path / "agent-server.json"
    path.write_text("not json")
    assert read_discovery_file(path) is None
    path.write_text(json.dumps({"url": "http://x", "token": "t"}))  # no pid
    assert read_discovery_file(path) is None
    assert read_discovery_file(tmp_path / "absent.json") is None


def test_discovery_finds_the_current_name_then_the_legacy_one(tmp_path, monkeypatch):
    # The App Server rename (FIB-1035): the server writes app-server.json, and a
    # client still finds a server that wrote agent-server.json, for one release.
    from fibsem.server import discovery

    current = tmp_path / "app-server.json"
    legacy = tmp_path / "agent-server.json"
    monkeypatch.setattr(discovery, "DISCOVERY_FILE", current)
    monkeypatch.setattr(discovery, "LEGACY_DISCOVERY_FILE", legacy)
    assert discovery.read_discovery_file() is None

    write_discovery_file("127.0.0.1", 8001, AuthConfig(token="old"), path=legacy)
    assert discovery.read_discovery_file()["token"] == "old"

    write_discovery_file("127.0.0.1", 8002, AuthConfig(token="new"), path=current)
    assert discovery.read_discovery_file()["token"] == "new"

    # Writing and removing with no path touch only the current name.
    remove_discovery_file()
    assert not current.exists() and legacy.exists()
    write_discovery_file("127.0.0.1", 8003, AuthConfig(token="again"))
    assert current.exists()
    assert discovery.read_discovery_file()["token"] == "again"
