"""Contract tests: the tool catalog maps onto real routes, and the discovery
file round-trips."""

import json
import os
import re

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fibsem import utils  # noqa: E402
from fibsem.ai.agent.catalog import CATALOG, GROUP_TITLES, NOT_TOOLS  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
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
    from fibsem.workflows.session import AppSession

    return build_server(
        microscope,
        app_context=AutoLamellaAppContext(AppSession()),
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


# A named service tool is its group and its route's command or query
# (``fm_acquire_overview`` -> ``imaging.fm_acquire_overview``,
# ``beam_acquire_overview`` -> ``imaging.acquire_overview``), a read with ``get_``
# before the query's name. These keep an older name, each for a reason.
NAMED_OTHERWISE = {
    # The beams' own command names, which the imaging service passes through.
    "beam_acquire_image": "acquire",
    "beam_auto_contrast": "autocontrast",
    # The service's raw move holds move_absolute (blocked for the agent);
    # renaming it would change every caller's move without a word.
    "stage_move_absolute": "safe_absolute_stage_movement",
    "stage_get_orientation": "stage_orientation",
    # The sample handling service's names are what the grid workflows and
    # scripts call; the tools say what they do in the grid group's words.
    "grid_load": "ensure_loaded",
    "grid_move_to": "move_to_grid",
}


def test_each_service_tool_is_named_for_its_route():
    mismatched = {}
    for tool in CATALOG:
        match = re.match(
            r"/microscope/services/\w+/(commands|queries)/(\w+)$", tool.path
        )
        if match is None:
            continue
        kind, route = match.groups()
        rest = tool.name.split("_", 1)[1]
        if route in (tool.name, rest) or (kind == "queries" and rest == f"get_{route}"):
            continue
        mismatched[tool.name] = route
    assert mismatched == NAMED_OTHERWISE


def test_catalog_names_unique_and_levels_valid():
    names = [t.name for t in CATALOG]
    assert len(names) == len(set(names))
    assert set(t.level for t in CATALOG) <= {"watch", "assist", "drive"}
    # A stop needs Drive (Patrick, 2026-10-14), so at Assist it goes in a
    # request; the agent's own mill it stops at any level, which the note says.
    by_name = {t.name: t for t in CATALOG}
    assert by_name["milling_stop"].level == "drive"
    assert "Assist or Drive, never at Watch" in by_name["milling_stop"].level_note
    assert by_name["workflow_stop"].level == "drive"
    # Only what drives the microscope takes control of it.
    assert all(t.level == "drive" for t in CATALOG if t.hardware)


# Group first, then verb (plans: mcp-tool-naming): sorted, a group reads as one
# block. A new tool joins a group in the catalog, or the group is added on purpose.
GROUPS = tuple(GROUP_TITLES)


def test_a_write_that_only_adds_is_not_destructive():
    by_name = {t.name: t for t in CATALOG}
    assert by_name["item_update"].destructive
    assert by_name["workflow_stop"].destructive
    assert not by_name["item_add"].destructive
    assert not by_name["request_create"].destructive
    assert not by_name["stage_get_position"].destructive


def test_every_tool_name_starts_with_its_group():
    stray = [t.name for t in CATALOG if t.name.split("_", 1)[0] not in GROUPS]
    assert stray == []
    # Every <group>_get(name) has a <group>_list giving the names it takes.
    names = {t.name for t in CATALOG}
    for group in ("protocol", "grid", "item", "request"):
        assert f"{group}_list" in names


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
