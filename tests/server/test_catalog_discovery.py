"""Contract tests: the tool catalog maps onto real routes, and the discovery
file round-trips."""

import json
import os

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fibsem import utils  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.catalog import CATALOG, tools_for_scopes  # noqa: E402
from fibsem.server.discovery import (  # noqa: E402
    read_discovery_file,
    remove_discovery_file,
    write_discovery_file,
)


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return build_server(microscope, auth=AuthConfig(token="t"))


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


def test_catalog_names_unique_and_scopes_valid():
    names = [t.name for t in CATALOG]
    assert len(names) == len(set(names))
    assert set(t.scope for t in CATALOG) <= {"read", "hardware"}


def test_tools_for_scopes_filters_hardware():
    read_only = tools_for_scopes({"read": True, "hardware": False})
    assert all(t.scope == "read" for t in read_only)
    assert any(t.name == "stop_milling" for t in read_only)
    armed = tools_for_scopes({"read": True, "hardware": True})
    assert len(armed) == len(CATALOG)


def test_discovery_round_trip(tmp_path):
    auth = AuthConfig.generate(arm_hardware=True, token="secret")
    path = tmp_path / "agent-server.json"
    write_discovery_file("0.0.0.0", 8001, auth, path=path)
    data = read_discovery_file(path)
    assert data is not None
    assert data["url"] == "http://127.0.0.1:8001"  # 0.0.0.0 is not connectable
    assert data["token"] == "secret"
    assert data["scopes"]["hardware"] is True
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
