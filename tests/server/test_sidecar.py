"""End-to-end sidecar tests: MCP tool calls through the real server app.

The TestClient is an httpx.Client, so the sidecar's HTTP layer runs unchanged
against the in-process ASGI app — full stack minus the socket."""

import asyncio
import json
import os

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.mcp.sidecar import build_sidecar  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.auth import AgentLevel  # noqa: E402
from fibsem.server.catalog import CATALOG  # noqa: E402

TOKEN = "test-token"


def _make_client(arm_hardware):
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    app = build_server(
        microscope, auth=AuthConfig.generate(arm_hardware=arm_hardware, token=TOKEN)
    )
    client = TestClient(app, raise_server_exceptions=False)
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    return client


def _capabilities(client):
    return client.get("/capabilities").json()


@pytest.fixture(scope="module")
def armed_sidecar():
    client = _make_client(arm_hardware=True)
    return build_sidecar(client, _capabilities(client))


def _tool_names(sidecar):
    listed = asyncio.run(sidecar.list_tools())
    return {t.name for t in getattr(listed, "tools", listed)}


def test_armed_sidecar_registers_the_full_microscope_catalog(armed_sidecar):
    # No app_context on this server, so app tools must not register even armed.
    expected = {t.name for t in CATALOG if t.router in ("microscope", "base")}
    assert _tool_names(armed_sidecar) == expected


def test_a_bench_server_without_hardware_leaves_the_hardware_tools_out():
    """Tools don't follow the scopes a level arms (FIB-1035): the server's 403
    is the gate. A bench server has no level, and only a restart arms its
    hardware, so its hardware tools are left out."""
    client = _make_client(arm_hardware=False)
    sidecar = build_sidecar(client, _capabilities(client))
    names = _tool_names(sidecar)
    assert names == {
        t.name
        for t in CATALOG
        if t.router in ("microscope", "base") and not t.hardware_only
    }
    assert "stop_milling" in names  # refused by the server, not hidden
    assert "acquire_image_preview" not in names


def test_a_server_with_a_level_offers_hardware_tools_at_watch():
    """The App Server: Drive arms hardware (FIB-1267), so hardware tools are
    offered at Watch too, and refused there by the server."""
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    auth = AuthConfig(token=TOKEN)
    auth.set_level(AgentLevel.WATCH)
    client = TestClient(
        build_server(microscope, auth=auth), raise_server_exceptions=False
    )
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    sidecar = build_sidecar(client, _capabilities(client))
    assert "move_stage_relative" in _tool_names(sidecar)
    result = asyncio.run(sidecar.call_tool("move_stage_relative", {"dx": 1e-6}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert "scope_not_armed" in text


def test_each_description_names_the_level_it_needs(armed_sidecar):
    listed = asyncio.run(armed_sidecar.list_tools())
    described = {t.name: t.description for t in getattr(listed, "tools", listed)}
    assert described["get_stage_position"].endswith("Works at Watch and Drive.")
    assert "Needs the agent at Drive" in described["stop_milling"]
    assert "Needs the agent at Drive" in described["move_stage_relative"]
    assert "(409)" in described["move_stage_relative"]


def test_read_tool_round_trip(armed_sidecar):
    result = asyncio.run(armed_sidecar.call_tool("get_stage_position", {}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert '"position"' in text


def test_move_tool_round_trip(armed_sidecar):
    result = asyncio.run(armed_sidecar.call_tool("move_stage_relative", {"dx": 1e-6}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert '"position"' in text


def test_stable_move_tool_round_trip(armed_sidecar):
    before = asyncio.run(armed_sidecar.call_tool("get_stage_position", {}))
    before = json.loads("".join(getattr(c, "text", "") for c in _contents(before)))
    result = asyncio.run(
        armed_sidecar.call_tool(
            "stable_move", {"dx": 5e-6, "dy": 0.0, "beam_type": "ELECTRON"}
        )
    )
    after = json.loads("".join(getattr(c, "text", "") for c in _contents(result)))
    assert after["position"]["x"] != pytest.approx(before["position"]["x"])


def test_image_tool_returns_image_content(armed_sidecar):
    result = asyncio.run(
        armed_sidecar.call_tool("acquire_image_preview", {"beam_type": "ION"})
    )
    contents = _contents(result)
    types = {getattr(c, "type", None) for c in contents}
    assert "image" in types
    image = next(c for c in contents if getattr(c, "type", None) == "image")
    assert getattr(image, "mime_type", None) == "image/jpeg"


def test_hardware_refusal_is_readable_text_not_an_error():
    # Registered tools + unarmed server = the server refuses; the agent should
    # see the structured refusal as text, not a raised exception.
    read_client = _make_client(arm_hardware=False)
    armed_shape = {
        "scopes": {"read": True, "control": False, "hardware": True},
        "routers": {"microscope": True, "app": False},
    }
    sidecar = build_sidecar(read_client, armed_shape)
    result = asyncio.run(sidecar.call_tool("move_stage_relative", {"dx": 1e-6}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert "scope_not_armed" in text


def test_service_query_tool_round_trip(armed_sidecar):
    """FIB-1359: a service query was reachable over raw HTTP only."""
    here = asyncio.run(armed_sidecar.call_tool("get_stage_position", {}))
    position = json.loads("".join(getattr(c, "text", "") for c in _contents(here)))
    result = asyncio.run(
        armed_sidecar.call_tool(
            "call_service_query",
            {
                "service": "stage_movement",
                "query": "project_stable_move",
                "kwargs": {
                    "dx": 0.0,
                    "dy": 0.0,
                    "beam_type": 1,
                    "base_position": position["position"],
                },
            },
        )
    )
    answer = json.loads("".join(getattr(c, "text", "") for c in _contents(result)))
    assert answer["result"]["x"] == pytest.approx(position["position"]["x"])


def _contents(result):
    # FastMCP.call_tool returns a content sequence; newer versions may return
    # (contents, structured). Normalize.
    if isinstance(result, tuple):
        result = result[0]
    return list(getattr(result, "content", result))
