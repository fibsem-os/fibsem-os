"""End-to-end sidecar tests: MCP tool calls through the real server app.

The TestClient is an httpx.Client, so the sidecar's HTTP layer runs unchanged
against the in-process ASGI app — full stack minus the socket."""

import asyncio
import os

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.mcp.sidecar import build_sidecar  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
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


def _scopes(client):
    return client.get("/capabilities").json()["scopes"]


@pytest.fixture(scope="module")
def armed_sidecar():
    client = _make_client(arm_hardware=True)
    return build_sidecar(client, _scopes(client))


def _tool_names(sidecar):
    listed = asyncio.run(sidecar.list_tools())
    return {t.name for t in getattr(listed, "tools", listed)}


def test_armed_sidecar_registers_the_full_catalog(armed_sidecar):
    assert _tool_names(armed_sidecar) == {t.name for t in CATALOG}


def test_read_only_sidecar_registers_read_tools_only():
    client = _make_client(arm_hardware=False)
    sidecar = build_sidecar(client, _scopes(client))
    names = _tool_names(sidecar)
    assert names == {t.name for t in CATALOG if t.scope == "read"}
    assert "stop_milling" in names
    assert "acquire_image_preview" not in names


def test_read_tool_round_trip(armed_sidecar):
    result = asyncio.run(armed_sidecar.call_tool("get_stage_position", {}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert '"position"' in text


def test_move_tool_round_trip(armed_sidecar):
    result = asyncio.run(armed_sidecar.call_tool("move_stage_relative", {"dx": 1e-6}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert '"position"' in text


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
    armed_shape = {"read": True, "control": False, "hardware": True}
    sidecar = build_sidecar(read_client, armed_shape)
    result = asyncio.run(sidecar.call_tool("move_stage_relative", {"dx": 1e-6}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert "scope_not_armed" in text


def _contents(result):
    # FastMCP.call_tool returns a content sequence; newer versions may return
    # (contents, structured). Normalize.
    if isinstance(result, tuple):
        result = result[0]
    return list(getattr(result, "content", result))
