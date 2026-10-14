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


def _make_client(level):
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    app = build_server(microscope, auth=AuthConfig.generate(level=level, token=TOKEN))
    client = TestClient(app, raise_server_exceptions=False)
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    return client


def _capabilities(client):
    return client.get("/capabilities").json()


@pytest.fixture(scope="module")
def armed_sidecar():
    client = _make_client("drive")
    return build_sidecar(client, _capabilities(client))


def _tool_names(sidecar):
    listed = asyncio.run(sidecar.list_tools())
    return {t.name for t in getattr(listed, "tools", listed)}


def test_armed_sidecar_registers_the_full_microscope_catalog(armed_sidecar):
    # No app_context on this server, so app tools must not register even armed.
    expected = {t.name for t in CATALOG if t.router in ("microscope", "base")}
    assert _tool_names(armed_sidecar) == expected


def test_hardware_tools_are_offered_at_watch():
    """Tools don't follow the level (FIB-1035): the server's 403 is the gate,
    so hardware tools are offered at Watch too, and refused there."""
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
    assert "level" in text


def test_each_description_names_the_level_it_needs(armed_sidecar):
    listed = asyncio.run(armed_sidecar.list_tools())
    described = {t.name: t.description for t in getattr(listed, "tools", listed)}
    assert described["get_stage_position"].endswith("Works at Watch, Assist and Drive.")
    assert "started itself it can stop at any level" in described["stop_milling"]
    assert "Needs the agent at Drive" in described["move_stage_relative"]
    assert "request_actions" in described["move_stage_relative"]
    assert "(409)" in described["move_stage_relative"]


def test_read_tool_round_trip(armed_sidecar):
    result = asyncio.run(armed_sidecar.call_tool("get_stage_position", {}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert '"position"' in text


def test_move_tool_round_trip(armed_sidecar):
    result = asyncio.run(armed_sidecar.call_tool("move_stage_relative", {"dx": 1e-6}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert '"position"' in text


def test_set_parameter_takes_a_number_sent_as_text(armed_sidecar):
    """The value's schema names no type, so a client may send 120e-6 as text."""
    result = asyncio.run(
        armed_sidecar.call_tool(
            "set_parameter",
            {"device": "electron", "parameter": "hfw", "value": "120e-6"},
        )
    )
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert json.loads(text)["value"] == pytest.approx(120e-6)


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
    # Registered tools + a server at Watch = the server refuses; the agent should
    # see the structured refusal as text, not a raised exception.
    read_client = _make_client("watch")
    armed_shape = {"routers": {"microscope": True, "app": False}, "level": "drive"}
    sidecar = build_sidecar(read_client, armed_shape)
    result = asyncio.run(sidecar.call_tool("move_stage_relative", {"dx": 1e-6}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert "level" in text


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


def _call_json(sidecar, tool, arguments):
    result = asyncio.run(sidecar.call_tool(tool, arguments))
    return json.loads("".join(getattr(c, "text", "") for c in _contents(result)))


def test_the_milling_angle_tools_speak_degrees(armed_sidecar):
    """The tools speak degrees; the stage movement command they call takes
    radians (FIB-853). Moving to 15 deg must read back as ~15 deg, not ~859."""
    moved = _call_json(
        armed_sidecar, "move_to_milling_angle", {"milling_angle_deg": 15.0}
    )
    assert moved == {"success": True}
    angle = _call_json(armed_sidecar, "get_milling_angle", {})
    assert angle["milling_angle_deg"] == pytest.approx(15.0, abs=0.5)
    orientation = _call_json(armed_sidecar, "get_stage_orientation", {})
    assert orientation == {"orientation": "MILLING"}


def _contents(result):
    # FastMCP.call_tool returns a content sequence; newer versions may return
    # (contents, structured). Normalize.
    if isinstance(result, tuple):
        result = result[0]
    return list(getattr(result, "content", result))


@pytest.mark.parametrize("device", ["electron", "chamber_camera"])
def test_a_command_that_images_answers_the_agent_an_image(armed_sidecar, device):
    """A beam's image or a camera frame from call_command comes back as a
    summary and an image the agent can look at, not bytes it can't read."""
    result = asyncio.run(
        armed_sidecar.call_tool(
            "call_command", {"device": device, "command": "acquire"}
        )
    )
    contents = _contents(result)
    assert any(getattr(c, "type", None) == "image" for c in contents)
    text = "".join(getattr(c, "text", "") for c in contents)
    answer = json.loads(text)
    assert answer["result"]["summary"]["shape"]
    assert answer["result"]["preview"]["image_index"] == 0
