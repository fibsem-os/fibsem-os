"""End-to-end sidecar tests: MCP tool calls through the real server app.

The TestClient is an httpx.Client, so the sidecar's HTTP layer runs unchanged
against the in-process ASGI app — full stack minus the socket."""

import asyncio
import json
import math
import os

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.ai.agent.catalog import CATALOG  # noqa: E402
from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.ai.agent.mcp.sidecar import build_sidecar  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402

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
    expected = {
        t.name for t in CATALOG if t.router in ("microscope", "base", "services")
    }
    assert _tool_names(armed_sidecar) == expected


def test_each_tool_takes_the_arguments_its_catalog_entry_names():
    """The catalog's params and optional are each tool's arguments: a request
    is checked against them at filing (FIB-1425), so they must be the
    sidecar's own, required and optional alike."""
    client = _make_client("drive")
    listed = asyncio.run(build_sidecar(client, None).list_tools())
    schemas = {
        t.name: getattr(t, "input_schema", None) or t.inputSchema
        for t in getattr(listed, "tools", listed)
    }
    for spec in CATALOG:
        schema = schemas[spec.name]
        # An acting tool also takes the evidence it cites.
        evidence = {"evidence"} if spec.takes_evidence else set()
        assert set(schema.get("properties", {})) == set(spec.params) | evidence, (
            spec.name
        )
        assert set(schema.get("required", [])) == (
            set(spec.params) - set(spec.optional)
        ), spec.name


def test_each_argument_carries_its_catalog_description():
    """The catalog's words for an argument reach the client in the input
    schema, where an agent reads them, not only in the catalog."""
    client = _make_client("drive")
    listed = asyncio.run(build_sidecar(client, None).list_tools())
    schemas = {
        t.name: getattr(t, "input_schema", None) or t.inputSchema
        for t in getattr(listed, "tools", listed)
    }
    for spec in CATALOG:
        properties = schemas[spec.name].get("properties", {})
        for name, argument in spec.arguments.items():
            assert properties[name].get("description") == argument.doc, (
                spec.name,
                name,
            )


def test_each_tool_takes_the_types_and_defaults_its_catalog_entry_declares():
    """The catalog declares each argument's type and default once
    (`ToolSpec.signature`): a tool the sidecar writes by hand takes exactly
    those, as one made from its entry does."""
    from mcp.server.mcpserver.utilities.func_metadata import func_metadata

    client = _make_client("drive")
    listed = asyncio.run(build_sidecar(client, None).list_tools())
    schemas = {
        t.name: getattr(t, "input_schema", None) or t.inputSchema
        for t in getattr(listed, "tools", listed)
    }
    for spec in CATALOG:
        properties = dict(schemas[spec.name]["properties"])
        properties.pop("evidence", None)

        def declared():
            pass

        declared.__name__ = spec.name
        declared.__signature__ = spec.signature
        expected = func_metadata(declared).arg_model.model_json_schema()
        for name, schema in properties.items():
            schema = {k: v for k, v in schema.items() if k != "description"}
            assert schema == expected["properties"][name], (spec.name, name)


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
    assert "stage_move_relative" in _tool_names(sidecar)
    assert "level" in _refused(sidecar, "stage_move_relative", {"dx": 1e-6})


def test_each_description_names_the_level_it_needs(armed_sidecar):
    listed = asyncio.run(armed_sidecar.list_tools())
    described = {t.name: t.description for t in getattr(listed, "tools", listed)}
    assert described["stage_get_position"].endswith("Works at Watch, Assist and Drive.")
    assert "started itself it can stop at Assist or Drive" in described["milling_stop"]
    assert "Needs the agent at Drive" in described["stage_move_relative"]
    assert "request_create" in described["stage_move_relative"]
    assert "(409)" in described["stage_move_relative"]


def test_each_tool_has_a_title_and_says_whether_it_changes_anything(armed_sidecar):
    listed = asyncio.run(armed_sidecar.list_tools())
    tools = {t.name: t for t in getattr(listed, "tools", listed)}
    move = tools["stage_move_absolute"]
    assert move.title == "Stage: Move absolute"
    assert move.annotations.read_only_hint is False
    assert move.annotations.destructive_hint is True
    position = tools["stage_get_position"]
    assert position.title == "Stage: Get position"
    assert position.annotations.read_only_hint is True
    assert tools["fm_acquire_image"].title == "FM: Acquire image"
    assert tools["device_set_parameter"].annotations.idempotent_hint is True
    assert move.annotations.idempotent_hint is False
    assert all(t.annotations.open_world_hint is False for t in tools.values())


def test_read_tool_round_trip(armed_sidecar):
    result = asyncio.run(armed_sidecar.call_tool("stage_get_position", {}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert '"position"' in text


def test_move_tool_round_trip(armed_sidecar):
    result = asyncio.run(armed_sidecar.call_tool("stage_move_relative", {"dx": 1e-6}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert '"position"' in text


def test_set_parameter_takes_a_number_sent_as_text(armed_sidecar):
    """The value's schema names no type, so a client may send 120e-6 as text."""
    result = asyncio.run(
        armed_sidecar.call_tool(
            "device_set_parameter",
            {"device": "electron", "parameter": "hfw", "value": "120e-6"},
        )
    )
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert json.loads(text)["value"] == pytest.approx(120e-6)


def test_set_service_parameter_sets_the_milling_angle(armed_sidecar):
    def milling_angle():
        result = asyncio.run(
            armed_sidecar.call_tool(
                "service_get_parameter",
                {"service": "stage_movement", "parameter": "milling_angle"},
            )
        )
        return json.loads("".join(getattr(c, "text", "") for c in _contents(result)))

    before = milling_angle()["value"]
    try:
        result = asyncio.run(
            armed_sidecar.call_tool(
                "service_set_parameter",
                {
                    "service": "stage_movement",
                    "parameter": "milling_angle",
                    "value": 21,
                },
            )
        )
        text = "".join(getattr(c, "text", "") for c in _contents(result))
        assert json.loads(text)["value"] == pytest.approx(21.0)
        assert milling_angle()["value"] == pytest.approx(21.0)
    finally:
        asyncio.run(
            armed_sidecar.call_tool(
                "service_set_parameter",
                {
                    "service": "stage_movement",
                    "parameter": "milling_angle",
                    "value": before,
                },
            )
        )


def test_vertical_move_works_from_either_view(armed_sidecar):
    """Coincidence from either beam: with the feature centred in the FIB, the
    offset is measured in the SEM view (beam_type ELECTRON)."""
    for arguments in (
        {"dy": 2e-6},
        {"dy": 2e-6, "beam_type": "ELECTRON", "relaxation": 0.8},
    ):
        before = _call_json(armed_sidecar, "stage_get_position", {})["position"]
        after = _call_json(armed_sidecar, "stage_vertical_move", arguments)["position"]
        assert after["z"] != pytest.approx(before["z"]), arguments


def test_stable_move_tool_round_trip(armed_sidecar):
    before = asyncio.run(armed_sidecar.call_tool("stage_get_position", {}))
    before = json.loads("".join(getattr(c, "text", "") for c in _contents(before)))
    result = asyncio.run(
        armed_sidecar.call_tool(
            "stage_stable_move", {"dx": 5e-6, "dy": 0.0, "beam_type": "ELECTRON"}
        )
    )
    after = json.loads("".join(getattr(c, "text", "") for c in _contents(result)))
    assert after["position"]["x"] != pytest.approx(before["position"]["x"])


def test_fm_stable_move_tool_reaches_the_service(armed_sidecar):
    # The demo has no FM: the service's own refusal comes back, so the tool reached it.
    text = _refused(armed_sidecar, "stage_fm_stable_move", {"dx": 5e-6, "dy": 0.0})
    assert "Fluorescence microscope is not available" in text


def test_spot_burn_estimate_tool_round_trip(armed_sidecar):
    settings = {
        "coordinates": [{"x": 0.2, "y": 0.2}, {"x": 0.8, "y": 0.8}],
        "exposure_time": 5.0,
    }
    answer = _call_json(
        armed_sidecar, "milling_estimate_spot_burn", {"settings": settings}
    )
    assert answer["seconds"] == pytest.approx(10.0)


def test_image_tool_returns_image_content(armed_sidecar):
    result = asyncio.run(
        armed_sidecar.call_tool("beam_acquire_image", {"beam_type": "ION"})
    )
    contents = _contents(result)
    types = {getattr(c, "type", None) for c in contents}
    assert "image" in types
    image = next(c for c in contents if getattr(c, "type", None) == "image")
    assert getattr(image, "mime_type", None) == "image/jpeg"


def _refused(sidecar, tool, arguments):
    """The text of a tool's refusal, which the sidecar flags an error: the MCP
    client gets is_error with this text, not a result to read as an answer."""
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError) as refused:
        asyncio.run(sidecar.call_tool(tool, arguments))
    return str(refused.value)


def test_a_hardware_refusal_is_an_error_with_the_server_s_text():
    # Registered tools + a server at Watch = the server refuses (403). The agent
    # gets it flagged an error, with the server's structured refusal as its text.
    read_client = _make_client("watch")
    armed_shape = {"routers": {"microscope": True, "app": False}, "level": "drive"}
    sidecar = build_sidecar(read_client, armed_shape)
    text = _refused(sidecar, "stage_move_relative", {"dx": 1e-6})
    assert "refused (403): " in text
    assert "level" in text


def test_service_query_tool_round_trip(armed_sidecar):
    """FIB-1359: a service query was reachable over raw HTTP only."""
    here = asyncio.run(armed_sidecar.call_tool("stage_get_position", {}))
    position = json.loads("".join(getattr(c, "text", "") for c in _contents(here)))
    result = asyncio.run(
        armed_sidecar.call_tool(
            "service_call_query",
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
    """The tools and the stage movement command all speak degrees (FIB-853).
    Moving to 15 deg must read back as ~15 deg, not ~859."""
    moved = _call_json(
        armed_sidecar, "stage_move_to_milling_angle", {"milling_angle_deg": 15.0}
    )
    assert moved == {"success": True}
    angle = _call_json(armed_sidecar, "stage_get_current_milling_angle", {})
    assert angle["milling_angle_deg"] == pytest.approx(15.0, abs=0.5)
    orientation = _call_json(armed_sidecar, "stage_get_orientation", {})
    assert orientation == {"orientation": "MILLING"}


def test_the_geometry_is_one_read(armed_sidecar):
    geometry = _call_json(armed_sidecar, "stage_get_geometry", {})["geometry"]
    assert set(geometry["views"]) == {"ELECTRON", "ION"}
    assert geometry["milling_angle"]["stored_deg"] is not None
    assert "FIB" in geometry["orientations"]


def test_the_milling_angle_move_takes_a_rotation_in_degrees(armed_sidecar):
    moved = _call_json(
        armed_sidecar,
        "stage_move_to_milling_angle",
        {"milling_angle_deg": 15.0, "rotation_deg": 10.0},
    )
    assert moved == {"success": True}
    position = _call_json(armed_sidecar, "stage_get_position", {})["position"]
    assert math.degrees(position["r"]) == pytest.approx(10.0, abs=0.1)


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
            "device_call_command", {"device": device, "command": "acquire"}
        )
    )
    contents = _contents(result)
    assert any(getattr(c, "type", None) == "image" for c in contents)
    text = "".join(getattr(c, "text", "") for c in contents)
    answer = json.loads(text)
    assert answer["result"]["summary"]["shape"]
    assert answer["result"]["preview"]["image_index"] == 0


def test_the_milling_state_is_answered_by_name(armed_sidecar):
    """The tool says IDLE, RUNNING, ...; the wire carries the enum's number."""
    from fibsem.ai.agent.mcp.sidecar import _MILLING_STATES
    from fibsem.structures import MillingState

    assert _MILLING_STATES == tuple(state.name for state in MillingState)
    assert [MillingState[name].value for name in _MILLING_STATES] == list(
        range(len(_MILLING_STATES))
    )
    result = asyncio.run(armed_sidecar.call_tool("milling_get_state", {}))
    text = "".join(getattr(c, "text", "") for c in _contents(result))
    assert json.loads(text) == {"state": "IDLE"}


def test_the_milling_angle_move_keeps_the_current_rotation(armed_sidecar):
    """With no rotation_deg the stage tilts and stays turned where it is; it
    used to swing back to the rotation reference."""
    _call_json(
        armed_sidecar,
        "stage_move_to_milling_angle",
        {"milling_angle_deg": 15.0, "rotation_deg": 10.0},
    )
    moved = _call_json(
        armed_sidecar, "stage_move_to_milling_angle", {"milling_angle_deg": 18.0}
    )
    assert moved == {"success": True}
    position = _call_json(armed_sidecar, "stage_get_position", {})["position"]
    assert math.degrees(position["r"]) == pytest.approx(10.0, abs=0.1)


def test_a_tool_taking_a_name_passes_it_on(armed_sidecar):
    """An overview's file name is `name`; it reached the route's arguments
    rather than clashing with the tool's own name (a TypeError every call)."""
    text = _refused(
        armed_sidecar,
        "beam_acquire_overview",
        {"settings": {}, "path": "/tmp/elsewhere", "name": "ov"},
    )
    assert "not_allowed" in text  # the server's answer: it got as far as the route


def test_a_server_error_is_flagged_and_says_failed():
    """A 5xx may have done part of its work (a move that failed partway has
    moved), so it is an error that says "failed", not a "refused" answer."""
    import httpx
    from mcp.server.mcpserver.exceptions import ToolError

    def answer(request):
        return httpx.Response(500, json={"detail": "the stage stopped"})

    client = httpx.Client(transport=httpx.MockTransport(answer), base_url="http://x")
    shape = {"routers": {"microscope": True, "app": False}, "level": "drive"}
    sidecar = build_sidecar(client, shape)
    with pytest.raises(ToolError, match=r"failed \(500\).*the stage stopped"):
        asyncio.run(sidecar.call_tool("stage_move_relative", {"dx": 1e-6}))


def test_an_unexpected_error_reaches_the_agent_with_its_message():
    """A tool that raises (here, the connection dropping) tells the agent what
    went wrong, not only "Error executing tool"."""
    import httpx
    from mcp.server.mcpserver.exceptions import ToolError

    def answer(request):
        raise httpx.ConnectError("the server went away")

    client = httpx.Client(transport=httpx.MockTransport(answer), base_url="http://x")
    shape = {"routers": {"microscope": True, "app": False}, "level": "drive"}
    sidecar = build_sidecar(client, shape)
    with pytest.raises(ToolError, match="ConnectError: the server went away"):
        asyncio.run(sidecar.call_tool("stage_get_position", {}))


def test_an_acting_tool_names_itself_to_the_server():
    """FIB-1426: the server describes the agent's call as a request's card does,
    from the tool and arguments the sidecar names; a read names nothing."""
    client = _make_client("drive")
    sidecar = build_sidecar(client, _capabilities(client))
    microscope = client.app.state.microscope
    recorded = []

    def record(kind, payload):
        recorded.append((kind, payload))

    microscope.session.record_signal.connect(record)
    try:
        asyncio.run(sidecar.call_tool("stage_get_position", {}))
        asyncio.run(sidecar.call_tool("stage_move_relative", {"dx": 2e-6}))
    finally:
        microscope.session.record_signal.disconnect(record)
    calls = [p for kind, p in recorded if kind == "action"]
    assert len(calls) == 1
    assert calls[0]["tool"] == "stage_move_relative"
    assert calls[0]["what"] == "Stage · Move 2 µm in x"
    assert calls[0]["level"] == "drive" and calls[0]["ok"]


def test_a_name_with_a_slash_reaches_its_route_whole():
    """An item named `Lamella 01/2` is one path parameter: the sidecar
    percent-encodes it once, in `_route`, so the server sees the whole name
    and not a longer path. Every app tool with a path parameter goes that way."""
    import httpx

    seen = []

    def answer(request):
        seen.append(request.url.raw_path.decode())  # as sent, before decoding
        return httpx.Response(200, json={"name": "Lamella 01/2"})

    client = httpx.Client(transport=httpx.MockTransport(answer), base_url="http://x")
    shape = {"routers": {"microscope": True, "app": True}, "level": "drive"}
    sidecar = build_sidecar(client, shape)
    asyncio.run(sidecar.call_tool("item_get", {"item_name": "Lamella 01/2"}))
    asyncio.run(
        sidecar.call_tool(
            "item_update",
            {"item_name": "Lamella 01/2", "patch": {}, "version": "v"},
        )
    )
    assert seen == [
        "/app/experiment/items/Lamella%2001%2F2",
        "/app/experiment/items/Lamella%2001%2F2",
    ]
