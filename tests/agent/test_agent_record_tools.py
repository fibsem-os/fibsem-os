"""The sidecar's record tools: on an agent host alone, or left off an
instrument whose agent keeps its record on a host."""

import pytest

from fibsem.ai.agent.catalog import RECORD_TOOLS

pytest.importorskip("mcp")


def _offered(record, capabilities=None):
    import asyncio

    from fibsem.ai.agent.mcp.sidecar import build_sidecar

    server = build_sidecar(object(), capabilities, record=record)
    return {tool.name for tool in asyncio.run(server.list_tools())}


def test_a_host_s_sidecar_offers_the_record_tools_alone():
    assert _offered(True) == set(RECORD_TOOLS)


def test_an_instrument_s_sidecar_leaves_them_to_the_host():
    every, without = _offered(None), _offered(False)
    assert RECORD_TOOLS <= every
    assert without == every - RECORD_TOOLS


def test_a_host_serving_a_data_folder_adds_the_image_and_file_tools():
    routers = {"app": False, "analysis": True, "files": True}
    offered = _offered(True, {"routers": routers})
    assert set(RECORD_TOOLS) < offered
    assert {"analysis_crop_image", "analysis_measure_histogram"} <= offered
    assert {"experiment_read_file", "experiment_list_files"} <= offered
    assert not any(name.startswith(("item_", "workflow_")) for name in offered)


def test_an_instrument_agent_on_the_gateways_agent_host_keeps_every_tool():
    """The gateway passes its calls to the agent host while the app is off:
    the app's tools stay offered, for when the app answers."""
    from fibsem.ai.agent.mcp.sidecar import _matched

    host = {"host": "agent", "routers": {"app": False}}
    assert _matched(host, None) is None
    assert _offered(None, _matched(host, None)) == _offered(None)
    # A host's own agent, and an app's answer, are as they were.
    assert _matched(host, True) is host
    app = {"routers": {"app": True}}
    assert _matched(app, None) is app
