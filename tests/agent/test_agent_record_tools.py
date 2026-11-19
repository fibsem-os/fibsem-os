"""The sidecar's record tools: on an agent host alone, or left off an
instrument whose agent keeps its record on a host."""

import pytest

from fibsem.ai.agent.catalog import RECORD_TOOLS

pytest.importorskip("mcp")


def _offered(record):
    import asyncio

    from fibsem.ai.agent.mcp.sidecar import build_sidecar

    server = build_sidecar(object(), None, record=record)
    return {tool.name for tool in asyncio.run(server.list_tools())}


def test_a_host_s_sidecar_offers_the_record_tools_alone():
    assert _offered(True) == set(RECORD_TOOLS)


def test_an_instrument_s_sidecar_leaves_them_to_the_host():
    every, without = _offered(None), _offered(False)
    assert RECORD_TOOLS <= every
    assert without == every - RECORD_TOOLS
