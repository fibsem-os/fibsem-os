"""What the agent's model calls used: one line per call, and the turn's sum."""

import asyncio
import json
import logging

import pytest

from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import usage
from fibsem.ai.agent.usage import Calls


def _usage(input=0, output=0, read=0, write=0):
    return {
        "input_tokens": input,
        "output_tokens": output,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write,
    }


def test_a_calls_blocks_are_one_line_and_the_turn_sums_its_calls(tmp_path):
    path = tmp_path / usage.FILE
    calls = Calls(path)
    # One call sent as two blocks, its output growing; then a specialist's call.
    calls.seen("m1", "opus", _usage(10, 5, 1000, 200), [])
    calls.seen("m1", "opus", _usage(10, 40, 1000, 200), ["stage_move"])
    calls.seen("m2", "sonnet", _usage(3, 7, 500, 0), ["image_acquire"], "toolu_1")
    assert calls.end_turn() == {
        "input": 13,
        "output": 47,
        "cache_read": 1500,
        "cache_write": 200,
    }
    # The context is the agent's own last call, not the specialist's.
    assert calls.context == 1210

    first, second = [json.loads(line) for line in path.read_text().splitlines()]
    assert first["turn"] == 1 and first["model"] == "opus"
    assert (first["output"], first["tools"]) == (40, ["stage_move"])
    assert "agent" not in first and second["agent"] == "toolu_1"

    calls.seen("m3", "opus", _usage(1, 1, 2000, 0), [])
    assert calls.end_turn()["cache_read"] == 2000
    assert json.loads(path.read_text().splitlines()[-1])["turn"] == 2


def test_a_usage_file_that_cant_be_written_costs_the_run_nothing(tmp_path):
    calls = Calls(tmp_path / "missing" / usage.FILE)
    calls.seen("m1", "opus", _usage(1, 2, 3, 4), [])
    assert calls.end_turn()["output"] == 2


def test_token_counts_read_short():
    assert [usage.short(n) for n in (412, 4120, 41_200, 1_000_000, 1_250_000)] == [
        "412",
        "4.1k",
        "41k",
        "1M",
        "1.2M",
    ]
    assert (
        usage.context_text(
            {"used": 412_000, "window": 1_000_000, "compact_at": 970_000}
        )
        == "Context 412k of 1M, compacts at 970k"
    )
    assert usage.context_text({"used": 5000}) == "Context 5k"
    assert usage.context_text(None) is None


class _Client:
    """The SDK's client stood in: one turn's messages, and its context."""

    def __init__(self, messages, context=None):
        self._messages, self._context = messages, context

    async def query(self, prompt):
        pass

    async def receive_response(self):
        for message in self._messages:
            yield message

    async def get_context_usage(self):
        if self._context is None:
            raise RuntimeError("an older CLI")
        return self._context


def _turn(messages, context=None, cost_before=None):
    from fibsem.ai.agent.harness import SdkAgent

    agent = SdkAgent(harness_mod.build_options(harness_mod.find_checkout()))
    agent._client = _Client(messages, context)
    agent._cost_usd = cost_before
    return asyncio.run(agent.turn("go"))


def _result(cost):
    from claude_agent_sdk import ResultMessage

    return ResultMessage(
        subtype="success",
        duration_ms=40_000,
        duration_api_ms=28_000,
        is_error=False,
        num_turns=3,
        session_id="s",
        total_cost_usd=cost,
    )


def test_a_turn_logs_its_tokens_time_and_context(caplog):
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import AssistantMessage, TextBlock

    caplog.set_level(logging.INFO, logger=harness_mod.log.name)
    said = AssistantMessage(
        content=[TextBlock("Done.")],
        model="opus",
        usage=_usage(1200, 3200, 410_000, 8100),
        message_id="m1",
    )
    context = {"totalTokens": 412_000, "rawMaxTokens": 1_000_000}
    context["autoCompactThreshold"] = 970_000
    end = _turn([said, _result(12.30)], context, cost_before=11.89)

    assert end.tokens["cache_read"] == 410_000
    assert end.context == {"used": 412_000, "window": 1_000_000, "compact_at": 970_000}
    (line,) = [r.message for r in caplog.records if r.message.startswith("Turn ended")]
    assert line == (
        "Turn ended: success, 3 steps, $0.4100 ($12.3000 so far). Tokens: 1.2k in, "
        "410k cached, 8.1k written, 3.2k out. 40 s, 28 s of it the model. "
        "Context 412k of 1M, compacts at 970k."
    )


def test_without_the_clis_figure_the_context_is_the_last_calls_size():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import AssistantMessage, TextBlock

    said = AssistantMessage(
        content=[TextBlock("Done.")],
        model="opus",
        usage=_usage(10, 5, 1000, 200),
        message_id="m1",
    )
    assert _turn([said, _result(1.0)]).context == {"used": 1210}
