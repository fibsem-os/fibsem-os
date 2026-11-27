"""Subagents in the background: which may run there, what they may do, the
operator's messages to a running one, and the CLI's messages between turns."""

import asyncio
import shutil
from pathlib import Path

import pytest

from fibsem.ai.agent import background
from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import watch as watch_mod

AGENTS = Path(__file__).resolve().parents[2] / ".claude" / "agents"
READERS = (
    "autolamella-verifier",
    "autolamella-investigator",
    "autolamella-auditor",
    "autolamella-site-finder",
)


@pytest.fixture
def runs(tmp_path):
    shutil.copytree(AGENTS, tmp_path / ".claude" / "agents")
    return background.Runs(tmp_path)


def _call(hook, tool, arguments, call_id="toolu_1", agent_id=None):
    given = {"tool_name": tool, "tool_input": arguments}
    if agent_id:
        given["agent_id"] = agent_id
    return asyncio.run(hook(given, call_id, None))


def _refused(answer) -> bool:
    out = answer.get("hookSpecificOutput") or {}
    return out.get("permissionDecision") == "deny"


@pytest.mark.parametrize("name", READERS)
def test_the_subagents_that_only_read_may_run_in_the_background(runs, name):
    assert runs.may_run_in_background(name)


def test_the_worker_and_a_subagent_with_no_tools_line_may_not(runs, tmp_path):
    assert not runs.may_run_in_background("fibsem-worker")
    (tmp_path / ".claude" / "agents" / "any.md").write_text(
        "---\nname: any\ndescription: every tool\n---\n\nDo it.\n"
    )
    assert not runs.may_run_in_background("any")
    assert not runs.may_run_in_background("not-there")


def test_a_background_call_is_refused_for_an_acting_subagent_a_third_or_a_nested_one(
    runs,
):
    hook = background.agent_hook(runs)
    worker = {"subagent_type": "fibsem-worker", "run_in_background": True}
    assert _refused(_call(hook, "Agent", worker))
    # Blocking, it goes as before.
    assert _call(hook, "Agent", {"subagent_type": "fibsem-worker"}, "w1") == {}
    finder = {"subagent_type": "autolamella-site-finder", "run_in_background": True}
    assert _call(hook, "Agent", finder, "b1") == {}
    assert _call(hook, "Agent", finder, "b2") == {}
    third = _call(hook, "Agent", finder, "b3")
    assert _refused(third) and "2 subagents" in str(third)
    # One answered: room again.
    runs.ended("b1")
    assert _call(hook, "Agent", finder, "b4") == {}
    # A subagent's own call runs in its turn.
    verifier = {"subagent_type": "autolamella-verifier", "run_in_background": True}
    assert _refused(_call(hook, "Agent", verifier, "n1", agent_id="worker-task"))
    # Code mode's shell never runs in the background.
    assert _refused(_call(hook, "Bash", {"command": "ls", "run_in_background": True}))
    assert _call(hook, "Bash", {"command": "ls"}) == {}


def test_a_background_run_only_reads(runs):
    runs.called("b1", "autolamella-site-finder", True)
    runs.started("b1", "task-b")
    runs.called("w1", "fibsem-worker", False)
    runs.started("w1", "task-w")
    hook = background.lease_hook(runs)
    move = "mcp__fibsem__stage_move_absolute"
    assert _refused(_call(hook, move, {}, agent_id="task-b"))
    assert _call(hook, "mcp__fibsem__grid_get_markers", {}, agent_id="task-b") == {}
    # Code mode's shell has no network: the fibsem tools are the only way.
    assert _call(hook, "Bash", {"command": "ls"}, agent_id="task-b") == {}
    assert _refused(_call(hook, "mcp__lab-notebook__write", {}, agent_id="task-b"))
    # The blocking worker and the lead keep the microscope.
    assert _call(hook, move, {}, agent_id="task-w") == {}
    assert _call(hook, move, {}) == {}


def test_the_operators_message_goes_to_its_run_after_its_next_call(runs):
    read, missed = [], []
    runs.on_delivered = lambda run, i, text: read.append((run.subagent, i, text))
    runs.on_unread = lambda run, i, text: missed.append((run.subagent, i, text))
    runs.called("b1", "autolamella-site-finder", True)
    runs.started("b1", "task-b")
    assert not runs.write("nope", "m0", "lost")
    assert runs.write("b1", "m1", "look at square C too")
    hook = background.deliver_hook(runs)
    # Another run's call, or the lead's, doesn't take it.
    assert _call(hook, "Read", {}, agent_id="task-other") == {}
    assert _call(hook, "Read", {}) == {}
    said = _call(hook, "mcp__fibsem__grid_get", {}, agent_id="task-b")
    context = said["hookSpecificOutput"]["additionalContext"]
    assert "look at square C too" in context and "id=m1" in context
    assert read == [("autolamella-site-finder", "m1", "look at square C too")]
    # Taken once.
    assert _call(hook, "Read", {}, agent_id="task-b") == {}
    # One it never reads goes to the lead as it ends.
    runs.write("b1", "m2", "and D")
    runs.ended("b1")
    assert missed == [("autolamella-site-finder", "m2", "and D")]


def test_the_message_line_to_a_run_carries_it():
    line = watch_mod._message_line({"id": "ab12", "text": "hurry", "run": "toolu_9"})
    assert line.kind == watch_mod.MESSAGE
    assert line.facts == {"run": "toolu_9", "said": "hurry"}
    assert line.message_ids == ("ab12",)
    assert "ended before it read it" in line


# --- the CLI's messages, read between turns ----------------------------------


class _Cli:
    """The SDK's client stood in: what each query answers, and messages the
    test puts on its stream while no turn runs."""

    def __init__(self):
        self.stream: asyncio.Queue = asyncio.Queue()
        self.turns = []

    async def query(self, prompt):
        for message in self.turns.pop(0):
            self.stream.put_nowait(message)

    async def receive_messages(self):
        while True:
            yield await self.stream.get()

    async def get_context_usage(self):
        raise RuntimeError("not said")


def _result(cost, origin=None):
    from claude_agent_sdk import ResultMessage

    return ResultMessage(
        subtype="success",
        duration_ms=1000,
        duration_api_ms=800,
        is_error=False,
        num_turns=1,
        session_id="s",
        total_cost_usd=cost,
        origin=origin,
    )


def _agent(runs):
    agent = harness_mod.SdkAgent(harness_mod.build_options(harness_mod.find_checkout()))
    agent.background = runs
    agent._client = cli = _Cli()
    return agent, cli


def test_a_background_run_stays_open_until_its_answer_and_its_turn_counts(runs):
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import (
        AssistantMessage,
        SystemMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    agent, cli = _agent(runs)
    own = []
    agent.on_own_turn = own.append
    call = ToolUseBlock(
        "b1",
        "Agent",
        {
            "subagent_type": "autolamella-site-finder",
            "description": "find sites",
            "prompt": "Grid 1's overviews.",
            "run_in_background": True,
        },
    )
    runs.called("b1", "autolamella-site-finder", True)
    started = ToolResultBlock("b1", "Async agent launched successfully. agentId: t1")
    cli.turns.append(
        [
            AssistantMessage([call], "opus"),
            SystemMessage("task_started", {"task_id": "t1", "tool_use_id": "b1"}),
            UserMessage([started]),
            AssistantMessage([TextBlock("Screening grid 2 meanwhile.")], "opus"),
            _result(1.0),
        ]
    )

    async def main():
        end = await agent.turn("go")
        kinds = [line["kind"] for line in agent.transcript]
        # The run's lines and its answer come while no turn of ours runs.
        reader = asyncio.ensure_future(agent.between_turns())
        cli.stream.put_nowait(
            AssistantMessage([TextBlock("Square 4 looks good.")], "opus", "b1")
        )
        cli.stream.put_nowait(
            SystemMessage(
                "task_notification",
                {
                    "task_id": "t1",
                    "tool_use_id": "b1",
                    "status": "completed",
                    "summary": "Sites 4 and 9.",
                },
            )
        )
        cli.stream.put_nowait(AssistantMessage([TextBlock("Read the sites.")], "opus"))
        cli.stream.put_nowait(_result(1.5, {"kind": "task-notification"}))
        for _ in range(50):
            if own:
                break
            await asyncio.sleep(0.01)
        reader.cancel()
        return end, kinds

    end, kinds = asyncio.run(main())
    assert end.cost_usd == 1.0
    # The call's step was not answered by its start.
    assert kinds.count("done") == 0
    step = next(line for line in agent.transcript if line["kind"] == "step")
    assert step["background"] is True and step["call"] == "b1"
    said = [line for line in agent.transcript if line.get("run") == step["seq"]]
    assert [line["text"] for line in said] == ["Square 4 looks good."]
    (done,) = [line for line in agent.transcript if line["kind"] == "done"]
    assert done["step"] == step["seq"] and done["text"] == "Sites 4 and 9."
    assert runs.get("b1") is None
    assert [turn.cost_usd for turn in own] == [1.5]


def test_a_turn_skips_the_end_of_one_the_cli_started_itself(runs):
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import AssistantMessage, TextBlock

    agent, cli = _agent(runs)
    own = []
    agent.on_own_turn = own.append
    cli.turns.append([AssistantMessage([TextBlock("Mine.")], "opus"), _result(2.0)])
    # The CLI's own turn ended just before ours was read.
    cli.stream.put_nowait(_result(1.5, {"kind": "task-notification"}))
    end = asyncio.run(agent.turn("go"))
    assert end.cost_usd == 2.0
    assert [turn.cost_usd for turn in own] == [1.5]


def test_without_runs_the_cli_runs_no_subagent_in_the_background(tmp_path):
    pytest.importorskip("claude_agent_sdk")
    checkout = harness_mod.find_checkout()
    off = harness_mod.build_options(checkout)
    assert off.env[harness_mod.BACKGROUND_OFF] == "1"
    on = harness_mod.build_options(checkout, runs=background.Runs(tmp_path))
    assert harness_mod.BACKGROUND_OFF not in on.env
    matchers = [m.matcher for m in on.hooks["PreToolUse"]]
    assert "Agent|Task|Bash" in matchers
    assert "TaskStop" in on.allowed_tools and "TaskStop" in on.tools


def test_the_operators_message_to_a_running_run_skips_the_leads_queue(runs):
    agent = harness_mod.SdkAgent(harness_mod.build_options(harness_mod.find_checkout()))
    agent.background = runs
    harness = harness_mod.Harness(agent, lambda: None)
    harness._items = asyncio.Queue()
    harness._urgent = asyncio.Event()
    runs.called("b1", "autolamella-site-finder", True)
    line = watch_mod._message_line({"id": "ab12", "text": "hurry", "run": "b1"})
    harness._put(harness._watcher, line)
    assert harness._items.empty()
    assert runs.get("b1").waiting == [("ab12", "hurry")]
    # A run that has gone: the lead reads it.
    gone = watch_mod._message_line({"id": "cd34", "text": "late", "run": "b9"})
    harness._put(harness._watcher, gone)
    assert harness._items.get_nowait()[1] is gone
