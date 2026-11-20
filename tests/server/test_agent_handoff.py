"""A fresh context for the same session (``context_reset``): the harness
against a real App Server, the SDK agent over a stand-in for the CLI, and the
pieces they share.

The App Server over a windowless Demo session and the real watch, as in
``test_agent_harness.py``; only the model is stood in. Nothing here runs the
Claude Code CLI: what a real CLI does with a new conversation is not tested.
"""

import asyncio
import uuid

import pytest

from fibsem.ai.agent import handoff as handoff_mod
from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import manager as sessions
from fibsem.ai.agent import mirror, playbooks, usage
from fibsem.ai.agent.harness import Harness, TurnEnd
from fibsem.ai.agent.session import AgentControl
from tests.server.test_agent_harness import (  # noqa: F401
    ScriptedAgent,
    _harness,
    _http,
    microscope,
    session,
)
from tests.waiting import wait_for

WORDS = harness_mod.texts()
# Its first sentence, which fills in nothing.
START = WORDS.handoff_start.split(". ")[0]
NOTE = {
    "done": "Screened grid 1; sites 01-04 approved by the operator.",
    "in_flight": "Nothing running; no proposal pending.",
    "next": "Start Setup on site 01.",
    "rules": "Stop thicker.",
}


class ResettingAgent(ScriptedAgent):
    """The scripted agent with what fibsem-agent's needs for a fresh context:
    ``new_context()``, a transcript and the skills it read."""

    def __init__(self, reply=None, during_reset=None):
        super().__init__(reply)
        self.transcript = []
        self.skills_read = ["fibsem-agent", "fibsem-microscope", "screen-grids"]
        self.conversations = []
        self.compacted = False
        self._during_reset = during_reset
        self._seq = 0

    def line(self, kind, text, **extra):
        self._seq += 1
        self.transcript.append({"seq": self._seq, "kind": kind, "text": text, "at": 0})
        return self._seq

    async def new_context(self):
        if self._during_reset is not None:
            await self._during_reset()
        conversation = str(uuid.uuid4())
        self.conversations.append((len(self.prompts), conversation))
        self.skills_read = []
        return conversation


def _handoff(kept=None, marks=None):
    return handoff_mod.Handoff(
        WORDS, marks, keep=kept.append if kept is not None else None
    )


def _handoff_text(session):
    with _http(session) as http:
        answer = http.get("/app/agent/handoff")
    assert answer.status_code == 200, answer.text
    return answer.json()["text"]


def test_a_reset_starts_a_new_conversation_when_the_turn_ends(session):
    """The tool queues it; the turn goes on to its end; the next turn is the
    fresh context's, with the note, the lines that came in around the reset,
    and the spend carried on."""
    control = AgentControl(session)
    sent = {}
    handoff = None

    async def reply(prompt):
        if len(agent.prompts) == 1:
            said = handoff.request(NOTE)
            assert "when this turn ends" in said
            # Still this turn: nothing has been reset.
            assert not agent.conversations
            # The operator writes while the turn goes on.
            sent["during_turn"] = control.message("how is it going?")
            await asyncio.sleep(1.0)
            return TurnEnd(cost_usd=2.0, session_id="scripted")
        # The fresh conversation's own total, starting again from 0.
        return TurnEnd(cost_usd=1.0, session_id="scripted")

    async def during_reset():
        # And while the new conversation starts.
        sent["during_reset"] = control.message("also check 02")
        await asyncio.sleep(1.5)

    agent = ResettingAgent(reply, during_reset)
    kept, contexts = [], []
    handoff = _handoff(kept)
    running = _harness(
        session,
        agent,
        budget_usd=2.5,
        handoff=handoff,
        keep_context=lambda conversation, spent: contexts.append((conversation, spent)),
    )
    # The second turn spends the session past its budget: spend carried on.
    assert running.ended(60) == 2
    assert len(agent.prompts) == 2
    ((at, conversation),) = agent.conversations
    assert at == 1, "the reset came before the turn that asked for it ended"
    fresh = agent.prompts[1]
    assert fresh.startswith(START)
    # The skills to read: the playbook's are given by the harness's control
    # (none here), the base skill first, and the others it had read.
    assert "in this order: fibsem-agent." in fresh
    assert "You had also read fibsem-microscope, screen-grids" in fresh
    assert "Start Setup on site 01." in fresh and "Stop thicker." in fresh
    assert handoff_mod.WAKE in fresh
    assert sent["during_turn"] in fresh and sent["during_reset"] in fresh
    assert fresh.index(handoff_mod.WAKE) > fresh.index("Start Setup on site 01.")

    assert contexts == [(conversation, 2.0)]
    assert len(kept) == 1 and "## Handoff to context 2, " in kept[0]
    assert handoff.contexts == 2

    reported = control.reported_session()
    assert reported["session_id"] == "scripted"
    assert reported["spent_usd"] == 3.0 and reported["contexts"] == 2
    assert reported["reason"] == "its budget is spent"
    # One session on the record, not two.
    kinds = [e["kind"] for e in session.recorder.buffer.events_since(0)["events"]]
    assert kinds.count("agent_session_started") == 1
    # Both messages were read as the fresh context's turn started.
    read = [
        i
        for e in session.recorder.buffer.events_since(0)["events"]
        if e["kind"] == "agent_messages_read"
        for i in e["payload"]["ids"]
    ]
    assert sent["during_turn"] in read and sent["during_reset"] in read

    # The note is kept beside the plan, and the chat shows where it started.
    text = _handoff_text(session)
    assert "## Handoff to context 2, " in text and "Start Setup on site 01." in text
    assert "agent/handoff-scripted.md in the experiment" in fresh
    with _http(session) as http:
        entries = http.get("/app/agent/transcript").json()["entries"]
    marker = [e for e in entries if e["kind"] == "context"]
    assert marker and marker[0]["text"].startswith("Fresh context: the agent started")


def test_a_second_reset_keeps_the_first_note(session):
    async def reply(prompt):
        if len(agent.prompts) <= 2:
            handoff.request({**NOTE, "next": f"Step {len(agent.prompts)}"})
        return TurnEnd(cost_usd=0.5, session_id="scripted")

    agent = ResettingAgent(reply)
    handoff = _handoff()
    running = _harness(session, agent, handoff=handoff)
    assert wait_for(lambda: len(agent.prompts) >= 3, timeout=30), "no third turn"
    assert running.stop() == 0
    assert len(agent.conversations) == 2
    assert agent.prompts[1].startswith(START) and "Step 1" in agent.prompts[1]
    assert agent.prompts[2].startswith(START) and "Step 2" in agent.prompts[2]
    text = _handoff_text(session)
    first = text.index("## Handoff to context 2, ")
    second = text.index("## Handoff to context 3, ")
    assert first < second and text.index("Step 1") < text.index("Step 2")
    assert AgentControl(session).reported_session()["contexts"] == 3


def test_nothing_changes_for_a_session_that_never_resets(session):
    async def reply(prompt):
        return TurnEnd(cost_usd=0.25, session_id="scripted")

    agent = ResettingAgent(reply)
    running = _harness(session, agent, handoff=_handoff(marks=[300_000]))
    assert wait_for(lambda: agent.prompts, timeout=20), "no kickoff"
    message = AgentControl(session).message("is 01 ok?")
    assert wait_for(lambda: len(agent.prompts) >= 2, timeout=20), "no second turn"
    assert running.stop() == 0
    assert not agent.conversations
    assert agent.prompts[0] == harness_mod.kickoff()
    assert agent.prompts[1].startswith(
        f"The watcher printed:\nMESSAGE id={message} from the operator: is 01 ok?"
    )
    assert START not in agent.prompts[1] and "Context: " not in agent.prompts[1]
    assert "contexts" not in AgentControl(session).reported_session()


def test_a_reset_that_cannot_start_leaves_the_agent_where_it_was():
    class Failing(ResettingAgent):
        async def new_context(self):
            raise RuntimeError("the CLI did not start")

    agent = Failing()
    handoff = _handoff()
    harness = Harness(agent, lambda: None, handoff=handoff)

    async def main():
        harness._stopping = asyncio.Event()
        harness._urgent = asyncio.Event()
        harness._items = asyncio.Queue()
        harness._report_read = lambda: asyncio.sleep(0)
        harness._send_transcript_every = lambda: asyncio.sleep(3600)
        await agent.start()
        handoff.request(NOTE)
        assert await harness._turn("The watcher printed:\nEVENT x")

    asyncio.run(main())
    assert handoff.contexts == 1 and handoff.pending is None
    said = handoff.into("The watcher printed:\nEVENT y")
    assert "could not start (the CLI did not start)" in said
    assert START not in said


def test_the_context_size_is_said_once_per_mark():
    sizes = iter([350_000, 360_000, 520_000, 90_000, 320_000])
    costs = iter([1.0, 1.5, 2.5, 3.0, 3.25])

    class Sized(ResettingAgent):
        async def turn(self, prompt):
            self.prompts.append(prompt)
            return TurnEnd(
                cost_usd=next(costs),
                context={"used": next(sizes), "window": 1_000_000},
            )

    agent = Sized()
    handoff = _handoff(marks=[300_000, 500_000, 700_000])
    harness = Harness(agent, lambda: None, handoff=handoff)
    said = []

    async def main():
        harness._stopping = asyncio.Event()
        harness._report_read = lambda: asyncio.sleep(0)
        harness._send_transcript_every = lambda: asyncio.sleep(3600)
        await agent.start()
        for _ in range(5):
            await harness._turn("x")
            said.append(handoff.into("x"))

    asyncio.run(main())
    marked = ["Context: " in prompt for prompt in said]
    # 300k once; 500k once; under both after a compaction, so 300k again.
    assert marked == [True, False, True, False, True]
    assert "Context: 350k of 1M tokens" in said[0]
    assert "the last one cost unknown" in said[0]
    assert "Context: 520k of 1M tokens" in said[2]
    assert "the last one cost $1.00" in said[2]
    assert "context_reset" in said[2]


def test_a_size_said_of_the_earlier_context_is_not_the_fresh_ones():
    handoff = _handoff(marks=[300_000])
    handoff.measure({"used": 800_000, "window": 1_000_000}, 4.0)
    handoff.request(NOTE)
    handoff.started(handoff.take(), ["fibsem-agent"], [], "handoff.md")
    said = handoff.into("The watcher printed:\nWAKE")
    assert said.startswith(START) and "Context: " not in said
    # The fresh context passes the mark again as it grows.
    handoff.measure({"used": 50_000}, 0.1)
    handoff.measure({"used": 310_000, "window": 1_000_000}, 0.2)
    assert "Context: 310k of 1M" in handoff.into("x")


def test_the_tool_refuses_a_note_without_its_parts():
    handoff = _handoff()
    assert handoff.request({"done": "x"}).startswith("Refused: the note needs")
    assert handoff.pending is None
    assert "fresh context starts" in handoff.request(NOTE)
    assert "replaces the one" in handoff.request({**NOTE, "next": "other"})
    assert handoff.take().parts["next"] == "other"
    no_words = handoff_mod.Handoff(
        harness_mod.instruments.Texts("x", "", "", "", "", "", "", "")
    )
    assert no_words.request(NOTE).startswith("Refused")


def test_the_tool_is_on_the_harness_server_and_compaction_is_told_what_to_keep():
    pytest.importorskip("claude_agent_sdk")
    checkout = harness_mod.find_checkout()
    handoff = _handoff()
    options = harness_mod.build_options(
        checkout,
        url="http://127.0.0.1:9",
        token="t0k",
        alarm=harness_mod.Alarm(),
        handoff=handoff,
    )
    assert harness_mod.HANDOFF_TOOL in options.allowed_tools
    server = options.mcp_servers[harness_mod.HARNESS_SERVER]["instance"]
    assert server is not None
    (matcher,) = options.hooks["PreCompact"]
    answer = asyncio.run(matcher.hooks[0]({"trigger": "auto"}, None, None))
    assert answer == {"systemMessage": WORDS.handoff_compact}
    assert "see the plan" in WORDS.handoff_compact
    # The CLI keeps its own compaction window unless the playbook sets one.
    assert handoff_mod.COMPACT_WINDOW_ENV not in options.env
    playbook = playbooks.load("fibsem-agent", checkout)
    playbook.session.compact_window = 600_000
    options = harness_mod.build_options(checkout, playbook=playbook)
    assert options.env[handoff_mod.COMPACT_WINDOW_ENV] == "600000"


def test_a_playbook_sets_the_marks_and_the_compact_window(tmp_path):
    path = tmp_path / "p.yaml"
    settings = playbooks._session(
        {"context_marks": [500000, 300000], "compact_window": 600000}, path
    )
    assert settings.context_marks == [300000, 500000]
    assert settings.compact_window == 600000
    for bad in ({"context_marks": 300000}, {"context_marks": [0]}):
        with pytest.raises(playbooks.PlaybookError, match="context_marks"):
            playbooks._session(bad, path)
    base = playbooks.shipped_base(harness_mod.find_checkout())
    assert base.session.context_marks == [300000, 500000, 700000]
    assert base.session.compact_window is None


def test_an_instrument_kind_without_handoff_words_takes_the_default_kinds(tmp_path):
    checkout = harness_mod.find_checkout()
    folder = tmp_path / harness_mod.instruments.INSTRUMENT_DIR
    for kind in ("fibsem", "other"):
        (folder / kind).mkdir(parents=True)
        for name in ("prompt.md", "intake.md", "reread.md"):
            source = checkout / harness_mod.instruments.INSTRUMENT_DIR / "fibsem"
            (folder / kind / name).write_text((source / name).read_text())
    (folder / "fibsem" / "handoff.md").write_text(
        (
            checkout / harness_mod.instruments.INSTRUMENT_DIR / "fibsem" / "handoff.md"
        ).read_text()
    )
    other = harness_mod.instruments.load("other", tmp_path)
    assert other.handoff_start == WORDS.handoff_start
    (folder / "fibsem" / "handoff.md").unlink()
    assert harness_mod.instruments.load("other", tmp_path).handoff_start == ""


def test_a_resume_carries_on_the_newest_context(tmp_path):
    folder = sessions.SessionFolder.open("s1", {}, root=tmp_path)
    record = folder.record()
    assert sessions.conversation(record) == "s1"
    assert sessions.context_spend(record) == 0.0
    folder.keep_context("c2", 12.5)
    folder.keep_context("c3", 30.0)
    folder.keep_handoff("## Handoff to context 2\n\nfirst")
    folder.keep_handoff("## Handoff to context 3\n\nsecond")
    record = folder.record()
    assert sessions.conversation(record) == "c3"
    assert sessions.context_spend(record) == 30.0
    text = (folder.folder / sessions.HANDOFF).read_text()
    assert text.index("first") < text.index("second")
    # The CLI's total for the resumed conversation adds to the spend before it.
    harness = Harness(
        ScriptedAgent(),
        lambda: None,
        resumed=True,
        spent_before_usd=40.0,
        spent_carried_usd=30.0,
    )
    harness._count_spend(12.0 + harness._carried_usd)
    assert harness._spent_usd == 42.0 and harness._run_spent() == 2.0


# --- the SDK agent over a stand-in for the CLI --------------------------------


def test_the_sdk_agent_starts_a_new_conversation_and_side_questions_follow(tmp_path):
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    from tests.agent.test_agent_aside import StandInCli

    clis = []

    def factory(options):
        cli = StandInCli(answer=f"answer {len(clis) + 1}")
        clis.append((options, cli))
        return ClaudeSDKClient(options, transport=cli)

    store = mirror.TranscriptMirror(tmp_path, "s1")
    options = ClaudeAgentOptions(session_id="s1", session_store=store)
    agent = harness_mod.SdkAgent(
        options,
        usage_path=tmp_path / usage.FILE,
        session_id="s1",
        client_factory=factory,
    )
    agent.skills_read = ["fibsem-agent"]

    async def main():
        await agent.start()
        first = await agent.side_question("before?", [], 5.0)
        conversation = await agent.new_context()
        old_ready, new_ready = clis[0][1].is_ready(), clis[1][1].is_ready()
        second = await agent.side_question("after?", [], 5.0)
        await agent.close()
        return first, conversation, second, old_ready, new_ready

    first, conversation, second, old_ready, new_ready = asyncio.run(main())
    assert (first, second) == ("answer 1", "answer 2")
    (old_options, old_cli), (new_options, new_cli) = clis
    assert not old_ready, "the earlier context's CLI was left running"
    assert new_ready
    assert new_options.session_id == conversation and new_options.resume is None
    assert old_options.session_id == "s1"
    assert agent.session_id == "s1" and agent.conversation_id == conversation
    assert store.session_id == conversation
    assert store.path({"session_id": conversation}) == tmp_path / mirror.FILE
    assert agent.skills_read == [] and agent.calls.segment == 2
    assert [q["question"] for q in new_cli.side_questions] == ["after?"]


def test_lines_held_for_a_digest_reach_the_fresh_context(session):
    """At Watch under digest, routine lines that come in while the agent asks
    for a fresh context, and while it starts, are held, then given at the top
    of the fresh context's first turn, after its handoff note."""
    from fibsem.ai.agent import wake_policy

    policy = wake_policy.WakeOn(
        wake_policy.Settings(watch="digest", every_min=60, after_lines=0)
    )
    buffer = session.recorder.buffer
    handoff = None

    async def reply(prompt):
        if len(agent.prompts) == 1:
            handoff.request(NOTE)
            buffer.append("task_started", {"item_name": "01-a", "task_name": "Setup"})
            await asyncio.sleep(1.0)
        return TurnEnd(session_id="scripted")

    async def during_reset():
        buffer.append("task_completed", {"item_name": "01-a", "task_name": "Setup"})
        await asyncio.sleep(1.0)

    agent = ResettingAgent(reply, during_reset)
    handoff = _handoff()
    running = _harness(session, agent, handoff=handoff, wake_on=policy)
    assert wait_for(lambda: len(agent.prompts) >= 2, timeout=30), "no fresh turn"
    assert running.stop() == 0
    fresh = agent.prompts[1]
    assert fresh.startswith(START)
    held = fresh.index("Held for you since ")
    assert fresh.index("Start Setup on site 01.") < held
    assert held < fresh.index(f"The watcher printed:\n{handoff_mod.WAKE}")
    assert "EVENT task_started 01-a Setup" in fresh
    assert "EVENT task_completed 01-a Setup" in fresh
    assert not policy.held
