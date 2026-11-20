"""A fresh context fibsem-agent starts itself: a resume whose last
conversation is too large or can't be loaded.

The harness against a real App Server, as in ``test_agent_handoff.py``; only
the model is stood in.
"""

import asyncio
import json

import pytest

from fibsem.ai.agent import handoff as handoff_mod
from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import instruments, playbooks, usage
from fibsem.ai.agent.harness import Harness, TurnEnd
from tests.server.test_agent_handoff import ResettingAgent
from tests.server.test_agent_harness import (  # noqa: F401
    _connect,
    _harness,
    microscope,
    session,
)
from tests.waiting import wait_for

WORDS = harness_mod.texts()
# Its first sentence, which fills in nothing.
RECOVERED = WORDS.handoff_recovered.split(":")[0]
NOTE = "## Handoff to context 2, 2026-11-19T20:00:00\n\n### Next\n\nPolish 22."


def _handoff():
    return handoff_mod.Handoff(WORDS)


def _usage(path, *calls):
    with open(path, "w", encoding="utf-8") as file:
        for call in calls:
            file.write(json.dumps(call) + "\n")


# --- deciding before the start ----------------------------------------------


def test_the_last_context_is_the_agents_last_call(tmp_path):
    path = tmp_path / usage.FILE
    assert usage.last_context(path) is None
    _usage(
        path,
        {"agent": None, "input": 5, "cache_read": 900_000, "cache_write": 1000},
        # A specialist's call is not the agent's context.
        {"agent": "toolu_1", "input": 5, "cache_read": 20_000, "cache_write": 0},
    )
    assert usage.last_context(path) == 901_005


def test_a_resume_reloads_only_a_context_below_the_limit(tmp_path):
    record = {"session_id": "s1"}
    path = tmp_path / usage.FILE
    _usage(path, {"input": 1, "cache_read": 953_000, "cache_write": 0})
    why = harness_mod._too_large_to_resume(record, tmp_path, 700_000, WORDS)
    assert why and "953k" in why and "700k" in why
    # Below it, no limit, no usage, or a new session: reload as before.
    assert harness_mod._too_large_to_resume(record, tmp_path, 1_000_000, WORDS) is None
    assert harness_mod._too_large_to_resume(record, tmp_path, None, WORDS) is None
    assert harness_mod._too_large_to_resume({}, tmp_path, 700_000, WORDS) is None
    assert (
        harness_mod._too_large_to_resume(record, tmp_path / "x", 700_000, WORDS) is None
    )


def test_the_latest_handoff_note_is_the_last_in_the_file(tmp_path):
    path = tmp_path / handoff_mod.FILE
    assert handoff_mod.latest_note(path) is None
    path.write_text(
        "## Handoff to context 2, a\n\nfirst\n\n## Handoff to context 3, b\n\nsecond\n"
    )
    assert handoff_mod.latest_note(path) == "## Handoff to context 3, b\n\nsecond"


def test_every_shipped_kind_has_words_for_it_and_the_base_playbook_sets_it():
    checkout = harness_mod.find_checkout()
    for kind in instruments.kinds(checkout):
        words = instruments.load(kind, checkout)
        assert "{" not in words.handoff_recovered.format(why="w", skills="s", note="n")
    base = playbooks.load("fibsem-agent", checkout)
    assert base.session.resume_max_context == 700_000


# --- the harness --------------------------------------------------------------


def test_a_resume_chosen_fresh_rebuilds_from_the_record(session):
    """`main` chose a fresh context: the first turn is the recovered prompt,
    with why and the latest note, and the new conversation is kept."""
    agent = ResettingAgent()
    agent.conversation_id = "fresh-1"
    contexts = []
    handoff = _handoff()
    running = _harness(
        session,
        agent,
        resumed=True,
        spent_before_usd=480.0,
        handoff=handoff,
        keep_context=lambda conversation, spent: contexts.append((conversation, spent)),
        recover="its last conversation was too large to reload (953k).",
        latest_note=lambda: NOTE,
    )
    assert wait_for(lambda: agent.prompts), "no first turn"
    running.stop()
    first = agent.prompts[0]
    assert first.startswith(RECOVERED)
    assert "too large to reload (953k)" in first and "Polish 22." in first
    assert handoff_mod.RECOVERED_WAKE in first
    assert WORDS.resumed not in first
    assert contexts == [("fresh-1", 480.0)]
    assert handoff.contexts == 2
    assert not agent.conversations, "the conversation was chosen before the start"


def test_a_resume_that_cannot_reload_starts_fresh_once(session):
    class Failing(ResettingAgent):
        async def start(self):
            self.starts = getattr(self, "starts", 0) + 1
            raise RuntimeError("No conversation found with session ID: s1")

    agent = Failing()
    contexts = []
    running = _harness(
        session,
        agent,
        resumed=True,
        handoff=_handoff(),
        keep_context=lambda conversation, spent: contexts.append((conversation, spent)),
    )
    assert wait_for(lambda: agent.prompts), "no first turn"
    running.stop()
    assert len(agent.conversations) == 1 and len(contexts) == 1
    first = agent.prompts[0]
    assert first.startswith(RECOVERED)
    assert "could not be reloaded (No conversation found" in first
    assert handoff_mod.NO_NOTE in first


def test_a_resumed_conversation_that_cannot_load_starts_fresh(session):
    """The first turn dies and the CLI said it couldn't load the conversation."""

    async def reply(prompt):
        if len(agent.prompts) == 1:
            harness_mod.CLI_SAID.append("RangeError: Invalid string length")
            return TurnEnd("no_result")
        return TurnEnd()

    harness_mod.CLI_SAID.clear()
    agent = ResettingAgent(reply)
    running = _harness(session, agent, resumed=True, handoff=_handoff())
    assert wait_for(lambda: len(agent.prompts) >= 2), "no fresh turn"
    running.stop()
    harness_mod.CLI_SAID.clear()
    assert agent.prompts[0].startswith(WORDS.resumed.split(".")[0])
    assert len(agent.conversations) == 1
    assert agent.prompts[1].startswith(RECOVERED)
    assert "Invalid string length" in agent.prompts[1]


def test_a_resume_that_fails_for_another_reason_stops_as_before(session):
    """A first turn that dies, or a start that fails, without the CLI saying
    the conversation can't be loaded: the old conversation is kept."""
    harness_mod.CLI_SAID.clear()
    agent = ResettingAgent(lambda prompt: _done(TurnEnd("no_result")))
    running = _harness(session, agent, resumed=True, handoff=_handoff())
    assert running.ended(30) == 1
    assert not agent.conversations

    class Failing(ResettingAgent):
        async def start(self):
            raise RuntimeError("Connection reset by peer")

    agent = Failing()
    harness = Harness(agent, _connect(session), resumed=True, handoff=_handoff())
    with pytest.raises(RuntimeError, match="Connection reset"):
        asyncio.run(harness.run())
    assert not agent.conversations and not agent.prompts


def test_what_counts_as_cannot_load():
    assert harness_mod.cannot_load("No conversation found with session ID: x")
    assert harness_mod.cannot_load("", "FATAL ERROR: JavaScript heap out of memory")
    assert harness_mod.cannot_load("SyntaxError: Unexpected token } in JSON")
    assert not harness_mod.cannot_load("Connection reset by peer", "API Error: 529")


def test_a_new_session_that_dies_still_stops(session):
    agent = ResettingAgent(lambda prompt: _done(TurnEnd("no_result")))
    running = _harness(session, agent, handoff=_handoff())
    assert running.ended(30) == 1
    assert not agent.conversations


async def _done(end):
    return end
