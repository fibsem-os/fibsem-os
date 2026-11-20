"""When the agent is told about its context's size: once past the playbook's
suggestion and at every natural point past its ceiling (`handoff.Ceiling`)."""

from fibsem.ai.agent import handoff as handoff_mod
from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import watch as watch_mod
from fibsem.ai.agent.harness import Harness
from tests.server.test_agent_harness import ScriptedAgent

ENDED = ["task_completed"]


def test_a_natural_point_is_an_end_with_nothing_failed_or_waiting():
    assert handoff_mod.natural_point(["task_completed"])
    assert handoff_mod.natural_point(["workflow_ended", "autofocus"])
    assert handoff_mod.natural_point(["digest_due"])
    assert not handoff_mod.natural_point(["autofocus"])
    for busy in ("task_failed", "proposal", "message", "mill_stage_started"):
        assert not handoff_mod.natural_point(["task_completed", busy])


def test_the_suggestion_is_said_once_and_only_at_a_natural_point():
    ceiling = handoff_mod.Ceiling(300_000, 500_000, 700_000)
    assert ceiling.line(250_000, ENDED) is None
    assert ceiling.line(320_000, ["autofocus"]) is None
    said = ceiling.line(320_000, ENDED)
    assert said and "natural point" in said and "320k" in said
    assert ceiling.line(340_000, ENDED) is None
    ceiling.fresh()
    assert ceiling.line(320_000, ENDED)


def test_past_the_ceiling_it_is_due_at_each_natural_point_then_firmly():
    ceiling = handoff_mod.Ceiling(300_000, 500_000, 700_000)
    said = [ceiling.line(520_000, ENDED) for _ in range(4)]
    assert all("ceiling of 500k" in line for line in said)
    assert all("is due" in line for line in said[:3])
    assert "asked 3 times" in said[3] and "compacts" in said[3] and "700k" in said[3]
    # Not between natural points.
    assert ceiling.line(530_000, ["proposal", "task_completed"]) is None
    ceiling.fresh()
    assert "is due" in ceiling.line(520_000, ENDED)


def test_no_settings_say_nothing():
    assert handoff_mod.Ceiling().line(900_000, ENDED) is None


def test_the_harness_adds_it_to_a_wake_at_a_natural_point():
    ceiling = handoff_mod.Ceiling(300_000, 500_000, 700_000)
    harness = Harness(
        ScriptedAgent(),
        lambda: None,
        handoff=handoff_mod.Handoff(harness_mod.texts()),
        ceiling=ceiling,
    )
    harness._context = {"used": 520_000, "window": 1_000_000}
    ended = watch_mod.Wake("task_completed", "EVENT task Setup completed on 02")
    milling = watch_mod.Wake("mill_stage_started", "EVENT mill started")
    assert "is due" in harness._ceiling_line([ended])
    assert harness._ceiling_line([ended, milling]) is None
    # A fresh context's size is not known until its first turn.
    harness._context = None
    assert harness._ceiling_line([ended]) is None
    # Without a handoff (no context_reset), nothing is said.
    harness = Harness(ScriptedAgent(), lambda: None, ceiling=ceiling)
    harness._context = {"used": 520_000}
    assert harness._ceiling_line([ended]) is None
