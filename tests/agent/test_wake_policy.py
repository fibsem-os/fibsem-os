"""What wakes the agent: the wake_on policy, its digest of held lines, and a
long command's end the agent already read.

The lines are the real watcher's (`watch._print_events` on event records as
the App Server serves them), and the queue is the harness's own `Wakes`."""

import asyncio
import json

import pytest

from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import playbooks, wake_lines, wake_policy
from fibsem.ai.agent import watch as watch_mod
from fibsem.ai.agent.wake_lines import fibsem as fibsem_lines
from fibsem.ai.agent.wake_policy import Settings, WakeOn

DIGEST = Settings(watch="digest", assist="all", drive="all")


def _lines(*events):
    """The watcher's lines for *events* (``(kind, payload)`` or full records)."""
    said = []
    records = [
        e if isinstance(e, dict) else {"kind": e[0], "payload": e[1]} for e in events
    ]
    for seq, record in enumerate(records, 1):
        record.setdefault("seq", seq)
    watch_mod._print_events(records, 0, said.append)
    return said


def _task(kind, item, task):
    return (kind, {"item_name": item, "task_name": task})


def _autofocus(verdict, status="completed"):
    return (
        "autofocus",
        {
            "beam_type": "ELECTRON",
            "status": status,
            "focus_score": 0.5,
            "check": {
                "verdict": verdict,
                "reasons": [] if verdict == "passed" else ["edge"],
            },
        },
    )


def _ended(run_id, state="done", actor=None, **payload):
    record = {
        "kind": "command_ended",
        "payload": {
            "run_id": run_id,
            "service": "milling",
            "command": "run_task",
            "state": state,
            **payload,
        },
    }
    if actor:
        record["actor"] = actor
    return record


def _policy(level="watch", settings=DIGEST, **kwargs):
    policy = WakeOn(settings, **kwargs)
    policy.at_level(level, initial=True)
    return policy


def _wake(kind, text=None, **facts):
    return watch_mod.Wake(kind, text or kind.upper(), **facts)


# --- the table ---------------------------------------------------------------


def test_every_line_the_watch_says_is_in_the_table():
    said = {
        watch_mod.PROPOSAL,
        watch_mod.PROPOSAL_RELEASED,
        watch_mod.SUGGESTION,
        watch_mod.REQUEST,
        watch_mod.COMMAND_ENDED,
        watch_mod.COMMAND_FAILED,
        watch_mod.AUTOFOCUS,
        watch_mod.ALIGNMENT,
        watch_mod.NOTE,
        watch_mod.MESSAGE,
        watch_mod.ASIDE,
        watch_mod.PLAN_EDITED,
        watch_mod.LEVEL,
        watch_mod.LEVEL_LOWERED,
        watch_mod.OPERATOR_STOP,
        watch_mod.WORKFLOW_STARTED,
        watch_mod.WORKFLOW_ENDED,
        watch_mod.WORKFLOW_FAILED,
        watch_mod.QUIET,
        watch_mod.GAP,
        watch_mod.RESTARTED,
        watch_mod.AGENT_OFF,
        watch_mod.AGENT_BACK,
        watch_mod.SERVER_UNAVAILABLE,
        watch_mod.SERVER_GONE,
        watch_mod.NO_EVENTS,
        watch_mod.UNREACHABLE,
        harness_mod.ALARM,
        harness_mod.DEADLINE,
        fibsem_lines.MILL_STAGE_STARTED,
        fibsem_lines.MILL_ENDED,
        fibsem_lines.MILL_FAILED,
        fibsem_lines.MILL_READING,
        fibsem_lines.STOP_RULE,
        *watch_mod.INTERESTING,
    }
    assert said == set(wake_lines.NEEDS)
    # The harness's own wake-ups and the operator always wake the agent.
    for kind in wake_lines.ALWAYS:
        assert wake_lines.needs(kind) == wake_lines.WAKE, kind
    assert wake_lines.needs(None) == wake_lines.needs("brand_new") == wake_lines.WAKE


ALWAYS_KINDS = sorted(
    kind for kind, need in wake_lines.NEEDS.items() if need == wake_lines.WAKE
)
POLICIES = [
    ("all", {}),
    ("digest", {}),
    ("digest", {"hold": ",".join(sorted(wake_policy._ROUTINE))}),
    ("all", {"hold": ",".join(sorted(wake_policy._ROUTINE))}),
]


@pytest.mark.parametrize("mode, extra", POLICIES)
def test_what_always_wakes_wakes_under_every_policy(mode, extra):
    for kind in ALWAYS_KINDS + ["a_kind_nobody_named"]:
        policy = _policy()
        assert policy.set(mode=mode, **extra).startswith("wake_on")
        wakes = harness_mod.Wakes(wake_on=policy)
        if kind == watch_mod.AGENT_BACK:
            # Back is news only after Off.
            assert wakes.add([watch_mod.AGENT_OFF_LINE]) is None
        if kind == watch_mod.AGENT_OFF:
            # Off holds every turn until the level is back, then wakes with it.
            assert wakes.add([_wake(kind)]) is None and not policy.held
            assert "AGENT_OFF" in wakes.add([watch_mod.AGENT_BACK_LINE])
            continue
        assert wakes.add([_wake(kind)]) is not None, kind
    # A routine kind that did not go as expected wakes it too.
    for kind in ("autofocus", "alignment", fibsem_lines.MILL_ENDED, "command_ended"):
        policy = _policy()
        policy.set(mode=mode, **extra)
        wakes = harness_mod.Wakes(wake_on=policy)
        assert wakes.add([_wake(kind, ok=False)]) is not None, kind


def test_a_failed_check_or_a_skip_wakes_at_watch():
    wakes = harness_mod.Wakes(wake_on=_policy())
    for event in (
        _autofocus("doubtful"),
        _autofocus("passed", status="failed"),
        ("task_skipped", {"item_name": "01", "task_name": "Mill", "skip_reason": "x"}),
        ("task_failed", {"item_name": "01", "task_name": "Mill"}),
        _ended("r9", state="failed", error="beam"),
    ):
        assert wakes.add(_lines(event)) is not None, event


# --- digest ------------------------------------------------------------------


def test_routine_lines_wait_and_come_in_order_with_the_next_wake():
    policy = _policy()
    wakes = harness_mod.Wakes(wake_on=policy)
    routine = _lines(
        _task("task_started", "01-a", "Setup"),
        _autofocus("passed"),
        _task("task_completed", "01-a", "Setup"),
        _task("task_completed", "02-b", "Setup"),
        _task("task_completed", "03-c", "Setup"),
        ("milling_stage_started", {"milling_task_name": "Rough", "stage_index": 0}),
        _ended("r1"),  # someone else's mill, done
    )
    assert len(routine) == 7
    for line in routine:
        assert wakes.add([line]) is None, line
    assert len(policy.held) == 7

    message = _lines(("operator_message", {"id": "m1", "text": "how is it going?"}))
    prompt = wakes.add(message)
    assert prompt is not None and not policy.held
    digest, rest = prompt.split("\n\nThe watcher printed:\n")
    assert digest.startswith("Held for you since ")
    assert "(wake_on digest), 7 line(s), delivered with this wake:" in digest
    assert digest.splitlines()[1:] == [
        str(routine[0]),
        str(routine[1]),
        "EVENT task_completed (3): 01-a Setup; 02-b Setup; 03-c Setup",
        str(routine[5]),
        str(routine[6]),
    ]
    assert rest.startswith("MESSAGE id=m1 from the operator: how is it going?")
    # Read once its turn starts, as before.
    assert wakes.read() == ["m1"]


def test_a_digest_past_its_size_says_what_it_left_out():
    policy = _policy(settings=Settings(watch="digest", every_min=0, after_lines=0))
    wakes = harness_mod.Wakes(wake_on=policy)
    held = [
        _wake("task_completed", f"EVENT task_completed {n:02d}-x Setup")
        for n in range(wake_policy.DIGEST_SHOWN + 10)
    ]
    for line in held:
        assert wakes.add([line]) is None
    prompt = wakes.add([_wake(watch_mod.QUIET, "QUIET: ...")])
    digest = prompt.split("\n\nThe watcher printed:\n")[0].splitlines()
    assert f"{len(held)} line(s)" in digest[0]
    assert digest[1] == (
        "(10 earlier held line(s) are not shown: 10 task_completed. "
        "app_get_events has them.)"
    )
    # The newest are each there.
    shown = digest[2]
    for line in held[10:]:
        assert line.split("EVENT task_completed ")[1] in shown
    assert "09-x" not in shown


def test_held_lines_come_once_enough_are_held():
    policy = _policy(settings=Settings(watch="digest", after_lines=3, every_min=0))
    wakes = harness_mod.Wakes(wake_on=policy)
    lines = _lines(*[_task("task_completed", f"0{n}", "Setup") for n in range(3)])
    assert wakes.add(lines[:1]) is None
    assert wakes.add(lines[1:2]) is None
    prompt = wakes.add(lines[2:])
    assert prompt.startswith("Held for you since ")
    assert "delivered at 3 lines held:" in prompt
    assert "The watcher printed:" not in prompt


def test_a_count_reached_on_a_large_cold_context_waits_for_a_line_that_wakes():
    big = wake_policy.LARGE_CONTEXT + 1
    idle = [wake_policy.CACHE_COLD_S + 1]
    policy = _policy(
        settings=Settings(watch="digest", after_lines=2, every_min=0),
        context=lambda: (idle[0], big),
    )
    wakes = harness_mod.Wakes(wake_on=policy)
    lines = _lines(*[_task("task_completed", f"0{n}", "Setup") for n in range(3)])
    for line in lines:
        assert wakes.add([line]) is None
    # Warm again (the agent was just woken by something else): the count counts.
    idle[0] = 1.0
    assert wakes.add([]) is not None


def test_held_lines_come_after_the_interval():
    async def main():
        rang = []
        policy = _policy(settings=Settings(watch="digest", every_min=0.001))
        policy.bind(asyncio.get_running_loop(), rang.append)
        wakes = harness_mod.Wakes(wake_on=policy)
        assert wakes.add(_lines(_task("task_started", "01", "Setup"))) is None
        await asyncio.sleep(0.3)
        assert [line.kind for line in rang] == [wake_policy.DIGEST_DUE]
        prompt = wakes.add(rang)
        assert prompt.startswith("Held for you since ")
        assert "delivered 0.001 min after the first was held:" in prompt
        assert "EVENT task_started 01 Setup" in prompt
        # A line that wakes first takes the hold, and the interval with it.
        assert wakes.add(_lines(_task("task_started", "02", "Setup"))) is None
        assert wakes.add([_wake(watch_mod.MESSAGE)]) is not None
        await asyncio.sleep(0.3)
        assert len(rang) == 1

    asyncio.run(main())


def test_the_digest_s_task_lines_still_call_their_reminder():
    said = []

    def remind(lines):
        said.append([line.kind for line in lines])
        return []

    wakes = harness_mod.Wakes(remind, wake_on=_policy())
    wakes.add(_lines(_task("task_completed", "01", "Setup")))
    wakes.add([_wake(watch_mod.MESSAGE)])
    assert said == [["task_completed", watch_mod.MESSAGE]]


def test_a_hold_kind_waits_under_all_and_a_wake_kind_wakes_under_digest():
    policy = _policy(level="drive")
    assert policy.mode == "all"
    policy.set(hold="autofocus")
    wakes = harness_mod.Wakes(wake_on=policy)
    assert wakes.add(_lines(_autofocus("passed"))) is None
    assert wakes.add(_lines(_task("task_completed", "01", "Setup"))) is not None

    policy = _policy()
    policy.set(wake="task_completed")
    wakes = harness_mod.Wakes(wake_on=policy)
    assert wakes.add(_lines(_task("task_started", "01", "Setup"))) is None
    assert wakes.add(_lines(_task("task_completed", "01", "Setup"))) is not None


# --- the level and the agent's own choice ------------------------------------


def _level(level, previous):
    return _lines(("agent_level_set", {"level": level, "previous": previous}))


def test_a_level_change_applies_that_level_s_default_and_says_so():
    policy = _policy(level="drive")
    wakes = harness_mod.Wakes(wake_on=policy)
    prompt = wakes.add(_level("watch", "drive"))
    assert policy.mode == "digest"
    assert (
        "What wakes you now: wake_on digest (every 30 min or 20 lines), the default at watch."
        in prompt
    )
    prompt = wakes.add(_level("drive", "watch"))
    assert policy.mode == "all"
    assert prompt.endswith(
        "What wakes you now: wake_on all, the default at drive.\n\n"
        + harness_mod.texts().reread_level
    )
    # Between two levels with the same default, nothing more is said.
    prompt = wakes.add(_level("assist", "drive"))
    assert "What wakes you" not in prompt


def test_the_agent_s_own_choice_holds_across_a_level_change():
    kept = []
    policy = _policy(level="drive", keep=kept.append)
    wakes = harness_mod.Wakes(wake_on=policy)
    policy.set(mode="all", every_min=10)
    assert kept[-1]["mode"] == "all" and kept[-1]["every_min"] == 10
    prompt = wakes.add(_level("watch", "drive"))
    assert policy.mode == "all"
    assert "still your own choice, wake_on all" in prompt
    # The operator's Stop puts the agent at Watch: still its choice.
    stop = _lines(
        {
            "kind": "action",
            "actor": "operator",
            "payload": {"stops": "agent", "what": "the agent", "level": "watch"},
        }
    )
    assert stop[0].facts == {"level": "watch"}
    wakes.add(stop)
    assert policy.mode == "all"
    # Back to the playbook's: the level's default from now on.
    assert "the playbook's default at watch" in policy.set(mode="default")
    assert policy.mode == "digest" and kept[-1] is None


def test_the_agent_s_choice_is_kept_for_a_resume():
    policy = _policy()
    policy.set(mode="digest", after_lines=5, wake="mill_ended")
    again = WakeOn(DIGEST)
    again.restore(policy.state())
    again.at_level("drive", initial=True)
    assert (again.mode, again.after_lines, again.wake) == (
        "digest",
        5,
        frozenset({"mill_ended"}),
    )
    fresh = WakeOn(DIGEST)
    fresh.restore(None)
    fresh.at_level("drive", initial=True)
    assert fresh.mode == "all"


def test_the_tool_refuses_what_it_cannot_do():
    policy = _policy()
    assert policy.set(mode="sometimes").startswith("Refused")
    assert "always wake you" in policy.set(hold="message")
    assert policy.set(every_min=-1).startswith("Refused")
    assert policy.set_by == "default"


def test_the_tool_says_the_policy_and_what_is_held():
    pytest.importorskip("claude_agent_sdk")
    policy = _policy()
    wakes = harness_mod.Wakes(wake_on=policy)
    wakes.add(_lines(_task("task_completed", "01", "Setup")))
    tool = harness_mod.wake_on_tool(policy)

    said = asyncio.run(tool.handler({}))["content"][0]["text"]
    assert said.startswith("wake_on digest (every 30 min or 20 lines)")
    assert "the playbook's default at watch" in said
    assert "Held now: 1 line(s) since " in said and "(1 task_completed)" in said
    said = asyncio.run(tool.handler({"mode": "all"}))["content"][0]["text"]
    assert said.startswith("wake_on all (your own choice).")
    assert policy.mode == "all"


def test_a_large_context_at_watch_is_told_what_a_wake_costs_once():
    used = [wake_policy.LARGE_CONTEXT + 50_000]
    policy = _policy(level="drive", context=lambda: (10.0, used[0]))
    wakes = harness_mod.Wakes(wake_on=policy)
    assert "re-reads all of it" not in wakes.add([_wake(watch_mod.MESSAGE)])
    prompt = wakes.add(_level("watch", "drive"))
    assert "Your context is 250k tokens, and every wake re-reads all of it" in prompt
    assert "re-reads all of it" not in wakes.add([_wake(watch_mod.MESSAGE)])
    # Told again on the next time at Watch, not before.
    wakes.add(_level("drive", "watch"))
    assert "re-reads all of it" in wakes.add(_level("watch", "drive"))


# --- the agent's own commands ------------------------------------------------


def test_a_command_end_the_agent_read_itself_waits_and_rides_along():
    policy = _policy(level="drive", settings=Settings(every_min=0.001, after_lines=1))
    wakes = harness_mod.Wakes(wake_on=policy)
    policy.saw_run_end("r1")

    (read,) = _lines(_ended("r1", actor="agent:claude"))
    assert wakes.add([read]) is None
    # It counts towards nothing: no digest is due on its account.
    assert policy.due() is None and policy._timer is None
    # A command it did not read wakes it: it ended after the agent went idle.
    (unread,) = _lines(_ended("r2", actor="agent:claude"))
    prompt = wakes.add([unread])
    assert prompt.startswith("Held for you since ")
    assert str(read) in prompt and prompt.endswith(f"The watcher printed:\n{unread}")


def test_a_failed_or_stopped_command_wakes_even_when_read():
    policy = _policy(level="drive")
    policy.saw_run_end("r1")
    wakes = harness_mod.Wakes(wake_on=policy)
    assert wakes.add(_lines(_ended("r1", state="failed", error="beam"))) is not None
    assert wakes.add(_lines(_ended("r1", state="cancelled"))) is not None


def test_the_agent_s_own_command_is_not_routine_under_digest():
    wakes = harness_mod.Wakes(wake_on=_policy())
    assert wakes.add(_lines(_ended("r1", actor="agent:claude"))) is not None
    # Its call's labels say so too.
    assert wakes.add(_lines(_ended("r2", level="drive"))) is not None
    assert wakes.add(_lines(_ended("r3", actor="workflow"))) is None


def _run_answer(state):
    return json.dumps(
        {
            "run": {
                "run_id": "r1",
                "service": "milling",
                "command": "run_task",
                "state": state,
                "result": {"status": "completed"},
            }
        }
    )


def test_the_ended_runs_a_result_shows():
    name = f"mcp__{harness_mod.SERVER}__service_wait_for_run"
    assert wake_policy.ended_runs(
        name, [{"type": "text", "text": _run_answer("done")}]
    ) == ["r1"]
    assert (
        wake_policy.ended_runs(name, [{"type": "text", "text": _run_answer("running")}])
        == []
    )
    assert wake_policy.ended_runs(name, [{"type": "text", "text": "not json"}]) == []
    other = f"mcp__{harness_mod.SERVER}__service_call_command"
    assert (
        wake_policy.ended_runs(other, [{"type": "text", "text": _run_answer("done")}])
        == []
    )


class _Agent:
    """What `_show` reads of the agent."""

    def __init__(self, policy):
        self.wake_on = policy
        self._steps = {}
        self.transcript = []
        self.skills_read = []

    def line(self, kind, text, **extra):
        self.transcript.append((kind, text))
        return len(self.transcript)


def test_the_harness_sees_the_agent_read_a_run_s_end():
    sdk = pytest.importorskip("claude_agent_sdk")
    policy = _policy(level="drive")
    agent = _Agent(policy)
    name = f"mcp__{harness_mod.SERVER}__service_wait_for_run"

    def call_and_answer(tool_id, state, parent=None):
        use = sdk.AssistantMessage(
            content=[
                sdk.ToolUseBlock(id=tool_id, name=name, input={"service": "milling"})
            ],
            model="m",
            parent_tool_use_id=parent,
        )
        harness_mod._show(use, agent)
        result = sdk.UserMessage(
            content=[
                sdk.ToolResultBlock(
                    tool_use_id=tool_id,
                    content=[{"type": "text", "text": _run_answer(state)}],
                    is_error=False,
                )
            ],
            parent_tool_use_id=parent,
        )
        harness_mod._show(result, agent)

    # A subagent's read is its own: the agent has not seen it.
    call_and_answer("t0", "done", parent="t-sub")
    assert policy._runs == []
    call_and_answer("t1", "running")
    assert policy._runs == []
    call_and_answer("t2", "done")
    assert policy._runs == ["r1"]


# --- all: as before ----------------------------------------------------------


def test_under_all_the_prompts_are_as_before():
    """A recorded run's lines, batch by batch, under the base playbook's
    policy at Drive, give exactly the prompts the watcher's lines alone did."""
    words = harness_mod.texts()
    base = playbooks.shipped_base(harness_mod.find_checkout())
    policy = WakeOn(Settings.of(base.session))
    policy.at_level("drive", initial=True)
    wakes = harness_mod.Wakes(words=words, wake_on=policy)
    batches = [
        _lines(("workflow_started", {})),
        _lines(_task("task_started", "01-a", "Setup")),
        _lines(_autofocus("passed"), _task("task_completed", "01-a", "Setup")),
        _lines(_ended("r1", actor="agent:claude")),
        _level("assist", "drive"),
        _lines(
            ("proposal_held", {"item_name": "01-a", "task_name": "Setup"}),
            ("milling_stage_started", {"milling_task_name": "Rough", "stage_index": 0}),
        ),
        _lines(("workflow_completed", {"task_counts": {"completed": 1}})),
    ]
    reread = "\n\n" + words.reread_level
    expected = [
        "The watcher printed:\n"
        + "\n".join(batch)
        + (
            reread
            if batch[0].kind in (watch_mod.LEVEL, watch_mod.LEVEL_LOWERED)
            else ""
        )
        for batch in batches
    ]
    assert [wakes.add(batch) for batch in batches] == expected


def test_the_base_playbook_sets_the_defaults():
    base = playbooks.shipped_base(harness_mod.find_checkout())
    settings = Settings.of(base.session)
    assert (settings.watch, settings.assist, settings.drive) == ("digest", "all", "all")
    assert (settings.every_min, settings.after_lines) == (30, 20)


def test_a_playbook_names_a_known_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    lab = tmp_path / "home" / "knowledge" / "playbooks"
    lab.mkdir(parents=True)
    (lab / "odd.yaml").write_text(
        "extends: autolamella\nsession:\n  wake_on_watch: sometimes\n", encoding="utf-8"
    )
    with pytest.raises(playbooks.PlaybookError, match="wake_on_watch"):
        playbooks.load("odd", harness_mod.find_checkout())
    (lab / "quiet.yaml").write_text(
        "extends: autolamella\nsession:\n  wake_on_drive: digest\n  digest_min: 10\n",
        encoding="utf-8",
    )
    found = playbooks.load("quiet", harness_mod.find_checkout())
    settings = Settings.of(found.session)
    assert (settings.watch, settings.drive, settings.every_min) == (
        "digest",
        "digest",
        10,
    )


def test_the_watcher_s_lines_carry_what_the_policy_reads():
    (passed, doubtful) = _lines(_autofocus("passed"), _autofocus("doubtful"))
    assert passed.facts == {"ok": True} and doubtful.facts == {"ok": False}
    (mill,) = _lines(
        (
            "milling",
            {
                "milling_run_id": "r1",
                "milling_task_name": "Trench",
                "status": "completed",
                "stages": [
                    {"name": "Rough", "status": "completed"},
                    {
                        "name": "Polish",
                        "status": "skipped",
                        "reason": "turned off in the protocol",
                    },
                ],
            },
        )
    )
    assert mill.kind == fibsem_lines.MILL_ENDED and mill.facts == {"ok": True}
    (early,) = _lines(
        (
            "milling",
            {
                "milling_run_id": "r1",
                "milling_task_name": "Polish",
                "status": "completed",
                "stages": [
                    {
                        "name": "Polish",
                        "status": "completed",
                        "end": {"kind": "ended", "by": "monitor", "reason": "drop"},
                    }
                ],
            },
        )
    )
    assert early.facts == {"ok": False}
    (level,) = _level("watch", "drive")
    assert level.facts == {"level": "watch"}
    (mine, theirs) = _lines(_ended("r1", actor="agent:claude"), _ended("r2"))
    assert mine.facts == {"run_id": "r1", "state": "done", "ok": False}
    assert theirs.facts["ok"] is True


def test_the_watch_cost_note_waits_while_a_context_mark_is_said():
    """The fresh-context mark line says the context's size: the policy's note
    at Watch is not said in the same turn, but on a later one."""
    marked = [True]
    policy = _policy(context=lambda: (10.0, wake_policy.LARGE_CONTEXT + 1))
    wakes = harness_mod.Wakes(wake_on=policy, sizes_said=lambda: marked[0])
    assert "context_reset" not in wakes.add([_wake(watch_mod.MESSAGE)])
    marked[0] = False
    assert "a fresh context with context_reset" in wakes.add([_wake(watch_mod.MESSAGE)])


def test_held_lines_come_with_the_harness_s_own_wake_up():
    """The harness's own ALARM wake (its timer, the step limit, a fresh
    context's first turn) takes the held lines with it: the hold lives in the
    harness, not in the model's context."""
    policy = _policy()
    wakes = harness_mod.Wakes(wake_on=policy)
    assert wakes.add(_lines(_task("task_completed", "01", "Setup"))) is None
    prompt = wakes.add([_wake(harness_mod.ALARM, "WAKE carry on")])
    assert prompt.startswith("Held for you since ")
    assert "EVENT task_completed 01 Setup" in prompt and not policy.held


def test_a_side_question_never_reaches_the_wake_queue():
    async def main():
        policy = _policy()
        harness = harness_mod.Harness(object(), lambda: None, wake_on=policy)
        harness._items, harness._urgent = asyncio.Queue(), asyncio.Event()
        harness._watcher = watcher = object()
        added = []
        harness._asides.add = added.append
        question = watch_mod.Aside("m1", "what is 01 doing?", "MESSAGE id=m1 ...")
        harness._put(watcher, question)
        assert harness._items.empty() and added == [question] and not policy.held

    asyncio.run(main())


def test_a_resume_is_not_kept_past_lines_that_wait():
    """The event cursor a resume starts from is kept only once every line
    before it has reached the agent: held lines are given again."""
    kept = []
    policy = _policy()
    harness = harness_mod.Harness(
        object(), lambda: None, keep_cursor=kept.append, wake_on=policy
    )
    policy.keep_held(_lines(_task("task_completed", "01", "Setup"))[0])
    harness._cursor_given = "s-5"
    harness._keep_given_cursor()
    assert kept == []
    policy.take()
    harness._keep_given_cursor()
    assert kept == ["s-5"]


def test_a_resume_keeps_the_cursor_before_the_first_held_line():
    """While lines wait for a digest, the cursor kept is the newest one before
    the first of them, not none: a resume gives again only what waits."""

    async def main():
        kept = []
        policy = _policy()
        harness = harness_mod.Harness(
            object(), lambda: None, keep_cursor=kept.append, wake_on=policy
        )
        harness._items, harness._urgent = asyncio.Queue(), asyncio.Event()
        harness._stopping = asyncio.Event()
        harness._watcher = watcher = object()
        harness._items.put_nowait((watcher, _wake(harness_mod.ALARM, "WAKE look")))
        harness._items.put_nowait((watcher, harness_mod._Cursor("s-1")))
        assert await harness._next_prompt()
        # A routine line is held; its batch's cursor is not kept.
        for item in _lines(_task("task_completed", "01", "Setup")) + [
            harness_mod._Cursor("s-2")
        ]:
            harness._items.put_nowait((watcher, item))
        waiting = asyncio.ensure_future(harness._next_prompt())
        for _ in range(5):
            await asyncio.sleep(0)
        waiting.cancel()
        assert policy.held and harness._cursor_given == "s-2"
        harness._keep_given_cursor()
        assert kept == ["s-1"]
        # Once the digest is delivered, the newest cursor is kept.
        policy.take()
        harness._keep_given_cursor()
        assert kept == ["s-1", "s-2"]

    asyncio.run(main())


def test_show_config_says_what_wakes_the_agent(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    assert harness_mod.main(["--show-config", "--json"]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["wake_on"] == {
        "watch": "digest",
        "assist": "all",
        "drive": "all",
        "digest_min": 30,
        "digest_lines": 20,
    }
    assert harness_mod._wake_on_text(config["wake_on"]) == (
        "Wakes:    every line at Drive and Assist; at Watch, routine lines in a "
        "digest every 30 min or 20 lines (the agent changes it with wake_on)"
    )
