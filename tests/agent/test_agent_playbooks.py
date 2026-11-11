"""Playbooks: the skills, reminders and rules a kind of job runs with, set at
intake and switched while the session runs."""

import asyncio
import json
import re

import pytest

from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import playbooks
from fibsem.ai.agent import reminders as reminder_mod

SERVER = harness_mod.SERVER


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    return tmp_path / "home" / "knowledge" / "playbooks"


@pytest.fixture
def checkout():
    found = harness_mod.find_checkout()
    assert found is not None
    return found


def write(folder, name, text):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.yaml").write_text(text, encoding="utf-8")


def call(tool, **tool_input):
    return {"tool_name": f"mcp__{SERVER}__{tool}", "tool_input": tool_input}


def test_the_shipped_playbooks_load_and_correlative_extends_autolamella(lab, checkout):
    assert {"autolamella", "autolamella-correlative", "fibsem-experiment"} <= set(
        playbooks.available(checkout)
    )
    correlative = playbooks.load("autolamella-correlative", checkout)
    assert correlative.chain == [
        "fibsem-agent",
        "autolamella",
        "autolamella-correlative",
    ]
    assert correlative.skills == [
        "fibsem-agent",
        "fibsem-microscope",
        "autolamella",
        "autolamella-correlative",
    ]
    optional = [o.skill for o in correlative.optional]
    assert "autolamella-polishing" in optional
    assert optional[-1] == "autolamella-coincidence-milling"
    polishing = correlative.optional_skill("autolamella-polishing")
    assert polishing.when == "before the first polishing pass"
    assert {h.when for h in polishing.hooks} == {"polishing_pass", "after_polish_pass"}
    summary = correlative.summary()
    assert summary["extends"] == ["fibsem-agent", "autolamella"]
    labels = [r["label"] for r in summary["reminders"]]
    assert "Placing a lamella at Setup" in labels
    # Parent first: autolamella's before_milling, then the correlative one's.
    milling = [r["say"] for r in summary["reminders"] if r["when"] == "before_milling"]
    assert len(milling) == 2 and "FM stack" in milling[1]


def test_the_labs_playbook_wins_and_a_bad_one_says_why(lab, checkout):
    write(lab, "autolamella", "description: ours\nskills: [autolamella]\n")
    assert playbooks.load("autolamella", checkout).description == "ours"
    assert playbooks.find("autolamella", checkout).is_relative_to(lab)

    write(lab, "loop-a", "extends: loop-b\n")
    write(lab, "loop-b", "extends: loop-a\n")
    with pytest.raises(playbooks.PlaybookError, match="loop"):
        playbooks.load("loop-a", checkout)
    write(lab, "orphan", "extends: no-such-playbook\n")
    with pytest.raises(playbooks.PlaybookError, match="no-such-playbook"):
        playbooks.load("orphan", checkout)
    write(lab, "typo", "skills: [no-such-skill]\n")
    with pytest.raises(playbooks.PlaybookError, match="no-such-skill"):
        playbooks.load("typo", checkout)
    write(lab, "wrong-when", "reminders:\n  someday: hello\n")
    with pytest.raises(playbooks.PlaybookError, match="someday"):
        playbooks.load("wrong-when", checkout)


def test_an_optional_skills_reminders_are_on_only_while_it_is_loaded(lab, checkout):
    changes = []
    reminders = playbooks.Reminders(on_change=lambda: changes.append(1))
    assert reminders.texts("polishing_pass") == []
    assert reminders.texts("hardware_error") == [], "none without a playbook"

    reminders.set(playbooks.load("autolamella", checkout), "named at launch")
    assert reminders.texts("setup_placement"), "the playbook's own are on at once"
    assert reminders.texts("hardware_error"), "and those of the base it extends"
    assert reminders.texts("polishing_pass") == []
    assert reminders.loaded("autolamella-polishing") is not None
    assert reminders.texts("polishing_pass")
    assert reminders.loaded("fibsem-agent") is None, "not an optional skill"
    assert reminders.finished("autolamella-polishing", "two passes, 180 nm")
    assert reminders.texts("polishing_pass") == []
    assert not reminders.finished("autolamella-polishing", "again")

    state = reminders.state()["playbook"]
    assert state["name"] == "autolamella" and state["hash"]
    rows = {row["skill"]: row for row in state["skills"]}
    assert rows["autolamella-polishing"] == {
        "skill": "autolamella-polishing",
        "state": "done",
        "summary": "two passes, 180 nm",
    }
    assert rows["screen-grids"]["state"] == "not loaded"
    assert changes

    resumed = playbooks.Reminders()
    assert resumed.restore({"playbook": state}, checkout) is None
    assert resumed.state() == reminders.state()
    assert "no-such" in resumed.restore({"playbook": {"name": "no-such"}}, checkout)


def test_setup_and_polishing_reminders_fire_on_their_own_tasks():
    setup = reminder_mod.HOOK_OPTIONS["setup_placement"]
    polishing = reminder_mod.HOOK_OPTIONS["polishing_pass"]
    assert setup.applies(call("proposal_decide", task_name="Setup Lamella"), SERVER)
    assert setup.applies(
        call("proposal_suggest", task_name="Select Milling Position"), SERVER
    )
    assert not setup.applies(call("proposal_decide", task_name="Polishing"), SERVER)
    assert polishing.applies(call("proposal_decide", task_name="Polishing"), SERVER)
    assert not polishing.applies(
        call("proposal_decide", task_name="Rough Mill"), SERVER
    )
    assert not polishing.applies(call("proposal_list", task_name="Polishing"), SERVER)


def test_a_reminder_is_said_only_when_on_and_its_call_matches(lab, checkout):
    reminders = playbooks.Reminders()
    hook = reminders.callback("setup_placement", SERVER)
    decide = call("proposal_decide", task_name="Setup Lamella")
    assert asyncio.run(hook(decide, "t1", {})) == {}
    reminders.set(playbooks.load("autolamella", checkout), "the operator")
    said = asyncio.run(hook(decide, "t2", {}))["hookSpecificOutput"]
    assert said["hookEventName"] == "PreToolUse"
    assert said["additionalContext"].startswith("Reminder: ")
    assert asyncio.run(hook(call("milling_stop"), "t3", {})) == {}


def test_the_playbook_tools_say_each_change_and_keep_it(lab, checkout):
    lines, kept = [], []
    control = harness_mod.PlaybookControl(
        playbooks.Reminders(), checkout, say=lines.append, keep=kept.append
    )
    listed = json.loads(control.list())
    assert listed["current"] is None
    assert "autolamella" in [row["name"] for row in listed["playbooks"]]

    assert control.set("no-such", "the operator").startswith("Refused")
    answer = control.set("autolamella", "agreed with the operator")
    assert "Running the autolamella playbook" in answer
    control.loaded("autolamella")  # a required skill: no line
    control.loaded("autolamella-polishing")
    assert control.done("screen-grids", "x").startswith("Refused")
    control.done("autolamella-polishing", "at 180 nm")
    assert lines == [
        "Playbook set: autolamella, agreed with the operator",
        "Skill loaded: autolamella-polishing, because before the first polishing pass",
        "Skill done: autolamella-polishing, at 180 nm",
    ]
    assert kept[-1]["playbook"]["skills"]
    assert "setup_placement" in {row["when"] for row in control.hooks()}

    answer = control.set("", "the operator")
    assert lines[-1] == "Playbook set: fibsem-agent, the operator"
    assert kept[-1]["playbook"]["name"] == "fibsem-agent"
    assert '"session"' not in answer, "a running session keeps its settings"


def test_show_config_names_the_playbook_it_starts_on(
    lab, checkout, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(harness_mod.Path, "home", lambda: tmp_path)
    argv = ["--show-config", "--json", "--playbook", "autolamella"]
    assert harness_mod.main(argv) == 0
    shown = json.loads(capsys.readouterr().out)["playbook"]
    assert shown["name"] == "autolamella"
    assert "autolamella-polishing" in [o["skill"] for o in shown["optional"]]

    write(lab, "broken", "extends: broken\n")
    assert harness_mod.main(["--show-config", "--playbook", "broken"]) == 1
    assert "extends itself" in capsys.readouterr().err


def test_a_session_setting_that_cant_be_used_says_why(lab, checkout):
    for text, said in (
        ("session:\n  level: max\n", "level is one of"),
        ("session:\n  api_key: sk\n", "unknown session setting(s) api_key"),
        ("session:\n  mode: everything\n", "mode is one of"),
        ("session:\n  effort: huge\n", "effort is one of"),
        ("session:\n  max_budget_usd: -1\n", "above 0"),
        ("session: [a]\n", "a mapping"),
    ):
        write(lab, "bad", text)
        with pytest.raises(playbooks.PlaybookError, match=re.escape(said)):
            playbooks.load("bad", checkout, harness_mod.EFFORTS)


def test_session_settings_add_up_along_extends_the_child_winning(lab, checkout):
    write(lab, "parent", "session:\n  effort: low\n  max_turns: 5\n  mode: code\n")
    write(lab, "child", "extends: parent\nsession:\n  max_turns: 9\n")
    found = playbooks.load("child", checkout).session
    assert (found.effort, found.max_turns, found.mode) == ("low", 9, "code")
    summary = playbooks.load("child", checkout).summary()["session"]
    assert {k: summary[k] for k in ("effort", "max_turns", "mode")} == {
        "effort": "low",
        "max_turns": 9,
        "mode": "code",
    }
    # Under it all, the shipped base playbook's: where the defaults live.
    base = playbooks.shipped_base(checkout).session
    assert summary["max_budget_usd"] == base.max_budget_usd
    assert summary["wake_max_min"] == base.wake_max_min


def test_a_finished_task_reminds_with_the_watchers_turn(lab, checkout):
    after = reminder_mod.HOOK_OPTIONS["after_task"]
    assert after.on_lines(["EVENT task_completed 01 Rough Mill"])
    assert not after.on_lines(["EVENT task_started 01 Rough Mill"])
    reminders = playbooks.Reminders()
    reminders.set(playbooks.load("autolamella", checkout), "the operator")
    wakes = harness_mod.Wakes(reminders.for_lines)
    done = ["EVENT task_completed 01 Rough Mill"]
    assert "Reminder:" not in wakes.add(done), "only while autolamella-lead is read"
    reminders.loaded("autolamella-lead")
    prompt = wakes.add(done)
    assert prompt.endswith(reminder_mod.reminder(reminders.texts("after_task")))
    assert "Reminder:" not in wakes.add(["EVENT task_started 01 Polishing"])
    # Never an SDK hook: no tool call means a task finished.
    pytest.importorskip("claude_agent_sdk")
    matchers = reminders.sdk_hooks(SERVER)
    assert reminder_mod.WATCHER not in matchers


def test_a_playbook_names_its_plan_headings_or_has_the_default(lab, checkout):
    assert playbooks.load("autolamella", checkout).plan == playbooks.PLAN
    write(lab, "short", "plan: [Brief, Steps]\n")
    write(lab, "shorter", "extends: short\n")
    assert playbooks.load("shorter", checkout).summary()["plan"] == ["Brief", "Steps"]
    write(lab, "bad-plan", "plan: ['## Brief']\n")
    with pytest.raises(playbooks.PlaybookError, match="headings"):
        playbooks.load("bad-plan", checkout)
