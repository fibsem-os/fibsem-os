"""``fibsem-agent``'s reminders (hooks), and what a session starts with: the
playbook named at launch, else the base ``fibsem-agent``, its session settings
and rules."""

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import playbooks, reminders, watch
from fibsem.ai.agent.catalog import CATALOG
from fibsem.ai.agent.reminders import HOOK_OPTIONS

SERVER = harness_mod.SERVER
TOOLS = sorted(spec.name for spec in CATALOG)


@pytest.fixture
def checkout():
    found = harness_mod.find_checkout()
    assert found is not None, "tests run from a checkout with .claude/skills"
    return found


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("FIBSEM_HOME", raising=False)
    return tmp_path


def write(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def lab(home: Path) -> Path:
    return home / ".fibsem" / "knowledge" / "playbooks"


@pytest.mark.parametrize("when", sorted(HOOK_OPTIONS))
def test_every_hook_option_fires_on_a_tool_that_exists(when):
    """A renamed tool fails here rather than quietly turning a reminder off."""
    option = HOOK_OPTIONS[when]
    if option.event == reminders.WATCHER:
        # The watcher prints "EVENT <kind> ..." for the kinds it watches.
        assert any(option.on_lines([f"EVENT {k} 01 Task"]) for k in watch.INTERESTING)
        return
    for part in option.tools.split("|"):
        assert any(re.fullmatch(part, tool) for tool in TOOLS), (when, part)


def call(tool, **extra):
    return {"tool_name": f"mcp__{SERVER}__{tool}", "tool_input": {}, **extra}


def test_a_hook_option_matches_its_tools_and_nothing_else():
    milling = HOOK_OPTIONS["before_milling"]
    assert milling.applies(call("milling_run_stage"), SERVER)
    assert not milling.applies(call("milling_stop"), SERVER)
    assert not milling.applies({"tool_name": "milling_run_stage"}, SERVER)
    assert not milling.applies({"tool_name": "mcp__other__milling_run_stage"}, SERVER)


def test_a_hardware_error_is_a_server_error_never_a_refusal():
    error = HOOK_OPTIONS["hardware_error"]
    assert error.applies(call("stage_move_absolute", error="failed (500): x"), SERVER)
    assert not error.applies(
        call("stage_move_absolute", error="refused (403): x"), SERVER
    )


def test_the_polish_pass_reminder_fires_only_on_an_ended_iterative_pass():
    after = HOOK_OPTIONS["after_polish_pass"]
    ended = {"stages": [{"strategy": {"facts": {"Pass data": "pass.json"}}}]}
    assert after.applies(call("service_wait_for_run", tool_response=ended), SERVER)
    assert not after.applies(
        call("service_wait_for_run", tool_response={"state": "running"}), SERVER
    )


def test_a_reminder_reaches_the_model_only_when_its_call_matches():
    hook = reminders.callback(
        HOOK_OPTIONS["before_milling"], ["Check depth.", "And current."], SERVER
    )
    said = asyncio.run(hook(call("milling_run_task"), "t1", {}))
    assert said["hookSpecificOutput"] == {
        "hookEventName": "PreToolUse",
        "additionalContext": "Reminder: Check depth. And current.",
    }
    assert asyncio.run(hook(call("milling_stop"), "t2", {})) == {}


def test_reminders_become_the_sdks_hooks_one_matcher_per_option():
    pytest.importorskip("claude_agent_sdk")
    hooks = [
        reminders.Hook("before_milling", "a"),
        reminders.Hook("before_milling", "b"),
        reminders.Hook("hardware_error", "c"),
    ]
    matchers = reminders.sdk_hooks(hooks, SERVER)
    assert set(matchers) == {"PreToolUse", "PostToolUseFailure"}
    (milling,) = matchers["PreToolUse"]
    assert milling.matcher == f"mcp__{SERVER}__(milling_run_.*)"
    assert re.fullmatch(milling.matcher, f"mcp__{SERVER}__milling_run_stage")
    assert reminders.sdk_hooks([], SERVER) is None


def test_the_base_playbook_holds_the_rules_every_session_starts_with(checkout):
    base = playbooks.load(playbooks.DEFAULT, checkout, harness_mod.EFFORTS)
    assert base.session.mode == "files" and base.session.max_budget_usd == 500
    assert [hook.when for hook in base.hooks] == ["hardware_error"]
    # The rules that hold on every task, read every turn after the harness's.
    prompt = harness_mod.system_prompt(playbook=base)
    assert "Rules of the fibsem-agent playbook you started on" in prompt
    assert "auto function" in prompt and "coincidence" in prompt
    assert "50 µm field of view" in prompt


def test_the_shipped_playbooks_build_on_the_base(checkout):
    base = playbooks.load(playbooks.DEFAULT, checkout, harness_mod.EFFORTS)
    for name in playbooks.available(checkout):
        found = playbooks.load(name, checkout, harness_mod.EFFORTS)
        assert found.description, name
        assert found.chain[0] == playbooks.DEFAULT, name
        assert found.session == base.session, "they keep the base's settings"
        assert found.rules[: len(base.rules)] == base.rules, name


def show(capsys, *argv):
    assert harness_mod.main(["--show-config", "--json", *argv]) == 0
    return json.loads(capsys.readouterr().out)


def test_show_config_prints_what_the_agent_runs_with_and_starts_nothing(
    home, capsys, checkout
):
    write(
        lab(home),
        "night",
        "extends: fibsem-agent\n"
        f"session:\n  effort: low\n  max_turns: 10\n  folder: {home / 'w'}\n"
        "  level: assist\n"
        "rules: [Prefer leaving a lamella unfinished.]\n"
        "reminders:\n  before_milling: Check the depth.\n",
    )
    config = show(capsys, "--playbook", "night")
    assert config["playbook"]["name"] == "night"
    assert config["playbook"]["extends"] == ["fibsem-agent"]
    assert config["level"] == "assist", "for the gateway's Level picker"
    assert (config["effort"], config["max_turns"]) == ("low", 10)
    assert config["max_budget_usd"] == 500, "the base's"
    assert (config["mode"], config["folder"]) == ("files", str(home / "w"))
    assert {
        "when": "before_milling",
        "label": "Before milling",
        "say": "Check the depth.",
    } in config["playbook"]["reminders"]
    assert config["system_prompt"].endswith("- Prefer leaving a lamella unfinished.")
    assert "Write" in config["tools"]

    config = show(capsys, "--playbook", "night", "--effort", "max", "--max-turns", "3")
    assert (config["effort"], config["max_turns"]) == ("max", 3), "flags win"

    config = show(capsys)
    assert config["playbook"]["name"] == "fibsem-agent", "the default"
    assert config["mode"] == "files"
    assert config["level"] is None


def test_with_no_folder_the_agent_works_in_its_sessions_folder(home, capsys):
    sessions = home / ".fibsem" / "agent" / "sessions"
    # Code mode is refused on Windows (no sandbox).
    modes = [["--files"]] + ([] if sys.platform == "win32" else [["--code"]])
    for argv in modes:
        folder = Path(show(capsys, *argv)["folder"])
        assert folder.name == "work" and folder.parent.parent == sessions
    write(lab(home), "own", "session:\n  mode: files\n")
    folder = Path(show(capsys, "--playbook", "own")["folder"])
    assert folder.name == "work" and folder.parent.parent == sessions
    write(lab(home), "nothing", "session:\n  mode: none\n")
    assert show(capsys, "--playbook", "nothing")["folder"] is None
    assert not sessions.exists(), "showing the config makes nothing"


def test_an_unknown_playbook_says_where_it_looked(home, capsys):
    assert harness_mod.main(["--playbook", "nope", "--show-config"]) == 1
    assert "No playbook 'nope'" in capsys.readouterr().err


def test_profile_is_gone(home, capsys):
    with pytest.raises(SystemExit):
        harness_mod.main(["--profile", "fibsem-agent", "--show-config"])


def test_the_lists_print_playbooks_and_hook_options(home, capsys):
    write(lab(home), "broken", "session:\n  mode: what\n")
    assert harness_mod.main(["--list-playbooks", "--json"]) == 0
    rows = {row["name"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["fibsem-agent"]["mode"] == "files"
    assert "mode is one of" in rows["broken"]["error"]

    assert harness_mod.main(["--list-hooks", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["when"] for row in rows] == list(HOOK_OPTIONS)


def test_the_session_carries_the_starting_playbooks_rules_and_reminders(checkout):
    pytest.importorskip("claude_agent_sdk")
    base = playbooks.load(playbooks.DEFAULT, checkout)
    options = harness_mod.build_options(checkout, playbook=base)
    assert options.skills == list(harness_mod.SKILLS)
    assert options.system_prompt["append"].endswith(f"- {base.rules[-1]}")
    assert list(options.hooks) == ["PostToolUseFailure"]
    assert harness_mod.build_options(checkout).hooks is None, "no playbook, no hooks"
