"""``fibsem-agent`` profiles and the reminders (hooks) they set."""

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import profiles
from fibsem.ai.agent.catalog import CATALOG
from fibsem.ai.agent.profiles import HOOK_OPTIONS, Profile, ProfileError

SERVER = harness_mod.SERVER
TOOLS = sorted(spec.name for spec in CATALOG)


@pytest.fixture
def checkout():
    found = harness_mod.find_checkout()
    assert found is not None, "tests run from a checkout with .claude/skills"
    return found


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles.Path, "home", lambda: tmp_path)
    return tmp_path


def write(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("when", sorted(HOOK_OPTIONS))
def test_every_hook_option_fires_on_a_tool_that_exists(when):
    """A renamed tool fails here rather than quietly turning a reminder off."""
    option = HOOK_OPTIONS[when]
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
    hook = profiles.callback(
        HOOK_OPTIONS["before_milling"], ["Check depth.", "And current."], SERVER
    )
    said = asyncio.run(hook(call("milling_run_task"), "t1", {}))
    assert said["hookSpecificOutput"] == {
        "hookEventName": "PreToolUse",
        "additionalContext": "Reminder from your profile: Check depth. And current.",
    }
    assert asyncio.run(hook(call("milling_stop"), "t2", {})) == {}


def test_reminders_become_the_sdks_hooks_one_matcher_per_option():
    pytest.importorskip("claude_agent_sdk")
    hooks = [
        profiles.Hook("before_milling", "a"),
        profiles.Hook("before_milling", "b"),
        profiles.Hook("hardware_error", "c"),
    ]
    matchers = profiles.sdk_hooks(hooks, SERVER)
    assert set(matchers) == {"PreToolUse", "PostToolUseFailure"}
    (milling,) = matchers["PreToolUse"]
    assert milling.matcher == f"mcp__{SERVER}__(milling_run_.*)"
    assert re.fullmatch(milling.matcher, f"mcp__{SERVER}__milling_run_stage")
    assert profiles.sdk_hooks([], SERVER) is None


def test_a_profile_loads_with_its_hash(tmp_path, checkout):
    path = write(
        tmp_path,
        "overnight",
        "model: opus\neffort: high\nmax_budget_usd: 40\nmode: files\nfolder: ~/w\n"
        "skills: [autolamella-agent]\nprompt: Overnight run.\n"
        "hooks:\n  - when: site\n    say: Coincidence checked here?\n",
    )
    profile = profiles.load(path, harness_mod.EFFORTS, checkout)
    assert profile.name == "overnight"
    assert (profile.model, profile.effort, profile.max_budget_usd) == (
        "opus",
        "high",
        40.0,
    )
    assert profile.hooks == [profiles.Hook("site", "Coincidence checked here?")]
    assert len(profile.hash) == 12
    path.write_text(path.read_text() + "max_turns: 10\n")
    assert profiles.load(path).hash != profile.hash, "an edit changes the hash"


@pytest.mark.parametrize(
    "text, said",
    [
        ("level: drive\n", "unknown setting(s) level"),
        ("api_key: sk\n", "unknown setting(s) api_key"),
        ("mode: everything\n", "mode is one of"),
        ("effort: huge\n", "effort is one of"),
        ("max_budget_usd: -1\n", "above 0"),
        ("skills: [no-such-skill]\n", "no skill no-such-skill"),
        ("hooks:\n  - when: lunch\n    say: eat\n", "no hook option 'lunch'"),
        ("hooks:\n  - when: site\n", "a when and a say"),
        ("- a list\n", "a mapping"),
    ],
)
def test_a_profile_that_cant_be_used_says_why(tmp_path, checkout, text, said):
    path = write(tmp_path, "bad", text)
    with pytest.raises(ProfileError, match=re.escape(said)):
        profiles.load(path, harness_mod.EFFORTS, checkout)


def test_this_machines_profile_wins_over_the_checkouts(home, tmp_path):
    checkout = tmp_path / "checkout"
    shipped = write(checkout / profiles.PROFILE_DIR, "default", "effort: low\n")
    assert profiles.find("default", checkout) == shipped
    own = write(home / ".fibsem" / "agent" / "profiles", "default", "effort: max\n")
    assert profiles.find("default", checkout) == own
    assert profiles.available(checkout) == [own]
    assert profiles.find("../default", checkout) is None


def test_the_shipped_default_profile_loads(checkout):
    path = checkout / profiles.PROFILE_DIR / f"{profiles.DEFAULT_PROFILE}.yaml"
    profile = profiles.load(path, harness_mod.EFFORTS, checkout)
    assert profile.mode == "none"
    assert [hook.when for hook in profile.hooks] == [
        "hardware_error",
        "after_polish_pass",
    ]
    # The rules that hold on every task, read every turn after the harness's.
    prompt = harness_mod.system_prompt(profile=profile)
    assert "auto function" in prompt and "coincidence" in prompt


def show(capsys, *argv):
    assert harness_mod.main(["--show-config", "--json", *argv]) == 0
    return json.loads(capsys.readouterr().out)


def test_show_config_prints_what_the_agent_runs_with_and_starts_nothing(
    home, capsys, checkout
):
    own = home / ".fibsem" / "agent" / "profiles"
    write(
        own,
        "night",
        f"effort: low\nmax_turns: 10\nmode: files\nfolder: {home / 'w'}\n"
        "prompt: Prefer leaving a lamella unfinished.\n"
        "hooks:\n  - when: before_milling\n    say: Check the depth.\n",
    )
    config = show(capsys, "--profile", "night")
    assert config["profile"]["name"] == "night"
    assert (config["effort"], config["max_turns"]) == ("low", 10)
    assert config["max_budget_usd"] == harness_mod.MAX_BUDGET_USD, "the default"
    assert (config["mode"], config["folder"]) == ("files", str(home / "w"))
    assert config["hooks"] == [
        {"when": "before_milling", "label": "Before milling", "say": "Check the depth."}
    ]
    assert config["system_prompt"].endswith("Prefer leaving a lamella unfinished.")
    assert "Write" in config["tools"]

    config = show(capsys, "--profile", "night", "--effort", "max", "--max-turns", "3")
    assert (config["effort"], config["max_turns"]) == ("max", 3), "flags win"

    config = show(capsys)
    assert config["profile"]["name"] == "fibsem-agent", "the default"
    assert config["mode"] == "none"
    assert [hook["when"] for hook in config["hooks"]] == [
        "hardware_error",
        "after_polish_pass",
    ]


def test_with_no_folder_the_agent_works_in_its_sessions_folder(home, capsys):
    sessions = home / ".fibsem" / "agent" / "sessions"
    # Code mode is refused on Windows (no sandbox).
    modes = [["--files"]] + ([] if sys.platform == "win32" else [["--code"]])
    for argv in modes:
        folder = Path(show(capsys, *argv)["folder"])
        assert folder.name == "work" and folder.parent.parent == sessions
    write(home / ".fibsem" / "agent" / "profiles", "own", "mode: files\n")
    folder = Path(show(capsys, "--profile", "own")["folder"])
    assert folder.name == "work" and folder.parent.parent == sessions
    assert not sessions.exists(), "showing the config makes nothing"


def test_an_unknown_profile_says_where_it_looked(home, capsys):
    assert harness_mod.main(["--profile", "nope", "--show-config"]) == 1
    assert "No profile 'nope'" in capsys.readouterr().err


def test_the_lists_print_profiles_and_hook_options(home, capsys):
    write(home / ".fibsem" / "agent" / "profiles", "broken", "mode: what\n")
    assert harness_mod.main(["--list-profiles", "--json"]) == 0
    rows = {row["name"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["fibsem-agent"]["mode"] == "none"
    assert "mode is one of" in rows["broken"]["error"]

    assert harness_mod.main(["--list-hooks", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["when"] for row in rows] == list(HOOK_OPTIONS)


def test_the_session_carries_the_profiles_skills_prompt_and_reminders(checkout):
    pytest.importorskip("claude_agent_sdk")
    profile = Profile(
        name="night",
        path=Path("night.yaml"),
        hash="abc",
        skills=["getting-started-dev"],
        prompt="Overnight.",
        hooks=[profiles.Hook("site", "Check the film.")],
    )
    options = harness_mod.build_options(checkout, profile=profile)
    assert options.skills == list(harness_mod.SKILLS) + ["getting-started-dev"]
    assert options.system_prompt["append"].endswith("Overnight.")
    assert list(options.hooks) == ["PreToolUse"]
    assert harness_mod.build_options(checkout).hooks is None, "no profile, no hooks"


def test_the_session_report_names_the_profile_and_its_reminders():
    profile = Profile(
        name="night",
        path=Path("night.yaml"),
        hash="abc",
        hooks=[profiles.Hook("site", "Check the film.")],
    )
    assert harness_mod._profile_report(profile) == {
        "profile": "night",
        "profile_hash": "abc",
        "hooks": [{"when": "site", "label": "At a site", "say": "Check the film."}],
    }
