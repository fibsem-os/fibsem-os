"""Playbooks: the skills, reminders and rules a kind of job runs with, set at
intake and switched while the session runs."""

import asyncio
import json
import re
import subprocess
import sys

import pytest

from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import playbooks
from fibsem.ai.agent import reminders as reminder_mod
from fibsem.ai.agent import watch as watch_mod

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
    done = [watch_mod.Wake("task_completed", "EVENT task_completed 01 Rough Mill")]
    started = [watch_mod.Wake("task_started", "EVENT task_started 01 Polishing")]
    assert after.on_lines(done)
    assert not after.on_lines(started)
    reminders = playbooks.Reminders()
    reminders.set(playbooks.load("autolamella", checkout), "the operator")
    wakes = harness_mod.Wakes(reminders.for_lines)
    assert "Reminder:" not in wakes.add(done), "only while autolamella-lead is read"
    reminders.loaded("autolamella-lead")
    prompt = wakes.add(done)
    assert prompt.endswith(reminder_mod.reminder(reminders.texts("after_task")))
    assert "Reminder:" not in wakes.add(started)
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


def test_the_mcp_servers_a_playbook_uses_are_named_and_the_base_names_fibsem(
    lab, checkout
):
    for name in playbooks.available(checkout):
        assert playbooks.load(name, checkout).mcp[0] == SERVER
    write(lab, "with-notebook", "extends: autolamella\nmcp: [notebook]\n")
    found = playbooks.load("with-notebook", checkout)
    assert found.mcp == [SERVER, "notebook"]
    assert found.summary()["mcp"] == [SERVER, "notebook"]
    write(lab, "bad-name", "mcp: [notebook/search]\n")
    with pytest.raises(playbooks.PlaybookError, match="mcp lists servers by name"):
        playbooks.load("bad-name", checkout)


def test_the_tools_loaded_from_the_start_add_up_and_name_fibsem_tools(lab, checkout):
    base = playbooks.load("fibsem-agent", checkout)
    assert "experiment_message_operator" in base.tools
    autolamella = playbooks.load("autolamella-correlative", checkout)
    assert autolamella.tools[: len(base.tools)] == base.tools
    assert "item_list" in autolamella.tools
    assert autolamella.summary()["tools"] == autolamella.tools
    # A lab's playbook gets the base's under its own, once each.
    write(lab, "mine", "tools: [item_get, app_get_status]\n")
    assert playbooks.load("mine", checkout).tools == base.tools + ["item_get"]
    # Another instrument kind's names its own.
    cryo = playbooks.load("cryo-fm", checkout)
    assert "beam_acquire_image" not in cryo.tools and "device_list" in cryo.tools
    write(lab, "bad-tool", "tools: [mcp__fibsem__app_get_status, no_such]\n")
    with pytest.raises(playbooks.PlaybookError, match="no fibsem tool .*no_such"):
        playbooks.load("bad-tool", checkout)


def test_the_sidecars_mark_the_playbooks_tools_and_show_config_lists_them(
    lab, checkout
):
    pytest.importorskip("claude_agent_sdk")
    playbook = playbooks.load("autolamella", checkout)
    options = harness_mod.build_options(
        checkout, playbook=playbook, extras={}, host="http://host:1"
    )
    flag = ["--always-load", ",".join(playbook.tools)]
    for name in (SERVER, harness_mod.RECORD_SERVER):
        args = options.mcp_servers[name]["args"]
        assert args[args.index("--always-load") :][:2] == flag
    bare = harness_mod.build_options(checkout, extras={})
    assert "--always-load" not in bare.mcp_servers[SERVER]["args"]
    import argparse

    args = argparse.Namespace(
        model=None, effort=None, max_budget_usd=1.0, max_turns=1, host=None
    )
    config = harness_mod.show_config(args, checkout, None, False, playbook, extras={})
    assert config["always_loaded"] == playbook.tools
    assert "Loaded:   server_get_capabilities" in harness_mod._config_text(config)


def test_the_sidecar_marks_the_tools_it_is_told_to_have_loaded_first():
    pytest.importorskip("mcp")
    from fibsem.ai.agent.mcp.sidecar import ALWAYS_LOAD, build_sidecar

    sidecar = build_sidecar(None, None, always_load={"app_get_status", "no_such"})
    listed = asyncio.run(sidecar.list_tools())
    marked = [
        t.name
        for t in listed
        if (t.model_dump(by_alias=True).get("_meta") or {}).get(ALWAYS_LOAD) is True
    ]
    assert marked == ["app_get_status"]


def test_each_shipped_playbooks_tools_are_on_its_instrument(lab, checkout):
    # A tool its instrument kind hides would be marked for nothing.
    pytest.importorskip("mcp")
    from fibsem.ai.agent.mcp.sidecar import build_sidecar

    offered = {}
    for name in playbooks.available(checkout):
        playbook = playbooks.load(name, checkout)
        kind = playbook.instrument
        if kind not in offered:
            listed = asyncio.run(build_sidecar(None, None, kind).list_tools())
            offered[kind] = {t.name for t in listed}
        assert set(playbook.tools) <= offered[kind], name


def _notebook(name="notebook"):
    from fibsem.ai.agent import extra_mcp

    return {
        name: extra_mcp.Server(name, ["search"], url="https://notebook.example/mcp")
    }


def test_a_server_the_playbook_names_must_be_set_up_and_fibsem_is_needed(lab, checkout):
    write(lab, "with-notebook", "mcp: [notebook]\n")
    playbook = playbooks.load("with-notebook", checkout)
    with pytest.raises(harness_mod.SetupRefused, match="notebook.*doesn't set up"):
        harness_mod.mcp_names(playbook, {})
    assert harness_mod.mcp_names(playbook, _notebook()) == [SERVER, "notebook"]
    # Without a playbook, the sidecar alone, as before.
    assert harness_mod.mcp_names(None, {}) == [SERVER]
    playbook.mcp = ["notebook"]
    with pytest.raises(harness_mod.SetupRefused, match="has no fibsem"):
        harness_mod.mcp_names(playbook, _notebook())


def test_the_session_gets_the_playbooks_servers_and_no_other_of_the_labs(lab, checkout):
    pytest.importorskip("claude_agent_sdk")
    write(lab, "with-notebook", "mcp: [notebook]\n")
    playbook = playbooks.load("with-notebook", checkout)
    extras = {**_notebook(), **_notebook("library")}
    options = harness_mod.build_options(checkout, playbook=playbook, extras=extras)
    # The playbook's, in its order; the lab's library it doesn't name is off.
    assert list(options.mcp_servers) == [SERVER, "notebook"]
    assert options.mcp_servers["notebook"]["url"] == "https://notebook.example/mcp"
    assert not any("library" in tool for tool in options.allowed_tools)
    assert "mcp__library__" not in options.system_prompt["append"]


def test_show_config_names_the_lab_servers_a_playbook_leaves_off(lab, checkout):
    import argparse

    write(lab, "with-notebook", "mcp: [notebook]\n")
    playbook = playbooks.load("with-notebook", checkout)
    extras = {**_notebook(), **_notebook("library")}
    args = argparse.Namespace(
        model=None, effort=None, max_budget_usd=1.0, max_turns=1, host=None
    )
    config = harness_mod.show_config(
        args, checkout, None, False, playbook, extras=extras
    )
    assert config["mcp"] == [SERVER, "notebook"]
    assert config["mcp_unused"] == ["library"]
    assert list(config["extra_mcp"]) == ["notebook"]
    assert "MCP off" in harness_mod._config_text(config)


def test_with_an_agent_host_fibsem_is_one_server_among_others(lab, checkout):
    write(lab, "notebook-only", "mcp: [notebook]\n")
    playbook = playbooks.load("notebook-only", checkout)
    playbook.mcp = ["notebook"]
    # The record and wake-ups go through the host: no instrument is needed.
    assert harness_mod.mcp_names(playbook, _notebook(), host=True) == ["notebook"]


def test_with_an_agent_host_the_record_tools_go_to_it(lab, checkout):
    pytest.importorskip("claude_agent_sdk")
    playbook = playbooks.load("autolamella", checkout)
    options = harness_mod.build_options(
        checkout,
        playbook=playbook,
        extras={},
        host="http://127.0.0.1:8030",
        host_token="host-secret",
    )
    record = options.mcp_servers[harness_mod.RECORD_SERVER]
    assert record["args"][-7:-2] == [
        "--url",
        "http://127.0.0.1:8030",
        "--token-env",
        "FIBSEM_HOST_TOKEN",
        "--record-only",
    ]
    assert "--no-record" in options.mcp_servers[SERVER]["args"]
    # The token in the environment, never on a command line.
    assert options.env["FIBSEM_HOST_TOKEN"] == "host-secret"
    assert "host-secret" not in json.dumps(options.mcp_servers)
    assert f"mcp__{harness_mod.RECORD_SERVER}__*" in options.allowed_tools


def test_an_instrument_with_servers_of_its_own_runs_through_a_host(lab, checkout):
    # A plugin's instrument kind, its microscope server over loopback HTTP and
    # an analysis server it starts, and no fibsem: the shipped cryo-fm kind
    # stands in for the plugin's.
    pytest.importorskip("claude_agent_sdk")
    from fibsem.ai.agent import extra_mcp, instruments

    write(lab, "own-servers", "instrument: cryo-fm\nmcp: [scope, analysis]\n")
    playbook = playbooks.load("own-servers", checkout)
    # Another kind than the base's: the base's fibsem is not under it.
    assert playbook.mcp == ["scope", "analysis"]
    extras = {
        "scope": extra_mcp.Server(
            "scope", ["image"], url="http://127.0.0.1:8765/mcp", about="the scope"
        ),
        "analysis": extra_mcp.Server(
            "analysis", ["measure"], command=["analysis-server"], about="analysis"
        ),
    }
    options = harness_mod.build_options(
        checkout,
        playbook=playbook,
        extras=extras,
        host="http://127.0.0.1:8030",
        host_token="t",
        words=instruments.load("cryo-fm", checkout),
    )
    assert list(options.mcp_servers)[:2] == ["scope", "analysis"]
    assert SERVER not in options.mcp_servers
    assert harness_mod.RECORD_SERVER in options.mcp_servers
    # The harness's tool line names its own servers' tools, not the fibsem
    # sidecar's; the rest of the prompt is the instrument kind's own words.
    prompt = options.system_prompt["append"]
    assert (
        "Your tools are the mcp__scope__*, mcp__analysis__*, "
        "mcp__fibsem_record__* tools" in prompt
    )
    assert "matching mcp__fibsem__ tool" not in prompt


def _no_instrument(lab, checkout, mcp="[notebook]"):
    write(
        lab,
        "advisor",
        f"description: Plans with a scientist.\ninstrument: none\nmcp: {mcp}\n",
    )
    return playbooks.load("advisor", checkout)


def test_a_playbook_may_have_no_instrument(lab, checkout):
    playbook = _no_instrument(lab, checkout)
    # None of the base's FIB-SEM skills or its fibsem server.
    assert playbook.instrument == "none"
    assert playbook.mcp == ["notebook"]
    assert "fibsem-microscope" not in playbook.skills
    assert harness_mod.mcp_names(playbook, _notebook(), host=True) == ["notebook"]
    # Its record is on a host: without one it is refused, by why.
    with pytest.raises(harness_mod.SetupRefused, match="no instrument.*--host"):
        harness_mod.mcp_names(playbook, _notebook())
    # And it has no fibsem server to name.
    with_fibsem = _no_instrument(lab, checkout, "[fibsem, notebook]")
    with pytest.raises(harness_mod.SetupRefused, match="no fibsem server"):
        harness_mod.mcp_names(with_fibsem, _notebook(), host=True)
    # No servers at all but the host's record is a session too.
    assert harness_mod.mcp_names(_no_instrument(lab, checkout, "[]"), {}, True) == []


def test_a_playbook_with_no_instrument_starts_against_a_bare_host(lab, checkout):
    pytest.importorskip("claude_agent_sdk")
    from fibsem.ai.agent import instruments

    playbook = _no_instrument(lab, checkout)
    words = instruments.load(playbook.instrument, checkout)
    options = harness_mod.build_options(
        checkout,
        playbook=playbook,
        extras=_notebook(),
        host="http://127.0.0.1:8030",
        host_token="t",
        words=words,
    )
    assert list(options.mcp_servers)[:2] == ["notebook", harness_mod.RECORD_SERVER]
    assert SERVER not in options.mcp_servers
    prompt = options.system_prompt["append"]
    assert "with no instrument" in prompt
    assert "mcp__notebook__*, mcp__fibsem_record__*" in prompt
    for microscope in ("microscope", "FIB", "SEM", "stage", "mcp__fibsem__"):
        assert microscope not in prompt
    # Its words name no instrument tool.
    for text in (words.start, words.resumed, words.goal, words.reread_level):
        assert "server_get_capabilities" not in text
        assert "app_get_status" not in text


def test_with_no_instrument_fibsem_agent_imports_no_microscope():
    code = (
        "import sys; from fibsem.ai.agent import harness, instruments; "
        "harness.find_checkout(); "
        "print([m for m in sys.modules if m.startswith(('fibsem.microscope', "
        "'fibsem.drivers', 'fibsem.server.server', 'fibsem.server.app_context'))])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_without_a_host_the_prompt_names_the_fibsem_tools_as_before(checkout):
    assert harness_mod._name_servers(harness_mod.NO_CODE, None) == harness_mod.NO_CODE
    assert harness_mod._name_servers(harness_mod.NO_CODE, [SERVER]) == (
        harness_mod.NO_CODE
    )


# --- wakes: what wakes the agent ---------------------------------------------------


def test_a_playbook_names_what_wakes_its_agent_and_the_nearest_wins(lab, checkout):
    write(lab, "quiet-lead", "extends: autolamella\nwakes: [run, decisions]\n")
    write(lab, "quieter", "extends: quiet-lead\nwakes: [decisions]\n")
    write(lab, "inherits", "extends: quiet-lead\n")
    assert playbooks.load("autolamella", checkout).wakes is None
    assert playbooks.load("quiet-lead", checkout).wakes == ["run", "decisions"]
    assert playbooks.load("quieter", checkout).wakes == ["decisions"]
    assert playbooks.load("inherits", checkout).wakes == ["run", "decisions"]


def test_wakes_names_known_groups(lab, checkout):
    write(lab, "odd", "extends: autolamella\nwakes: [run, stage_moves]\n")
    with pytest.raises(playbooks.PlaybookError, match="stage_moves"):
        playbooks.load("odd", checkout)


def test_every_line_the_watch_says_has_a_group_or_always_wakes():
    from fibsem.ai.agent import wake_lines

    grouped = set().union(*wake_lines.GROUPS.values()) | wake_lines.ALWAYS
    lines = {
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
        *watch_mod.INTERESTING,
    }
    assert lines <= grouped


def _wake(kind):
    return watch_mod.Wake(kind, kind.upper())


def test_a_line_the_playbook_leaves_out_waits_for_the_next_turn():
    from fibsem.ai.agent import wake_lines

    wakes = harness_mod.Wakes(waking=lambda: wake_lines.waking(["run"]))
    assert wakes.add([_wake("autofocus")]) is None
    assert wakes.add([_wake("note")]) is None
    prompt = wakes.add([_wake("workflow_ended")])
    # One turn, with the lines held before it.
    assert "AUTOFOCUS" in prompt and "NOTE" in prompt
    assert "WORKFLOW_ENDED" in prompt


def test_the_operator_always_wakes_the_agent():
    from fibsem.ai.agent import wake_lines

    wakes = harness_mod.Wakes(waking=lambda: wake_lines.waking([]))
    assert wakes.add([_wake("workflow_started")]) is None
    assert wakes.add([_wake(watch_mod.MESSAGE)]) is not None
    assert wakes.add([_wake(watch_mod.LEVEL)]) is not None
    assert wakes.add([_wake(harness_mod.ALARM)]) is not None


def test_with_no_wakes_every_line_wakes_the_agent():
    assert harness_mod.Wakes(waking=lambda: None).add([_wake("autofocus")])
    assert harness_mod.Wakes().add([_wake("autofocus")])
