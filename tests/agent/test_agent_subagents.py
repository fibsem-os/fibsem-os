"""Subagents as lab config: the lab's own beside the shipped ones, a lab
copy replacing a shipped one, every one checked, and a playbook saying which
are on."""

import pytest

from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import playbooks, sources, subagents

TOOL = "mcp__fibsem__app_get_status"


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    folder = tmp_path / "home" / "knowledge" / "agents"
    folder.mkdir(parents=True)
    return folder


@pytest.fixture
def checkout():
    found = harness_mod.find_checkout()
    assert found is not None
    return found


def _subagent(name, tools=TOOL, extra=""):
    return (
        f"---\nname: {name}\ndescription: Checks a lamella.\ntools: {tools}\n"
        f"{extra}---\n\nLook and answer.\n"
    )


def test_a_lab_subagent_is_offered_and_a_lab_copy_replaces_a_shipped_one(
    lab, checkout, tmp_path
):
    (lab / "my-checker.md").write_text(_subagent("my-checker"), encoding="utf-8")
    (lab / "autolamella-verifier.md").write_text(
        _subagent("autolamella-verifier"), encoding="utf-8"
    )
    assert sources.lab_agents() == ["autolamella-verifier", "my-checker"]
    assert harness_mod.subagents_for() == list(harness_mod.SPECIALISTS) + ["my-checker"]
    assert harness_mod.subagent_tools(harness_mod.subagents_for()) == [
        "Agent(my-checker)"
    ]
    assert sources.agent_file("autolamella-verifier", checkout).parent == lab

    run = tmp_path / "run"
    sources.assemble(run, checkout, ["fibsem-agent"], harness_mod.subagents_for())
    agents = run / ".claude" / "agents"
    assert "Look and answer." in (agents / "autolamella-verifier.md").read_text()
    assert (agents / "my-checker.md").is_file()
    assert (agents / "autolamella-auditor.md").is_file()


def test_the_session_allows_the_labs_new_subagents(lab, checkout):
    pytest.importorskip("claude_agent_sdk")
    (lab / "my-checker.md").write_text(_subagent("my-checker"), encoding="utf-8")
    options = harness_mod.build_options(checkout, url="http://127.0.0.1:9", token="t")
    assert "Agent(my-checker)" in options.allowed_tools
    # the file and shell tools stay denied outside code mode, whatever it lists
    assert "Bash" in options.disallowed_tools


def test_a_subagent_is_checked(lab, checkout):
    for shipped in harness_mod.SPECIALISTS:
        subagents.check_file(shipped, checkout, harness_mod.EFFORTS)
    with pytest.raises(subagents.SubagentError, match="no tool mcp__fibsem__vent_now"):
        subagents.check(_subagent("x", tools=f"{TOOL}, mcp__fibsem__vent_now"))
    with pytest.raises(subagents.SubagentError, match="no tool WebFetch"):
        subagents.check(_subagent("x", tools="WebFetch"))
    with pytest.raises(subagents.SubagentError, match="effort"):
        subagents.check(_subagent("x", extra="effort: huge\n"), harness_mod.EFFORTS)
    with pytest.raises(subagents.SubagentError, match="unknown setting"):
        subagents.check(_subagent("x", extra="permissionMode: bypass\n"))
    with pytest.raises(subagents.SubagentError, match="front matter"):
        subagents.check("Just text.")
    (lab / "broken.md").write_text(_subagent("broken", tools="Nope"), encoding="utf-8")
    with pytest.raises(subagents.SubagentError, match="broken.md: no tool Nope"):
        subagents.check_file("broken", checkout)


def test_a_subagents_file_round_trips_through_its_settings():
    text = _subagent("x", tools=f"{TOOL}, Read", extra="model: opus\n")
    data, body = subagents.front_matter(text)
    assert subagents.render(data, body) == text.replace("\n\nLook", "\n\nLook", 1)


def test_a_playbook_turns_its_subagents_on_and_its_optional_skills_too(lab, checkout):
    found = playbooks.load("autolamella-correlative", checkout)
    lead = found.optional_skill("autolamella-lead")
    assert lead.agents == list(harness_mod.SPECIALISTS)
    on = playbooks.Reminders()
    assert on.subagents() == [], "no playbook, no subagents"
    on.set(found, "named by the operator")
    assert on.subagents() == []
    on.loaded("autolamella-lead")
    assert on.subagents() == list(harness_mod.SPECIALISTS)
    assert found.summary()["optional"][0]["agents"] == list(harness_mod.SPECIALISTS)

    ours = lab.parent / "playbooks" / "ours.yaml"
    ours.parent.mkdir(parents=True)
    ours.write_text(
        "extends: autolamella\nagents: [no-such-subagent]\n", encoding="utf-8"
    )
    with pytest.raises(playbooks.PlaybookError, match="no subagent no-such-subagent"):
        playbooks.load("ours", checkout)
    (lab / "my-checker.md").write_text(_subagent("my-checker"), encoding="utf-8")
    ours.write_text("extends: autolamella\nagents: [my-checker]\n", encoding="utf-8")
    on.set(playbooks.load("ours", checkout), "named by the operator")
    assert on.subagents() == ["my-checker"]


def test_the_transcript_says_which_subagents_are_on(lab, checkout):
    said = []
    control = harness_mod.PlaybookControl(
        playbooks.Reminders(), checkout, say=said.append
    )
    control.set("autolamella", "named by the operator")
    control.loaded("autolamella-lead")
    assert said[-1] == "Subagents on: " + ", ".join(harness_mod.SPECIALISTS)
