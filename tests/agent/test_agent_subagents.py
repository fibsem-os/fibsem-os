"""Subagents as lab config: the lab's own beside the shipped ones, a lab
copy replacing a shipped one, every one checked, and a playbook saying which
are on."""

import pytest

from fibsem.ai.agent import extra_mcp, playbooks, sources, subagents
from fibsem.ai.agent import harness as harness_mod

TOOL = "mcp__fibsem__app_get_status"


def _all():
    """What a session started on no particular playbook loads."""
    return harness_mod.session_names(harness_mod.find_checkout())


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
    agents = _all().agents
    assert agents[-1] == "my-checker" and agents.count("autolamella-verifier") == 1
    assert harness_mod.subagent_tools(agents)[-1] == "Agent(my-checker)"
    assert sources.agent_file("autolamella-verifier", checkout).parent == lab

    run = tmp_path / "run"
    sources.assemble(run, checkout, ["fibsem-agent"], agents)
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
    for shipped in _all().agents:
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


def test_front_matter_reads_windows_line_endings():
    # a checkout on Windows writes the shipped agent files with them
    text = "---\nname: x\ntools: Read, Grep\n---\n\nLook.\n"
    data, body = subagents.front_matter(text.replace("\n", "\r\n"))
    assert data == {"name": "x", "tools": "Read, Grep"}
    assert body.strip() == "Look."
    plain = "No front matter.\r\n"
    assert subagents.front_matter(plain) == (None, plain)


def test_a_subagents_file_round_trips_through_its_settings():
    text = _subagent("x", tools=f"{TOOL}, Read", extra="model: opus\n")
    data, body = subagents.front_matter(text)
    assert subagents.render(data, body) == text.replace("\n\nLook", "\n\nLook", 1)


def test_a_playbook_turns_its_subagents_on_and_its_optional_skills_too(lab, checkout):
    found = playbooks.load("autolamella-correlative", checkout)
    lead = found.optional_skill("autolamella-lead")
    assert lead.agents == _all().agents
    on = playbooks.Reminders()
    assert on.subagents() == [], "no playbook, no subagents"
    on.set(found, "named by the operator")
    assert on.subagents() == []
    on.loaded("autolamella-lead")
    assert on.subagents() == _all().agents
    assert found.summary()["optional"][0]["agents"] == _all().agents

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


def test_the_site_finder_is_on_with_screening(lab, checkout):
    found = playbooks.load("autolamella", checkout)
    assert found.optional_skill("screen-grids").agents == ["autolamella-site-finder"]
    assert "autolamella-site-finder" in found.optional_skill("autolamella-lead").agents
    on = playbooks.Reminders()
    on.set(found, "named by the operator")
    on.loaded("screen-grids")
    assert on.subagents() == ["autolamella-site-finder"]
    # Named by two of its optional skills, it is offered once.
    assert found.all_agents().count("autolamella-site-finder") == 1
    assert _all().agents.count("autolamella-site-finder") == 1
    # The correlative playbook screens grids too.
    correlative = playbooks.load("autolamella-correlative", checkout)
    assert correlative.optional_skill("screen-grids").agents == [
        "autolamella-site-finder"
    ]


def test_the_transcript_says_which_subagents_are_on(lab, checkout):
    said = []
    control = harness_mod.PlaybookControl(
        playbooks.Reminders(), checkout, say=said.append
    )
    control.set("autolamella", "named by the operator")
    control.loaded("autolamella-lead")
    assert said[-1] == "Subagents on: " + ", ".join(_all().agents)


def test_a_specialist_gets_the_tools_an_extra_server_gives_specialists(
    lab, checkout, tmp_path
):
    (tmp_path / "home" / "agent").mkdir()
    (tmp_path / "home" / "agent" / "mcp.yaml").write_text(
        "notebook:\n  url: https://notebook.example/mcp\n"
        "  tools: [search, append_entry]\n  subagents: [search]\n",
        encoding="utf-8",
    )
    subagents.check(_subagent("x", tools=f"{TOOL}, mcp__notebook__search"))
    # The agent's alone: in tools, not in subagents.
    with pytest.raises(
        subagents.SubagentError, match="no tool mcp__notebook__append_entry"
    ):
        subagents.check(_subagent("x", tools=f"{TOOL}, mcp__notebook__append_entry"))

    anything = "---\nname: anything\ndescription: Looks.\n---\n\nLook.\n"
    (lab / "anything.md").write_text(anything, encoding="utf-8")
    original = sources.agent_file("autolamella-verifier", checkout)
    before = original.read_bytes()
    run = tmp_path / "run"
    agents = ["autolamella-verifier", "anything"]
    sources.assemble(run, checkout, ["fibsem-agent"], agents)
    copied = run / ".claude" / "agents"
    assembled = {p.name: p.read_bytes() for p in copied.iterdir()}
    subagents.add_tools(run, agents, [])
    assert {p.name: p.read_bytes() for p in copied.iterdir()} == assembled

    subagents.add_tools(run, agents, extra_mcp.subagent_allowed(extra_mcp.load()))
    # The session's copies only: the shipped file is as it was.
    assert original.read_bytes() == before
    shipped, _ = subagents.front_matter(before.decode("utf-8"))
    data, body = subagents.front_matter(
        (copied / "autolamella-verifier.md").read_text("utf-8")
    )
    assert subagents.names(data["tools"]) == subagents.names(shipped["tools"]) + [
        "mcp__notebook__search"
    ]
    assert data["model"] == shipped["model"] and body.strip()
    # No tools line is every tool already: a line would narrow it.
    assert (copied / "anything.md").read_text("utf-8") == anything


def _worker(checkout):
    data, body = subagents.front_matter(
        (checkout / ".claude" / "agents" / f"{subagents.WORKER}.md").read_text(
            encoding="utf-8"
        )
    )
    return data, body


def test_every_tool_is_the_workers_or_the_leads(checkout):
    from fibsem.ai.agent.catalog import CATALOG

    data, _ = _worker(checkout)
    listed = subagents.names(data["tools"])
    fibsem = {
        t[len("mcp__fibsem__") :] for t in listed if t.startswith("mcp__fibsem__")
    }
    catalog = {spec.name for spec in CATALOG}
    # A new tool is a choice: the worker's line or the lead-only list.
    assert subagents.LEAD_ONLY <= catalog
    assert fibsem == catalog - subagents.LEAD_ONLY, (
        "add the new tool to fibsem-worker.md's tools line, or to "
        "subagents.LEAD_ONLY when it is the lead's job"
    )
    # The record, the operator and the run are the lead's.
    for lead in ("plan_set_section", "experiment_message_operator", "workflow_start"):
        assert lead in subagents.LEAD_ONLY
    # It may answer its own item's proposal, and start only the Verifier.
    assert "mcp__fibsem__proposal_decide" in listed
    agents = [t for t in listed if t.startswith("Agent")]
    assert agents == ["Agent(autolamella-verifier)"]


def test_the_worker_file_is_checked_and_an_agent_limit_is_a_tool(lab, checkout):
    subagents.check_file(subagents.WORKER, checkout)
    limited = _subagent("my-lead", tools=f"{TOOL}, Agent(my-checker)")
    assert subagents.check(limited)["name"] == "my-lead"
    with pytest.raises(subagents.SubagentError, match="no tool Agent"):
        subagents.check(_subagent("bad", tools="Agent(no spaces allowed here)"))
    # A worker may start the Verifier: one level of nesting.
    assert harness_mod.CLI_ENV["CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH"] == "2"


def test_the_worker_is_on_with_the_lead_and_asked_for_at_intake(lab, checkout):
    found = playbooks.load("autolamella", checkout)
    assert subagents.WORKER in found.optional_skill("autolamella-lead").agents
    on = playbooks.Reminders()
    on.set(found, "named by the operator")
    assert subagents.WORKER not in on.subagents(), "only in lead mode"
    on.loaded("autolamella-lead")
    assert subagents.WORKER in on.subagents()
    # Whether the lead uses it is the operator's answer at intake.
    lead = (
        checkout / ".claude" / "skills" / "autolamella-lead" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "polishing to a worker (*Workers*)? Default: yes." in " ".join(lead.split())
