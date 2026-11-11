"""The agent's configuration from its playbooks: the skills and subagents a
session loads, the shipped base playbook's defaults, and the words for its
instrument kind."""

import pytest

from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import instruments, library, playbooks


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    folder = tmp_path / "home" / "knowledge" / "playbooks"
    folder.mkdir(parents=True)
    return folder


@pytest.fixture
def checkout():
    found = harness_mod.find_checkout()
    assert found is not None
    return found


def _loaded(name, checkout):
    playbook = playbooks.load(name, checkout) if name else None
    return harness_mod.session_names(checkout, playbook)


def test_a_session_loads_what_its_playbooks_name(lab, checkout):
    everything = _loaded(None, checkout)
    assert everything.skills[0] == "fibsem-agent"
    assert {"autolamella", "autolamella-lead", "fibsem-experiment"} <= set(
        everything.skills
    )
    assert "autolamella-verifier" in everything.agents
    # The checkout's developer skills are never loaded.
    assert "review-changes" not in everything.skills
    # The base playbook is a start with no job yet: intake may set any.
    assert _loaded(playbooks.DEFAULT, checkout) == everything

    autolamella = _loaded("autolamella", checkout)
    assert "fibsem-experiment" not in autolamella.skills
    assert "autolamella-correlative" not in autolamella.skills
    assert autolamella.skills[:3] == [
        "fibsem-agent",
        "fibsem-microscope",
        "autolamella",
    ]
    assert "autolamella-verifier" in autolamella.agents

    experiment = _loaded("fibsem-experiment", checkout)
    assert experiment.skills == [
        "fibsem-agent",
        "fibsem-microscope",
        "fibsem-experiment",
    ]
    assert experiment.agents == []


def test_the_agent_tools_are_the_sessions_subagents(lab, checkout):
    pytest.importorskip("claude_agent_sdk")
    experiment = playbooks.load("fibsem-experiment", checkout)
    options = harness_mod.build_options(checkout, playbook=experiment)
    assert not any(t.startswith("Agent(") for t in options.allowed_tools)
    lead = playbooks.load("autolamella", checkout)
    options = harness_mod.build_options(checkout, playbook=lead)
    assert "Agent(autolamella-verifier)" in options.allowed_tools


def test_playbook_set_refuses_a_playbook_the_session_did_not_load(lab, checkout):
    loaded = _loaded("fibsem-experiment", checkout)
    control = harness_mod.PlaybookControl(
        playbooks.Reminders(), checkout, loaded=loaded
    )
    said = control.set("autolamella", "named by the operator")
    assert said.startswith("Refused:") and "restart the agent" in said
    assert control.reminders.playbook is None
    assert control.set("fibsem-experiment", "named by the operator").startswith(
        "Running"
    )
    # Back to the base is always open: it names only the base skill.
    assert control.set("", "by the agent").startswith("Running")


def test_the_knowledge_window_lists_the_skills_the_playbooks_name(lab, checkout):
    names = library._agent_skills()
    assert names == playbooks.shipped_names(checkout)["skills"]
    assert "fibsem-agent" in names and "screen-grids" in names
    assert "review-changes" not in names


def test_the_defaults_are_the_shipped_base_playbooks(lab, checkout):
    base = playbooks.shipped_base(checkout)
    assert base.instrument == "fibsem"
    for key in harness_mod.REQUIRED:
        assert getattr(base.session, key) is not None, key
    assert harness_mod.Alarm().max_min == base.session.wake_max_min
    assert harness_mod.Deadlines().warn_s == base.session.deadline_warn_min * 60
    # A lab copy of the base without them still gets them.
    (lab / f"{playbooks.DEFAULT}.yaml").write_text(
        "description: Ours.\nsession:\n  max_turns: 7\n", encoding="utf-8"
    )
    ours = playbooks.load(playbooks.DEFAULT, checkout)
    assert ours.session.max_turns == 7
    assert ours.session.quiet_min == base.session.quiet_min
    assert ours.skills == ["fibsem-agent", "fibsem-microscope"]


def test_the_session_settings_show_where_each_comes_from(lab, checkout):
    (lab / "ours.yaml").write_text(
        "extends: autolamella\nsession:\n  max_turns: 7\n", encoding="utf-8"
    )
    session = library.read("playbooks", "ours").session
    assert session["max_turns"] == {"value": 7, "source": "ours"}
    assert session["quiet_min"]["source"] == playbooks.DEFAULT


def test_the_words_come_from_the_instrument_kinds_files(lab, checkout):
    assert instruments.kinds(checkout) == ["fibsem"]
    words = harness_mod.texts()
    assert words.kind == "fibsem"
    prompt = harness_mod.harness_prompt(words=words, deadline_warn_min=45)
    assert prompt.startswith("You run a FIB/SEM microscope")
    assert "wakes you 45 min before it" in prompt
    assert "{" not in harness_mod.kickoff(goal="mill {01}")[:20]
    assert "mill {01}" in harness_mod.kickoff(goal="mill {01}")
    assert "<!--" not in harness_mod.kickoff()


def test_an_unknown_instrument_kind_fails_at_load(lab, checkout):
    (lab / "other.yaml").write_text(
        "extends: fibsem-agent\ninstrument: cryo-fm\n", encoding="utf-8"
    )
    with pytest.raises(playbooks.PlaybookError, match="no instrument kind 'cryo-fm'"):
        playbooks.load("other", checkout)


def test_an_instrument_kinds_files_are_checked(tmp_path):
    folder = tmp_path / instruments.INSTRUMENT_DIR / "fibsem"
    folder.mkdir(parents=True)
    (folder / "prompt.md").write_text("Hello {tools} {nobody}\n")
    (folder / "intake.md").write_text(
        "## start\n\na\n\n## resumed\n\nb\n\n## goal\n\n{goal}\n"
    )
    (folder / "reread.md").write_text("## proposals\n\na\n\n## plan\n\nb\n")
    with pytest.raises(instruments.InstrumentError, match="nobody"):
        instruments.load("fibsem", tmp_path)
    (folder / "prompt.md").write_text("Hello {tools}\n")
    with pytest.raises(instruments.InstrumentError, match="## level"):
        instruments.load("fibsem", tmp_path)
    (folder / "reread.md").write_text(
        "## proposals\n\na\n\n## plan\n\nb\n\n## level\n\nc\n"
    )
    assert instruments.load("fibsem", tmp_path).reread_level == "c"
    with pytest.raises(instruments.InstrumentError, match="No instrument kind"):
        instruments.load("cryo-fm", tmp_path)


def test_show_config_says_what_it_loads_and_its_timers(lab, capsys):
    import json

    assert (
        harness_mod.main(["--playbook", "fibsem-experiment", "--show-config", "--json"])
        == 0
    )
    config = json.loads(capsys.readouterr().out)
    assert config["skills"] == [
        "fibsem-agent",
        "fibsem-microscope",
        "fibsem-experiment",
    ]
    assert config["specialists"] == []
    assert config["instrument"] == "fibsem"
    assert (
        config["quiet_min"]
        == playbooks.shipped_base(harness_mod.find_checkout()).session.quiet_min
    )
    assert harness_mod.main(["--show-config", "--json"]) == 0
    config = json.loads(capsys.readouterr().out)
    assert "autolamella-verifier" in config["specialists"]


def test_the_base_skills_go_only_to_playbooks_of_its_instrument_kind(tmp_path, lab):
    # A checkout shipping two instrument kinds: the base playbook's and a
    # hypothetical standalone cryo-FM's.
    root = tmp_path / "checkout"
    shipped = root / ".claude" / "fibsem-agent"
    (shipped / "playbooks").mkdir(parents=True)
    (shipped / "playbooks" / "fibsem-agent.yaml").write_text(
        "instrument: fibsem\n"
        "skills: [fibsem-agent, fibsem-microscope]\n"
        "session: {max_turns: 40}\n"
    )
    for kind in ("fibsem", "cryo-fm"):
        (shipped / "instruments" / kind).mkdir(parents=True)
        (shipped / "instruments" / kind / "prompt.md").write_text("words")
    for skill in ("fibsem-agent", "fibsem-microscope", "fibsem-experiment", "cryo-fm"):
        (root / ".claude" / "skills" / skill).mkdir(parents=True)
        (root / ".claude" / "skills" / skill / "SKILL.md").write_text("---\n---\n")
    (lab / "cryo-fm.yaml").write_text(
        "instrument: cryo-fm\nskills: [fibsem-agent, cryo-fm]\n"
    )
    (lab / "fib-work.yaml").write_text("skills: [fibsem-experiment]\n")

    other = playbooks.load("cryo-fm", root)
    assert other.instrument == "cryo-fm"
    assert other.skills == ["fibsem-agent", "cryo-fm"]
    # The base's session settings still hold for every kind.
    assert other.session.max_turns == 40

    fib = playbooks.load("fib-work", root)
    assert fib.instrument == "fibsem"
    assert fib.skills == ["fibsem-agent", "fibsem-microscope", "fibsem-experiment"]
