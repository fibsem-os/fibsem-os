"""Where the agent's skills, subagents and playbooks come from: the lab's own
in the knowledge base first, then the shipped ones (a checkout's .claude, or
the package's copy of it)."""

from pathlib import Path

import pytest

from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import manager, playbooks, sources


def _all():
    """What a session started on no particular playbook loads."""
    return harness_mod.session_names(harness_mod.find_checkout())


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """An empty state root: the lab's folders under it."""
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    return tmp_path / "home" / "knowledge"


@pytest.fixture
def checkout():
    found = harness_mod.find_checkout()
    assert found is not None
    return found


def _skill(folder: Path, name: str, text: str) -> None:
    (folder / name).mkdir(parents=True)
    (folder / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {text}\n---\n\n{text}\n", encoding="utf-8"
    )


def test_a_lab_skill_wins_over_the_shipped_one_and_a_new_one_is_offered(
    lab, checkout, tmp_path
):
    _skill(lab / "skills", "screen-grids", "our own screening")
    _skill(lab / "skills", "yeast-lamellae", "what works on yeast here")
    assert sources.lab_skills() == ["screen-grids", "yeast-lamellae"]
    assert _all().skills[-1] == "yeast-lamellae"
    assert _all().skills.count("screen-grids") == 1

    run = tmp_path / "run"
    (run / ".claude" / "settings.json").parent.mkdir(parents=True)
    (run / ".claude" / "settings.json").write_text("{}", encoding="utf-8")
    from_lab = sources.assemble(run, checkout, _all().skills, _all().agents)
    assert from_lab == ["screen-grids", "yeast-lamellae"]
    skills = run / ".claude" / "skills"
    assert "our own screening" in (skills / "screen-grids" / "SKILL.md").read_text()
    assert (skills / "autolamella" / "read_record.py").is_file(), "folders, whole"
    assert not (run / ".claude" / "settings.json").exists(), "rebuilt every start"
    for name in _all().agents:
        assert (run / ".claude" / "agents" / f"{name}.md").is_file()


def test_a_skill_found_nowhere_is_refused(lab, checkout, tmp_path):
    with pytest.raises(FileNotFoundError, match="no-such-skill"):
        sources.assemble(tmp_path / "run", checkout, ["no-such-skill"])


def test_the_labs_playbook_wins_over_the_shipped(lab, checkout):
    shipped = playbooks.find(playbooks.DEFAULT, checkout)
    assert shipped.is_relative_to(checkout)
    ours = lab / "playbooks" / f"{playbooks.DEFAULT}.yaml"
    ours.parent.mkdir(parents=True)
    ours.write_text("skills: [yeast-lamellae]\n", encoding="utf-8")
    assert playbooks.find(playbooks.DEFAULT, checkout) == ours
    with pytest.raises(playbooks.PlaybookError, match="no skill yeast-lamellae"):
        playbooks.load(playbooks.DEFAULT, checkout)
    _skill(lab / "skills", "yeast-lamellae", "what works on yeast here")
    found = playbooks.load(playbooks.DEFAULT, checkout)
    # The shipped base stays under the lab's copy: its skill and defaults hold.
    assert found.skills == ["fibsem-agent", "fibsem-microscope", "yeast-lamellae"]
    assert found.session.max_turns == playbooks.shipped_base(checkout).session.max_turns
    assert found.instrument == "fibsem"


def test_the_wheel_ships_skills_subagents_and_playbooks(checkout, tmp_path):
    sources.ship(checkout, tmp_path)
    shipped = tmp_path / ".claude"
    for name in _all().skills:
        assert (shipped / "skills" / name / "SKILL.md").is_file()
    for name in _all().agents:
        assert (shipped / "agents" / f"{name}.md").is_file()
    playbook = shipped / "fibsem-agent" / "playbooks" / f"{playbooks.DEFAULT}.yaml"
    assert playbook.is_file()
    assert (shipped / "fibsem-agent" / "instruments" / "fibsem" / "prompt.md").is_file()
    assert not list(shipped.rglob("__pycache__"))
    assert not (shipped / "settings.json").exists(), "only the agent's files"
    # A pip install has no checkout: the shipped copy is found instead.
    assert sources.find_root(str(tmp_path)) == tmp_path.resolve()


def test_a_session_without_a_work_folder_runs_in_its_own_run_folder(tmp_path, checkout):
    folder = manager.session_folder("abc", tmp_path)
    assert harness_mod._run_folder(None, folder, checkout) == folder / manager.RUN
    manager.SessionFolder.open("abc", {"cwd": str(checkout)}, root=tmp_path)
    folder = manager.session_folder("abc", tmp_path)
    assert harness_mod._run_folder("abc", folder, checkout) == checkout, (
        "a session from before run/ resumes where it ran"
    )
    run = manager.session_folder("def", tmp_path) / manager.RUN
    manager.SessionFolder.open("def", {"cwd": str(run)}, root=tmp_path)
    assert manager.last_session(None, tmp_path, own=manager.RUN) == "def"
    assert manager.last_session(None, tmp_path) is None, "work/ is another mode"
    assert manager.last_session(checkout, tmp_path) == "abc"
