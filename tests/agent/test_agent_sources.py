"""Where the agent's skills, specialists and profiles come from: the lab's own
in the knowledge base first, then the shipped ones (a checkout's .claude, or
the package's copy of it)."""

from pathlib import Path

import pytest

from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import manager, profiles, sources


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
    assert harness_mod.skills_for()[-1] == "yeast-lamellae"
    assert harness_mod.skills_for().count("screen-grids") == 1

    run = tmp_path / "run"
    (run / ".claude" / "settings.json").parent.mkdir(parents=True)
    (run / ".claude" / "settings.json").write_text("{}", encoding="utf-8")
    from_lab = sources.assemble(
        run, checkout, harness_mod.skills_for(), harness_mod.SPECIALISTS
    )
    assert from_lab == ["screen-grids", "yeast-lamellae"]
    skills = run / ".claude" / "skills"
    assert "our own screening" in (skills / "screen-grids" / "SKILL.md").read_text()
    assert (skills / "autolamella" / "read_record.py").is_file(), "folders, whole"
    assert not (run / ".claude" / "settings.json").exists(), "rebuilt every start"
    for name in harness_mod.SPECIALISTS:
        assert (run / ".claude" / "agents" / f"{name}.md").is_file()


def test_a_skill_found_nowhere_is_refused(lab, checkout, tmp_path):
    with pytest.raises(FileNotFoundError, match="no-such-skill"):
        sources.assemble(tmp_path / "run", checkout, ["no-such-skill"])


def test_the_labs_profile_wins_then_this_machines_then_the_shipped(
    lab, checkout, tmp_path, monkeypatch
):
    monkeypatch.setattr(profiles.Path, "home", lambda: tmp_path)
    shipped = profiles.find(profiles.DEFAULT_PROFILE, checkout)
    assert shipped.is_relative_to(checkout)
    machine = profiles.machine_dir() / f"{profiles.DEFAULT_PROFILE}.yaml"
    machine.parent.mkdir(parents=True)
    machine.write_text("effort: high\n", encoding="utf-8")
    assert profiles.find(profiles.DEFAULT_PROFILE, checkout) == machine
    ours = lab / "profiles" / f"{profiles.DEFAULT_PROFILE}.yaml"
    ours.parent.mkdir(parents=True)
    ours.write_text("effort: low\nskills: [yeast-lamellae]\n", encoding="utf-8")
    assert profiles.find(profiles.DEFAULT_PROFILE, checkout) == ours
    with pytest.raises(profiles.ProfileError, match="no skill yeast-lamellae"):
        profiles.load(ours, checkout=checkout)
    _skill(lab / "skills", "yeast-lamellae", "what works on yeast here")
    assert profiles.load(ours, checkout=checkout).skills == ["yeast-lamellae"]


def test_the_wheel_ships_skills_specialists_and_profiles(checkout, tmp_path):
    sources.ship(checkout, tmp_path)
    shipped = tmp_path / ".claude"
    for name in harness_mod.SKILLS:
        assert (shipped / "skills" / name / "SKILL.md").is_file()
    for name in harness_mod.SPECIALISTS:
        assert (shipped / "agents" / f"{name}.md").is_file()
    assert (shipped / "fibsem-agent" / f"{profiles.DEFAULT_PROFILE}.yaml").is_file()
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
