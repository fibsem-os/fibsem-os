"""A plugin adds a lab's own instrument kind, with its playbooks and skills:
installed, its agent folder is read after what fibsemOS ships, and never
replaces it."""

from pathlib import Path

import pytest

from fibsem.ai.agent import harness, instruments, playbooks, sources
from fibsem.plugins import loader

INTAKE = "## start\n\nStart.\n\n## resumed\n\nAgain.\n\n## goal\n\n{goal}\n"
REREAD = "## proposals\n\na\n\n## plan\n\nb\n\n## level\n\nc\n"


class _EntryPoint:
    def __init__(self, target):
        self.name, self.value, self._target = "lab", "lab_instrument:agent", target
        self.dist = type("Dist", (), {"name": "lab-instrument", "version": "1"})()

    def load(self):
        return self._target


def _plugin(root: Path, kind: str) -> Path:
    claude = root / ".claude"
    words = claude / "fibsem-agent" / "instruments" / kind
    words.mkdir(parents=True)
    (words / "prompt.md").write_text("You run the lab's instrument. {tools}")
    (words / "intake.md").write_text(INTAKE)
    (words / "reread.md").write_text(REREAD)
    (claude / "fibsem-agent" / "playbooks").mkdir(parents=True)
    (claude / "fibsem-agent" / "playbooks" / f"{kind}.yaml").write_text(
        f"description: The lab's instrument.\ninstrument: {kind}\n"
        f"skills: [fibsem-agent, {kind}]\n"
    )
    (claude / "skills" / kind).mkdir(parents=True)
    (claude / "skills" / kind / "SKILL.md").write_text("---\nname: x\n---\n")
    return root


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """A plugin naming its agent folder, as an installed package would."""
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    root = _plugin(tmp_path / "plugin", "lab-scope")
    loader.clear_cache()
    monkeypatch.setattr(
        loader, "_entry_points", lambda group: iter([_EntryPoint(lambda: root)])
    )
    yield root
    loader.clear_cache()


def test_a_plugin_adds_its_instrument_kind_playbook_and_skills(installed):
    checkout = harness.find_checkout()
    assert sources.plugin_roots() == [installed.resolve()]
    assert "lab-scope" in instruments.kinds(checkout)
    assert instruments.load("lab-scope", checkout).prompt.startswith("You run the lab")

    playbook = playbooks.load("lab-scope", checkout)
    assert playbook.instrument == "lab-scope"
    assert "lab-scope" in playbooks.available(checkout)
    assert sources.skill_dir("lab-scope", checkout).is_relative_to(installed)
    # The shipped ones are still found first.
    assert sources.skill_dir("fibsem-agent", checkout).is_relative_to(checkout)
    assert playbooks.load("fibsem-agent", checkout).path.is_relative_to(checkout)


def test_a_plugin_cant_replace_a_shipped_instrument_kind(tmp_path, installed):
    checkout = harness.find_checkout()
    _plugin(tmp_path / "other", "fibsem")
    loader.clear_cache()
    roots = [installed.resolve(), (tmp_path / "other").resolve()]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sources, "plugin_roots", lambda: roots)
        words = instruments.load("fibsem", checkout)
    assert not words.prompt.startswith("You run the lab")


def test_a_folder_with_no_claude_folder_is_left_out(tmp_path, monkeypatch):
    loader.clear_cache()
    monkeypatch.setattr(
        loader, "_entry_points", lambda group: iter([_EntryPoint(str(tmp_path))])
    )
    try:
        assert sources.plugin_roots() == []
    finally:
        loader.clear_cache()


def test_a_plugins_kind_is_offered_the_tools_on_its_devices(installed):
    from fibsem.ai.agent.catalog import tools_for_capabilities

    checkout = harness.find_checkout()
    capabilities = {"routers": {"services": True, "app": False}}
    folder = installed / ".claude" / "fibsem-agent" / "instruments" / "lab-scope"

    def offered():
        devices = instruments.devices("lab-scope", checkout)
        return {t.name for t in tools_for_capabilities(capabilities, devices)}

    # No devices.yaml: the tools for any instrument only.
    assert "device_list" in offered() and "stage_get_position" not in offered()
    (folder / "devices.yaml").write_text("devices: [stage, camera]\n")
    assert "stage_get_position" in offered()
    assert "milling_run_stage" not in offered()

    (folder / "devices.yaml").write_text("devices: stage\n")
    with pytest.raises(instruments.InstrumentError, match="list of names"):
        instruments.devices("lab-scope", checkout)
    (folder / "devices.yaml").write_text("tools: [milling_run_stage]\n")
    with pytest.raises(instruments.InstrumentError, match="not"):
        instruments.devices("lab-scope", checkout)
