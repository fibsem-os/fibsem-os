"""A protocol from before task protocols opens as a task protocol.

The conversion itself is pinned in `tests/test_saved_shapes.py`; here, what
opening one does to the file, and what it refuses.
"""

import shutil
from pathlib import Path

import pytest
import yaml

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.codec.upgrades import pre_task_protocol

LEGACY = (
    Path(__file__).parents[2]
    / "fibsem"
    / "applications"
    / "autolamella"
    / "protocol"
    / "legacy"
)


@pytest.fixture
def old_protocol(tmp_path):
    path = tmp_path / "protocol.yaml"
    shutil.copy(LEGACY / "protocol-waffle.yaml", path)
    return path


def test_opening_one_saves_it_as_a_task_protocol_and_keeps_the_original(
    old_protocol,
):
    original = old_protocol.read_text()

    protocol = AutoLamellaTaskProtocol.load(old_protocol)

    assert list(protocol.workflow_config.workflow) == [
        "Trench Milling",
        "Undercut",
        "Setup Lamella Position",
        "Mill Fiducial",
        "Rough Milling",
        "Polishing",
    ]
    assert Path(f"{old_protocol}.before-v1").read_text() == original
    saved = yaml.safe_load(old_protocol.read_text())
    assert not pre_task_protocol.is_pre_task(saved)
    again = AutoLamellaTaskProtocol.load(old_protocol)
    assert again.to_dict() == protocol.to_dict()


def test_the_copy_is_made_once(old_protocol):
    original = old_protocol.read_text()
    AutoLamellaTaskProtocol.load(old_protocol)
    old_protocol.write_text(original.replace("waffle-baseline", "edited"))

    AutoLamellaTaskProtocol.load(old_protocol)

    assert Path(f"{old_protocol}.before-v1").read_text() == original


def test_a_task_protocol_is_left_as_it_is(tmp_path):
    path = tmp_path / "protocol.yaml"
    shutil.copy(LEGACY.parent / "task-protocol.yaml", path)
    original = path.read_text()

    AutoLamellaTaskProtocol.load(path)

    assert path.read_text() == original
    assert not Path(f"{path}.before-v1").exists()


def test_one_that_cannot_be_written_still_opens(old_protocol, monkeypatch):
    def refuse(self, filename):
        raise PermissionError("read-only")

    monkeypatch.setattr(AutoLamellaTaskProtocol, "save", refuse)
    original = old_protocol.read_text()

    protocol = AutoLamellaTaskProtocol.load(old_protocol)

    assert "Undercut" in protocol.task_config
    assert old_protocol.read_text() == original


def test_an_experiment_opens_with_its_old_protocol(tmp_path):
    experiment = Experiment(path=tmp_path, name="old")
    folder = Path(experiment.path)
    folder.mkdir(parents=True, exist_ok=True)
    experiment.save()
    shutil.copy(LEGACY / "protocol-on-grid.yaml", folder / "protocol.yaml")

    loaded = Experiment.load(folder / "experiment.yaml")

    assert loaded.task_protocol is not None
    assert "Rough Milling" in loaded.task_protocol.task_config
    assert (folder / "protocol.yaml.before-v1").exists()


@pytest.mark.parametrize("name", ["liftout", "serial-liftout"])
def test_a_lift_out_protocol_is_refused(name):
    data = yaml.safe_load((LEGACY / f"protocol-{name}.yaml").read_text())

    with pytest.raises(ValueError, match="lift-out"):
        AutoLamellaTaskProtocol.from_dict(data)


def test_milling_its_method_needs_must_be_there():
    data = {"options": {"method": "on-grid"}, "milling": {}}

    with pytest.raises(ValueError, match="mill_rough"):
        AutoLamellaTaskProtocol.from_dict(data)


def test_a_stage_without_a_type_finds_its_pattern():
    from fibsem.milling.patterning.patterns import UndercutPattern

    stage = pre_task_protocol._stage({}, "undercut", None)

    assert stage["pattern"]["name"] == UndercutPattern.name
