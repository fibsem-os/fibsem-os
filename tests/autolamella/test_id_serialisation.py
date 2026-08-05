"""The `_id` -> `id` rename must not move the on-disk format.

FIB-495 renamed the attribute because it is the system's most externally-facing
identifier -- persisted, read from 26 call sites, and carried in image metadata and
hook payloads -- while the underscore marked it internal.

The serialised keys deliberately did not change, so no existing experiment.yaml
needs migrating. That is the load-bearing claim, and it is the one worth pinning:
a later cleanup that "tidies" the key would silently orphan the id in every
experiment ever written, and every image whose metadata joins on it.
"""

import os
from pathlib import Path

import pytest
import yaml

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    Lamella,
)


def test_the_experiment_id_is_still_written_under_underscore_id(tmp_path: Path) -> None:
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)
    exp.save()

    written = yaml.safe_load((Path(exp.path) / "experiment.yaml").read_text())

    assert written["_id"] == exp.id
    assert "id" not in written, "the key moved -- existing experiments would lose their id"


def test_an_experiment_written_before_the_rename_still_loads(tmp_path: Path) -> None:
    """The point of keeping the key: files on disk predate the attribute name."""
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)
    exp.save()

    path = Path(exp.path) / "experiment.yaml"
    on_disk = yaml.safe_load(path.read_text())
    original_id = on_disk["_id"]

    reloaded = Experiment.load(path)

    assert reloaded.id == original_id


def test_the_lamella_id_keeps_its_own_key(tmp_path: Path) -> None:
    """Lamella already serialised under "id", not "_id" -- the two were inconsistent
    before this rename and still are. Pinned so the asymmetry is deliberate rather
    than something a future reader assumes is a typo and "fixes"."""
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")

    assert lamella.to_dict()["id"] == lamella.id

    restored = Lamella.from_dict(lamella.to_dict())
    assert restored.id == lamella.id


def test_the_protocol_id_round_trips() -> None:
    protocol = AutoLamellaTaskProtocol()

    written = protocol.to_dict()
    assert written["_id"] == protocol.id

    assert AutoLamellaTaskProtocol.from_dict(written).id == protocol.id


def test_a_missing_id_does_not_raise(tmp_path: Path) -> None:
    """Older or hand-edited files may have no id at all."""
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    data = lamella.to_dict()
    del data["id"]

    assert Lamella.from_dict(data).id == ""
