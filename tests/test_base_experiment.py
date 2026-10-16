"""What every experiment is, on one whose items are not lamellae."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fibsem.workflows.experiment import EXPERIMENT_FILENAME, BaseExperiment


class Positions(BaseExperiment):
    application_software = "volume"

    def __init__(self, path, name="volume"):
        super().__init__(path=path, name=name)
        self.positions = [
            SimpleNamespace(id="p1", name="P1", path=Path(self.path) / "P1"),
            SimpleNamespace(id="p2", name="P2", path=Path(self.path) / "P2"),
        ]

    def items(self):
        return list(self.positions)


@pytest.fixture
def experiment(tmp_path):
    experiment = Positions(tmp_path)
    Path(experiment.path).mkdir()
    return experiment


def test_an_item_is_found_by_id(experiment):
    assert experiment.get_item_by_id("p2").name == "P2"
    assert experiment.get_item_by_id("missing") is None
    assert experiment.item_path(experiment.positions[0]) == Path(experiment.path) / "P1"


def test_saving_writes_its_identity(experiment):
    experiment.user = "Ada"

    experiment.save()

    saved = yaml.safe_load((Path(experiment.path) / EXPERIMENT_FILENAME).read_text())
    assert saved["_id"] == experiment.id
    assert saved["metadata"] == {"user": "Ada"}
    assert not experiment.unsaved


def test_a_save_that_fails_leaves_it_unsaved(tmp_path):
    experiment = Positions(tmp_path)  # its folder was never made

    with pytest.raises(OSError):
        experiment.save()

    assert experiment.unsaved


def test_the_saved_identity_reads_back(experiment, tmp_path):
    experiment.project = "cells"
    again = Positions(tmp_path)

    again._read_identity(experiment.to_dict())

    assert again.id == experiment.id
    assert again.project == "cells"
    assert again.created_at == experiment.created_at


def test_the_named_operator_decides(experiment):
    experiment.user = "Ada"

    assert experiment.author().name == "Ada"
