"""experiment.yaml says which application wrote it, and only that one opens it."""

import os
from pathlib import Path

import pytest
import yaml

from fibsem.applications.autolamella.structures import Experiment
from fibsem.config import peek_experiment
from fibsem.workflows.experiment import EXPERIMENT_FILENAME, load_experiment
from fibsem.workflows.saved_format import FORMAT_VERSION, WrongApplication


@pytest.fixture
def saved(tmp_path) -> Path:
    experiment = Experiment(path=tmp_path, name="exp")
    os.makedirs(experiment.path)
    experiment.save()
    return Path(experiment.path) / EXPERIMENT_FILENAME


def _rewrite(path: Path, **changes) -> dict:
    data = yaml.safe_load(path.read_text())
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    path.write_text(yaml.safe_dump(data))
    return data


def test_a_save_names_the_app_and_format(saved):
    data = yaml.safe_load(saved.read_text())

    assert data["app"] == "autolamella"
    assert data["format_version"] == FORMAT_VERSION


def test_a_file_from_before_the_tag_opens_as_autolamella(saved):
    _rewrite(saved, app=None, format_version=None)

    experiment = load_experiment(saved.parent)

    assert isinstance(experiment, Experiment)
    assert peek_experiment(str(saved)).app == "autolamella"


def test_another_apps_file_is_refused_and_left_alone(saved):
    _rewrite(saved, app="volume", slices=[1, 2, 3])
    before = saved.read_text()

    with pytest.raises(WrongApplication, match="volume experiment"):
        Experiment.load(saved)

    assert saved.read_text() == before
    assert peek_experiment(str(saved)).app == "volume"


def test_an_app_nothing_here_reads_is_named(saved):
    _rewrite(saved, app="tomography")

    with pytest.raises(KeyError, match="tomography"):
        load_experiment(saved)
