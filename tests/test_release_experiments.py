"""Every released version's experiment opens here, saves and opens again
(engineering review R16). The files are in tests/fixtures/experiments, each
written by the release it is named for."""

import shutil
from pathlib import Path

import pytest

from fibsem.applications.autolamella.structures import Experiment
from fibsem.workflows.experiment import (
    EXPERIMENT_FILENAME,
    backup_path,
    load_experiment,
)
from fibsem.workflows.saved_format import FORMAT_VERSION
from fibsem.workflows.status import TaskStatus

RELEASES = Path(__file__).parent / "fixtures" / "experiments"
VERSIONS = sorted(p.name for p in RELEASES.iterdir() if p.is_dir())
TASKS = ["Setup Lamella Position", "Mill Fiducial", "Rough Milling"]


@pytest.fixture(params=VERSIONS)
def folder(request, tmp_path) -> Path:
    """A release's experiment, copied to where this test can write."""
    return Path(shutil.copytree(RELEASES / request.param, tmp_path / "experiment"))


def _what_matters(experiment: Experiment) -> dict:
    """The saved experiment without where it is on disk."""
    data = experiment.to_dict()
    data.pop("path", None)
    return data


def test_there_is_a_release_to_open():
    assert VERSIONS, "no release experiments in tests/fixtures/experiments"


def test_it_opens_with_its_lamellae_and_their_history(folder):
    experiment = load_experiment(folder)
    assert isinstance(experiment, Experiment)
    assert not experiment.read_only
    by_name = {p.name: p for p in experiment.positions}
    assert sorted(by_name) == ["first", "second", "untouched"]
    for name in ("first", "second"):
        history = by_name[name].task_history
        assert [s.name for s in history] == TASKS
        assert all(s.status is TaskStatus.Completed for s in history)
        assert all(s.end_timestamp is not None for s in history)
        assert by_name[name].task_state.status is TaskStatus.Completed
    assert by_name["untouched"].task_history == []
    assert by_name["untouched"].task_state.status is TaskStatus.NotStarted
    # Its lamellae are where the experiment is now, not where it was written.
    assert Path(by_name["first"].path) == folder / "first"


def test_its_protocol_opens_with_every_task(folder):
    experiment = load_experiment(folder)
    assert experiment.task_protocol is not None
    assert set(TASKS) <= set(experiment.task_protocol.task_config)
    for lamella in experiment.positions:
        assert set(TASKS) <= set(lamella.task_config)


def test_it_saves_in_this_format_and_opens_the_same(folder):
    path = folder / EXPERIMENT_FILENAME
    as_released = path.read_text(encoding="utf-8")
    experiment = load_experiment(folder)
    experiment.save()
    assert backup_path(path).read_text(encoding="utf-8") == as_released
    again = load_experiment(folder)
    assert again.read_format == FORMAT_VERSION
    assert _what_matters(again) == _what_matters(experiment)
