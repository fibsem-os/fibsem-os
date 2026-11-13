"""An experiment.yaml in another format than this build writes (engineering
review R15, R17): a newer one is never written, and an older one is kept
beside the first save over it."""

import pytest
import yaml

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.structures.settings import MicroscopeState
from fibsem.workflows.experiment import (
    EXPERIMENT_FILENAME,
    FibsemExperiment,
    NewerFormat,
    backup_path,
    load_experiment,
)
from fibsem.workflows.saved_format import FORMAT_VERSION


def _saved_as(tmp_path, version, **extra):
    """An AutoLamella experiment on disk that says it is in format *version*,
    with *extra* top-level keys this build does not know; its file."""
    experiment = Experiment(path=tmp_path, name="formats")
    (tmp_path / "formats").mkdir()
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.add_new_lamella(
        microscope_state=MicroscopeState(),
        task_config=experiment.task_protocol.task_config,
        name="one",
    )
    experiment.save()
    path = tmp_path / "formats" / EXPERIMENT_FILENAME
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["format_version"] = version
    data.update(extra)
    path.write_text(yaml.safe_dump(data, indent=4), encoding="utf-8")
    return path


def test_a_newer_experiment_opens_and_is_never_written(tmp_path):
    path = _saved_as(tmp_path, FORMAT_VERSION + 1, from_the_future={"kept": True})
    before = path.read_text(encoding="utf-8")
    experiment = load_experiment(path)
    assert experiment.read_only
    assert [p.name for p in experiment.positions] == ["one"]
    with pytest.raises(NewerFormat, match="newer version of fibsemOS"):
        experiment.save()
    assert experiment.unsaved
    assert path.read_text(encoding="utf-8") == before
    assert not backup_path(path).exists()


def test_an_older_experiment_is_kept_beside_its_first_save(tmp_path):
    path = _saved_as(tmp_path, FORMAT_VERSION - 1)
    before = path.read_text(encoding="utf-8")
    experiment = load_experiment(path)
    assert not experiment.read_only
    experiment.save()
    assert backup_path(path).read_text(encoding="utf-8") == before
    assert yaml.safe_load(path.read_text())["format_version"] == FORMAT_VERSION
    # A later save leaves the copy of the original alone.
    experiment.positions[0].name = "renamed"
    experiment.save()
    assert backup_path(path).read_text(encoding="utf-8") == before


def test_a_file_from_before_the_format_tag_is_an_older_one(tmp_path):
    path = _saved_as(tmp_path, None)
    experiment = load_experiment(path)
    experiment.save()
    assert backup_path(path).exists()


def test_a_current_or_new_experiment_gets_no_copy(tmp_path):
    path = _saved_as(tmp_path, FORMAT_VERSION)
    load_experiment(path).save()
    assert not backup_path(path).exists()
    plain = FibsemExperiment.create(tmp_path, "plain")
    plain.save()
    assert not backup_path(tmp_path / "plain" / EXPERIMENT_FILENAME).exists()


def test_a_copy_that_cannot_be_made_does_not_stop_the_save(tmp_path, monkeypatch):
    import shutil

    path = _saved_as(tmp_path, FORMAT_VERSION - 1)
    experiment = load_experiment(path)

    def refuse(*args, **kwargs):
        raise OSError("read-only folder")

    monkeypatch.setattr(shutil, "copy2", refuse)
    experiment.save()
    assert yaml.safe_load(path.read_text())["format_version"] == FORMAT_VERSION
