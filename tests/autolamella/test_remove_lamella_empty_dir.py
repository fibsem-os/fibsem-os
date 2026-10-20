"""Removing a lamella removes its directory only when it is empty."""

from pathlib import Path

from fibsem.applications.autolamella.poses import remove_lamella
from fibsem.applications.autolamella.structures import Experiment, Lamella


def _experiment_with_lamella(tmp_path):
    experiment = Experiment(path=tmp_path, name="remove-exp")
    lamella = Lamella(
        path=Path(experiment.path) / "01-test", number=1, petname="01-test"
    )
    lamella.path.mkdir(parents=True)
    experiment.positions.append(lamella)
    return experiment, lamella


def test_an_empty_directory_goes(tmp_path):
    experiment, lamella = _experiment_with_lamella(tmp_path)
    assert remove_lamella(experiment, lamella)

    assert not lamella.path.exists()
    assert Path(experiment.path).exists()


def test_a_directory_with_files_stays_untouched(tmp_path):
    experiment, lamella = _experiment_with_lamella(tmp_path)
    (lamella.path / "empty").mkdir()
    (lamella.path / "images").mkdir()
    image = lamella.path / "images" / "ref.tif"
    image.write_bytes(b"x")

    assert remove_lamella(experiment, lamella)

    assert image.exists()
    assert (lamella.path / "empty").exists()


def test_a_missing_directory_is_fine(tmp_path):
    experiment, lamella = _experiment_with_lamella(tmp_path)
    lamella.path.rmdir()

    assert remove_lamella(experiment, lamella)


def test_a_directory_with_only_empty_folders_stays(tmp_path):
    experiment, lamella = _experiment_with_lamella(tmp_path)
    (lamella.path / "empty").mkdir()

    assert remove_lamella(experiment, lamella)

    assert (lamella.path / "empty").exists()
