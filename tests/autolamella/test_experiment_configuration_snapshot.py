"""An experiment keeps a copy of the microscope configuration it ran with.

It already copied the protocol and recorded the instrument's identity. The
configuration -- calibration and defaults -- was recorded nowhere, so once a holder
was recalibrated or a limit changed, nothing could say what an earlier run used.
"""

import os
from pathlib import Path

import pytest

from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.structures import CONFIGURATION_VERSION, SystemSettings


@pytest.fixture
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo")
    yield scope
    scope.disconnect()


def _saved_experiment(tmp_path: Path) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-experiment")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)
    exp.save()  # registration writes only for an experiment that has a file
    return exp


def _snapshots(folder: Path):
    return sorted(
        p.name for p in folder.iterdir() if p.name.startswith("configuration")
    )


def test_registering_copies_the_configuration_in(microscope, tmp_path):
    experiment = _saved_experiment(tmp_path)
    folder = Path(experiment.path)

    experiment.register_metadata(microscope)

    copy = utils.load_yaml(str(folder / "configuration.yaml"))
    assert copy["version"] == CONFIGURATION_VERSION
    assert set(copy) >= {"info", "hardware", "calibration", "defaults"}
    # It reads back as the configuration the session ran with. Compared through
    # the file's own contents: what is fitted (GIS, manipulator) is asked of the
    # instrument at connect and is deliberately not in a configuration file.
    back = SystemSettings.from_dict(copy)
    assert utils._plain(back.to_dict()) == utils._plain(microscope.system.to_dict())
    assert back.stage == microscope.system.stage


def test_an_unchanged_configuration_writes_nothing_new(microscope, tmp_path):
    experiment = _saved_experiment(tmp_path)
    folder = Path(experiment.path)
    experiment.register_metadata(microscope)
    first = (folder / "configuration.yaml").stat().st_mtime_ns

    experiment.register_metadata(microscope)

    assert _snapshots(folder) == ["configuration.yaml"]
    assert (folder / "configuration.yaml").stat().st_mtime_ns == first


def test_a_changed_configuration_overwrites_the_copy(microscope, tmp_path):
    """The copy is the latest session's, the same rule the session record follows."""
    experiment = _saved_experiment(tmp_path)
    folder = Path(experiment.path)
    experiment.register_metadata(microscope)

    microscope.system.stage.shuttle_pre_tilt = 12.0  # a recalibrated holder
    experiment.register_metadata(microscope)

    assert _snapshots(folder) == ["configuration.yaml"]
    now = utils.load_yaml(str(folder / "configuration.yaml"))
    assert SystemSettings.from_dict(now).stage.shuttle_pre_tilt == 12.0


def test_an_experiment_without_a_file_is_not_written_to(microscope, tmp_path):
    """Constructing an experiment does not touch the disk (FIB-420), and neither
    does registering one that has no file yet."""
    experiment = Experiment(path=tmp_path, name="unsaved")
    experiment.task_protocol = AutoLamellaTaskProtocol()

    experiment.register_metadata(microscope)

    assert not os.path.exists(experiment.path)


def test_a_copy_that_cannot_be_written_does_not_fail_registration(
    microscope, tmp_path, monkeypatch
):
    experiment = _saved_experiment(tmp_path)

    def refuse(*args, **kwargs):
        raise PermissionError("read-only share")

    monkeypatch.setattr(utils, "_write_configuration_file", refuse)
    experiment.register_metadata(microscope)

    assert experiment.session is not None
    assert not (Path(experiment.path) / "configuration.yaml").exists()
