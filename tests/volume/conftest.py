"""A volume experiment on the demo microscope, with positions ready to slice."""

import pytest

from fibsem import acquire
from fibsem.applications.volume.experiment import VolumeExperiment
from fibsem.applications.volume.position import FIB_REFERENCE, SEM_REFERENCE
from fibsem.applications.volume.tasks import milling as milling_module
from fibsem.structures.settings import BeamType, ImageSettings, Point
from fibsem.workflows import task as task_module


@pytest.fixture
def experiment(tmp_path):
    return VolumeExperiment.create(tmp_path, "vol")


def set_up_position(experiment, microscope, name: str, iterations: int = 3):
    """A position at the current stage pose, with both alignment references."""
    position = experiment.add_position(
        name, microscope.get_microscope_state(), Point(0, 1e-6), iterations=iterations
    )
    folder = experiment.position_path(position)
    folder.mkdir(parents=True)
    for beam, filename in (
        (BeamType.ION, FIB_REFERENCE),
        (BeamType.ELECTRON, SEM_REFERENCE),
    ):
        settings = ImageSettings(beam_type=beam, hfw=50e-6, resolution=(256, 170))
        acquire.acquire_image(microscope, settings).save(str(folder / filename))
    return position


@pytest.fixture
def position(experiment, microscope):
    return set_up_position(experiment, microscope, "aspen")


@pytest.fixture
def milled(monkeypatch):
    """The configs the slice task and the shared milling loop hand to the mill."""
    configs = []
    real = milling_module.run_milling_task

    def run(microscope, config, **kwargs):
        configs.append(config)
        return real(microscope, config, **kwargs)

    monkeypatch.setattr(milling_module, "run_milling_task", run)
    monkeypatch.setattr(task_module, "run_milling_task", run)
    return configs
