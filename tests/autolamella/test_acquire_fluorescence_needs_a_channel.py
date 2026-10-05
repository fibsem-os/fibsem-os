"""Acquire Fluorescence Image with no channels configured fails up front,
naming the problem, rather than with an IndexError from the autofocus's
first-channel fallback (FIB-1067). On the simulated Arctis, which has a
fluorescence microscope; nothing is acquired."""

import os

import pytest
from psygnal.containers import EventedDict

import fibsem.config as fconfig
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (
    AcquireFluorescenceImageConfig,
    AcquireFluorescenceImageTask,
)

ACQUIRE = "Acquire Fluorescence Image"
CONFIG = os.path.join(fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(config_path=CONFIG, setup_logging=False)
    if microscope.fm is None:
        pytest.skip("no fluorescence microscope in the simulator")
    yield microscope
    microscope.disconnect()


def test_no_channels_is_refused_by_name(microscope, tmp_path):
    config = AcquireFluorescenceImageConfig(task_name=ACQUIRE)
    assert config.channel_settings == [] and config.autofocus_settings.enabled
    exp = Experiment(path=tmp_path, name="fm-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        microscope.get_microscope_state(), EventedDict({ACQUIRE: config})
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    task = AcquireFluorescenceImageTask(
        microscope=microscope,
        config=config,
        lamella=lamella,
        parent_ui=None,
        task_manager=None,
    )

    with pytest.raises(ValueError, match="No fluorescence channels configured"):
        task._run()
