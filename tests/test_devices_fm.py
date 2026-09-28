"""The FM's parts as devices, over the simulated FM classes: each parameter and
command does what the FM class it adapts always did."""

import numpy as np
import pytest

from fibsem.devices.core import ParameterReadOnly
from fibsem.devices.drivers.fm import bind_fm_devices
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings


@pytest.fixture
def fm():
    microscope = FluorescenceMicroscope()
    return microscope, bind_fm_devices(microscope)


def test_the_fm_is_a_group_and_four_parts(fm):
    _, devices = fm
    assert sorted(devices) == [
        "camera",
        "filter_set",
        "fm",
        "light_source",
        "objective",
    ]
    assert devices["fm"].parameters == {}
    assert "acquire_channel" in devices["fm"].commands


def test_parameters_read_and_write_the_fm_classes(fm):
    microscope, devices = fm
    camera, light = devices["camera"], devices["light_source"]
    camera.exposure_time.set_value(0.25)
    assert microscope.camera.exposure_time == 0.25
    camera.binning.set_value(2)
    assert microscope.camera.binning == 2
    light.power.set_value(0.3)
    assert microscope.light_source.power == 0.3
    devices["filter_set"].excitation_wavelength.set_value(450)
    assert microscope.filter_set.excitation_wavelength == 450


def test_metadata_comes_from_the_fm_classes(fm):
    microscope, devices = fm
    exposure = devices["camera"].exposure_time.limits
    assert (exposure.min, exposure.max) == microscope.camera.exposure_time_limits
    assert devices["camera"].binning.choices == list(
        microscope.camera.available_binnings
    )
    assert devices["light_source"].power.limits.max == 1.0
    assert devices["light_source"].power.set_value(1.5) == 1.0  # clipped


def test_the_objective_moves_only_through_commands(fm):
    microscope, devices = fm
    objective = devices["objective"]
    with pytest.raises(ParameterReadOnly):
        objective.position.set_value(0.0)
    seen = []
    objective.state.changed.connect(seen.append)
    objective.state.get_value()
    objective.insert()
    assert objective.state.cached == "Inserted" == microscope.objective.state
    assert seen == ["Inserted"]
    objective.retract()
    assert objective.state.cached == "Retracted"


def test_a_camera_frame_is_the_fm_class_frame(fm):
    microscope, devices = fm
    frame = devices["camera"].acquire()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == microscope.camera.resolution[::-1]


def test_a_channel_sets_up_the_parts_and_signals_their_changes(fm):
    _, devices = fm
    power = devices["light_source"].power
    power.get_value()
    seen = []
    power.changed.connect(seen.append)
    channel = ChannelSettings(excitation_wavelength=450, power=0.2, exposure_time=0.01)
    frame = devices["fm"].acquire_channel(channel.to_dict())
    assert isinstance(frame, np.ndarray)
    assert seen == [0.2]
    assert devices["filter_set"].excitation_wavelength.cached == 450
    assert devices["camera"].exposure_time.cached == 0.01
