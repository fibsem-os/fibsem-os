"""The FM's parts as devices, over the simulated FM classes: each parameter and
command does what the FM class it adapts always did."""

import numpy as np
import pytest

from fibsem.devices.core import ParameterReadOnly
from fibsem.devices.drivers.fm import bind_fm_devices
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import (
    OBJECTIVE_STATES,
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
    objective_device_state,
    objective_state_name,
)
from fibsem.structures import InsertableDeviceState


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
    assert objective.state.cached is InsertableDeviceState.INSERTED
    assert microscope.objective.state == "Inserted"  # the FM class keeps its name
    assert seen == [InsertableDeviceState.INSERTED]
    objective.retract()
    assert objective.state.cached is InsertableDeviceState.RETRACTED


def test_every_objective_state_name_survives_the_device_state():
    for name in OBJECTIVE_STATES:
        assert objective_state_name(objective_device_state(name)) == name
    assert objective_device_state("Parked") is InsertableDeviceState.UNKNOWN


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


def test_the_emission_filter_is_one_typed_choice(fm):
    microscope, devices = fm
    filters = devices["filter_set"]
    assert filters.emission_filter.choices == [
        EmissionFilter("Reflection"),
        EmissionFilter("Fluorescence"),
    ]
    filters.emission_filter.set_value(EmissionFilter("Fluorescence"))
    assert microscope.filter_set.emission_wavelength == "Fluorescence"
    filters.emission_filter.set_value(REFLECTION)
    assert microscope.filter_set.emission_wavelength is None
    assert filters.emission_filter.get_value() == REFLECTION
    with pytest.raises(ValueError):
        filters.emission_filter.set_value(EmissionFilter("GFP", 510.0, 560.0))


class _BandFilters:
    """Single emission bands keyed by their bottom edge in nm, as Odemis has them."""

    available_excitation_wavelengths = (470.0,)
    available_emission_wavelengths = (None, 425.0, 510.0)
    emission_bands = {425.0: (425.0, 475.0), 510.0: (510.0, 560.0)}
    excitation_wavelength = 470.0
    emission_wavelength = None


def test_a_band_carries_both_edges_when_the_fm_class_knows_them():
    from fibsem.devices.drivers.fm import FMFilterSet

    old = _BandFilters()
    filters = FMFilterSet(old).connect()
    green = EmissionFilter("510–560 nm", low=510.0, high=560.0)
    assert filters.emission_filter.choices == [
        REFLECTION,
        EmissionFilter("425–475 nm", low=425.0, high=475.0),
        green,
    ]
    filters.emission_filter.set_value(green)
    assert old.emission_wavelength == 510.0
    assert filters.emission_filter.get_value() == green


def test_a_thermo_multi_band_reads_as_its_filter():
    from fibsem.devices.drivers.fm import FMFilterSet

    class ThermoLike:
        available_excitation_wavelengths = (488.0,)
        available_emission_wavelengths = (None, "Fluorescence")
        excitation_wavelength = 488.0
        emission_wavelength = 488.0  # what Thermo reports in fluorescence mode

    filters = FMFilterSet(ThermoLike()).connect()
    assert filters.emission_filter.get_value() == EmissionFilter("Fluorescence")
