"""The FM as devices: a group and four parts, on the Demo FM devices, which do what
the simulated FM does. What every FM device driver shares (channels, frames, live
view, nearest excitation) is checked here; each driver's own tests check the rest."""

import time

import numpy as np
import pytest

from fibsem.devices.core import ParameterReadOnly
from fibsem.devices.drivers.demo import bind_demo_fm
from fibsem.fm.structures import (
    OBJECTIVE_STATES,
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
    objective_device_state,
    objective_state_name,
)
from fibsem.microscopes.simulator import SIM_CAMERA_EXPOSURE_LIMITS
from fibsem.structures import InsertableDeviceState


@pytest.fixture
def fm():
    return bind_demo_fm()


def test_the_fm_is_a_group_and_four_parts(fm):
    assert sorted(fm) == [
        "camera",
        "filter_set",
        "fm",
        "light_source",
        "objective",
    ]
    # Its one parameter is where a running sequence is; the hardware is the parts'.
    assert list(fm["fm"].parameters) == ["progress"]
    assert not fm["fm"].progress.settable
    assert {"acquire_channel", "acquire_z_stack", "cancel"} <= set(fm["fm"].commands)


def test_parameters_read_and_write_the_parts(fm):
    camera, light = fm["camera"], fm["light_source"]
    camera.exposure_time.set_value(0.25)
    assert camera.sim_exposure_time == 0.25
    camera.binning.set_value(2)
    assert camera.sim_binning == 2
    light.power.set_value(0.3)
    assert light.sim_power == 0.3
    fm["filter_set"].excitation_wavelength.set_value(450)
    assert fm["filter_set"].sim_excitation_wavelength == 450


def test_the_parts_report_their_limits(fm):
    exposure = fm["camera"].exposure_time.limits
    assert (exposure.min, exposure.max) == SIM_CAMERA_EXPOSURE_LIMITS
    assert fm["camera"].binning.choices == [1, 2, 4, 8]
    assert fm["light_source"].power.limits.max == 1.0
    assert fm["light_source"].power.set_value(1.5) == 1.0  # clipped


def test_the_objective_moves_only_through_commands(fm):
    objective = fm["objective"]
    with pytest.raises(ParameterReadOnly):
        objective.position.set_value(0.0)
    seen = []
    objective.state.changed.connect(seen.append)
    objective.state.get_value()
    objective.insert()
    assert objective.state.cached is InsertableDeviceState.INSERTED
    assert seen == [InsertableDeviceState.INSERTED]
    objective.retract()
    assert objective.state.cached is InsertableDeviceState.RETRACTED


def test_every_objective_state_name_survives_the_device_state():
    for name in OBJECTIVE_STATES:
        assert objective_state_name(objective_device_state(name)) == name
    assert objective_device_state("Parked") is InsertableDeviceState.UNKNOWN


def test_a_camera_frame_is_the_binned_resolution(fm):
    frame = fm["camera"].acquire()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == tuple(fm["camera"].resolution.get_value())[::-1]


def test_a_channel_sets_up_the_parts_and_signals_their_changes(fm):
    power = fm["light_source"].power
    power.get_value()
    seen = []
    power.changed.connect(seen.append)
    channel = ChannelSettings(excitation_wavelength=450, power=0.2, exposure_time=0.01)
    frame = fm["fm"].acquire_channel(channel.to_dict())
    assert isinstance(frame, np.ndarray)
    assert seen == [0.2]
    assert fm["filter_set"].excitation_wavelength.cached == 450
    assert fm["camera"].exposure_time.cached == 0.01


def test_the_emission_filter_is_one_typed_choice(fm):
    filters = fm["filter_set"]
    fluorescence = EmissionFilter("Fluorescence", multi_band=True)
    assert filters.emission_filter.choices == [REFLECTION, fluorescence]
    filters.emission_filter.set_value(fluorescence)
    assert filters.sim_emission_filter == fluorescence
    filters.emission_filter.set_value(REFLECTION)
    assert filters.emission_filter.get_value() == REFLECTION
    with pytest.raises(ValueError):
        filters.emission_filter.set_value(EmissionFilter("GFP", 510.0, 560.0))


def test_an_excitation_between_bands_selects_the_nearest(fm, caplog):
    """FIB-1094: 488 nm means the band at 450 on a filter set of 365/450/550/635,
    as the Odemis driver has always read it, rather than a refusal."""
    excitation = fm["filter_set"].excitation_wavelength

    with caplog.at_level("WARNING"):
        written = excitation.set_value(488)

    assert written == 450
    assert fm["filter_set"].sim_excitation_wavelength == 450
    assert excitation.cached == 450
    assert "nearest" in caplog.text


def test_the_old_api_snaps_too_and_caches_what_was_applied(fm):
    excitation = fm["filter_set"].excitation_wavelength
    seen = []
    excitation.changed.connect(seen.append)

    excitation.write_through(600)

    assert fm["filter_set"].sim_excitation_wavelength == 635
    assert excitation.cached == 635
    assert seen == [635]


def test_a_parameter_not_declared_nearest_still_refuses(fm):
    with pytest.raises(ValueError):
        fm["camera"].binning.set_value(3)


def test_a_frame_carries_what_it_was_taken_with(fm):
    channel = ChannelSettings(excitation_wavelength=550, power=0.3, exposure_time=0.05)
    frame = fm["fm"].acquire_frame(channel.to_dict())
    assert frame.metadata["exposure_time"] == 0.05
    assert frame.metadata["power"] == 0.3
    assert frame.metadata["excitation_wavelength"] == 550
    assert frame.metadata["pixel_size"] == list(fm["camera"].pixel_size.get_value())
    assert EmissionFilter.from_dict(frame.metadata["emission_filter"]) == (
        fm["filter_set"].emission_filter.cached
    )
    assert "acquisition_date" in frame.metadata


def test_a_frame_with_the_current_settings_is_the_camera_frame(fm):
    frame = fm["fm"].acquire_frame()
    assert frame.data.shape == tuple(fm["camera"].resolution.get_value())[::-1]
    assert frame.metadata["exposure_time"] == fm["camera"].sim_exposure_time
    assert frame.metadata["objective_position"] == fm["objective"].sim_position
    assert "acquisition_date" in frame.metadata


def test_live_view_sets_up_the_channel_and_stops_on_request(fm):
    group = fm["fm"]
    assert {"start_live", "stop_live"} <= set(group.commands)
    channel = ChannelSettings(excitation_wavelength=550, power=0.4, exposure_time=0.03)
    group.start_live(channel.to_dict())
    try:
        assert group.is_live
        assert fm["light_source"].sim_power == 0.4
        assert fm["light_source"].power.cached == 0.4
        frame = group.acquire_frame()
        assert frame.data.ndim == 2
    finally:
        group.stop_live()
    assert not group.is_live
    group.stop_live()  # safe when not live


def test_live_view_stops_by_itself_when_nobody_asks_for_a_frame(fm, caplog):
    group = fm["fm"]
    group.live_timeout = 0.2
    stopped = []
    group._stop_live = lambda: stopped.append(True)
    group.start_live()
    deadline = time.monotonic() + 3
    while group.is_live and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not group.is_live and stopped == [True]
    assert "stopping live view" in caplog.text


def test_live_view_keeps_going_while_frames_are_asked_for(fm):
    group = fm["fm"]
    group.live_timeout = 0.3
    group.start_live()
    try:
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            group.acquire_frame()
            time.sleep(0.05)
        assert group.is_live
    finally:
        group.stop_live()


def test_the_simulator_has_no_hardware_units_for_power_or_gain(fm):
    """Its power and gain are fractions already, so there is nothing more to show."""
    from fibsem.fm.microscope import FluorescenceMicroscope

    api = FluorescenceMicroscope(fm)
    assert api.light_source.power_native_scale is None
    assert api.camera.gain_native_scale is None
    assert "native_max" not in fm["light_source"].describe()["power"]
