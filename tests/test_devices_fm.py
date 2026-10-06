"""The FM's parts as devices, over the simulated FM classes: each parameter and
command does what the FM class it adapts always did."""

import time

import numpy as np
import pytest

from fibsem.devices.core import ParameterReadOnly
from fibsem.devices.drivers.fm import bind_fm_devices
from fibsem.fm.structures import (
    OBJECTIVE_STATES,
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
    objective_device_state,
    objective_state_name,
)
from fibsem.microscopes.simulator import SimulatedFluorescenceMicroscope
from fibsem.structures import InsertableDeviceState


@pytest.fixture
def fm():
    microscope = SimulatedFluorescenceMicroscope()
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
    # Its one parameter is where a running sequence is; the hardware is the parts'.
    assert list(devices["fm"].parameters) == ["progress"]
    assert not devices["fm"].progress.settable
    assert {"acquire_channel", "acquire_z_stack", "cancel"} <= set(
        devices["fm"].commands
    )


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
    gain = devices["camera"].gain.limits
    assert (gain.min, gain.max) == (0.0, 1.0)  # a fraction, as power is


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
        EmissionFilter("Fluorescence", multi_band=True),
    ]
    filters.emission_filter.set_value(EmissionFilter("Fluorescence", multi_band=True))
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
    assert filters.emission_filter.get_value() == EmissionFilter(
        "Fluorescence", multi_band=True
    )


def test_a_dual_band_filter_carries_both_bands():
    from fibsem.devices.drivers.fm import FMFilterSet

    class DualBand(_BandFilters):
        available_emission_wavelengths = (None, 505.0)
        emission_bands = {505.0: ((505.0, 535.0), (600.0, 650.0))}

    old = DualBand()
    filters = FMFilterSet(old).connect()
    dual = EmissionFilter("505–535 / 600–650 nm", bands=[(505, 535), (600, 650)])
    assert filters.emission_filter.choices == [REFLECTION, dual]
    assert dual.multi_band and dual.low == 505.0 and dual.centre is None
    filters.emission_filter.set_value(dual)
    assert old.emission_wavelength == 505.0


def test_an_excitation_between_bands_selects_the_nearest(fm, caplog):
    """FIB-1094: 488 nm means the band at 450 on a filter set of 365/450/550/635,
    as the Odemis driver has always read it, rather than a refusal."""
    microscope, devices = fm
    excitation = devices["filter_set"].excitation_wavelength

    with caplog.at_level("WARNING"):
        written = excitation.set_value(488)

    assert written == 450
    assert microscope.filter_set.excitation_wavelength == 450
    assert excitation.cached == 450
    assert "nearest" in caplog.text


def test_the_old_api_snaps_too_and_caches_what_was_applied(fm):
    microscope, devices = fm
    excitation = devices["filter_set"].excitation_wavelength
    seen = []
    excitation.changed.connect(seen.append)

    excitation.write_through(600)

    assert microscope.filter_set.excitation_wavelength == 635
    assert excitation.cached == 635
    assert seen == [635]


def test_a_write_caches_the_value_read_back(fm):
    """A driver may adjust a nearest parameter again; the cache holds what it applied."""
    microscope, devices = fm
    excitation = devices["filter_set"].excitation_wavelength
    original = type(microscope.filter_set).excitation_wavelength

    class Adjusting(type(microscope.filter_set)):
        @original.setter
        def excitation_wavelength(self, value):
            original.fset(self, value + 1e-9)  # float noise from a unit conversion

    microscope.filter_set.__class__ = Adjusting
    assert excitation.set_value(550) == pytest.approx(550 + 1e-9, abs=0)
    assert excitation.cached == microscope.filter_set.excitation_wavelength


def test_a_parameter_not_declared_nearest_still_refuses(fm):
    _, devices = fm
    with pytest.raises(ValueError):
        devices["camera"].binning.set_value(3)


def test_a_frame_carries_what_the_fm_class_stamped(fm, monkeypatch):
    """The FM class's own metadata, so a driver's per-frame values (odemis: date,
    pixel size, exposure) reach the coordinator rather than a later read."""
    microscope, devices = fm
    stamped = []
    acquire_image = microscope.acquire_image

    def acquire_and_keep(settings=None):
        image = acquire_image(settings)
        image.metadata.acquisition_date = "2026-10-02T08:00:00"  # as odemis would
        stamped.append(image.metadata)
        return image

    monkeypatch.setattr(microscope, "acquire_image", acquire_and_keep)
    channel = ChannelSettings(excitation_wavelength=550, power=0.3, exposure_time=0.05)
    frame = devices["fm"].acquire_frame(channel.to_dict())
    (md,) = stamped
    assert frame.metadata["acquisition_date"] == "2026-10-02T08:00:00"
    assert frame.metadata["pixel_size"] == [md.pixel_size_x, md.pixel_size_y]
    assert frame.metadata["exposure_time"] == 0.05
    assert frame.metadata["power"] == 0.3
    assert frame.metadata["excitation_wavelength"] == 550
    assert EmissionFilter.from_dict(frame.metadata["emission_filter"]) == (
        devices["filter_set"].emission_filter.cached
    )


def test_a_frame_with_the_current_settings_is_the_camera_frame(fm):
    microscope, devices = fm
    microscope.camera._use_counter = False
    np.random.seed(0)
    expected = microscope.camera.acquire_image()
    np.random.seed(0)
    frame = devices["fm"].acquire_frame()
    assert np.array_equal(frame.data, expected)
    assert frame.metadata["exposure_time"] == microscope.camera.exposure_time
    assert frame.metadata["objective_position"] == microscope.objective.position
    assert "acquisition_date" in frame.metadata


def test_live_view_sets_up_the_channel_and_stops_on_request(fm):
    microscope, devices = fm
    group = devices["fm"]
    assert {"start_live", "stop_live"} <= set(group.commands)
    channel = ChannelSettings(excitation_wavelength=550, power=0.4, exposure_time=0.03)
    group.start_live(channel.to_dict())
    try:
        assert group.is_live
        assert microscope.light_source.power == 0.4
        assert devices["light_source"].power.cached == 0.4
        frame = group.acquire_frame()
        assert frame.data.ndim == 2
    finally:
        group.stop_live()
    assert not group.is_live
    group.stop_live()  # safe when not live


def test_live_view_uses_the_fm_class_live_view_when_it_has_one(fm, monkeypatch):
    """odemis: the stream stays active between pulled frames, light on once."""
    microscope, devices = fm
    calls = []
    monkeypatch.setattr(
        microscope.camera, "_start_fast_acquisition", lambda: None, raising=False
    )
    monkeypatch.setattr(
        microscope, "start_acquisition", lambda s=None: calls.append(("start", s))
    )
    monkeypatch.setattr(microscope, "stop_acquisition", lambda: calls.append("stop"))
    devices["fm"].start_live()
    devices["fm"].stop_live()
    assert calls == [("start", None), "stop"]


def test_live_view_stops_by_itself_when_nobody_asks_for_a_frame(fm, caplog):
    _, devices = fm
    group = devices["fm"]
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
    _, devices = fm
    group = devices["fm"]
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
    from fibsem.fm.api import DeviceFluorescenceMicroscope

    api = DeviceFluorescenceMicroscope(fm[1])
    assert api.light_source.power_native_scale is None
    assert api.camera.gain_native_scale is None
    assert "native_max" not in fm[1]["light_source"].describe()["power"]
