"""The Odemis FM drivers drive the METEOR's odemis components directly.

Each test builds the FM devices over stub odemis components (``_odemis_stubs``) and
checks what they leave the components doing: the light's power per source, the filter
wheel's position, the camera's settings and subscriptions, the focuser's position.

What has to hold:

- the light is on only while the camera exposes (or live view runs), with only the
  selected source at its power, and off afterwards, even when the exposure fails;
- excitation and power are settings for the next exposure; while live, a change
  reaches the light at once;
- the emission filter is the wheel's position, and writing it moves the wheel.

Nothing here has run on a METEOR.
"""

import sys
import time

import pytest

from fibsem.fm.structures import REFLECTION, ChannelSettings, EmissionFilter
from fibsem.util.timestamps import iso_from_posix
from tests.fm import _odemis_stubs as stubs

MAX_POWER = 0.4


@pytest.fixture(scope="module")
def odemis():
    saved = {}
    for name in stubs.ODEMIS_MODULE_NAMES + stubs.FIBSEM_ODEMIS_MODULE_NAMES:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    stubs.install_odemis_stubs()
    import fibsem.drivers.odemis.devices as drivers
    import fibsem.fm.odemis as fm_odemis

    yield fm_odemis, drivers

    stubs.remove_odemis_stubs()
    sys.modules.update(saved)


def _components(state: str = "retracted"):
    components = stubs.default_components()
    focus = components["focus"]
    if "inserted" in state:
        focus.position._value = {"z": 8.0e-3}
    if "gain" in state:
        components["ccd"].gain = stubs.FakeVA(1.0, range=(0.0, 1.0))
    if "no-favourites" in state:
        focus._metadata = {}
    if "lit" in state:
        components["light"].power._value = [0.0, 0.1, 0.0, 0.0]
    return components


def _bind(odemis, state: str = "retracted", components=None):
    _, drivers = odemis
    components = stubs.use_components(components or _components(state))
    return components, drivers.bind_odemis_fm()


def _off():
    return [0.0, 0.0, 0.0, 0.0]


def _only(index: int, watts: float):
    power = _off()
    power[index] = watts
    return power


class _LightLog:
    """Every power the light was set to, in order."""

    def __init__(self, light):
        self.powers = []
        light.power._setter = lambda value: self.powers.append(list(value))


CHANNELS = {
    "fluorescence": ChannelSettings(
        excitation_wavelength=450,
        emission_wavelength=500.0,
        power=0.3,
        exposure_time=0.2,
        gain=0.5,
    ),
    "reflection": ChannelSettings(
        excitation_wavelength=550,
        emission_wavelength=None,
        power=0.1,
        exposure_time=0.05,
    ),
    "label": ChannelSettings(
        excitation_wavelength=635,
        emission_wavelength="Fluorescence",
        power=1.0,
        exposure_time=0.5,
    ),
}


def wait_for(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


# -- connect ----------------------------------------------------------------------------


def test_the_drivers_are_the_fm_devices(odemis):
    _, devices = _bind(odemis, "gain")
    assert {name: type(d).__name__ for name, d in devices.items()} == {
        "fm": "OdemisFM",
        "camera": "OdemisFMCamera",
        "light_source": "OdemisFMLightSource",
        "filter_set": "OdemisFMFilterSet",
        "objective": "OdemisFMObjective",
    }
    assert sorted(devices["camera"].parameters) == [
        "binning",
        "exposure_time",
        "gain",
        "mount_transform",
        "offset",
        "pixel_size",
        "resolution",
    ]
    assert not devices["camera"].offset.settable
    assert devices["camera"].binning.choices == [1, 2, 4, 8, 16]
    assert REFLECTION in devices["filter_set"].emission_filter.choices
    assert sorted(devices["filter_set"].excitation_wavelength.choices) == pytest.approx(
        [365.0, 450.0, 550.0, 635.0]
    )


def test_connecting_leaves_the_light_and_wheel_alone(odemis):
    components = _components()
    log = _LightLog(components["light"])
    _bind(odemis, components=components)
    assert log.powers == []
    assert components["filter"].position.value == {"band": 2}


def test_with_the_light_off_the_excitation_is_the_one_below_the_wheels_filter(odemis):
    """As odemis's FluoStream starts: the source closest below the emission, at no
    power. The wheel is at 500-550 nm, so 450 nm."""
    _, devices = _bind(odemis)
    assert devices["filter_set"].excitation_wavelength.get_value() == pytest.approx(450)
    assert devices["light_source"].power.get_value() == 0.0


def test_with_the_light_on_the_excitation_is_the_brightest_source(odemis):
    _, devices = _bind(odemis, "lit")
    assert devices["filter_set"].excitation_wavelength.get_value() == pytest.approx(450)
    assert devices["light_source"].power.get_value() == pytest.approx(0.1 / MAX_POWER)


# -- settings ---------------------------------------------------------------------------


def test_power_and_excitation_are_settings_for_the_next_exposure(odemis):
    components, devices = _bind(odemis)
    log = _LightLog(components["light"])
    devices["light_source"].power.write_through(0.5)
    devices["filter_set"].excitation_wavelength.write_through(640)  # nearest: 635
    assert log.powers == []
    assert devices["light_source"].power.get_value() == 0.5
    assert devices["filter_set"].excitation_wavelength.get_value() == pytest.approx(635)


def test_power_is_clipped_to_a_fraction(odemis):
    _, devices = _bind(odemis)
    devices["light_source"].power.write_through(1.5)
    assert devices["light_source"].power.get_value() == 1.0
    power = devices["light_source"].power
    assert (power.limits.min, power.limits.max) == (0.0, 1.0)


def test_the_emission_filter_is_the_wheels_position(odemis):
    components, devices = _bind(odemis)
    wheel = components["filter"]
    param = devices["filter_set"].emission_filter
    assert param.get_value().low == pytest.approx(500)

    target = next(f for f in param.choices if f != REFLECTION and f.low > 680 - 1)
    param.write_through(target)
    assert wheel.position.value == {"band": 4}

    param.write_through(REFLECTION)
    assert wheel.position.value == {"band": 0}
    assert param.get_value() == REFLECTION


def test_the_wheel_is_not_moved_when_it_is_there_already(odemis):
    components, devices = _bind(odemis)
    wheel = components["filter"]
    param = devices["filter_set"].emission_filter
    before = wheel.position.set_count
    param.write_through(param.get_value())
    assert wheel.position.set_count == before


def test_an_unknown_emission_filter_is_refused(odemis):
    _, devices = _bind(odemis)
    with pytest.raises(ValueError):
        devices["filter_set"].write_emission_filter(
            EmissionFilter("nowhere", bands=((1.0, 2.0),))
        )


def test_fluorescence_takes_the_band_above_the_excitation(odemis):
    components, devices = _bind(odemis)
    devices["filter_set"].excitation_wavelength.write_through(550)
    devices["filter_set"].select_fluorescence()
    assert components["filter"].position.value == {"band": 3}  # 590 nm
    assert devices["filter_set"].emission_filter.get_value().low == pytest.approx(590)


# -- acquisitions -----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CHANNELS))
def test_an_acquisition_lights_only_the_selected_source_for_the_exposure(odemis, name):
    components, devices = _bind(odemis, "gain")
    log = _LightLog(components["light"])
    settings = CHANNELS[name]

    frame = devices["fm"].acquire_frame(settings.to_dict())

    source = {450: 1, 550: 2, 635: 3}[settings.excitation_wavelength]
    assert log.powers == [_only(source, settings.power * MAX_POWER), _off()]
    assert components["ccd"].exposureTime.value == settings.exposure_time
    assert frame.metadata["exposure_time"] == settings.exposure_time
    assert frame.metadata["excitation_wavelength"] == pytest.approx(
        settings.excitation_wavelength
    )
    assert frame.metadata["power"] == pytest.approx(settings.power)
    assert frame.data.shape == (2048, 2048)


def test_an_acquisition_with_the_current_settings_changes_nothing_else(odemis):
    components, devices = _bind(odemis, "lit")
    log = _LightLog(components["light"])
    devices["fm"].acquire_frame(None)
    assert log.powers == [_only(1, 0.1), _off()]
    assert components["filter"].position.value == {"band": 2}


def test_the_light_goes_off_when_an_exposure_fails(odemis):
    components, devices = _bind(odemis, "lit")
    log = _LightLog(components["light"])

    def fail(asap=True):
        raise RuntimeError("camera gone")

    components["ccd"].data.get = fail
    with pytest.raises(RuntimeError, match="camera gone"):
        devices["camera"].acquire()
    assert log.powers[-1] == _off()


def test_a_frame_carries_odemis_time_with_its_offset(odemis):
    """odemis stamps MD_ACQ_DATE (POSIX) at exposure; the frame keeps it, written
    with this machine's offset, over the time `acquire_frame` took before it."""
    _, devices = _bind(odemis, "inserted")
    frame = devices["fm"].acquire_frame(CHANNELS["fluorescence"].to_dict())
    assert frame.metadata["acquisition_date"] == iso_from_posix(1_780_000_000.0)


# -- live view --------------------------------------------------------------------------


def test_live_view_keeps_the_light_on_and_the_camera_acquiring(odemis):
    components, devices = _bind(odemis, "inserted")
    camera = components["ccd"]
    log = _LightLog(components["light"])
    group = devices["fm"]
    group.live_timeout = None
    settings = CHANNELS["fluorescence"]

    group.start_live(settings.to_dict())
    assert len(camera.data.listeners) == 1
    frames = [group.acquire_frame() for _ in range(3)]
    assert len(frames) == 3
    # One switch on for the whole of live view, not one per frame.
    assert log.powers == [_only(1, 0.3 * MAX_POWER)]

    devices["light_source"].power.write_through(0.5)  # reaches the light at once
    devices["filter_set"].excitation_wavelength.write_through(635)
    assert log.powers[-2:] == [_only(1, 0.5 * MAX_POWER), _only(3, 0.5 * MAX_POWER)]

    group.stop_live()
    assert log.powers[-1] == _off()
    assert camera.data.listeners == []
    assert not group.is_live


def test_live_view_stops_by_itself_when_nobody_pulls(odemis):
    components, devices = _bind(odemis, "inserted")
    group = devices["fm"]
    group.live_timeout = 0.2
    group.start_live(None)
    assert wait_for(lambda: not group.is_live, timeout=3.0)
    assert components["light"].power.value == _off()
    assert components["ccd"].data.listeners == []


def test_after_live_view_an_acquisition_switches_the_light_itself(odemis):
    components, devices = _bind(odemis, "lit")
    group = devices["fm"]
    group.live_timeout = None
    group.start_live(None)
    group.stop_live()
    log = _LightLog(components["light"])
    devices["camera"].acquire()
    assert log.powers == [_only(1, 0.1), _off()]


# -- the camera and objective -----------------------------------------------------------


def test_a_camera_without_gain_has_no_gain_parameter(odemis):
    _, devices = _bind(odemis)
    assert "gain" not in devices["camera"].parameters


def _camera_with_gain(odemis, va):
    components = _components()
    components["ccd"].gain = va
    return _bind(odemis, components=components)[1]["camera"]


def test_gain_is_a_fraction_of_the_cameras_range(odemis):
    va = stubs.FakeVA(4.0, range=(0.0, 16.0))
    camera = _camera_with_gain(odemis, va)

    assert camera.gain.get_value() == pytest.approx(0.25)
    camera.gain.write_through(0.5)
    assert va.value == pytest.approx(8.0)
    assert (camera.gain.limits.min, camera.gain.limits.max) == (0.0, 1.0)
    camera.gain.write_through(1.5)  # clipped to the camera's maximum
    assert va.value == pytest.approx(16.0)


def test_gain_with_set_values_takes_the_nearest(odemis):
    va = stubs.FakeVA(2.0, choices={1.0, 2.0, 4.0})
    camera = _camera_with_gain(odemis, va)

    assert camera.gain.get_value() == pytest.approx(0.5)
    camera.gain.write_through(0.9)  # 3.6 in camera units
    assert va.value == 4.0


def test_gain_without_a_range_is_not_offered(odemis):
    # with no range or choices there is no maximum to scale it to a fraction by
    camera = _camera_with_gain(odemis, stubs.FakeVA(3.0))

    assert "gain" not in camera.parameters


def test_binning_is_checked_against_the_cameras_binnings(odemis):
    components, devices = _bind(odemis)
    devices["camera"].binning.write_through(2)
    assert components["ccd"].binning.value == (2, 2)
    with pytest.raises(ValueError):
        devices["camera"].write_binning(3)


def test_the_objective_inserts_and_retracts_to_its_favourites(odemis):
    components, devices = _bind(odemis)
    objective = devices["objective"]
    assert objective.state.get_value().name == "RETRACTED"
    assert objective.insert() is True
    assert components["focus"].position.value == {"z": 8.0e-3}
    assert objective.state.get_value().name == "INSERTED"
    assert objective.retract() is True
    assert components["focus"].position.value == {"z": -1.0e-3}


def test_an_objective_without_favourites_does_not_move(odemis):
    components, devices = _bind(odemis, "no-favourites")
    assert devices["objective"].insert() is False
    assert components["focus"].position.value == {"z": -1.0e-3}


def test_an_objective_move_is_clipped_to_the_user_limit(odemis):
    components, devices = _bind(odemis)
    objective = devices["objective"]
    objective.limit_position.write_through(5.0e-3)
    objective.move_absolute(9.0e-3)
    assert components["focus"].position.value == {"z": 5.0e-3}


# -- the FM API over the devices --------------------------------------------------------


def _api_fm(odemis, state="inserted"):
    fm_odemis, _ = odemis
    components, devices = _bind(odemis, state)
    devices["fm"].live_timeout = None
    return components, fm_odemis.DeviceOdemisFluorescenceMicroscope(devices)


def test_the_api_acquires_with_the_channel_and_switches_the_light(odemis):
    components, fm = _api_fm(odemis)
    log = _LightLog(components["light"])
    image = fm.acquire_image(CHANNELS["fluorescence"])
    channel = image.metadata.channels[0]
    assert channel.excitation_wavelength == pytest.approx(450)
    assert channel.emission_wavelength == pytest.approx(500)
    assert channel.power == pytest.approx(0.3)
    assert log.powers == [_only(1, 0.3 * MAX_POWER), _off()]


def test_the_api_maps_a_fluorescence_emission_to_a_band(odemis):
    components, fm = _api_fm(odemis)
    fm.filter_set.excitation_wavelength = 450
    fm.filter_set.emission_wavelength = "Fluorescence"
    assert components["filter"].position.value == {"band": 2}
    assert fm.filter_set.emission_wavelength == pytest.approx(500)


def test_the_api_z_stack_turns_the_light_off_after_each_frame(odemis):
    from fibsem.fm.acquisition import acquire_z_stack
    from fibsem.fm.structures import ZParameters

    components, fm = _api_fm(odemis)
    log = _LightLog(components["light"])
    acquire_z_stack(
        fm,
        [CHANNELS["fluorescence"], CHANNELS["reflection"]],
        ZParameters(zmin=-1e-6, zmax=1e-6),
    )
    assert len(log.powers) > 2
    assert log.powers[-1] == _off()
    assert all(p == _off() for p in log.powers[1::2])


def test_api_live_view_runs_and_stops_the_light(odemis):
    components, fm = _api_fm(odemis)
    camera = components["ccd"]
    pulled = []
    real_get = camera.data.get

    def counting_get(asap=True):
        pulled.append(asap)
        return real_get(asap)

    camera.data.get = counting_get
    fm.start_acquisition(CHANNELS["fluorescence"])
    assert wait_for(lambda: len(pulled) >= 3)
    assert components["light"].power.value == _only(1, 0.3 * MAX_POWER)
    fm.stop_acquisition()
    assert not fm.is_streaming
    assert wait_for(lambda: components["light"].power.value == _off())
    assert camera.data.listeners == []
    assert all(asap is False for asap in pulled)


def test_an_odemis_microscope_builds_its_fm_from_the_devices(odemis):
    from types import SimpleNamespace

    import fibsem.drivers.odemis.microscope as odemis_microscope
    from fibsem.drivers.odemis.devices import build_odemis_fm
    from fibsem.drivers.registry import BuildContext
    from fibsem.structures import (
        CameraImageTransform,
        DeviceEntry,
        FluorescenceSystemSettings,
    )

    microscope = odemis_microscope.OdemisThermoMicroscope.__new__(
        odemis_microscope.OdemisThermoMicroscope
    )
    microscope.system = SimpleNamespace(
        fm=FluorescenceSystemSettings(mount_transform=CameraImageTransform.FLIP_X)
    )
    stubs.use_components(_components("inserted"))
    fm = build_odemis_fm(
        DeviceEntry.from_dict(microscope.system.fm.to_dict(), name="fm"),
        BuildContext(microscope=microscope),
    )
    assert type(fm).__name__ == "DeviceOdemisFluorescenceMicroscope"
    assert fm.parent is microscope
    assert sorted(fm.devices) == [
        "camera",
        "filter_set",
        "fm",
        "light_source",
        "objective",
    ]
    assert fm.devices["fm"].live_timeout is None
    assert all(d.parent is microscope for d in fm.devices.values())
    # As the configuration's fm entry states it.
    assert fm.mount_transform is CameraImageTransform.FLIP_X
