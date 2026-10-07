"""A simulated instrument: the configuration's manufacturer, with `sim: {enabled: true}`.

The Demo stands in for it and shows that manufacturer's profile
(`fibsem.drivers.demo.profiles`). ThermoFisher is the only profile so far.
"""

import pytest

from fibsem import manufacturers, utils
from fibsem.devices import entries
from fibsem.drivers import registry
from fibsem.drivers.demo.devices import DemoBeam, DemoStage
from fibsem.drivers.demo.microscope import DemoMicroscope
from fibsem.drivers.demo.profiles import THERMOFISHER_PROFILE, demo_profile
from fibsem.structures import BeamType, DeviceEntry


def _system(manufacturer, sim_enabled=True):
    system = utils.load_microscope_configuration(None, None).system
    system.info.manufacturer = manufacturers.normalize_manufacturer(manufacturer)
    system.sim = {**(system.sim or {}), "enabled": sim_enabled}
    return system


@pytest.mark.parametrize(
    "manufacturer", ["ThermoFisher", "thermo", "Demo", "demo", None, "Unknown"]
)
def test_demo_and_thermofisher_are_the_thermofisher_profile(manufacturer):
    assert demo_profile(manufacturer) is THERMOFISHER_PROFILE


@pytest.mark.parametrize("manufacturer", ["Tescan", "Odemis", "JEOL"])
def test_an_instrument_without_a_profile_is_refused_by_name(manufacturer):
    with pytest.raises(NotImplementedError, match=f"cannot simulate a {manufacturer}"):
        demo_profile(manufacturer)


def test_sim_enabled_is_what_makes_a_configuration_simulated():
    assert registry.is_simulated(_system("ThermoFisher"))
    assert not registry.is_simulated(_system("ThermoFisher", sim_enabled=False))
    system = _system("ThermoFisher")
    system.sim = {}
    assert not registry.is_simulated(system)


def test_a_simulated_thermofisher_is_the_demo_with_its_devices():
    microscope = registry.connect_microscope(_system("ThermoFisher"))

    assert isinstance(microscope, DemoMicroscope)
    assert microscope.profile is THERMOFISHER_PROFILE
    assert microscope.manufacturer == manufacturers.THERMOFISHER
    # The configuration still names the instrument; its images say it was the Demo.
    assert microscope.system.info.manufacturer == manufacturers.THERMOFISHER
    assert microscope.system.info.model == "DemoMicroscope"
    # Built by the Demo's builders, not AutoScript's.
    assert isinstance(microscope.beams[BeamType.ELECTRON], DemoBeam)
    assert isinstance(microscope.stage, DemoStage)


def test_a_simulated_tescan_is_refused_until_it_has_a_profile():
    with pytest.raises(NotImplementedError, match="cannot simulate a Tescan"):
        registry.connect_microscope(_system("Tescan"))


def test_the_beams_offer_the_profiles_values():
    microscope = registry.connect_microscope(_system("ThermoFisher"))
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        beam = microscope.beams[beam_type]
        assert list(beam.voltage.metadata.choices) == list(
            THERMOFISHER_PROFILE.voltages[beam_type]
        )
        assert list(beam.detector_type.metadata.choices) == list(
            THERMOFISHER_PROFILE.detector_types
        )


@pytest.mark.parametrize("driver, another", [("Demo", False), ("ThermoFisher", True)])
def test_on_a_simulated_thermofisher_the_demo_is_the_own_driver(
    monkeypatch, driver, another
):
    microscope = registry.connect_microscope(_system("ThermoFisher"))
    entry = DeviceEntry(name="manipulator", type="manipulator", driver=driver)
    monkeypatch.setattr(
        entries, "configured_device_entries", lambda _: {"manipulator": entry}
    )
    assert microscope._built_by_another_driver("manipulator") is another
