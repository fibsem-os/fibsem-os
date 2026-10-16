"""The milling service on the Demo, and the old milling methods over it.

The Demo mills with the shared demo code through `microscope.milling`. That
`finish_milling` puts the beam back as `setup_milling` found it is in the contract
suite (`test_finish_milling_puts_the_beam_back`), for both demos.
"""

import dataclasses

import pytest

from fibsem import config as cfg
from fibsem import utils
from fibsem.devices.core import Device
from fibsem.drivers.demo.services import DemoMilling
from fibsem.services import Service
from fibsem.services.milling import DRAW_HOOKS, Milling, UnsupportedPattern
from fibsem.structures import (
    BeamType,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    MillingState,
)


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        config_path=cfg.DEFAULT_CONFIGURATION_PATH,
        manufacturer="Demo",
        setup_logging=False,
    )
    return microscope


def _rectangle() -> FibsemRectangleSettings:
    return FibsemRectangleSettings(
        width=10e-6, height=5e-6, depth=1e-6, centre_x=0, centre_y=0
    )


def _patterns():
    return [
        _rectangle(),
        FibsemLineSettings(start_x=0, start_y=0, end_x=1e-6, end_y=0, depth=1e-6),
        FibsemCircleSettings(centre_x=0, centre_y=0, radius=1e-6, depth=1e-6),
    ]


def _recipe(**kwargs) -> FibsemMillingSettings:
    values = dict(milling_current=7.6e-9, milling_voltage=30e3, hfw=80e-6)
    values.update(kwargs)
    return FibsemMillingSettings(**values)


def _conditions(microscope, beam_type=BeamType.ION):
    return tuple(
        microscope._read_beam(key, beam_type) for key in ("voltage", "current", "hfw")
    )


def test_the_demo_has_a_milling_service_that_is_not_a_device(microscope):
    milling = microscope.milling
    assert isinstance(milling, DemoMilling)
    assert isinstance(milling, Service) and not isinstance(milling, Device)
    assert milling not in microscope.beams.values()
    assert milling.ion is microscope.beams[BeamType.ION]
    assert milling.electron is microscope.beams[BeamType.ELECTRON]
    assert milling.resources is microscope.resources
    assert sorted(milling.parameters) == ["progress", "state"]
    assert {
        "setup",
        "draw",
        "prepare",
        "start",
        "stop",
        "pause",
        "resume",
        "clear",
        "restore",
        "run",
    } <= set(milling.commands)
    assert "estimate_time" in milling.queries


def test_prepare_applies_the_recipe_and_draws(microscope):
    milling = microscope.milling
    prepared = milling.prepare(_recipe(patterning_mode="Parallel"), _patterns())
    assert _conditions(microscope) == (30e3, 7.6e-9, 80e-6)
    assert microscope._connection.milling_system.patterning_mode == "Parallel"
    assert len(microscope._connection.milling_system.patterns) == 3
    assert prepared["milling_current"] == 7.6e-9
    assert milling.estimate_time() == {"seconds": 15, **prepared}
    assert microscope.estimate_milling_time() == 15


def test_a_current_the_beam_does_not_list_is_warned_of(microscope):
    listed = sorted(microscope.beams[BeamType.ION].current.choices)
    asked = (listed[0] + listed[1]) / 2
    prepared = microscope.milling.prepare(_recipe(milling_current=asked), _patterns())
    warning = prepared["warning"]
    assert "is not a current the milling beam lists" in warning
    assert utils.format_current(listed[0]) in warning
    assert utils.format_current(listed[1]) in warning
    assert microscope.milling.estimate_time()["warning"] == warning


def test_a_driver_that_lists_no_currents_has_no_warning(microscope, monkeypatch):
    from fibsem.devices.core import ParameterMetadata

    current = microscope.beams[BeamType.ION].current
    monkeypatch.setattr(current, "_metadata_source", ParameterMetadata())
    prepared = microscope.milling.prepare(_recipe(milling_current=1.234e-9), [])
    assert prepared["warning"] is None


def test_a_setup_clears_what_was_drawn(microscope):
    microscope.milling.prepare(_recipe(), _patterns())
    microscope.setup_milling(_recipe())
    assert microscope._connection.milling_system.patterns == []


def test_the_old_methods_go_through_the_service(microscope):
    microscope.setup_milling(_recipe())
    microscope.draw_patterns(_patterns())
    assert len(microscope._connection.milling_system.patterns) == 3

    seen = []
    microscope.milling.state.get_value()
    microscope.milling.changed.connect(lambda name, value: seen.append(value))
    microscope.start_milling()
    assert microscope.get_milling_state() is MillingState.RUNNING
    microscope.pause_milling()
    assert microscope.milling.state.get_value() is MillingState.PAUSED
    microscope.resume_milling()
    microscope.get_milling_state()
    microscope.stop_milling()
    assert microscope.get_milling_state() is MillingState.IDLE
    assert seen == [
        MillingState.RUNNING,
        MillingState.PAUSED,
        MillingState.RUNNING,
        MillingState.IDLE,
    ]
    microscope.clear_patterns()
    assert microscope._connection.milling_system.patterns == []


def test_run_milling_mills_what_the_service_drew(microscope):
    microscope.setup_milling(_recipe())
    microscope.draw_patterns([_rectangle()])
    microscope.run_milling()
    assert microscope.get_milling_state() is MillingState.IDLE
    assert microscope._connection.milling_system.patterns == []


def test_restore_goes_through_the_beam_device(microscope):
    beam = microscope.beams[BeamType.ION]
    microscope.setup_milling(_recipe())
    seen = []
    beam.changed.connect(lambda name, value: seen.append(name))
    microscope.milling.restore()
    assert {"current", "hfw"} <= set(seen)


def test_the_service_needs_its_hooks():
    for hook in ("_setup", "_start", "_estimate", "_clear"):
        assert getattr(DemoMilling, hook) is not getattr(Milling, hook), hook


def test_the_demo_draws_every_pattern_type():
    assert set(DemoMilling.supported_pattern_types()) == set(DRAW_HOOKS)


class _Unknown(FibsemPatternSettings):
    """A pattern type no driver draws."""

    volume = 0.0


def test_a_pattern_type_the_driver_does_not_draw_raises(microscope):
    microscope.setup_milling(_recipe())
    with pytest.raises(UnsupportedPattern, match="cannot draw _Unknown patterns"):
        microscope.draw_pattern(_Unknown())
    assert microscope._connection.milling_system.patterns == []


def test_nothing_is_drawn_when_one_pattern_cannot_be(microscope):
    """Every pattern is checked first, so a stage is never half drawn."""
    microscope.setup_milling(_recipe())
    with pytest.raises(UnsupportedPattern):
        microscope.draw_patterns(_patterns()[:2] + [_Unknown()])
    assert microscope._connection.milling_system.patterns == []


def test_a_subclass_of_a_drawn_type_is_drawn_as_it(microscope):
    class Notched(FibsemRectangleSettings):
        pass

    microscope.setup_milling(_recipe())
    microscope.draw_pattern(
        Notched(width=1e-6, height=1e-6, depth=1e-6, centre_x=0, centre_y=0)
    )
    assert len(microscope._connection.milling_system.patterns) == 1


def test_without_an_ion_beam_the_demo_does_not_mill(microscope):
    """A configuration with the ion column off builds no ion beam, so no milling
    service: there is nothing to mill with, so milling raises, stopping does nothing
    and the state reads idle."""
    from copy import deepcopy

    from fibsem.drivers.demo.microscope import DemoMicroscope

    system = deepcopy(microscope.system)
    system.ion.enabled = False
    electron_only = DemoMicroscope(system)
    assert BeamType.ION not in electron_only.beams
    assert electron_only.milling is None
    with pytest.raises(ValueError, match="ION beam is not enabled"):
        electron_only.draw_patterns([_rectangle()])
    assert electron_only._connection.milling_system.patterns == []
    with pytest.raises(ValueError, match="ION beam is not enabled"):
        electron_only.clear_patterns()
    electron_only.stop_milling()
    assert electron_only.get_milling_state() is MillingState.IDLE


def test_the_demo_says_which_settings_it_mills_with(microscope):
    supported = microscope.milling.supported_settings()
    assert set(supported) == {
        "milling_channel",
        "hfw",
        "milling_current",
        "milling_voltage",
        "application_file",
        "patterning_mode",
    }
    ion = microscope.beams[BeamType.ION]
    # a beam's own parameter carries the beam's metadata, not a copy
    assert supported["milling_current"] is ion.parameters["current"].metadata
    assert list(supported["application_file"].choices) == list(
        microscope._connection.milling_system.application_files
    )
    assert supported["patterning_mode"].choices == ("Serial", "Parallel")
    assert supported["milling_channel"].choices == (BeamType.ION, BeamType.ELECTRON)


def test_the_demo_says_which_directions_it_scans(microscope):
    from fibsem.drivers.demo.simulator import SIMULATOR_SCAN_DIRECTIONS

    supported = microscope.milling.supported_pattern_settings()
    assert set(supported) == {"scan_direction"}
    assert list(supported["scan_direction"].choices) == SIMULATOR_SCAN_DIRECTIONS


def test_the_demo_mills_with_the_settings_it_says(microscope):
    from tests.fixtures.milling_reads import fields_setup_reads

    read = fields_setup_reads(microscope.milling, _recipe())
    assert read == set(microscope.milling.supported_settings())


def test_a_channel_with_no_beam_has_no_settings(microscope):
    del microscope.milling._roles["electron"]
    assert microscope.milling.supported_settings(BeamType.ELECTRON) == {}
    assert microscope.milling.supported_settings()["milling_channel"].choices == (
        BeamType.ION,
    )


@pytest.mark.parametrize(
    "version, polygons", [("4.11", False), ("4.12", True)], ids=["4.11", "4.12"]
)
def test_autoscript_draws_polygons_from_4_12(monkeypatch, version, polygons):
    from packaging.version import parse as parse_version

    from fibsem.drivers.autoscript import microscope as autoscript
    from fibsem.drivers.autoscript.services import AutoScriptMilling

    monkeypatch.setattr(autoscript, "AUTOSCRIPT_VERSION", parse_version(version))
    types = AutoScriptMilling.supported_pattern_types()
    assert (FibsemPolygonSettings in types) is polygons
    assert set(types) | {FibsemPolygonSettings} == set(DRAW_HOOKS)


def test_the_drawn_patterns_are_announced_until_they_are_removed(microscope):
    # FIB-1375: the app draws an agent's mill from these; it has no config of it.
    milling = microscope.milling
    seen = []
    milling.patterns_changed.connect(lambda patterns: seen.append(len(patterns)))
    milling.prepare(_recipe(), _patterns()[:2])
    milling.draw([_rectangle()])
    assert len(milling.drawn_patterns) == 3
    milling.run()
    assert milling.drawn_patterns == []
    milling.prepare(_recipe(), [_rectangle()])
    microscope.clear_patterns()
    assert seen == [2, 3, 0, 1, 0]


# -- the recipe as each driver reads it (FIB-1437) ------------------------------------


def test_a_protocol_saved_with_tescan_fields_at_the_top_loads_them_under_tescan():
    recipe = FibsemMillingSettings.from_dict(
        {"milling_current": 2e-9, "preset": "30 keV; 1 nA", "rate": 0.3}
    )
    assert recipe.milling_current == 2e-9
    assert recipe.driver_settings == {"tescan": {"preset": "30 keV; 1 nA", "rate": 0.3}}
    saved = recipe.to_dict()
    assert saved["tescan.preset"] == "30 keV; 1 nA" and "preset" not in saved
    assert FibsemMillingSettings.from_dict(saved) == recipe


def test_a_driver_reads_its_keys_and_keeps_the_others(microscope):
    milling = microscope.milling
    recipe = FibsemMillingSettings(
        milling_current=2e-9, driver_settings={"tescan.rate": 0.3}
    )
    driver = milling.driver_settings(recipe)
    assert driver.milling_current == 2e-9
    assert not hasattr(driver, "rate")
    edited = milling.with_driver_settings(
        recipe, dataclasses.replace(driver, milling_current=4e-9)
    )
    assert edited.milling_current == 4e-9
    assert edited.driver_settings == {"tescan": {"rate": 0.3}}


def test_the_agent_sees_the_recipe_this_instrument_mills_with(microscope):
    from fibsem.schema import arguments_schema

    schema = arguments_schema(
        type(microscope.milling).setup, microscope.milling.schema_defs()
    )
    recipe = schema["$defs"]["FibsemMillingSettings"]["properties"]
    assert "milling_current" in recipe and "driver_settings" not in recipe
    assert not any(key.startswith("tescan.") for key in recipe)
    beam = microscope.milling.ion.current
    assert recipe["milling_current"]["enum"] == list(beam.metadata.choices)
    # a command that takes no recipe has no recipe schema
    assert "$defs" not in arguments_schema(
        type(microscope.milling).estimate_time, microscope.milling.schema_defs()
    )


def test_a_driver_with_settings_of_its_own_needs_a_prefix():
    from fibsem.services.milling import DriverMillingSettings

    @dataclasses.dataclass
    class Own(DriverMillingSettings):
        aperture: int = 1

    with pytest.raises(TypeError, match="settings_prefix"):

        class Unprefixed(DemoMilling):
            settings_class = Own

    class Prefixed(DemoMilling):
        settings_class = Own
        settings_prefix = "probe"

    assert Prefixed.setting_key("aperture") == "probe.aperture"
    assert Prefixed.setting_key("hfw") == "hfw"


def test_only_a_shared_field_is_taken_shared():
    from fibsem.structures import shared_milling_setting

    with pytest.raises(ValueError, match="not a shared milling setting"):
        shared_milling_setting("rate")
