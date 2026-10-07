"""The milling service on the Demo, and the old milling methods over it.

The Demo mills with the shared demo code through `microscope.milling`. That
`finish_milling` puts the beam back as `setup_milling` found it is in the contract
suite (`test_finish_milling_puts_the_beam_back`), for both demos.
"""

import numpy as np
import pytest

from fibsem import config as cfg
from fibsem import utils
from fibsem.devices.core import Device
from fibsem.services import Service
from fibsem.services.drivers.demo import DemoMilling
from fibsem.services.milling import Milling
from fibsem.structures import (
    BeamType,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
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
        microscope.get(key, beam_type) for key in ("voltage", "current", "hfw")
    )


def test_the_demo_has_a_milling_service_that_is_not_a_device(microscope):
    milling = microscope.milling
    assert isinstance(milling, DemoMilling)
    assert isinstance(milling, Service) and not isinstance(milling, Device)
    assert milling not in microscope.beams.values()
    assert milling.ion is microscope.beams[BeamType.ION]
    assert milling.electron is microscope.beams[BeamType.ELECTRON]
    assert milling.resources is microscope.resources
    assert sorted(milling.parameters) == ["state"]
    assert {
        "setup",
        "draw",
        "prepare",
        "start",
        "stop",
        "pause",
        "resume",
        "estimate",
        "clear",
        "restore",
    } <= set(milling.commands)


def test_prepare_applies_the_recipe_and_draws(microscope):
    milling = microscope.milling
    milling.prepare(_recipe(patterning_mode="Parallel"), _patterns())
    assert _conditions(microscope) == (30e3, 7.6e-9, 80e-6)
    assert microscope.milling_system.patterning_mode == "Parallel"
    assert len(microscope.milling_system.patterns) == 3
    assert milling.estimate() == microscope.estimate_milling_time() == 15


def test_a_setup_clears_what_was_drawn(microscope):
    microscope.milling.prepare(_recipe(), _patterns())
    microscope.setup_milling(_recipe())
    assert microscope.milling_system.patterns == []


def test_the_old_methods_go_through_the_service(microscope):
    microscope.setup_milling(_recipe())
    microscope.draw_patterns(_patterns())
    assert len(microscope.milling_system.patterns) == 3

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
    assert microscope.milling_system.patterns == []


def test_run_milling_mills_what_the_service_drew(microscope):
    microscope.setup_milling(_recipe())
    microscope.draw_patterns([_rectangle()])
    microscope.run_milling(milling_current=7.6e-9, milling_voltage=30e3)
    assert microscope.get_milling_state() is MillingState.IDLE
    assert microscope.milling_system.patterns == []


def test_restore_goes_through_the_beam_device(microscope):
    beam = microscope.beams[BeamType.ION]
    microscope.setup_milling(_recipe())
    seen = []
    beam.changed.connect(lambda name, value: seen.append(name))
    microscope.milling.restore()
    assert {"current", "hfw"} <= set(seen)


def test_the_service_needs_its_hooks():
    for hook in ("_setup", "_draw", "_start", "_estimate", "_clear"):
        assert getattr(DemoMilling, hook) is not getattr(Milling, hook), hook


def test_without_an_ion_beam_the_demo_mills_with_its_own_code(microscope):
    """A configuration with the ion column off builds no ion beam, so no milling
    service; the milling methods fall through to the shared demo code."""
    from copy import deepcopy

    from fibsem.microscopes.device_demo import DemoMicroscope

    system = deepcopy(microscope.system)
    system.ion.enabled = False
    electron_only = DemoMicroscope(system)
    assert BeamType.ION not in electron_only.beams
    assert electron_only.milling is None
    electron_only.draw_patterns([_rectangle()])
    assert len(electron_only.milling_system.patterns) == 1
    assert electron_only.get_milling_state() is MillingState.IDLE
    electron_only.clear_patterns()
    assert electron_only.milling_system.patterns == []


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
        microscope.milling_system.application_files
    )
    assert supported["patterning_mode"].choices == ("Serial", "Parallel")
    assert supported["milling_channel"].choices == (BeamType.ION, BeamType.ELECTRON)


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
