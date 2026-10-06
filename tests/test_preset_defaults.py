"""A column set by preset (the Tescan ion column) has a preset as its default.

The Tescan API refuses to set the ion column's voltage or current directly, so a
default voltage and current there did nothing. The preset is what sets it (FIB-1084).
"""

import threading

import pytest

from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import BeamType


def test_only_the_tescan_ion_column_is_set_by_preset():
    tescan = object.__new__(TescanMicroscope)
    tescan._connection_lock = threading.RLock()

    assert tescan.beam_uses_presets(BeamType.ION)
    assert not tescan.beam_uses_presets(BeamType.ELECTRON)


def test_other_backends_are_not():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    try:
        assert not microscope.beam_uses_presets(BeamType.ION)
        assert not microscope.beam_uses_presets(BeamType.ELECTRON)
    finally:
        microscope.disconnect()


@pytest.fixture
def preset_ion(monkeypatch):
    """The Demo simulator with its ion column declared preset-driven, reporting an
    active preset the way the Tescan backend does."""
    active = {"preset": "30 keV; 150 pA"}
    monkeypatch.setattr(
        FibsemMicroscope,
        "beam_uses_presets",
        lambda self, beam_type: beam_type is BeamType.ION,
    )
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    original_get = microscope.get_preset

    def get_preset(beam_type):
        if beam_type is BeamType.ION:
            return active["preset"]
        return original_get(beam_type)

    def set_preset(preset, beam_type):
        active["preset"] = preset
        return preset

    microscope.get_preset, microscope.set_preset = get_preset, set_preset
    microscope._active = active
    yield microscope
    microscope.disconnect()


def test_capturing_takes_the_active_preset(preset_ion):
    preset_ion.system.ion.beam.preset = None

    preset_ion.capture_defaults(BeamType.ION)

    assert preset_ion.system.ion.beam.preset == "30 keV; 150 pA"


def test_capturing_keeps_the_preset_when_none_is_known(preset_ion):
    """The Tescan backend knows the active preset only once one has been set or an
    image taken; before that, capturing must not erase the configured one."""
    preset_ion.system.ion.beam.preset = "30 keV; 1 nA"
    preset_ion._active["preset"] = None

    preset_ion.capture_defaults(BeamType.ION)

    assert preset_ion.system.ion.beam.preset == "30 keV; 1 nA"


def test_a_column_not_set_by_preset_captures_none(preset_ion):
    preset_ion.system.electron.beam.preset = None

    preset_ion.capture_defaults(BeamType.ELECTRON)

    assert preset_ion.system.electron.beam.preset is None


def test_applying_the_defaults_activates_the_preset(preset_ion):
    preset_ion.system.ion.beam.preset = "30 keV; 1 nA"

    preset_ion.apply_defaults()

    assert preset_ion._active["preset"] == "30 keV; 1 nA"
