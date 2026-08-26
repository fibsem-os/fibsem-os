"""Milling time estimation: the preset-driven dose model vs the legacy table.

On preset-driven backends (TESCAN) the beam conditions come from the selected
preset, so the legacy estimate — a silicon sputter-rate table keyed on the
unused ``milling_current`` field — is wrong twice over (wrong current scale,
wrong material). The dose model ``t = volume / (rate × current)`` uses the same
inputs DrawBeam computes the real exposure from: the stage's own etch rate and
the current parsed from the preset name.

The model is registered process-wide by the TESCAN driver
(``set_preset_driven_estimation``): the planning stack estimates through the
``FibsemMillingStage.estimated_time`` property with no microscope in scope, so
it cannot be a call-site parameter. The hard requirement locked in here is that
nothing changes for any other backend: unregistered — the default — must be
byte-identical to the legacy behaviour, and so must every preset the model
cannot read.
"""

import threading

import pytest

from fibsem.milling.base import (
    FibsemMillingStage,
    estimate_milling_time,
    estimate_stage_milling_time,
    estimate_total_milling_time,
    parse_current_from_preset,
    set_preset_driven_estimation,
)
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.structures import CrossSectionPattern, FibsemMillingSettings

RATE = 1.3e-8  # m3/A/s (the cryo-lamella default)
PRESET_100PA = "30 keV; 100 pA"

# 10 x 1 x 1 um rectangle = 1e-17 m3; t = 1e-17 / (1.3e-8 * 100e-12) = 7.6923 s
DOSE_MODEL_SECONDS = pytest.approx(7.6923, abs=1e-3)


@pytest.fixture(autouse=True)
def _legacy_estimation_by_default():
    """Reset the process-wide registration around every test (it is global state)."""
    set_preset_driven_estimation(False)
    yield
    set_preset_driven_estimation(False)


def make_stage(
    preset: str = PRESET_100PA,
    rate: float = RATE,
    milling_current: float = 2.0e-9,
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle,
    pattern_time: float = 0,
) -> FibsemMillingStage:
    pattern = RectanglePattern(width=10e-6, height=1e-6, depth=1e-6)
    pattern.cross_section = cross_section
    pattern.time = pattern_time
    return FibsemMillingStage(
        milling=FibsemMillingSettings(
            preset=preset, rate=rate, milling_current=milling_current
        ),
        pattern=pattern,
    )


# ---------------------------------------------------------------------------
# preset-name parsing (names are free-form on the instrument)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset, expected",
    [
        ("30 keV; 100 pA", 100e-12),
        ("30 keV; 2nA", 2e-9),  # the old field default; vendor style without a space
        ("30 keV; 1 nA; my cool preset", 1e-9),
        ("15 keV; 1.5 nA", 1.5e-9),
        ("20 keV; 500 uA", 500e-6),
        ("20 keV; 500 µA", 500e-6),
    ],
)
def test_parse_current_from_preset(preset, expected):
    assert parse_current_from_preset(preset) == pytest.approx(expected)


@pytest.mark.parametrize(
    "preset",
    [
        None,
        "",
        "my cool preset",  # no current token at all
        "30 keV",  # voltage only
        "slot 2A",  # bare "A" is noise, not a beam current
        "2 mA range",  # unprefixed/mA deliberately rejected
        "nAmeless",  # unit letters inside a word
    ],
)
def test_parse_current_rejects_non_current_names(preset):
    assert parse_current_from_preset(preset) is None


def test_parse_current_takes_the_first_token():
    assert parse_current_from_preset("100 pA (was 1 nA)") == pytest.approx(100e-12)


# ---------------------------------------------------------------------------
# the hard requirement: nothing changes unless the TESCAN model is registered
# ---------------------------------------------------------------------------


def test_unregistered_is_identical_to_the_legacy_estimate():
    stage = make_stage()
    assert estimate_stage_milling_time(stage) == estimate_milling_time(
        stage.pattern, stage.milling.milling_current
    )
    assert stage.estimated_time == estimate_milling_time(
        stage.pattern, stage.milling.milling_current
    )
    assert estimate_total_milling_time([stage, stage]) == pytest.approx(
        2 * estimate_milling_time(stage.pattern, stage.milling.milling_current)
    )


def test_unreadable_preset_falls_back_to_legacy_even_when_registered():
    set_preset_driven_estimation(True)
    stage = make_stage(preset="my cool preset")
    assert estimate_stage_milling_time(stage) == estimate_milling_time(
        stage.pattern, stage.milling.milling_current
    )


def test_unusable_rate_falls_back_to_legacy_even_when_registered():
    set_preset_driven_estimation(True)
    stage = make_stage(rate=0.0)
    assert estimate_stage_milling_time(stage) == estimate_milling_time(
        stage.pattern, stage.milling.milling_current
    )


# ---------------------------------------------------------------------------
# the dose model
# ---------------------------------------------------------------------------


def test_registered_uses_the_dose_model():
    set_preset_driven_estimation(True)
    stage = make_stage()
    assert estimate_stage_milling_time(stage) == DOSE_MODEL_SECONDS
    assert stage.estimated_time == DOSE_MODEL_SECONDS
    assert estimate_total_milling_time([stage]) == DOSE_MODEL_SECONDS


def test_dose_model_keeps_the_cleaning_cross_section_factor():
    set_preset_driven_estimation(True)
    stage = make_stage(cross_section=CrossSectionPattern.CleaningCrossSection)
    assert estimate_stage_milling_time(stage) == pytest.approx(0.66 * 7.6923, abs=1e-3)


def test_explicit_pattern_time_wins_in_both_models():
    stage = make_stage(pattern_time=42.0)
    assert estimate_stage_milling_time(stage) == 42.0
    set_preset_driven_estimation(True)
    assert estimate_stage_milling_time(stage) == 42.0


def test_dose_model_ignores_the_dead_milling_current_field():
    set_preset_driven_estimation(True)
    a = make_stage(milling_current=20e-12)
    b = make_stage(milling_current=120e-9)
    assert estimate_stage_milling_time(a) == estimate_stage_milling_time(b)


# ---------------------------------------------------------------------------
# driver registration
# ---------------------------------------------------------------------------


def test_tescan_construction_registers_and_disconnect_unregisters(monkeypatch):
    """TescanMicroscope.__init__ registers the model; disconnect() hands the
    legacy model back (so a later non-TESCAN session estimates as before)."""
    import os

    import fibsem.config as cfg
    from fibsem import utils
    from fibsem.microscopes import tescan as tescan_module
    from fibsem.microscopes.tescan import TescanMicroscope

    stage = make_stage()
    legacy = estimate_milling_time(stage.pattern, stage.milling.milling_current)

    config_path = os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    system = utils.load_microscope_configuration(config_path).system

    # __init__ only guards on the SDK's availability, it does not use it
    monkeypatch.setattr(tescan_module, "TESCAN_API_AVAILABLE", True)
    microscope = TescanMicroscope(system_settings=system)
    assert estimate_stage_milling_time(stage) == DOSE_MODEL_SECONDS

    class FakeConnection:
        def Disconnect(self):
            pass

    microscope.connection = FakeConnection()
    microscope._connection_lock = threading.RLock()
    microscope.disconnect()
    assert estimate_stage_milling_time(stage) == legacy
