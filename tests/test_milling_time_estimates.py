"""Milling time estimation: the preset-driven dose model vs the legacy table.

On preset-driven backends (TESCAN) the beam conditions come from the selected
preset, so the legacy estimate — a silicon sputter-rate table keyed on the
unused ``milling_current`` field — is wrong twice over (wrong current scale,
wrong material). The dose model ``t = volume / (rate × current)`` uses the same
inputs DrawBeam computes the real exposure from: the stage's own etch rate and
the current parsed from the preset name.

The TESCAN driver supplies the model (``TescanMicroscope.estimate_stage_milling_time``)
and ``utils.setup_session`` installs it for the session on connect
(``set_milling_time_estimator``): the planning stack estimates through the
``FibsemMillingStage.estimated_time`` property with no microscope in scope, so
it cannot be a call-site parameter. The hard requirement locked in here is that
nothing changes for any other backend: no estimator, or the base class default,
must be byte-identical to the legacy behaviour, and so must every preset the
model cannot read.
"""

import threading

import pytest

from fibsem.drivers.tescan.microscope import TescanMicroscope, parse_current_from_preset
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import (
    FibsemMillingStage,
    estimate_milling_time,
    estimate_stage_milling_time,
    estimate_total_milling_time,
    set_milling_time_estimator,
    using_milling_time_estimator,
)
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.structures import CrossSectionPattern, FibsemMillingSettings

RATE = 1.3e-8  # m3/A/s (the cryo-lamella default)
PRESET_100PA = "30 keV; 100 pA"

# 10 x 1 x 1 um rectangle = 1e-17 m3; t = 1e-17 / (1.3e-8 * 100e-12) = 7.6923 s
DOSE_MODEL_SECONDS = pytest.approx(7.6923, abs=1e-3)


@pytest.fixture(autouse=True)
def _legacy_estimation_by_default():
    """Reset the session-level estimator around every test (it is global state)."""
    with using_milling_time_estimator(None):
        yield


def use_tescan_model() -> None:
    set_milling_time_estimator(TescanMicroscope.estimate_stage_milling_time)


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
        ("30 keV; 2nA", 2e-9),  # the FibsemMillingSettings default, no space
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
# the hard requirement: nothing changes unless the TESCAN model is installed
# ---------------------------------------------------------------------------


def test_no_estimator_is_identical_to_the_legacy_estimate():
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


def test_unreadable_preset_falls_back_to_legacy_with_the_tescan_model():
    use_tescan_model()
    stage = make_stage(preset="my cool preset")
    assert estimate_stage_milling_time(stage) == estimate_milling_time(
        stage.pattern, stage.milling.milling_current
    )


def test_unusable_rate_falls_back_to_legacy_with_the_tescan_model():
    use_tescan_model()
    stage = make_stage(rate=0.0)
    assert estimate_stage_milling_time(stage) == estimate_milling_time(
        stage.pattern, stage.milling.milling_current
    )


# ---------------------------------------------------------------------------
# the dose model
# ---------------------------------------------------------------------------


def test_tescan_model_uses_the_dose_model():
    use_tescan_model()
    stage = make_stage()
    assert estimate_stage_milling_time(stage) == DOSE_MODEL_SECONDS
    assert stage.estimated_time == DOSE_MODEL_SECONDS
    assert estimate_total_milling_time([stage]) == DOSE_MODEL_SECONDS


def test_dose_model_keeps_the_cleaning_cross_section_factor():
    use_tescan_model()
    stage = make_stage(cross_section=CrossSectionPattern.CleaningCrossSection)
    assert estimate_stage_milling_time(stage) == pytest.approx(0.66 * 7.6923, abs=1e-3)


def test_explicit_pattern_time_wins_in_both_models():
    stage = make_stage(pattern_time=42.0)
    assert estimate_stage_milling_time(stage) == 42.0
    use_tescan_model()
    assert estimate_stage_milling_time(stage) == 42.0


def test_dose_model_ignores_the_dead_milling_current_field():
    use_tescan_model()
    a = make_stage(milling_current=20e-12)
    b = make_stage(milling_current=120e-9)
    assert estimate_stage_milling_time(a) == estimate_stage_milling_time(b)


# ---------------------------------------------------------------------------
# the driver supplies the model; connecting installs it
# ---------------------------------------------------------------------------


def test_base_class_default_is_identical_to_the_legacy_estimate():
    stage = make_stage()
    set_milling_time_estimator(FibsemMicroscope.estimate_stage_milling_time)
    assert estimate_stage_milling_time(stage) == estimate_milling_time(
        stage.pattern, stage.milling.milling_current
    )


def test_using_restores_the_previous_estimator():
    stage = make_stage()
    use_tescan_model()
    with using_milling_time_estimator(lambda _: 1.0):
        assert estimate_stage_milling_time(stage) == 1.0
    assert estimate_stage_milling_time(stage) == DOSE_MODEL_SECONDS


def test_connecting_installs_the_drivers_model():
    """setup_session replaces whatever was installed with the connected driver's
    model: the Demo backend's default, so the table, after a TESCAN session."""
    from fibsem import utils

    stage = make_stage()
    use_tescan_model()
    microscope, _ = utils.setup_session(manufacturer="Demo")
    assert estimate_stage_milling_time(stage) == estimate_milling_time(
        stage.pattern, stage.milling.milling_current
    )
    microscope.disconnect()


def _make_tescan(monkeypatch) -> TescanMicroscope:
    import os

    import fibsem.config as cfg
    from fibsem import utils
    from fibsem.drivers.tescan import microscope as tescan_module

    config_path = os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    system = utils.load_microscope_configuration(config_path).system
    # __init__ only guards on the SDK's availability, it does not use it
    monkeypatch.setattr(tescan_module, "TESCAN_API_AVAILABLE", True)
    return TescanMicroscope(system_settings=system)


def test_tescan_construction_does_not_switch_the_model(monkeypatch):
    stage = make_stage()
    _make_tescan(monkeypatch)
    assert estimate_stage_milling_time(stage) == estimate_milling_time(
        stage.pattern, stage.milling.milling_current
    )


def test_tescan_disconnect_keeps_the_model(monkeypatch):
    """ETAs quoted after a disconnect still describe the instrument this session
    plans for; only connecting another backend replaces the model."""
    stage = make_stage()
    microscope = _make_tescan(monkeypatch)
    use_tescan_model()

    class FakeConnection:
        def Disconnect(self):
            pass

    microscope.connection = FakeConnection()
    microscope._connection_lock = threading.RLock()
    microscope.disconnect()
    assert estimate_stage_milling_time(stage) == DOSE_MODEL_SECONDS
