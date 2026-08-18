"""Per-task forward duration estimates (FIB-666).

Each task config reports its own expected wall-clock, so a new task type ships its
estimate with itself instead of being added to a central table that falls behind.
These tests pin what each override adds, not the constants underneath.
"""

import pytest

from fibsem import timing
from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (
    AcquireFluorescenceImageConfig,
)
from fibsem.applications.autolamella.workflows.tasks.fiducial import MillFiducialTaskConfig
from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
    SpotBurnFiducialTaskConfig,
)
from fibsem.fm.structures import ChannelSettings
from fibsem.fm.timing import estimate_acquisition_time
from fibsem.structures import Point


# ── the base estimate ────────────────────────────────────────────────────────

def test_base_estimate_exceeds_the_scan_only_figure():
    """estimated_time counts scan arithmetic and milling; estimated_duration adds the
    measured per-operation costs the task actually pays."""
    cfg = MillFiducialTaskConfig()
    assert cfg.estimated_duration > cfg.estimated_time


def test_base_estimate_counts_reference_imaging_at_both_ends():
    cfg = MillFiducialTaskConfig()
    cfg.milling = {}
    assert cfg.estimated_duration == pytest.approx(
        2 * timing.reference_image_cost(cfg.reference_imaging)
    )


# ── spot burn: one exposure per coordinate ───────────────────────────────────

def test_spot_burn_adds_an_exposure_per_coordinate():
    cfg = SpotBurnFiducialTaskConfig(exposure_time=10)
    baseline = cfg.estimated_duration

    cfg.coordinates = [Point(x=0.1, y=0.1), Point(x=0.2, y=0.2), Point(x=0.3, y=0.3)]

    assert cfg.estimated_duration == pytest.approx(baseline + 3 * 10)


def test_spot_burn_with_no_points_costs_only_the_base():
    with_points = SpotBurnFiducialTaskConfig(
        exposure_time=10, coordinates=[Point(x=0.1, y=0.1)]
    )
    without = SpotBurnFiducialTaskConfig(exposure_time=10)
    assert without.estimated_duration < with_points.estimated_duration


def test_spot_burn_scales_with_exposure_time():
    points = [Point(x=0.1, y=0.1), Point(x=0.2, y=0.2)]
    short = SpotBurnFiducialTaskConfig(exposure_time=5, coordinates=list(points))
    long = SpotBurnFiducialTaskConfig(exposure_time=10, coordinates=list(points))
    assert long.estimated_duration - short.estimated_duration == pytest.approx(2 * 5)


# ── fluorescence: delegate to the FM timing model ────────────────────────────

def _channel(exposure: float) -> ChannelSettings:
    return ChannelSettings(
        name="DAPI",
        excitation_wavelength=365,
        emission_wavelength=450,
        power=0.5,
        exposure_time=exposure,
    )


def test_fluorescence_adds_the_channel_acquisition():
    cfg = AcquireFluorescenceImageConfig()
    baseline = cfg.estimated_duration

    cfg.channel_settings = [_channel(0.5), _channel(0.2)]

    assert cfg.estimated_duration == pytest.approx(
        baseline + estimate_acquisition_time(cfg.channel_settings, cfg.zparams)
    )


def test_fluorescence_scales_with_channel_count():
    one = AcquireFluorescenceImageConfig(channel_settings=[_channel(0.5)])
    two = AcquireFluorescenceImageConfig(channel_settings=[_channel(0.5), _channel(0.5)])
    assert two.estimated_duration > one.estimated_duration


# ── every task answers ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "config_cls",
    [MillFiducialTaskConfig, SpotBurnFiducialTaskConfig, AcquireFluorescenceImageConfig],
)
def test_every_config_reports_a_positive_duration(config_cls):
    """A task with no override still answers, via the base implementation."""
    assert config_cls().estimated_duration > 0
