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
from fibsem.fm.structures import AutoFocusSettings, ChannelSettings, FocusSweepPass
from fibsem.fm.timing import estimate_acquisition_time, estimate_autofocus_time
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
    """Each point costs its exposure plus a blank/park/unblank cycle."""
    cfg = SpotBurnFiducialTaskConfig(exposure_time=10)
    baseline = cfg.estimated_duration

    cfg.coordinates = [Point(x=0.1, y=0.1), Point(x=0.2, y=0.2), Point(x=0.3, y=0.3)]

    assert cfg.estimated_duration == pytest.approx(
        baseline + 3 * (10 + timing.SPOT_BURN_POINT_OVERHEAD_S)
    )


def test_spot_burn_counts_the_move_alignment_and_current_change():
    """The burn alone left the estimate at 0.71x of machine time; the rest is here."""
    cfg = SpotBurnFiducialTaskConfig(exposure_time=10)
    expected = (
        2 * timing.reference_image_cost(cfg.reference_imaging)
        + timing.stage_move_cost(1)
        + timing.REFERENCE_ALIGNMENT_S
        + 2 * timing.BEAM_CURRENT_CHANGE_S
    )
    assert cfg.estimated_duration == pytest.approx(expected)


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
    """Adding channels adds both the acquisition and the autofocus sweep -- with no
    channel there is nothing to focus on, so the sweep costs nothing."""
    cfg = AcquireFluorescenceImageConfig()
    baseline = cfg.estimated_duration
    assert estimate_autofocus_time(cfg.autofocus_settings, []) == 0.0

    cfg.channel_settings = [_channel(0.5), _channel(0.2)]

    assert cfg.estimated_duration == pytest.approx(
        baseline
        + estimate_acquisition_time(cfg.channel_settings, cfg.zparams)
        + estimate_autofocus_time(cfg.autofocus_settings, cfg.channel_settings)
    )


def test_fluorescence_scales_with_channel_count():
    one = AcquireFluorescenceImageConfig(channel_settings=[_channel(0.5)])
    two = AcquireFluorescenceImageConfig(channel_settings=[_channel(0.5), _channel(0.5)])
    assert two.estimated_duration > one.estimated_duration


def test_fluorescence_counts_the_move_and_the_objective():
    """_run moves the stage and drives the objective before it acquires anything."""
    cfg = AcquireFluorescenceImageConfig(channel_settings=[_channel(0.5)])
    cfg.autofocus_settings = AutoFocusSettings(passes=[FocusSweepPass(enabled=False)])
    cfg.retract_objective = False

    expected = (
        2 * timing.reference_image_cost(cfg.reference_imaging)
        + timing.stage_move_cost(1)
        + timing.OBJECTIVE_INSERT_S
        + timing.OBJECTIVE_FOCUS_MOVE_S
        + estimate_acquisition_time(cfg.channel_settings, cfg.zparams)
    )
    assert cfg.estimated_duration == pytest.approx(expected)


def test_retracting_the_objective_costs_a_second_move():
    kept = AcquireFluorescenceImageConfig(retract_objective=False)
    retracted = AcquireFluorescenceImageConfig(retract_objective=True)
    assert retracted.estimated_duration - kept.estimated_duration == pytest.approx(
        timing.OBJECTIVE_RETRACT_S
    )


def test_autofocus_is_counted_only_when_a_pass_is_enabled():
    channels = [_channel(0.5)]
    off = AcquireFluorescenceImageConfig(
        channel_settings=channels,
        autofocus_settings=AutoFocusSettings(passes=[FocusSweepPass(enabled=False)]),
    )
    on = AcquireFluorescenceImageConfig(
        channel_settings=channels,
        autofocus_settings=AutoFocusSettings(passes=[FocusSweepPass(enabled=True)]),
    )
    assert on.estimated_duration > off.estimated_duration


# ── the autofocus sweep is counted, not assumed ──────────────────────────────

def test_autofocus_estimate_scales_with_the_steps_it_will_take():
    """The sweep is arithmetic: one image per step, and n_steps comes off the pass."""
    channels = [_channel(0.5)]
    coarse = AutoFocusSettings(passes=[FocusSweepPass(search_range=50e-6, step_size=5e-6)])
    fine = AutoFocusSettings(passes=[FocusSweepPass(search_range=50e-6, step_size=1e-6)])
    assert estimate_autofocus_time(fine, channels) > estimate_autofocus_time(coarse, channels)


def test_autofocus_estimate_ignores_disabled_passes():
    channels = [_channel(0.5)]
    both = AutoFocusSettings(
        passes=[FocusSweepPass(enabled=True), FocusSweepPass(enabled=True)]
    )
    one = AutoFocusSettings(
        passes=[FocusSweepPass(enabled=True), FocusSweepPass(enabled=False)]
    )
    assert estimate_autofocus_time(one, channels) < estimate_autofocus_time(both, channels)


def test_autofocus_estimate_is_zero_when_every_pass_is_off():
    settings = AutoFocusSettings(passes=[FocusSweepPass(enabled=False)])
    assert estimate_autofocus_time(settings, [_channel(0.5)]) == 0.0


def test_autofocus_estimate_needs_a_channel_to_focus_on():
    settings = AutoFocusSettings(passes=[FocusSweepPass(enabled=True)])
    assert estimate_autofocus_time(settings, []) == 0.0


def test_autofocus_estimate_uses_the_named_channel():
    """Resolved the way the acquisition path resolves it, so a slow focus channel is
    not costed as though it were the fast one listed first."""
    fast, slow = _channel(0.01), _channel(1.0)
    slow.name = "slow"
    settings = AutoFocusSettings(
        passes=[FocusSweepPass(enabled=True)], channel_name="slow"
    )
    named = estimate_autofocus_time(settings, [fast, slow])
    defaulted = estimate_autofocus_time(
        AutoFocusSettings(passes=[FocusSweepPass(enabled=True)]), [fast, slow]
    )
    assert named > defaulted


# ── every task answers ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "config_cls",
    [MillFiducialTaskConfig, SpotBurnFiducialTaskConfig, AcquireFluorescenceImageConfig],
)
def test_every_config_reports_a_positive_duration(config_cls):
    """A task with no override still answers, via the base implementation."""
    assert config_cls().estimated_duration > 0
