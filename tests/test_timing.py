"""Cost primitives behind the forward duration estimates (FIB-666).

The constants themselves are measurements and will move as more instruments are
sampled; these tests pin the *shape* of each primitive — what it counts, what it
refuses to count — rather than the numbers, so a re-measurement does not have to
rewrite the suite.
"""

import pytest

from fibsem import timing
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig, MillingTaskAcquisitionSettings
from fibsem.structures import (
    FibsemMillingSettings,
    ImageSettings,
    MillingAlignment,
    ReferenceImageParameters,
)


def _imaging() -> ImageSettings:
    return ImageSettings(resolution=(1536, 1024), dwell_time=1e-6)


def _stage(current: float = 2e-9, **pattern) -> FibsemMillingStage:
    pattern = pattern or dict(width=10e-6, height=5e-6, depth=1e-6)
    return FibsemMillingStage(
        milling=FibsemMillingSettings(milling_current=current),
        pattern=RectanglePattern(**pattern),
    )


# ── image_cost ───────────────────────────────────────────────────────────────


def test_image_cost_adds_the_fixed_overhead_to_the_scan():
    img = _imaging()
    assert timing.image_cost(img) == pytest.approx(
        img.estimated_time + timing.IMAGE_OVERHEAD_S
    )


def test_image_cost_scales_with_count():
    img = _imaging()
    assert timing.image_cost(img, 4) == pytest.approx(4 * timing.image_cost(img, 1))


def test_image_cost_of_nothing_is_zero():
    assert timing.image_cost(None) == 0.0
    assert timing.image_cost(_imaging(), 0) == 0.0
    assert timing.image_cost(_imaging(), -1) == 0.0


def test_image_overhead_dominates_at_typical_settings():
    """The reason estimates built from scan time alone came out 30-40x short: at
    1536x1024 and 1 us the fixed cost is larger than the scan it wraps."""
    img = _imaging()
    assert timing.IMAGE_OVERHEAD_S > img.estimated_time


# ── reference_image_cost ─────────────────────────────────────────────────────


def test_reference_image_cost_counts_every_fov_and_beam():
    params = ReferenceImageParameters(
        imaging=_imaging(),
        acquire_image1=True,
        acquire_image2=True,
        acquire_sem=True,
        acquire_fib=True,
    )
    assert timing.reference_image_cost(params) == pytest.approx(
        timing.image_cost(params.imaging, 4)
    )


def test_reference_image_cost_drops_deselected_beams():
    both = ReferenceImageParameters(
        imaging=_imaging(), acquire_sem=True, acquire_fib=True
    )
    one = ReferenceImageParameters(
        imaging=_imaging(), acquire_sem=True, acquire_fib=False
    )
    assert timing.reference_image_cost(one) == pytest.approx(
        timing.reference_image_cost(both) / 2
    )


def test_reference_image_cost_exceeds_the_scan_only_estimate():
    """The gap is the whole point: estimated_time counts scan time and nothing else."""
    params = ReferenceImageParameters(imaging=_imaging())
    assert timing.reference_image_cost(params) > params.estimated_time


def test_reference_image_cost_of_nothing_is_zero():
    assert timing.reference_image_cost(None) == 0.0


# ── stage_move_cost ──────────────────────────────────────────────────────────


def test_stage_move_cost_scales_with_count():
    assert timing.stage_move_cost(3) == pytest.approx(3 * timing.STAGE_MOVE_ABSOLUTE_S)


def test_relative_moves_are_cheaper_than_absolute():
    assert timing.stage_move_cost(1, absolute=False) < timing.stage_move_cost(
        1, absolute=True
    )


def test_no_move_costs_nothing():
    assert timing.stage_move_cost(0) == 0.0


# ── alignment_cost ───────────────────────────────────────────────────────────


def test_alignment_costs_two_passes_of_steps_plus_one_frames():
    """A stage configured for 3 steps acquired 4 frames per pass, and ran two passes --
    once at the imaging current, once after switching to the milling current."""
    alignment = MillingAlignment(enabled=True, steps=3, imaging=_imaging())
    per_pass = timing.ALIGNMENT_AUTOCONTRAST_S + 4 * timing.ALIGNMENT_IMAGE_S
    assert timing.alignment_cost(alignment) == pytest.approx(2 * per_pass)


def test_alignment_pass_count_is_adjustable():
    alignment = MillingAlignment(enabled=True, steps=3, imaging=_imaging())
    assert timing.alignment_cost(alignment, passes=1) == pytest.approx(
        timing.alignment_cost(alignment, passes=2) / 2
    )
    assert timing.alignment_cost(alignment, passes=0) == 0.0


def test_alignment_frames_are_cheaper_than_full_frames():
    """They are reduced-area: ~0.8 s against ~3.5 s for the frame the alignment's own
    ImageSettings describes, so costing them through image_cost overstates them."""
    assert timing.ALIGNMENT_IMAGE_S < timing.image_cost(_imaging(), 1)


def test_disabled_alignment_costs_nothing():
    assert (
        timing.alignment_cost(
            MillingAlignment(enabled=False, steps=3, imaging=_imaging())
        )
        == 0.0
    )
    assert timing.alignment_cost(None) == 0.0


# ── milling_cost ─────────────────────────────────────────────────────────────


def test_milling_cost_sums_the_stages():
    a, b = _stage(), _stage(current=7.6e-9)
    assert timing.milling_cost([a, b]) == pytest.approx(
        a.estimated_time + b.estimated_time
    )


def test_milling_cost_ignores_disabled_stages():
    """A stage switched off is not milled, so it is not quoted -- the same rule the
    task config's own estimate had to be corrected to follow (FIB-687)."""
    enabled, disabled = _stage(), _stage(current=7.6e-9)
    disabled.enabled = False
    assert timing.milling_cost([enabled, disabled]) == pytest.approx(
        enabled.estimated_time
    )


def test_milling_cost_of_nothing_is_zero():
    assert timing.milling_cost(None) == 0.0
    assert timing.milling_cost([]) == 0.0


# ── milling_task_cost ────────────────────────────────────────────────────────


def test_milling_task_cost_includes_alignment_and_acquisition():
    stage = _stage()
    acq = MillingTaskAcquisitionSettings(
        acquire_sem=True, acquire_fib=False, imaging=_imaging()
    )
    cfg = FibsemMillingTaskConfig(stages=[stage], acquisition=acq)
    cfg.alignment = MillingAlignment(enabled=True, steps=2, imaging=_imaging())

    expected = (
        stage.estimated_time
        + timing.alignment_cost(cfg.alignment)
        + timing.image_cost(acq.imaging, 1)
        + 2 * timing.BEAM_CURRENT_CHANGE_S
    )
    assert timing.milling_task_cost(cfg) == pytest.approx(expected)


def test_milling_task_cost_counts_the_final_refresh_image():
    """A task that acquires nothing of its own still ends with one FIB image."""
    acq = MillingTaskAcquisitionSettings(
        acquire_sem=False,
        acquire_fib=False,
        acquire_final_image=True,
        imaging=_imaging(),
    )
    cfg = FibsemMillingTaskConfig(stages=[_stage()], acquisition=acq)
    cfg.alignment = MillingAlignment(enabled=False)
    assert timing.milling_task_cost(cfg) == pytest.approx(
        cfg.stages[0].estimated_time
        + timing.image_cost(acq.imaging, 1)
        + 2 * timing.BEAM_CURRENT_CHANGE_S
    )


def test_milling_task_cost_charges_the_current_change_both_ways():
    """Into the milling current and back out again -- ~40 s that milling's own
    estimate never sees."""
    cfg = FibsemMillingTaskConfig(stages=[_stage()])
    cfg.alignment = MillingAlignment(enabled=False)
    cfg.acquisition = MillingTaskAcquisitionSettings(
        acquire_sem=False, acquire_fib=False, acquire_final_image=False
    )
    assert timing.milling_task_cost(cfg) == pytest.approx(
        cfg.stages[0].estimated_time + 2 * timing.BEAM_CURRENT_CHANGE_S
    )


def test_a_task_that_mills_nothing_pays_no_current_change():
    """Every stage disabled: the current is never switched, so it is not charged."""
    stage = _stage()
    stage.enabled = False
    cfg = FibsemMillingTaskConfig(stages=[stage])
    cfg.alignment = MillingAlignment(enabled=False)
    cfg.acquisition = MillingTaskAcquisitionSettings(
        acquire_sem=False, acquire_fib=False, acquire_final_image=False
    )
    assert timing.milling_task_cost(cfg) == 0.0


def test_milling_task_cost_of_nothing_is_zero():
    assert timing.milling_task_cost(None) == 0.0
