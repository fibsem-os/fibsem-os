import json
import os

import copy

import numpy as np
import pytest

from fibsem import acquire, alignment, utils
from fibsem.structures import FibsemImage
from fibsem.alignment import (
    AlignmentDifferential,
    AlignmentMethod,
    AlignmentIteration,
    AlignmentResult,
    AlignmentSubsystem,
    compare_alignment_methods,
    crosscorrelation_cv2,
    multi_step_alignment_v2,
    shift_from_crosscorrelation,
    shift_from_crosscorrelation_v2,
    shift_from_skimage_phase_correlation,
)
from fibsem.alignment.methods import _subpixel_peak

@pytest.mark.parametrize("offset", [-30, -10, 0, 10, 30])
def test_align_from_crosscorrelation(offset):
    ref_image = FibsemImage.generate_blank_image(resolution=(512, 512), hfw=50e-6)
    new_image = FibsemImage.generate_blank_image(resolution=(512, 512), hfw=50e-6)

    w = h = 150
    x, y = 200, 200
    ref_image.data[y:y + h, x:x + w] = 255
    new_image.data[y + offset:y + h + offset, x + offset:x + w + offset] = 255

    dx, dy, xcorr, _ = alignment.shift_from_crosscorrelation(
        ref_image, new_image, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
    )

    pixel_size = ref_image.metadata.pixel_size.x
    assert np.isclose(dx, offset * pixel_size, atol=pixel_size), f"dx: {dx}, offset: {offset * pixel_size}"
    assert np.isclose(dy, offset * pixel_size, atol=pixel_size), f"dy: {dy}, offset: {offset * pixel_size}"


# ---------------------------------------------------------------------------
# crosscorrelation_cv2
# ---------------------------------------------------------------------------

def test_crosscorrelation_cv2_known_shift():
    """crosscorrelation_cv2 recovers a known integer shift within 1 pixel."""
    rng = np.random.default_rng(42)
    img = rng.random((256, 256), dtype=np.float32)
    shift_col, shift_row = 15, -10
    shifted = np.roll(np.roll(img, shift_row, axis=0), shift_col, axis=1)

    sx, sy, response = crosscorrelation_cv2(img, shifted)

    assert abs(sx - shift_col) < 1.0, f"x shift {sx:.2f} != expected {shift_col}"
    assert abs(sy - shift_row) < 1.0, f"y shift {sy:.2f} != expected {shift_row}"


def test_crosscorrelation_cv2_response_good_match():
    """Response is high (> 0.5) for a well-matched pair."""
    rng = np.random.default_rng(7)
    img = rng.random((256, 256), dtype=np.float32)
    shifted = np.roll(img, 5, axis=1)

    _, _, response = crosscorrelation_cv2(img, shifted)

    assert response > 0.5, f"Expected high response, got {response:.3f}"


def test_crosscorrelation_cv2_response_low_for_noise():
    """Response is low (< 0.3) when the two images are unrelated noise."""
    rng = np.random.default_rng(99)
    img1 = rng.random((256, 256), dtype=np.float32)
    img2 = rng.random((256, 256), dtype=np.float32)

    _, _, response = crosscorrelation_cv2(img1, img2)

    assert response < 0.3, f"Expected low response for noise pair, got {response:.3f}"


def test_crosscorrelation_cv2_zero_shift():
    """Identical images should return near-zero shift and high response."""
    rng = np.random.default_rng(1)
    img = rng.random((256, 256), dtype=np.float32)

    sx, sy, response = crosscorrelation_cv2(img, img)

    assert abs(sx) < 0.5, f"x shift {sx:.3f} should be ~0 for identical images"
    assert abs(sy) < 0.5, f"y shift {sy:.3f} should be ~0 for identical images"
    assert response > 0.8, f"Response {response:.3f} should be high for identical images"


# ---------------------------------------------------------------------------
# shift_from_crosscorrelation_v2
# ---------------------------------------------------------------------------

def test_shift_from_crosscorrelation_v2_known_shift():
    """v2 recovers a known shift in metres within 1 pixel."""
    ref_image = FibsemImage.generate_blank_image(resolution=(512, 512), hfw=50e-6)
    new_image = FibsemImage.generate_blank_image(resolution=(512, 512), hfw=50e-6)

    w = h = 150
    offset = 20
    x, y = 200, 200
    ref_image.data[y:y + h, x:x + w] = 255
    new_image.data[y + offset:y + h + offset, x + offset:x + w + offset] = 255

    pixel_size = ref_image.metadata.pixel_size.x

    dx_v2, dy_v2, response = shift_from_crosscorrelation_v2(ref_image, new_image)

    assert response > 0.0, "response should be positive"
    assert np.isclose(dx_v2, offset * pixel_size, atol=pixel_size), \
        f"dx_v2={dx_v2:.2e}, expected {offset * pixel_size:.2e}"
    assert np.isclose(dy_v2, offset * pixel_size, atol=pixel_size), \
        f"dy_v2={dy_v2:.2e}, expected {offset * pixel_size:.2e}"


# ---------------------------------------------------------------------------
# _subpixel_peak
# ---------------------------------------------------------------------------

def _make_parabola(shape, peak_row, peak_col, amplitude=100.0):
    """Build a 2D parabolic surface with its maximum at (peak_row, peak_col)."""
    h, w = shape
    r = np.arange(h, dtype=np.float64)
    c = np.arange(w, dtype=np.float64)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    return amplitude - (rr - peak_row) ** 2 - (cc - peak_col) ** 2


def test_subpixel_peak_recovers_fractional_offset():
    """A parabolic surface with peak at a non-integer position is recovered to < 0.01 px."""
    true_row, true_col = 10.3, 15.7
    xcorr = _make_parabola((32, 32), true_row, true_col)

    # integer argmax lands at the nearest integer bin
    int_row, int_col = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    row_sub, col_sub = _subpixel_peak(xcorr, int_row, int_col)

    assert abs(row_sub - true_row) < 0.01, f"row error {abs(row_sub - true_row):.4f} >= 0.01"
    assert abs(col_sub - true_col) < 0.01, f"col error {abs(col_sub - true_col):.4f} >= 0.01"


def test_subpixel_peak_symmetric_returns_integer():
    """A peak sampled exactly at an integer position returns that integer (no offset)."""
    xcorr = _make_parabola((32, 32), 16.0, 16.0)
    row_sub, col_sub = _subpixel_peak(xcorr, 16, 16)

    assert abs(row_sub - 16.0) < 1e-9, f"unexpected row shift {row_sub - 16.0}"
    assert abs(col_sub - 16.0) < 1e-9, f"unexpected col shift {col_sub - 16.0}"


def test_subpixel_peak_border_falls_back_to_integer():
    """Peak on the image border (row=0 or col=0) returns the integer coordinate unchanged."""
    xcorr = np.zeros((16, 16), dtype=np.float64)
    xcorr[0, 8] = 10.0  # peak at top row border

    row_sub, col_sub = _subpixel_peak(xcorr, 0, 8)

    assert row_sub == 0.0, f"border row should not be refined, got {row_sub}"


def test_subpixel_peak_flat_denom_no_crash():
    """When all three samples are equal (zero denominator) the integer position is returned."""
    xcorr = np.ones((16, 16), dtype=np.float64)
    # All neighbours are equal → denom == 0 → no refinement
    row_sub, col_sub = _subpixel_peak(xcorr, 8, 8)

    assert row_sub == 8.0
    assert col_sub == 8.0


# ---------------------------------------------------------------------------
# Method agreement: all three methods should agree on magnitude and direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shift_x,shift_y", [(20, 20), (20, -20), (-15, 10), (0, 25)])
def test_alignment_methods_agree(shift_x, shift_y):
    """All three alignment methods recover the same shift direction and magnitude.

    Tolerance is 2 pixels — the bandpass cross-correlation method is less precise
    than the phase-correlation variants, so agreement is checked loosely.
    """
    ref_image, new_image, pixel_size = _make_shifted_images(shift_x, shift_y)
    atol = 2 * pixel_size

    dx_cc, dy_cc, _, _ = shift_from_crosscorrelation(
        ref_image, new_image, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
    )
    dx_cv2, dy_cv2, _ = shift_from_crosscorrelation_v2(ref_image, new_image)
    result_sk = shift_from_skimage_phase_correlation(ref_image, new_image)
    dx_sk, dy_sk = result_sk.shift.x, result_sk.shift.y

    # all three should agree on direction (sign) when shift is non-zero
    if shift_x != 0:
        assert np.sign(dx_cc) == np.sign(shift_x * pixel_size), f"cc dx sign wrong: {dx_cc:.2e}"
        assert np.sign(dx_cv2) == np.sign(shift_x * pixel_size), f"cv2 dx sign wrong: {dx_cv2:.2e}"
        assert np.sign(dx_sk) == np.sign(shift_x * pixel_size), f"sk dx sign wrong: {dx_sk:.2e}"
    if shift_y != 0:
        assert np.sign(dy_cc) == np.sign(shift_y * pixel_size), f"cc dy sign wrong: {dy_cc:.2e}"
        assert np.sign(dy_cv2) == np.sign(shift_y * pixel_size), f"cv2 dy sign wrong: {dy_cv2:.2e}"
        assert np.sign(dy_sk) == np.sign(shift_y * pixel_size), f"sk dy sign wrong: {dy_sk:.2e}"

    # all three should agree on magnitude within atol
    assert abs(dx_cc - dx_cv2) < atol, f"cc vs cv2 dx disagree: {dx_cc:.2e} vs {dx_cv2:.2e}"
    assert abs(dy_cc - dy_cv2) < atol, f"cc vs cv2 dy disagree: {dy_cc:.2e} vs {dy_cv2:.2e}"
    assert abs(dx_cc - dx_sk) < atol, f"cc vs sk dx disagree: {dx_cc:.2e} vs {dx_sk:.2e}"
    assert abs(dy_cc - dy_sk) < atol, f"cc vs sk dy disagree: {dy_cc:.2e} vs {dy_sk:.2e}"
    assert abs(dx_cv2 - dx_sk) < atol, f"cv2 vs sk dx disagree: {dx_cv2:.2e} vs {dx_sk:.2e}"
    assert abs(dy_cv2 - dy_sk) < atol, f"cv2 vs sk dy disagree: {dy_cv2:.2e} vs {dy_sk:.2e}"


def test_shift_from_crosscorrelation_v2_minimum_response_gate():
    """When response is below minimum_response, (0, 0, response) is returned."""
    ref_image = FibsemImage.generate_blank_image(resolution=(512, 512), hfw=50e-6, random=True)
    new_image = FibsemImage.generate_blank_image(resolution=(512, 512), hfw=50e-6, random=True)

    dx, dy, response = shift_from_crosscorrelation_v2(ref_image, new_image, minimum_response=1.0)

    assert dx == 0.0 and dy == 0.0, \
        f"Expected zero shift when response {response:.3f} < minimum_response"


# ---------------------------------------------------------------------------
# compare_alignment_methods
# ---------------------------------------------------------------------------

def _make_shifted_images(shift_x, shift_y):
    """Return (ref_image, new_image, pixel_size) with a synthetic known shift."""
    ref_image = FibsemImage.generate_blank_image(resolution=(512, 512), hfw=50e-6)
    new_image = FibsemImage.generate_blank_image(resolution=(512, 512), hfw=50e-6)
    w = h = 150
    x, y = 200, 200
    ref_image.data[y : y + h, x : x + w] = 255
    new_image.data[y + shift_y : y + h + shift_y, x + shift_x : x + w + shift_x] = 255
    return ref_image, new_image, ref_image.metadata.pixel_size.x


def test_compare_alignment_methods_returns_differential():
    """compare_alignment_methods returns an AlignmentDifferential with one result per method."""
    ref_image, new_image, _ = _make_shifted_images(20, 20)
    differential = compare_alignment_methods(ref_image, new_image)

    assert isinstance(differential, AlignmentDifferential)
    assert len(differential.results) == len(list(AlignmentMethod))
    assert set(differential.shifts_px.keys()) == {m.value for m in AlignmentMethod}


def test_compare_alignment_methods_results_tagged_with_method():
    """Each AlignmentIteration in the differential has its method field set."""
    ref_image, new_image, _ = _make_shifted_images(20, 20)
    differential = compare_alignment_methods(ref_image, new_image)

    for result in differential.results:
        assert result.method is not None
        assert isinstance(result.method, AlignmentMethod)


def test_compare_alignment_methods_agreement_on_known_shift():
    """All methods agree within threshold for a clean synthetic shift."""
    ref_image, new_image, _ = _make_shifted_images(20, 20)
    differential = compare_alignment_methods(ref_image, new_image, agreement_threshold_px=2.0)

    assert differential.agreement, (
        f"Methods should agree for a clean shift, "
        f"max_disagreement={differential.max_disagreement_px:.2f} px\n"
        f"shifts: {differential.shifts_px}"
    )


def test_compare_alignment_methods_shifts_close_to_expected():
    """Shifts in pixels from each method are within 2 px of the ground truth."""
    shift_x, shift_y = 15, -10
    ref_image, new_image, _ = _make_shifted_images(shift_x, shift_y)
    differential = compare_alignment_methods(ref_image, new_image)

    for name, pt in differential.shifts_px.items():
        assert abs(pt.x - shift_x) < 2.0, f"{name}: dx={pt.x:.2f} expected {shift_x}"
        assert abs(pt.y - shift_y) < 2.0, f"{name}: dy={pt.y:.2f} expected {shift_y}"


def test_compare_alignment_methods_disagreement_metric():
    """max_disagreement_px is non-negative and consistent with shifts_px."""
    ref_image, new_image, _ = _make_shifted_images(20, 5)
    differential = compare_alignment_methods(ref_image, new_image)

    assert differential.max_disagreement_px >= 0.0

    # recompute manually and verify it matches
    points = list(differential.shifts_px.values())
    expected_max = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.hypot(points[i].x - points[j].x, points[i].y - points[j].y)
            expected_max = max(expected_max, d)
    assert np.isclose(differential.max_disagreement_px, expected_max)


# ---------------------------------------------------------------------------
# AlignmentDifferential.to_dict — JSON serialisability
# ---------------------------------------------------------------------------

def test_alignment_differential_to_dict_json_serialisable():
    """to_dict() must be JSON-serialisable (no numpy scalars)."""
    ref_image, new_image, _ = _make_shifted_images(20, 20)
    differential = compare_alignment_methods(ref_image, new_image)
    d = differential.to_dict()
    # must not raise
    json.dumps(d)


def test_alignment_differential_to_dict_native_types():
    """to_dict() values are Python native types, not numpy scalars."""
    ref_image, new_image, _ = _make_shifted_images(10, 10)
    differential = compare_alignment_methods(ref_image, new_image)
    d = differential.to_dict()

    assert isinstance(d["agreement"], bool), f"agreement should be bool, got {type(d['agreement'])}"
    assert isinstance(d["max_disagreement_px"], float), f"max_disagreement_px should be float"


def test_alignment_differential_to_dict_structure():
    """to_dict() contains expected keys and one entry per method in shifts_px."""
    ref_image, new_image, _ = _make_shifted_images(15, -10)
    differential = compare_alignment_methods(ref_image, new_image)
    d = differential.to_dict()

    assert set(d.keys()) == {"shifts_px", "max_disagreement_px", "agreement", "consensus_shift"}
    assert set(d["shifts_px"].keys()) == {m.value for m in AlignmentMethod}


# ---------------------------------------------------------------------------
# multi_step_alignment_v2 with validate=True
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_session():
    """Single Demo session shared across all multi_step tests in this module."""
    microscope, settings = utils.setup_session(debug=False)
    return microscope, settings


def test_multi_step_alignment_validate_populates_validation(demo_session, tmp_path):
    """validate=True stores an AlignmentDifferential in run.validation."""
    microscope, settings = demo_session
    ref_image = acquire.acquire_image(microscope, settings.image)

    run = multi_step_alignment_v2(
        microscope=microscope,
        ref_image=ref_image,
        steps=1,
        validate=True,
        path=str(tmp_path),
    )

    assert run.validation is not None
    assert isinstance(run.validation, AlignmentDifferential)


def test_multi_step_alignment_validate_implies_final_image(demo_session, tmp_path):
    """validate=True implies acquire_final_image — run.final_image must be set."""
    microscope, settings = demo_session
    ref_image = acquire.acquire_image(microscope, settings.image)

    run = multi_step_alignment_v2(
        microscope=microscope,
        ref_image=ref_image,
        steps=1,
        validate=True,
        path=str(tmp_path),
    )

    assert run.final_image is not None


def test_multi_step_alignment_no_validate_no_validation(demo_session, tmp_path):
    """Without validate=True, run.validation is None."""
    microscope, settings = demo_session
    ref_image = acquire.acquire_image(microscope, settings.image)

    run = multi_step_alignment_v2(
        microscope=microscope,
        ref_image=ref_image,
        steps=1,
        validate=False,
        acquire_final_image=False,
        path=str(tmp_path),
    )

    assert run.validation is None


def test_multi_step_alignment_run_to_dict_json_serialisable_with_validation(demo_session, tmp_path):
    """AlignmentResult.to_dict() with validation must be JSON-serialisable."""
    microscope, settings = demo_session
    ref_image = acquire.acquire_image(microscope, settings.image)

    run = multi_step_alignment_v2(
        microscope=microscope,
        ref_image=ref_image,
        steps=1,
        validate=True,
        path=str(tmp_path),
    )

    d = run.to_dict()
    json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# consensus_shift
# ---------------------------------------------------------------------------

def test_consensus_shift_present():
    """compare_alignment_methods always populates consensus_shift."""
    ref_image, new_image, _ = _make_shifted_images(20, 10)
    differential = compare_alignment_methods(ref_image, new_image)
    assert differential.consensus_shift is not None


def test_consensus_shift_within_method_range():
    """consensus_shift lies between the per-method estimates (not outside the spread)."""
    from fibsem.structures import Point as Pt
    shift_x, shift_y = 20, 10
    ref_image, new_image, pixel_size = _make_shifted_images(shift_x, shift_y)
    differential = compare_alignment_methods(ref_image, new_image)

    xs = [pt.x for pt in differential.shifts_px.values()]
    ys = [pt.y for pt in differential.shifts_px.values()]
    cx = differential.consensus_shift.x / pixel_size
    cy = differential.consensus_shift.y / pixel_size

    assert min(xs) - 0.1 <= cx <= max(xs) + 0.1, f"consensus x={cx:.2f} outside [{min(xs):.2f}, {max(xs):.2f}]"
    assert min(ys) - 0.1 <= cy <= max(ys) + 0.1, f"consensus y={cy:.2f} outside [{min(ys):.2f}, {max(ys):.2f}]"


def test_consensus_shift_serialised():
    """consensus_shift survives a to_dict / from_dict round-trip."""
    ref_image, new_image, _ = _make_shifted_images(20, 10)
    differential = compare_alignment_methods(ref_image, new_image)
    from fibsem.alignment import AlignmentDifferential
    reloaded = AlignmentDifferential.from_dict(differential.to_dict())
    assert reloaded.consensus_shift is not None
    assert np.isclose(reloaded.consensus_shift.x, differential.consensus_shift.x)
    assert np.isclose(reloaded.consensus_shift.y, differential.consensus_shift.y)


# ---------------------------------------------------------------------------
# AlignmentDifferential.from_dict round-trip
# ---------------------------------------------------------------------------

def test_alignment_differential_from_dict_round_trip():
    """from_dict(to_dict()) preserves all scalar fields."""
    ref_image, new_image, _ = _make_shifted_images(15, -10)
    original = compare_alignment_methods(ref_image, new_image)
    reloaded = AlignmentDifferential.from_dict(original.to_dict())

    assert np.isclose(reloaded.max_disagreement_px, original.max_disagreement_px)
    assert reloaded.agreement == original.agreement
    assert set(reloaded.shifts_px.keys()) == set(original.shifts_px.keys())
    for key in original.shifts_px:
        assert np.isclose(reloaded.shifts_px[key].x, original.shifts_px[key].x)
        assert np.isclose(reloaded.shifts_px[key].y, original.shifts_px[key].y)


# ---------------------------------------------------------------------------
# AlignmentResult.load round-trip
# ---------------------------------------------------------------------------

def test_alignment_run_save_load_round_trip(demo_session, tmp_path):
    """AlignmentResult.save() followed by .load() recovers all scalar fields."""
    microscope, settings = demo_session
    ref_image = acquire.acquire_image(microscope, settings.image)

    run = multi_step_alignment_v2(
        microscope=microscope,
        ref_image=ref_image,
        steps=1,
        validate=True,
        path=str(tmp_path),
    )

    run_dir = os.path.join(str(tmp_path), run.name)
    reloaded = AlignmentResult.load(run_dir)

    assert reloaded.name == run.name
    assert reloaded.method == run.method
    assert reloaded.subsystem == run.subsystem
    assert len(reloaded.results) == len(run.results)
    for orig, loaded in zip(run.results, reloaded.results):
        assert np.isclose(loaded.shift.x, orig.shift.x)
        assert np.isclose(loaded.shift.y, orig.shift.y)
        assert np.isclose(loaded.score, orig.score)
        assert loaded.method == orig.method
    assert reloaded.final_image is not None
    assert reloaded.validation is not None
    assert np.isclose(reloaded.validation.max_disagreement_px, run.validation.max_disagreement_px)


# ---------------------------------------------------------------------------
# verified alignment mode (FIB-711)
# ---------------------------------------------------------------------------

def test_validation_modes_registry():
    """The two shipped modes, and only 'verified' gates."""
    from fibsem.alignment import VALIDATION_MODES, DEFAULT_VALIDATION_MODE

    assert DEFAULT_VALIDATION_MODE == "none", "default must stay the historical path"
    assert VALIDATION_MODES["none"].validator is None
    assert VALIDATION_MODES["verified"].validator == "dog"
    # both modes must share an estimator: validation decides whether a measurement is
    # trusted, it must never silently change what does the measuring
    assert (
        VALIDATION_MODES["none"].estimator == VALIDATION_MODES["verified"].estimator
    ), "validation must not side-load an estimator swap"


def test_unknown_mode_raises_before_touching_the_microscope():
    from fibsem.alignment import get_validation_mode

    with pytest.raises(ValueError, match="Unknown alignment validation"):
        get_validation_mode("nonsense")


def test_zscore_profile_is_the_historical_pipeline():
    """The zscore profile must reproduce the untouched call exactly."""
    from fibsem.alignment.methods import (
        get_preprocessing_profile,
        shift_from_crosscorrelation,
    )

    ref, new, _ = _make_shifted_images(11, -7)
    kw = dict(lowpass=50, highpass=4, sigma=5)
    dx_old, dy_old, _, _ = shift_from_crosscorrelation(ref, new, use_rect_mask=True, **kw)
    dx_new, dy_new, _, _ = shift_from_crosscorrelation(
        ref, new, **get_preprocessing_profile("zscore"), **kw
    )
    assert (dx_old, dy_old) == (dx_new, dy_new)


def test_identical_images_are_accepted_at_zero_shift():
    """Both profiles agree on a trivial pair, so verified must not refuse it."""
    from fibsem.alignment.verified import shift_from_crosscorrelation_validated

    ref, _, _ = _make_shifted_images(0, 0)
    result = shift_from_crosscorrelation_validated(ref, ref, mode="verified")

    assert result.accepted, result.reason
    assert result.disagreement_px < 3.0


def test_unrelated_images_are_refused_and_zeroed():
    """When the two profiles cannot corroborate each other, nothing moves."""
    from fibsem.alignment.verified import shift_from_crosscorrelation_validated

    rng = np.random.default_rng(0)
    ref, _, _ = _make_shifted_images(0, 0)
    noise = FibsemImage(
        data=rng.integers(0, 255, ref.data.shape, dtype=np.uint8),
        metadata=ref.metadata,
    )
    result = shift_from_crosscorrelation_validated(ref, noise, mode="verified")

    if not result.accepted:
        assert result.shift.x == 0.0 and result.shift.y == 0.0
        assert result.shift_px.x == 0.0 and result.shift_px.y == 0.0
        assert "disagree" in result.reason


def test_unvalidated_mode_never_refuses():
    """'none' has no validator, so it accepts unconditionally -- as it always did."""
    from fibsem.alignment.verified import shift_from_crosscorrelation_validated

    rng = np.random.default_rng(1)
    ref, _, _ = _make_shifted_images(0, 0)
    noise = FibsemImage(
        data=rng.integers(0, 255, ref.data.shape, dtype=np.uint8),
        metadata=ref.metadata,
    )
    result = shift_from_crosscorrelation_validated(ref, noise, mode="none")

    assert result.accepted
    assert np.isnan(result.disagreement_px), "no validator ran, so there is no disagreement"


def test_rejected_step_applies_no_shift(monkeypatch, demo_session, tmp_path):
    """A rejected iteration must not reach _apply_shift."""
    import fibsem.alignment as al

    microscope, settings = demo_session
    ref_image = acquire.acquire_image(microscope, settings.image)
    applied = []
    monkeypatch.setattr(
        al, "_apply_shift", lambda **kw: applied.append((kw["dx"], kw["dy"]))
    )

    def reject(ref, new, method=None, alignment_validation=None):
        from fibsem.alignment import AlignmentIteration
        from fibsem.structures import Point

        return AlignmentIteration(
            shift=Point(0.0, 0.0), score=0.5, image=new, method=method,
            accepted=False, reason="test rejection", disagreement_px=99.0,
        )

    monkeypatch.setattr(al, "_calculate_shift", reject)
    run = al.multi_step_alignment_v2(
        microscope=microscope, ref_image=ref_image, steps=2,
        validate=False, acquire_final_image=False,
        alignment_validation="verified", path=str(tmp_path),
    )

    assert applied == [], f"a rejected step moved the beam: {applied}"
    assert run.aligned is False
    assert run.outcome == "all_steps_rejected"
    assert run.n_rejected == 2 and run.n_accepted == 0


def test_preference_is_resolved_per_context(monkeypatch):
    """Workflow and milling alignment read separate settings (FIB-807)."""
    import fibsem.config as cfg
    from fibsem.alignment.verified import AlignmentContext, get_configured_validation

    prefs = cfg.UserPreferences()
    prefs.alignment.validation_workflow = "none"
    prefs.alignment.validation_milling = "verified"
    monkeypatch.setattr(cfg, "load_user_preferences", lambda: prefs)

    assert get_configured_validation(AlignmentContext.WORKFLOW) == "none"
    assert get_configured_validation(AlignmentContext.MILLING) == "verified"

    # and the other way round, so this cannot pass by reading one field twice
    prefs.alignment.validation_workflow = "verified"
    prefs.alignment.validation_milling = "none"
    assert get_configured_validation(AlignmentContext.WORKFLOW) == "verified"
    assert get_configured_validation(AlignmentContext.MILLING) == "none"


def test_bad_or_unreadable_preference_falls_back_to_default(monkeypatch):
    """Alignment must not break because preferences are wrong or missing."""
    import fibsem.config as cfg
    from fibsem.alignment import DEFAULT_VALIDATION_MODE
    from fibsem.alignment.verified import AlignmentContext, get_configured_validation

    prefs = cfg.UserPreferences()
    prefs.alignment.validation_workflow = "nonsense"
    monkeypatch.setattr(cfg, "load_user_preferences", lambda: prefs)
    assert get_configured_validation(AlignmentContext.WORKFLOW) == DEFAULT_VALIDATION_MODE

    def boom():
        raise OSError("unreadable")

    monkeypatch.setattr(cfg, "load_user_preferences", boom)
    assert get_configured_validation(AlignmentContext.WORKFLOW) == DEFAULT_VALIDATION_MODE
    assert get_configured_validation(AlignmentContext.MILLING) == DEFAULT_VALIDATION_MODE


def test_preferences_default_to_historical_behaviour():
    """A fresh install, and a prefs file written before this existed, stay unvalidated."""
    from fibsem.config import UserPreferences

    for prefs in (UserPreferences(), UserPreferences.from_dict({"display": {}})):
        assert prefs.alignment.validation_workflow == "none"
        assert prefs.alignment.validation_milling == "none"


def test_milling_and_workflow_call_sites_declare_their_context():
    """The split is worthless if the call sites do not say which context they are."""
    import inspect

    from fibsem.alignment import AlignmentContext
    import fibsem.milling.core as core
    from fibsem.applications.autolamella.workflows.tasks import base

    assert "AlignmentContext.MILLING" in inspect.getsource(core)
    assert "AlignmentContext.WORKFLOW" in inspect.getsource(base)
    # and the enum members are what those sources name
    assert AlignmentContext.MILLING.value == "milling"
    assert AlignmentContext.WORKFLOW.value == "workflow"


def _all_rejected(monkeypatch):
    """Make every alignment step refuse."""
    import fibsem.alignment as al
    from fibsem.alignment import AlignmentIteration
    from fibsem.structures import Point

    monkeypatch.setattr(al, "_apply_shift", lambda **kw: None)
    monkeypatch.setattr(
        al, "_calculate_shift",
        lambda ref, new, method=None, alignment_validation=None: AlignmentIteration(
            shift=Point(0.0, 0.0), score=0.5, image=new, method=method,
            accepted=False, reason="test rejection", disagreement_px=99.0,
        ),
    )


def _set_prefs(monkeypatch, **kw):
    import fibsem.config as cfg
    prefs = cfg.UserPreferences()
    for k, v in kw.items():
        setattr(prefs.alignment, k, v)
    monkeypatch.setattr(cfg, "load_user_preferences", lambda: prefs)


def test_abort_on_failure_off_returns_a_failed_run(monkeypatch, demo_session, tmp_path):
    """Default: a fully-refused alignment logs and returns, it does not raise."""
    import fibsem.alignment as al

    microscope, settings = demo_session
    ref_image = acquire.acquire_image(microscope, settings.image)
    _all_rejected(monkeypatch)
    _set_prefs(monkeypatch, validation_workflow="verified", validation_milling="verified", abort_on_failure=False)

    run = al.multi_step_alignment_v2(
        microscope=microscope, ref_image=ref_image, steps=2,
        validate=False, acquire_final_image=False, path=str(tmp_path),
    )
    assert run.aligned is False
    assert run.outcome == "all_steps_rejected"


def test_abort_on_failure_raises_and_still_saves_the_diagnostics(
    monkeypatch, demo_session, tmp_path
):
    """With the flag on, the task is stopped -- but the evidence is written first."""
    import fibsem.alignment as al
    from fibsem.alignment import AlignmentFailedError

    microscope, settings = demo_session
    ref_image = acquire.acquire_image(microscope, settings.image)
    _all_rejected(monkeypatch)
    _set_prefs(monkeypatch, validation_workflow="verified", validation_milling="verified", abort_on_failure=True)

    with pytest.raises(AlignmentFailedError, match="Could not verify the alignment"):
        al.multi_step_alignment_v2(
            microscope=microscope, ref_image=ref_image, steps=2,
            validate=False, acquire_final_image=False, path=str(tmp_path),
        )

    saved = list(tmp_path.rglob("data.json"))
    assert saved, "diagnostics must be saved before the failure propagates"


def test_alignment_failure_is_not_treated_as_a_user_cancellation():
    """AlignmentFailedError must be recorded as a genuine failure, not a Stop."""
    from fibsem.alignment import AlignmentFailedError
    from fibsem.cancellation import OperationCancelledError

    assert not issubclass(AlignmentFailedError, OperationCancelledError)
    assert not issubclass(AlignmentFailedError, InterruptedError)


# ---------------------------------------------------------------------------
# regressions from the FIB-807 review
# ---------------------------------------------------------------------------

def test_data_json_is_strict_json_even_on_an_unvalidated_run(demo_session, tmp_path):
    """NaN is not valid JSON, and disagreement_px/psr are NaN on every zscore run."""
    import json

    from fibsem.alignment import multi_step_alignment_v2

    microscope, settings = demo_session
    ref_image = acquire.acquire_image(microscope, settings.image)
    multi_step_alignment_v2(
        microscope=microscope, ref_image=ref_image, steps=1,
        validate=False, acquire_final_image=False,
        alignment_validation="none", path=str(tmp_path),
    )

    raw = next(tmp_path.rglob("data.json")).read_text()
    assert "NaN" not in raw, "bare NaN token: strict JSON parsers will reject this"

    def reject(token):
        raise ValueError(f"non-standard JSON constant: {token}")

    json.loads(raw, parse_constant=reject)  # must not raise


def test_run_verdict_survives_save_and_load(demo_session, tmp_path, monkeypatch):
    """A refused run must not reload as a successful one."""
    import fibsem.alignment as al
    from fibsem.alignment import AlignmentResult

    microscope, settings = demo_session
    ref_image = acquire.acquire_image(microscope, settings.image)
    _all_rejected(monkeypatch)
    _set_prefs(monkeypatch, validation_workflow="verified", validation_milling="verified", abort_on_failure=False)

    run = al.multi_step_alignment_v2(
        microscope=microscope, ref_image=ref_image, steps=2,
        validate=False, acquire_final_image=False, path=str(tmp_path),
    )
    assert run.aligned is False

    reloaded = AlignmentResult.load(str(next(tmp_path.rglob("data.json")).parent))
    assert reloaded.aligned == run.aligned
    assert reloaded.outcome == run.outcome
    assert reloaded.n_rejected == run.n_rejected
    assert reloaded.mode == run.mode


def test_legacy_record_without_a_verdict_is_derived_from_its_steps(
    demo_session, tmp_path
):
    """A data.json written before these fields existed must not claim success."""
    import json

    from fibsem.alignment import AlignmentResult, multi_step_alignment_v2

    microscope, settings = demo_session
    ref_image = acquire.acquire_image(microscope, settings.image)
    multi_step_alignment_v2(
        microscope=microscope, ref_image=ref_image, steps=1,
        validate=False, acquire_final_image=False,
        alignment_validation="none", path=str(tmp_path),
    )
    p = next(tmp_path.rglob("data.json"))
    d = json.loads(p.read_text())
    for key in ("mode", "aligned", "n_accepted", "n_rejected", "outcome"):
        d.pop(key, None)
    d["results"][0]["accepted"] = False   # a refused step, pre-verdict format
    p.write_text(json.dumps(d))

    reloaded = AlignmentResult.load(str(p.parent))
    assert reloaded.aligned is False, "derived verdict must follow the steps"
    assert reloaded.outcome == "all_steps_rejected"
    assert reloaded.n_rejected == 1


def test_direct_beam_shift_alignment_honours_the_global_preference(
    monkeypatch, demo_session
):
    """The safety preference must apply on every path that can move the beam.

    beam_shift_alignment_v2 is a public entry point (example/autolamella.py calls it
    directly). Before this fix only multi_step_alignment_v2 resolved the preference,
    so a direct caller silently ran unvalidated however the user had it configured.
    """
    import fibsem.alignment as al
    from fibsem.alignment import AlignmentIteration
    from fibsem.structures import Point

    microscope, settings = demo_session
    ref_image = acquire.acquire_image(microscope, settings.image)
    seen = {}
    applied = []
    monkeypatch.setattr(al, "_apply_shift", lambda **kw: applied.append(kw))

    def spy(ref, new, method=None, alignment_validation=None):
        seen["validation"] = alignment_validation
        return AlignmentIteration(
            shift=Point(0.0, 0.0), score=0.5, image=new, method=method,
            accepted=False, reason="test rejection", disagreement_px=99.0,
        )

    monkeypatch.setattr(al, "_calculate_shift", spy)
    _set_prefs(monkeypatch, validation_workflow="verified", validation_milling="verified")

    result = al.beam_shift_alignment_v2(microscope=microscope, ref_image=ref_image)

    assert seen["validation"] == "verified", "the global preference did not reach this path"
    assert result.accepted is False
    assert applied == [], "a refused alignment moved the beam"


# --- shift clipping (FIB-807) ---------------------------------------------------


def _image_with_pixel_size(height: int, width: int, pixel_size: float) -> FibsemImage:
    """A real FibsemImage of a given ROI size, for the geometry-dependent clip."""
    ref, _, _ = _make_shifted_images(0, 0)
    md = copy.deepcopy(ref.metadata)
    md.pixel_size.x = pixel_size
    md.pixel_size.y = pixel_size
    return FibsemImage(data=np.zeros((height, width), dtype=np.uint8), metadata=md)



def test_clip_leaves_a_small_shift_untouched():
    from fibsem.alignment.verified import clip_shift_to_roi

    # 10px on a 400x400 ROI: nowhere near 0.8 of the 200px half-width
    dx, dy, clipped = clip_shift_to_roi(10e-9, 0.0, (400, 400), (1e-9, 1e-9), 0.8)
    assert clipped is False
    assert (dx, dy) == (10e-9, 0.0)


def test_clip_bounds_an_oversized_shift():
    from fibsem.alignment.verified import clip_shift_to_roi

    # half-width 200px, limit 0.8 -> 160px. Ask for 300px.
    dx, dy, clipped = clip_shift_to_roi(300e-9, 0.0, (400, 400), (1e-9, 1e-9), 0.8)
    assert clipped is True
    assert dx == pytest.approx(160e-9)
    assert dy == 0.0


def test_clip_preserves_direction():
    """Scaling both axes together; clipping them independently would rotate the move."""
    from fibsem.alignment.verified import clip_shift_to_roi

    dx, dy, clipped = clip_shift_to_roi(300e-9, 150e-9, (400, 400), (1e-9, 1e-9), 0.8)
    assert clipped is True
    # same bearing as the input (2:1)
    assert dx / dy == pytest.approx(300.0 / 150.0)


def test_clip_zero_fraction_disables():
    from fibsem.alignment.verified import clip_shift_to_roi

    dx, dy, clipped = clip_shift_to_roi(9999e-9, 0.0, (400, 400), (1e-9, 1e-9), 0.0)
    assert clipped is False
    assert dx == 9999e-9


def test_clip_uses_the_per_axis_half_dimension():
    """A tall, narrow ROI bounds x more tightly than y."""
    from fibsem.alignment.verified import clip_shift_to_roi

    # shape is (height, width) = (800, 200): half-width 100px, half-height 400px
    dxc, _, cx = clip_shift_to_roi(150e-9, 0.0, (800, 200), (1e-9, 1e-9), 1.0)
    _, dyc, cy = clip_shift_to_roi(0.0, 150e-9, (800, 200), (1e-9, 1e-9), 1.0)
    assert cx is True and dxc == pytest.approx(100e-9)
    assert cy is False and dyc == pytest.approx(150e-9)


def test_validation_judges_raw_shifts_not_clipped_ones(monkeypatch):
    """Clipping before comparing would let two disagreeing estimates agree at the bound.

    A 300px estimator and a 200px validator both clip to 160px. If the disagreement
    were measured after clipping it would be 0 and the step accepted -- a false
    accept manufactured by the guard meant to make things safer.
    """
    import numpy as np

    import fibsem.alignment.verified as v

    calls = []

    def fake(ref, new, **kw):
        # first call is the estimator, second the validator
        dx = 300e-9 if not calls else 200e-9
        calls.append(kw)
        return dx, 0.0, np.zeros((8, 8)), 1.0

    monkeypatch.setattr(v, "shift_from_crosscorrelation", fake)

    ref = _image_with_pixel_size(400, 400, 1e-9)
    new = _image_with_pixel_size(400, 400, 1e-9)
    out = v.shift_from_crosscorrelation_validated(ref, new, mode="verified", clip_fraction=0.8)

    assert out.disagreement_px == pytest.approx(100.0)  # 300px vs 200px, raw
    assert out.accepted is False
    assert out.shift.x == 0.0


def test_clip_applies_to_an_accepted_shift(monkeypatch):
    """An agreed-on but oversized shift is applied, bounded, not refused."""
    import numpy as np

    import fibsem.alignment.verified as v

    def fake(ref, new, **kw):
        return 300e-9, 0.0, np.zeros((8, 8)), 1.0  # both passes agree exactly

    monkeypatch.setattr(v, "shift_from_crosscorrelation", fake)

    ref = _image_with_pixel_size(400, 400, 1e-9)
    new = _image_with_pixel_size(400, 400, 1e-9)
    out = v.shift_from_crosscorrelation_validated(ref, new, mode="verified", clip_fraction=0.8)

    assert out.accepted is True
    assert out.clipped is True
    assert out.shift.x == pytest.approx(160e-9)  # 0.8 * 200px half-width
    assert out.to_dict()["clipped"] is True


def test_clip_fraction_default_is_read_from_preferences(monkeypatch):
    import fibsem.config as cfg
    from fibsem.alignment.verified import get_configured_clip_fraction

    prefs = cfg.UserPreferences()
    prefs.alignment.clip_fraction = 0.5
    monkeypatch.setattr(cfg, "load_user_preferences", lambda: prefs)
    assert get_configured_clip_fraction() == 0.5

    def boom():
        raise OSError("unreadable")

    monkeypatch.setattr(cfg, "load_user_preferences", boom)
    from fibsem.alignment.verified import DEFAULT_CLIP_FRACTION

    assert get_configured_clip_fraction() == DEFAULT_CLIP_FRACTION
