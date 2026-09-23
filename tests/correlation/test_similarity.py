"""The closed-form 2D similarity behind three-point alignment (FIB-1030).

Hand-built cases: a known turn, scale and offset applied to points must come back
exactly; noise must come back as residuals and not as a reflection; too few or
degenerate pairs must refuse rather than answer.
"""

import math

import numpy as np
import pytest

from fibsem.correlation.similarity import SimilarityFit, fit_similarity


def _apply(points, scale, rotation_deg, tx, ty):
    theta = math.radians(rotation_deg)
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    return scale * np.asarray(points, dtype=float) @ rotation.T + np.array([tx, ty])


SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
TRIANGLE = np.array([[0.0, 0.0], [10.0, 0.0], [3.0, 8.0]])


@pytest.mark.parametrize(
    "scale, rotation, tx, ty",
    [(1.0, 0.0, 0.0, 0.0), (1.0, 30.0, 5.0, -2.0), (1.7, -100.0, -40.0, 12.0)],
)
def test_a_known_similarity_comes_back_exactly(scale, rotation, tx, ty):
    dst = _apply(TRIANGLE, scale, rotation, tx, ty)
    fit = fit_similarity(TRIANGLE, dst)
    assert fit.scale == pytest.approx(scale)
    assert fit.rotation == pytest.approx(rotation)
    assert fit.translation == pytest.approx((tx, ty))
    assert fit.rms == pytest.approx(0.0, abs=1e-9)
    assert fit.apply(TRIANGLE) == pytest.approx(dst)


def test_two_pairs_are_enough():
    dst = _apply(SQUARE[:2], 2.0, 45.0, 1.0, 1.0)
    fit = fit_similarity(SQUARE[:2], dst)
    assert (fit.scale, fit.rotation) == (pytest.approx(2.0), pytest.approx(45.0))


def test_fixing_the_scale_fits_only_the_turn_and_offset():
    dst = _apply(TRIANGLE, 1.3, 20.0, 4.0, 4.0)
    fit = fit_similarity(TRIANGLE, dst, fix_scale=True)
    assert fit.scale == 1.0
    assert fit.rotation == pytest.approx(20.0)
    # It cannot fit the scale it was denied, and says so in the residual.
    assert fit.rms > 0.5
    held = fit_similarity(TRIANGLE, dst, fix_scale=True, scale=1.3)
    assert held.scale == 1.3
    assert held.rms == pytest.approx(0.0, abs=1e-9)


def test_noise_becomes_residuals_and_the_rms_is_their_root_mean_square():
    rng = np.random.default_rng(0)
    dst = _apply(SQUARE, 1.0, 10.0, 0.0, 0.0) + rng.normal(0.0, 0.2, SQUARE.shape)
    fit = fit_similarity(SQUARE, dst)
    assert fit.residuals.shape == (4,)
    assert fit.rms == pytest.approx(math.sqrt(float((fit.residuals**2).mean())))
    assert fit.rms < 0.5
    assert fit.rotation == pytest.approx(10.0, abs=2.0)


def _best_proper_rms(src, dst):
    """The least residual any proper similarity can reach, by brute force over the
    turn: for each angle the optimal scale and offset are closed-form."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    best = np.inf
    for degrees in np.linspace(-180.0, 180.0, 7201):
        theta = math.radians(degrees)
        c, s = math.cos(theta), math.sin(theta)
        turned = src_c @ np.array([[c, -s], [s, c]]).T
        scale = max(float((turned * dst_c).sum() / (turned**2).sum()), 0.0)
        residual = np.linalg.norm(scale * turned - dst_c, axis=1)
        best = min(best, float(np.sqrt((residual**2).mean())))
    return best


def test_a_mirrored_target_is_never_fitted_as_a_reflection():
    """A reflection would hide a wrong camera transform behind a good residual: the
    fit stays a proper rotation, the best one there is, and reports the misfit. A
    fit that let the reflection through would report a turn and scale that are not
    the least-squares answer among rotations, and so a larger residual than this."""
    mirrored = TRIANGLE * np.array([-1.0, 1.0])
    fit = fit_similarity(TRIANGLE, mirrored)
    assert np.linalg.det(fit.matrix()[:2, :2]) > 0
    assert fit.rms > 1.0
    assert fit.rms == pytest.approx(_best_proper_rms(TRIANGLE, mirrored), rel=1e-3)


@pytest.mark.parametrize("n", [0, 1])
def test_too_few_pairs_refuse(n):
    with pytest.raises(ValueError, match="at least two"):
        fit_similarity(TRIANGLE[:n], TRIANGLE[:n])


def test_coincident_sources_refuse():
    with pytest.raises(ValueError, match="coincide"):
        fit_similarity(np.zeros((3, 2)), TRIANGLE)


def test_mismatched_pairs_refuse():
    with pytest.raises(ValueError, match="against"):
        fit_similarity(TRIANGLE, SQUARE)


def test_the_matrix_agrees_with_apply():
    fit = SimilarityFit(scale=1.5, rotation=33.0, translation=(2.0, -1.0))
    homogeneous = np.c_[TRIANGLE, np.ones(3)] @ fit.matrix().T
    assert homogeneous[:, :2] == pytest.approx(fit.apply(TRIANGLE))
