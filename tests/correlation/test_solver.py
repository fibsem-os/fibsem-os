"""The rigid FM->FIB solver: synthetic cases with known answers, and the saved fits.

The synthetic cases are built from a known rotation, scale and translation, so the
solver must recover them exactly without noise and fit no worse than the truth with
it. The saved fits are the nine in ``fixtures/nominal_transform.json``; there the
solver must do at least as well as the pyto fit it replaced (``pyto_oracle``).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fibsem.correlation import solver
from fibsem.correlation.geometry import branch_of, rotation_angle_deg

from . import pyto_oracle
from .test_nominal_transform import ENTRIES, IDS, _isotropic_coords, _nominal


def _case(seed: int, n: int = 8, noise: float = 0.0, depth: float = 0.3):
    """Fiducials through a random rotation, scale and offset: ``(fm, fib, R, s, t)``."""
    rng = np.random.default_rng(seed)
    rotation = Rotation.random(random_state=seed).as_matrix()
    scale = float(rng.uniform(0.3, 3.0))
    offset = rng.uniform(-500, 500, 2)
    fm = rng.uniform(-300, 300, (n, 3))
    fm[:, 2] *= depth
    fib = scale * fm @ rotation[:2].T + offset + rng.normal(0, noise, (n, 2))
    return fm, fib, rotation, scale, offset


# ── the model ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(12))
def test_unseeded_recovers_an_exact_transform(seed):
    fm, fib, rotation, scale, offset = _case(seed)
    fit = solver.fit_unseeded(fm, fib)
    assert fit.rms < 1e-6
    assert rotation_angle_deg(fit.rotation, rotation) < 1e-4
    assert fit.scale == pytest.approx(scale, rel=1e-6)
    assert np.allclose(fit.translation, [*offset, 0.0], atol=1e-5)
    assert np.allclose(fit.project(fm)[:, :2], fib, atol=1e-5)


@pytest.mark.parametrize("seed", range(12))
def test_unseeded_fits_noise_no_worse_than_the_truth(seed):
    fm, fib, rotation, scale, offset = _case(seed, n=6, noise=4.0)
    fit = solver.fit_unseeded(fm, fib)
    truth = solver.fit_rigid(fm, fib, [(rotation, scale)])
    assert fit.rms <= truth.rms + 1e-9


def test_a_seed_converges_from_tens_of_degrees_out():
    fm, fib, rotation, scale, _ = _case(3)
    off = Rotation.from_euler("xyz", [25, -20, 15], degrees=True).as_matrix()
    fit = solver.fit_from_seed(fm, fib, off @ rotation, scale * 1.3)
    assert rotation_angle_deg(fit.rotation, rotation) < 1e-4
    assert fit.scale == pytest.approx(scale, rel=1e-6)


def test_coplanar_fiducials_leave_two_equal_branches():
    """Without depth the mirror projects identically: the fit cannot tell them apart."""
    fm, fib, rotation, scale, _ = _case(5, depth=0.0)
    mirror = rotation.copy()
    mirror[:2, 2] *= -1
    mirror[2] = np.cross(mirror[0], mirror[1])
    a = solver.fit_from_seed(fm, fib, rotation, scale)
    b = solver.fit_from_seed(fm, fib, mirror, scale)
    assert a.rms < 1e-6 and b.rms < 1e-6
    assert rotation_angle_deg(a.rotation, b.rotation) > 1.0


def test_the_same_picks_give_the_same_fit():
    fm, fib, *_ = _case(9, noise=3.0)
    a = solver.fit_unseeded(fm, fib)
    b = solver.fit_unseeded(fm, fib)
    assert np.array_equal(a.rotation, b.rotation) and a.scale == b.scale


def test_rms_is_the_root_mean_squared_distance():
    fm, fib, rotation, scale, _ = _case(2, noise=2.0)
    fit = solver.fit_rigid(fm, fib, [(rotation, scale)])
    distances = np.linalg.norm(fit.project(fm)[:, :2] - fib, axis=1)
    assert fit.rms == pytest.approx(np.sqrt(np.mean(distances**2)))
    assert np.allclose(fit.residuals, fit.project(fm)[:, :2] - fib)


def test_translation_about_a_centre_keeps_the_projection():
    fm, fib, *_ = _case(4)
    fit = solver.fit_unseeded(fm, fib)
    centre = np.array([100.0, -40.0, 20.0])
    t = fit.translation_about(centre)
    moved = fit.scale * (fm - centre) @ fit.rotation.T + fit.scale * centre + t
    assert np.allclose(moved, fit.project(fm))


def test_too_few_pairs_or_a_bad_seed_is_refused():
    fm, fib, *_ = _case(1, n=2)
    with pytest.raises(ValueError, match="at least 3"):
        solver.fit_unseeded(fm, fib)
    fm, fib, *_ = _case(1)
    with pytest.raises(ValueError, match="3x3"):
        solver.fit_rigid(fm, fib, [(np.zeros(3), 1.0)])


def test_eulers_are_the_saved_convention():
    """``R = Rz(psi) Rx(theta) Rz(phi)``, theta in [0, 180]: what runs have saved."""
    from fibsem.correlation.pyto.rigid_3d import Rigid3D

    for seed in range(20):
        rotation = Rotation.random(random_state=seed).as_matrix()
        angles = solver.eulers_deg(rotation)
        assert 0.0 <= angles[1] <= 180.0
        assert np.allclose(
            angles, np.degrees(Rigid3D.extract_euler(rotation, mode="x", ret="one"))
        )
        assert np.allclose(solver.rotation_from_eulers_deg(angles), rotation)


def test_the_cover_is_the_sixty_rotations_of_the_icosahedron():
    cover = solver._COVER
    assert len(cover) == 60
    keys = {tuple(np.round(r, 6).ravel()) for r in cover}
    assert len(keys) == 60
    assert all(tuple(np.round(a @ b, 6).ravel()) in keys for a in cover for b in cover)


# ── against the pyto fits it replaced, on the saved fiducials ─────────────


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_unseeded_matches_pyto_on_the_saved_fits(entry):
    nominal = _nominal(entry)
    fib, fm = _isotropic_coords(entry, nominal)
    fit = solver.fit_unseeded(fm, fib)
    r_old, _, rms_old = pyto_oracle.unseeded(fm, fib)
    assert fit.rms <= rms_old + 1e-3
    assert branch_of(fit.rotation, nominal)[0] == branch_of(r_old, nominal)[0]


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
@pytest.mark.parametrize("mirrored", [False, True], ids=["nominal", "mirror"])
def test_seeded_matches_pyto_on_the_saved_fits(entry, mirrored):
    nominal = _nominal(entry)
    seed = nominal.mirrored() if mirrored else nominal
    fib, fm = _isotropic_coords(entry, nominal)
    fit = solver.fit_from_seed(fm, fib, seed.rotation, seed.scale)
    r_old, _, rms_old = pyto_oracle.from_seed(fm, fib, seed.rotation, seed.scale)
    assert fit.rms <= rms_old + 1e-3
    assert rotation_angle_deg(fit.rotation, r_old) < 0.5
