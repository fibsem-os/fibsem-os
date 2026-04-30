"""Pytest tests for fibsem.correlation.refractive_index."""

import pytest

from fibsem.correlation.refractive_index import (
    SliceScalingFactorLUT,
    ZetaParams,
    get_lut,
    lookup_zeta,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lut() -> SliceScalingFactorLUT:
    return SliceScalingFactorLUT()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lut_loads(lut: SliceScalingFactorLUT) -> None:
    """LUT constructs without error and exposes grid arrays."""
    assert lut is not None


def test_known_values(lut: SliceScalingFactorLUT) -> None:
    """Spot-check against values computed in the reference notebook."""
    # Meteor bead case: tilt=15, depth=4, NA=0.8, n2=1.29, wavelength=0.58 → ~1.45
    zeta_bead = lut.lookup(tilt_deg=15, depth_um=4, NA=0.8, n2=1.29, wavelength_um=0.58)
    assert abs(zeta_bead - 1.45) < 0.01, f"Expected ~1.45, got {zeta_bead:.4f}"

    # Cell targeting case: tilt=15, depth=4, NA=0.7, n2=1.35, wavelength=0.515 → ~1.47
    zeta_cell = lut.lookup(tilt_deg=15, depth_um=4, NA=0.7, n2=1.35, wavelength_um=0.515)
    assert abs(zeta_cell - 1.47) < 0.01, f"Expected ~1.47, got {zeta_cell:.4f}"


def test_lookup_zeta_convenience(lut: SliceScalingFactorLUT) -> None:
    """Module-level lookup_zeta matches the class method."""
    zeta_class = lut.lookup(15, 4.0, 0.8, 1.34, 0.570)
    zeta_fn = lookup_zeta(15, 4.0, 0.8, 1.34, 0.570)
    assert abs(zeta_class - zeta_fn) < 1e-9


def test_lookup_params(lut: SliceScalingFactorLUT) -> None:
    """lookup_params is consistent with lookup."""
    params = ZetaParams(tilt_deg=10, depth_um=5.0, NA=0.7, n2=1.30, wavelength_um=0.520)
    assert lut.lookup_params(params) == lut.lookup(
        params.tilt_deg, params.depth_um, params.NA, params.n2, params.wavelength_um
    )


def test_grid_bounds_exposed(lut: SliceScalingFactorLUT) -> None:
    """Grid axis arrays are exposed with correct min/max."""
    assert lut.tilts.min() == pytest.approx(0.0)
    assert lut.tilts.max() == pytest.approx(30.0)
    assert lut.nas.min() == pytest.approx(0.2)
    assert lut.nas.max() == pytest.approx(0.9)
    assert lut.depths.min() == pytest.approx(0.5)
    assert lut.depths.max() == pytest.approx(14.5)
    assert lut.n2s.min() == pytest.approx(1.22)
    assert lut.n2s.max() == pytest.approx(1.46)
    assert lut.wavelengths.min() == pytest.approx(0.42)
    assert lut.wavelengths.max() == pytest.approx(0.72)


def test_bounds_error(lut: SliceScalingFactorLUT) -> None:
    """Querying outside the grid bounds raises ValueError."""
    with pytest.raises(ValueError):
        lut.lookup(tilt_deg=999, depth_um=4, NA=0.8, n2=1.34, wavelength_um=0.570)


def test_zero_tilt(lut: SliceScalingFactorLUT) -> None:
    """tilt=0 is a valid grid edge and returns a sensible float."""
    zeta = lut.lookup(tilt_deg=0, depth_um=5.0, NA=0.8, n2=1.30, wavelength_um=0.520)
    assert isinstance(zeta, float)
    assert 1.0 < zeta < 3.0


def test_singleton_get_lut() -> None:
    """get_lut() returns the same object on repeated calls."""
    a = get_lut()
    b = get_lut()
    assert a is b
