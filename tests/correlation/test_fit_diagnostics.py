"""Tests for the fit diagnostics — the compute → data → render split (FIB-281).

Each fit function computes a refined coordinate and returns a ``FitDiagnostic``
(data, no matplotlib); ``plot_fit_diagnostic`` renders it. Tests assert on the
data (panel presence, marker positions) rather than on figure internals, plus a
couple of render checks.

  * FIB hole            -> XY only        (has_z False)
  * reflection hole     -> z + XY, inverted signal
  * fluorescence target -> z + XY
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from fibsem.correlation.fit_diagnostics import FitDiagnostic, plot_fit_diagnostic
from fibsem.correlation.util import (
    hole_fitting_FIB,
    hole_fitting_reflection,
    target_fitting_fluorescence,
)


def _blob_2d(size: int, cx: float, cy: float, sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


def _refl_vol(cx: float, nz: int = 21, size: int = 60) -> np.ndarray:
    vol = np.full((nz, size, size), 200.0, dtype=np.float32)
    for zi in range(nz):  # a dark hole, deepest at z=10
        vol[zi] -= 160.0 * np.exp(-((zi - 10) ** 2) / (2 * 2.0**2)) * _blob_2d(size, cx, cx)
    return vol


def _fluor_vol(cx: float, nz: int = 21, size: int = 40) -> np.ndarray:
    vol = np.zeros((nz, size, size), dtype=np.float32)
    for zi in range(nz):  # a bright target, brightest at z=10
        vol[zi] += 200.0 * np.exp(-((zi - 10) ** 2) / (2 * 2.0**2)) * _blob_2d(size, cx, cx)
    return vol


# --- the fit fns return a FitDiagnostic (data, no matplotlib) --------------


def test_fib_hole_fit_returns_xy_only_diagnostic():
    img = (200.0 - 160.0 * _blob_2d(40, 20, 20)).astype(np.float32)
    xr, yr, d = hole_fitting_FIB(img, x=20, y=20, cutout=15)

    assert isinstance(d, FitDiagnostic)
    assert np.isfinite(xr) and np.isfinite(yr)
    assert not d.has_z              # FIB has no z panel
    assert d.fitted_xy is not None  # a clean fit has a fitted marker


def test_reflection_fit_returns_inverted_z_and_xy_diagnostic():
    xr, yr, zr, d = hole_fitting_reflection(_refl_vol(30), 30, 30, 10, 2)

    assert isinstance(d, FitDiagnostic)
    assert d.has_z and d.z_inverted  # reflection: z panel, signal inverted
    assert 8 <= zr <= 12
    assert d.z_fitted == pytest.approx(zr)


def test_fluorescence_fit_returns_z_and_xy_diagnostic():
    xr, yr, zr, d = target_fitting_fluorescence(_fluor_vol(20), 20, 20, 10, 5)

    assert isinstance(d, FitDiagnostic)
    assert d.has_z and not d.z_inverted
    assert 8 <= zr <= 12
    assert d.fitted_xy is None  # default has no XY fitting -> no fitted marker


# --- FIB-282: the input marker tracks the sub-pixel click ------------------


def test_fib_input_marker_tracks_subpixel_click():
    # The input marker must sit at the sub-pixel click, not the integer ROI
    # centre — otherwise a no-change fit draws input vs fitted ~1px apart.
    cx = 20.6
    img = (200.0 - 160.0 * _blob_2d(40, cx, cx)).astype(np.float32)
    xr, yr, d = hole_fitting_FIB(img, cx, cx, cutout=15)

    expected = 15 + (cx - round(cx))  # cutout + fractional click, not the centre
    assert d.input_xy == pytest.approx((expected, expected), abs=1e-6)
    # the hole sits on the click, so the fit shouldn't move it and the markers
    # should land on top of each other.
    assert xr == pytest.approx(cx, abs=0.3)
    assert d.fitted_xy == pytest.approx(d.input_xy, abs=0.25)


def test_reflection_input_marker_tracks_subpixel_click():
    cx = 30.6
    _, _, _, d = hole_fitting_reflection(_refl_vol(cx), cx, cx, 10, 2)

    expected = 15 + (cx - round(cx))  # xy_cutout is 15
    assert d.input_xy == pytest.approx((expected, expected), abs=1e-6)
    assert d.fitted_xy == pytest.approx(d.input_xy, abs=0.5)


def test_fluorescence_input_marker_tracks_subpixel_click():
    cx = 20.6
    _, _, _, d = target_fitting_fluorescence(_fluor_vol(cx), cx, cx, z=10, cutout=5)

    expected = 5 + (cx - round(cx))  # cutout is 5
    assert d.input_xy == pytest.approx((expected, expected), abs=1e-6)


def test_fluorescence_xy_fit_marker_coincides_at_subpixel_click():
    cx = 20.6
    _, _, _, d = target_fitting_fluorescence(
        _fluor_vol(cx), cx, cx, z=10, cutout=5, use_xy_fitting=True
    )
    assert d.fitted_xy is not None
    assert d.fitted_xy == pytest.approx(d.input_xy, abs=0.5)


# --- plot_fit_diagnostic renders the data ----------------------------------


def _xy_only():
    return FitDiagnostic(title="t", roi_xy=np.zeros((10, 10)), input_xy=(5.0, 5.0))


def _with_z():
    return FitDiagnostic(
        title="t", roi_xy=np.zeros((10, 10)), input_xy=(5.0, 5.0),
        z_axis=np.arange(5), z_signal=np.arange(5.0), z_fit=np.arange(5.0),
        z_input=2.0, z_fitted=2.5,
    )


def test_plot_fit_diagnostic_panel_counts():
    assert len(plot_fit_diagnostic(_xy_only()).axes) == 1   # XY only
    assert len(plot_fit_diagnostic(_with_z()).axes) == 2    # z + XY


def test_plot_fit_diagnostic_dark_vs_light_facecolor():
    from matplotlib.colors import to_hex

    assert to_hex(plot_fit_diagnostic(_with_z(), dark=True).get_facecolor()) == "#1e2124"
    assert to_hex(plot_fit_diagnostic(_with_z(), dark=False).get_facecolor()) == "#ffffff"
