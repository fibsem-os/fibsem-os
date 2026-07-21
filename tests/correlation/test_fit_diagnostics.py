"""Smoke tests for the fit diagnostic figures (FIB-260).

The change is presentational (confirmation-friendly figures), so these lock the
figure layout — each fit runs on a clean synthetic feature, returns finite
coordinates, and produces the expected number of panels:

  * FIB hole            -> 1 panel  (XY only)
  * reflection hole     -> 2 panels (z + XY)
  * fluorescence target -> 2 panels (z + XY)
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from fibsem.correlation.util import (
    hole_fitting_FIB,
    hole_fitting_reflection,
    target_fitting_fluorescence,
)


def _blob_2d(size: int, cx: int, cy: int, sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


def test_fib_hole_fit_single_panel():
    # a dark hole on a bright background
    img = (200.0 - 160.0 * _blob_2d(40, 20, 20)).astype(np.float32)
    xr, yr, fig = hole_fitting_FIB(img, x=20, y=20, cutout=15)

    assert np.isfinite(xr) and np.isfinite(yr)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1  # XY only — no z for a FIB image


def test_reflection_hole_fit_z_and_xy_panels():
    nz = 21
    vol = np.full((nz, 60, 60), 200.0, dtype=np.float32)
    for zi in range(nz):
        depth = np.exp(-((zi - 10) ** 2) / (2 * 2.0**2))  # hole deepest at z=10
        vol[zi] -= 160.0 * depth * _blob_2d(60, 30, 30)

    xr, yr, zr, fig = hole_fitting_reflection(vol, x=30, y=30, z=10, cutout=2)

    assert np.isfinite(xr) and np.isfinite(yr) and np.isfinite(zr)
    assert 8 <= zr <= 12  # near the true hole z
    assert len(fig.axes) == 2  # consolidated: one z panel + XY hero


def test_fluorescence_target_fit_z_and_xy_panels():
    nz = 21
    vol = np.zeros((nz, 40, 40), dtype=np.float32)
    for zi in range(nz):
        bright = np.exp(-((zi - 10) ** 2) / (2 * 2.0**2))  # brightest at z=10
        vol[zi] += 200.0 * bright * _blob_2d(40, 20, 20)

    xr, yr, zr, fig = target_fitting_fluorescence(vol, x=20, y=20, z=10, cutout=5)

    assert np.isfinite(zr)
    assert 8 <= zr <= 12
    assert len(fig.axes) == 2
