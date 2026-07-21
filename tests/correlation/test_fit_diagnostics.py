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
import pytest
from matplotlib.figure import Figure

from fibsem.correlation.util import (
    hole_fitting_FIB,
    hole_fitting_reflection,
    target_fitting_fluorescence,
)


def _blob_2d(size: int, cx: float, cy: float, sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


def _marker(ax, label: str):
    """(x, y) of the diagnostic marker with the given legend label."""
    for line in ax.lines:
        if line.get_label() == label:
            return float(line.get_xdata()[0]), float(line.get_ydata()[0])
    raise AssertionError(f"no marker labelled {label!r}")


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


def test_fib_input_marker_tracks_subpixel_click():
    # FIB-282: the input marker must sit at the sub-pixel click, not the integer
    # ROI centre — otherwise a no-change fit draws input vs fitted ~1px apart.
    cx = 20.6
    img = (200.0 - 160.0 * _blob_2d(40, cx, cx)).astype(np.float32)
    xr, yr, fig = hole_fitting_FIB(img, cx, cx, cutout=15)

    ix, iy = _marker(fig.axes[0], "input")
    expected = 15 + (cx - round(cx))  # cutout + fractional click, not the centre
    assert ix == pytest.approx(expected, abs=1e-6)
    assert iy == pytest.approx(expected, abs=1e-6)

    # the hole sits exactly on the click, so the fit shouldn't move it and the
    # input/fitted markers should land on top of each other.
    assert xr == pytest.approx(cx, abs=0.3)
    fx, fy = _marker(fig.axes[0], "fitted")
    assert abs(ix - fx) < 0.25 and abs(iy - fy) < 0.25


def test_reflection_input_marker_tracks_subpixel_click():
    cx = 30.6
    nz = 21
    vol = np.full((nz, 60, 60), 200.0, dtype=np.float32)
    for zi in range(nz):
        depth = np.exp(-((zi - 10) ** 2) / (2 * 2.0**2))
        vol[zi] -= 160.0 * depth * _blob_2d(60, cx, cx)

    _, _, _, fig = hole_fitting_reflection(vol, cx, cx, z=10, cutout=2)

    ax_xy = fig.axes[1]  # z panel is axes[0]; the XY hero is axes[1]
    ix, iy = _marker(ax_xy, "input")
    expected = 15 + (cx - round(cx))  # xy_cutout is 15
    assert ix == pytest.approx(expected, abs=1e-6)
    assert iy == pytest.approx(expected, abs=1e-6)

    fx, fy = _marker(ax_xy, "fitted")
    assert abs(ix - fx) < 0.5 and abs(iy - fy) < 0.5


def _fluorescence_vol(cx, nz=21, size=40):
    vol = np.zeros((nz, size, size), dtype=np.float32)
    for zi in range(nz):
        bright = np.exp(-((zi - 10) ** 2) / (2 * 2.0**2))
        vol[zi] += 200.0 * bright * _blob_2d(size, cx, cx)
    return vol


def test_fluorescence_input_marker_tracks_subpixel_click():
    # Default path (no XY fitting): only the input marker is drawn, and it must
    # still sit at the sub-pixel click rather than the integer ROI centre.
    cx = 20.6
    _, _, _, fig = target_fitting_fluorescence(
        _fluorescence_vol(cx), cx, cx, z=10, cutout=5
    )
    ix, iy = _marker(fig.axes[1], "input")  # XY hero is axes[1]
    expected = 5 + (cx - round(cx))  # cutout is 5
    assert ix == pytest.approx(expected, abs=1e-6)
    assert iy == pytest.approx(expected, abs=1e-6)


def test_fluorescence_xy_fit_markers_coincide_at_subpixel_click():
    # With XY fitting on, the fitted marker should land on the input marker when
    # the target sits exactly on the (sub-pixel) click.
    cx = 20.6
    _, _, _, fig = target_fitting_fluorescence(
        _fluorescence_vol(cx), cx, cx, z=10, cutout=5, use_xy_fitting=True
    )
    ax_xy = fig.axes[1]
    ix, iy = _marker(ax_xy, "input")
    fx, fy = _marker(ax_xy, "fitted")
    assert abs(ix - fx) < 0.5 and abs(iy - fy) < 0.5
