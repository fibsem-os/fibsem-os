"""Locating a feature in an image by fitting a Gaussian to it.

Each fitter refines a click to the centre of the feature under it -- a burnt hole
in a FIB image, a hole in a reflection z-stack, a fluorescent target in an FM
z-stack -- and returns the refined position with a :class:`FitDiagnostic`, the data
needed to draw the fit's diagnostic figure. Nothing here draws; rendering is
:func:`fibsem.correlation.fit_diagnostics.plot_fit_diagnostic`.

Migrated from tdct.beadPos and refactored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit


@dataclass
class FitDiagnostic:
    """Everything needed to render a fit's diagnostic figure — no matplotlib.

    The XY panel is always present; the z panel is present only when
    :attr:`z_signal` is set (the FIB hole fit is XY-only). Marker positions are
    in ROI pixel coordinates; :attr:`input_xy` carries the sub-pixel click so the
    input marker lands exactly where the user clicked (FIB-282).
    """

    title: str
    roi_xy: np.ndarray  # image shown in the XY panel
    input_xy: Tuple[float, float]  # red input marker (ROI coords)
    fitted_xy: Optional[Tuple[float, float]] = None  # green marker, or None → not drawn
    xy_title: str = "XY"
    xy_message: Optional[str] = None  # shown when there is no fitted marker
    # z panel — left None for an XY-only (FIB) fit
    z_axis: Optional[np.ndarray] = None
    z_signal: Optional[np.ndarray] = None
    z_fit: Optional[np.ndarray] = None
    z_input: Optional[float] = None
    z_fitted: Optional[float] = None
    z_inverted: bool = False  # reflection: signal is inverted

    @property
    def has_z(self) -> bool:
        """True for the FM fits (z + XY); False for the XY-only FIB fit."""
        return self.z_signal is not None


def fit_guass1d(
    data: np.ndarray, show: bool = False, ax=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 1D Gaussian to the data
    Args:
        data: 1D numpy array
        show: show the plot of the fit (for debugging)
        ax: optional matplotlib axes to plot on
    Returns:
        popt: optimal parameters
        pcov: covariance matrix
    """

    data = data - data.min()  # shift data to 0
    p0 = [data.max(), data.argmax(), 1]  # initial guess
    x = np.arange(len(data))  # x values
    popt, pcov = curve_fit(gauss1d, x, data, p0=p0)

    # plot the data and the fit
    if ax is not None:
        ax.plot(data, label="Data")
        ax.plot(gauss1d(x, *popt), label="Gaussian 1D fit")
        ax.legend()
    elif show:
        import matplotlib.pyplot as plt

        plt.title("1D Gaussian fit")
        plt.plot(data, label="Data")
        plt.plot(gauss1d(x, *popt), label="Gaussian 1D fit")
        plt.legend()
        plt.show()

    return popt, pcov


def gauss1d(x: np.ndarray, A: float, mu: float, sigma: float) -> float:
    """Gaussian 1D fit
    Args:
        x: x values
        A: magnitude
        mu: offset on x axis
        sigma: width
    Returns:
        y: gaussian values
    """
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


##### 2D GAUSSIAN FIT #####


def gauss2d_offset(coords, a, x0, y0, sigma_x, sigma_y, offset):
    x, y = coords  # unpack the coordinates
    return (
        a
        * np.exp(
            -(((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2))
        )
        + offset
    )


def fit_gauss_2d_mod(
    slc: np.ndarray,
    show: bool = False,
    center: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a dark 2D Gaussian to ``slc``. The fit starts from ``center``
    (x, y in ``slc``'s frame), or from the middle of ``slc`` when the caller
    cut the window around the feature; a window clipped by the image's edge
    is not centred on it, and a start in the wrong place lands the fit on
    nothing (FIB-980)."""
    y_indices, x_indices = np.indices(slc.shape)
    # Flatten the coordinate arrays and the slice data for fitting.
    x_data = x_indices.ravel()
    y_data = y_indices.ravel()
    slc_data = slc.ravel()

    # define initial guess
    if center is None:
        center = (slc.shape[1] / 2, slc.shape[0] / 2)
    p0 = [
        float(np.min(slc)) - float(np.max(slc)),
        center[0],
        center[1],
        1,
        1,
        np.max(slc),
    ]

    popt, pcov = curve_fit(
        gauss2d_offset, (x_data, y_data), slc_data, p0=p0, maxfev=10000
    )
    return popt, pcov


# A spot burn in the FIB image is a dark spot a few pixels across. A fitted
# Gaussian outside these widths, or bright, is a fit to something else: the
# trench, the bright halo, a gradient (FIB-1216).
_BURN_SIGMA_PX = (0.5, 12.0)


def _burn_shaped(popt: np.ndarray, shape: Tuple[int, int]) -> bool:
    """Whether a dark-Gaussian fit landed on something burn-like in its window."""
    amplitude, xo, yo, sx, sy = popt[:5]
    lo, hi = _BURN_SIGMA_PX
    return bool(
        np.all(np.isfinite(popt[:5]))
        and amplitude < 0
        and 0 <= xo < shape[1]
        and 0 <= yo < shape[0]
        and lo <= abs(sx) <= hi
        and lo <= abs(sy) <= hi
    )


def hole_fitting_FIB(
    img: np.ndarray,
    x: float,
    y: float,
    cutout: int = 15,
):
    """Refine selection of hole in FIB image.

    The fit starts at the click, as it always has, and that answer stands when
    it is burn-shaped (:func:`_burn_shaped`). From a click more than ~4 px off
    the burn it is not -- the Gaussian starts 1 px wide at the click and has
    nothing to pull on -- so the fit starts again at the window's darkest point
    after smoothing, which finds a burn up to the window's edge. When neither
    finds a burn the fit fails (:class:`FitOutsideWindow`) rather than handing
    back the click as an unchanged fit (FIB-1216).

    Args:
        img: 2D numpy array (Y,X)
        x,y initial coordinates from the user click (may be sub-pixel)
        cutout: size of the cutout around the point in x,y
    Returns:
        xr, yr: refined x, y coordinates
        diagnostic: FitDiagnostic for the (XY-only) diagnostic figure
    Raises:
        FitOutsideWindow: no burn-shaped dark spot in the region searched.
    """
    # The click may be sub-pixel; round for integer slicing but keep the
    # fraction so the input marker is drawn exactly where the user clicked
    # (otherwise a no-change fit shows the markers up to ~1px apart).
    xi, yi = int(round(x)), int(round(y))
    # Cut out a box around the point, clamped to the image. A plain slice with
    # a negative start wraps to the far end of the array: a click within the
    # cutout of the top or left edge gave an empty window or one from the wrong
    # side of the image. The offsets are kept so the fit comes back absolute.
    height, width = img.shape
    y0, y1 = max(0, yi - cutout), min(height, yi + cutout)
    x0, x1 = max(0, xi - cutout), min(width, xi + cutout)
    roi = img[y0:y1, x0:x1]

    def _fit(center):
        try:
            popt, _ = fit_gauss_2d_mod(roi, show=False, center=center)
        except Exception as e:  # maxfev, an empty or flat window
            logging.debug(f"FIB hole fit from {center}: {e}")
            return None
        return popt if _burn_shaped(popt, roi.shape) else None

    # the click, in the clamped window's frame (the window's middle when the
    # window is not clipped)
    popt = _fit(center=(xi - x0, yi - y0))
    if popt is None and roi.size:
        smooth = ndimage.gaussian_filter(np.asarray(roi, dtype=float), sigma=2)
        dark_y, dark_x = np.unravel_index(np.argmin(smooth), smooth.shape)
        popt = _fit(center=(float(dark_x), float(dark_y)))
    if popt is None:
        raise FitOutsideWindow(
            f"no burn-shaped dark spot in the region searched: x {x0}-{x1 - 1}, "
            f"y {y0}-{y1 - 1}"
        )
    xopt, yopt = popt[1], popt[2]

    # back to the coordinates of the original image; _burn_shaped kept the fit
    # inside the window, so it is inside the image
    xr = xopt + x0
    yr = yopt + y0

    # --- Diagnostic (XY only — no z for the FIB image) ---
    diagnostic = FitDiagnostic(
        title="FIB hole fit",
        roi_xy=roi,
        input_xy=(x - x0, y - y0),  # the sub-pixel click, in the window's frame
        fitted_xy=(xopt, yopt),
        xy_title="XY",
    )
    return xr, yr, diagnostic


def target_fitting_fluorescence(
    img: np.ndarray,
    x: float,
    y: float,
    z: int,
    cutout: int = 5,
    use_xy_fitting: bool = False,
) -> tuple:
    """Refine selection of target in fluorescence image.
    Args:
        img: 3D numpy array (Z,Y,X), interpolated to isotropic pixel size
        x,y,z initial coordinates from the user click (x, y may be sub-pixel)
        cutout: size of the cutout around the point in x,y. z uses 3x this value
        use_xy_fitting: whether to use xy fitting or just return the input x, y
    Returns:
        xr, yr, zr: refined x, y, z coordinates
        diagnostic: FitDiagnostic for the z + XY diagnostic figure
    """
    # round the (possibly sub-pixel) click for slicing; keep the fraction for
    # the input marker so it lands exactly where the user clicked (FIB-282).
    xc, yc = int(round(x)), int(round(y))
    # Clamped to the image: a plain slice with a negative start wraps to the
    # far edge, so a click near the top or left edge gave an empty window and
    # the fit raised on it (FIB-1220, as FIB-980 for the reflection fit). The
    # offsets keep the fitted x, y absolute.
    height, width = img.shape[1:]
    y0, y1 = max(0, yc - cutout), min(height, yc + cutout)
    x0, x1 = max(0, xc - cutout), min(width, xc + cutout)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"({x:.0f}, {y:.0f}) is outside the image")
    roi = img[:, y0:y1, x0:x1]
    intensity = np.mean(roi, axis=(1, 2))
    popt_z, _ = fit_guass1d(intensity)
    zi = popt_z[1]

    slc_init = img[int(zi), y0:y1, x0:x1]
    err = None
    if use_xy_fitting:
        try:
            popt_xy, _ = fit_gauss_2d_mod(slc_init, show=False)
            xopt, yopt = popt_xy[1], popt_xy[2]
        except Exception as e:
            logging.warning(f"Error in initial XY fit: {e}")
            xopt, yopt = xc - x0, yc - y0  # fallback to the click if fit fails
            err = e

        if not (0 <= xopt < x1 - x0 and 0 <= yopt < y1 - y0):
            logging.warning(
                f"XY fit out of bounds, returning original x, y. xopt: {xopt}, yopt: {yopt}, cutout: {cutout}"
            )
            xopt, yopt = xc - x0, yc - y0

        xi = xopt + x0
        yi = yopt + y0
    else:
        xi, yi = x, y
        xopt, yopt = xc - x0, yc - y0  # fit result is the click for plotting

    # --- confirmation-friendly diagnostic: z (left) + XY hero (right) ---
    z_axis = np.arange(len(intensity))
    diagnostic = FitDiagnostic(
        title="Fluorescence target fit",
        roi_xy=slc_init,
        input_xy=(x - x0, y - y0),
        fitted_xy=(xopt, yopt) if (use_xy_fitting and err is None) else None,
        xy_title=f"XY  @ z = {zi:.1f}",
        xy_message="XY fit failed" if err is not None else None,
        z_axis=z_axis,
        z_signal=intensity,
        z_fit=gauss1d(z_axis, *popt_z) + intensity.min(),
        z_input=z,
        z_fitted=zi,
    )
    return xi, yi, zi, diagnostic

    # vals = np.array(zvalues).astype(np.float32)
    # ch_idx = np.argmax(vals[:, 0])

    # logging.info(f"solution found: Channel Index: {ch_idx}: xyz: {xyz_vals[ch_idx]}")
    # return ch_idx, xyz_vals[ch_idx]


# Fewer planes than this around the click and the z fit is a line through two
# points; the fit refuses and says so rather than returning a number.
_MIN_Z_PLANES = 3


class FitOutsideWindow(ValueError):
    """A fit whose answer lies outside the region it searched.

    A Gaussian fitted to a profile with no clear feature can put its centre
    anywhere -- measured: 980 px away in xy, 2.5 mm deep in z -- and that
    number used to come back as the fit, so the confirmation offered it as a
    success (FIB-1219). Raised instead; the caller keeps the point where it
    was and reports the fit as failed.
    """


def hole_fitting_reflection(da, x, y, z, cutout) -> tuple:
    from scipy.ndimage import gaussian_filter

    # round the (possibly sub-pixel) click for slicing; keep the fraction for
    # the input marker so a no-change fit shows the markers coincident.
    xi, yi = int(round(x)), int(round(y))
    zmin = 10
    zmax = 5
    nz, height, width = da.shape

    # Both windows are clamped to the stack and the image. A plain slice with
    # a negative start wraps to the end of the array: a click below plane 10,
    # or within the cutout of the top or left edge, gave an empty window and
    # the fit raised on it (FIB-980). The offsets are kept so the fitted z and
    # xy still come back absolute.
    zmin1 = max(0, int(z) - zmin)
    zmax1 = min(nz, int(z) + zmax)
    if zmax1 - zmin1 < _MIN_Z_PLANES:
        raise ValueError(
            f"only {zmax1 - zmin1} plane(s) around z = {int(z)} to fit through; "
            f"the z fit needs at least {_MIN_Z_PLANES}"
        )
    y0, y1 = max(0, yi - cutout), min(height, yi + cutout + 1)
    x0, x1 = max(0, xi - cutout), min(width, xi + cutout + 1)

    roi = da[zmin1:zmax1, y0:y1, x0:x1]
    intensity = np.mean(roi, axis=(1, 2))
    intensity = intensity.max() - intensity  # invert: the hole is dark

    popt, _ = fit_guass1d(intensity)
    zopt = popt[1]
    zreal = zopt + zmin1  # back to absolute z
    # The planes searched span zmin1 - 0.5 .. zmax1 - 0.5; a peak beyond them
    # is an extrapolation from a profile with no dip in it, not a fit.
    if not np.isfinite(zreal) or not (zmin1 - 0.5 <= zreal <= zmax1 - 0.5):
        raise FitOutsideWindow(
            f"z fit fell outside the planes searched: {zreal:.1f}, "
            f"searched {zmin1}-{zmax1 - 1}"
        )

    # xy fitting on the fitted z-slice
    xy_cutout = 15
    z_fitted = min(max(round(zreal), 0), nz - 1)
    fy0, fy1 = max(0, yi - xy_cutout), min(height, yi + xy_cutout + 1)
    fx0, fx1 = max(0, xi - xy_cutout), min(width, xi + xy_cutout + 1)
    roi_fitted = da[z_fitted, fy0:fy1, fx0:fx1]
    popt_xy, _ = fit_gauss_2d_mod(roi_fitted, center=(xi - fx0, yi - fy0))
    xopt, yopt = popt_xy[1], popt_xy[2]
    if not (
        np.isfinite(xopt)
        and np.isfinite(yopt)
        and 0 <= xopt < roi_fitted.shape[1]
        and 0 <= yopt < roi_fitted.shape[0]
    ):
        raise FitOutsideWindow(
            f"xy fit fell outside the region searched: "
            f"({xopt + fx0:.0f}, {yopt + fy0:.0f}), searched x {fx0}-{fx1 - 1}, "
            f"y {fy0}-{fy1 - 1}"
        )
    xopt_real = xopt + fx0
    yopt_real = yopt + fy0

    # --- confirmation-friendly diagnostic ---
    # Lead with the "did it land on the feature?" view (ROI + input/fitted
    # markers); a compact z panel answers "did z land right?". The old figure
    # had a raw-profile panel with six reference lines and a second z panel in a
    # different (cutout-relative) frame with the same labels — dropped.

    # z: signal + gaussian fit (grey). The hole is dark, so the signal is
    # inverted (z_inverted) — the peak is the hole.
    z_axis = np.arange(zmin1, zmin1 + len(intensity))
    gauss_curve = gauss1d(np.arange(len(intensity)), *popt) + intensity.min()

    diagnostic = FitDiagnostic(
        title="Reflection hole fit",
        roi_xy=gaussian_filter(roi_fitted, sigma=1),
        input_xy=(x - fx0, y - fy0),  # the sub-pixel click, in the window's frame
        # A 2D fit outside the window raised above (FIB-1219), so this is in it.
        fitted_xy=(xopt, yopt),
        xy_title=f"XY  @ z = {zreal:.1f}",
        z_axis=z_axis,
        z_signal=intensity,
        z_fit=gauss_curve,
        z_input=z,
        z_fitted=zreal,
        z_inverted=True,
    )
    return xopt_real, yopt_real, zreal, diagnostic
