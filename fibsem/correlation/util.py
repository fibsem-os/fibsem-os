from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

from fibsem.correlation.fit_diagnostics import FitDiagnostic

# migrated from tdct.beadPos and refactored


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


## Gaussian 2D fit from http://scipy.github.io/old-wiki/pages/Cookbook/FittingData
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a Gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: (
        height
        * np.exp(
            -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2
        )
    )


#### INTERPOLATION ####

INTERPOLATION_METHODS = ["linear", "cubic"]


def interpolate_z_stack(
    image: np.ndarray, pixelsize_in: float, pixelsize_out: float, method: str = "linear"
) -> np.ndarray:
    """Interpolate a 3D image array along the z-axis using scipy's zoom function."""

    # check for multi-channel images, only single channel images are supported currently
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(
                f"Muli-channel images are not supported, got an image of shape {image.shape}, expected (1, Z, Y, X) or (Z, Y, X)"
            )

        # remove the channel dimension
        logging.info(f"Squeezing channel dimension from image of shape {image.shape}")
        image = np.squeeze(image, axis=0)

    if image.ndim != 3:
        raise ValueError(f"image must be a ZYX array, but got {image.ndim}")

    if method not in INTERPOLATION_METHODS:
        raise ValueError(
            f"interpolation method  must be in {INTERPOLATION_METHODS} got {method}"
        )

    # interpolate the image
    return scipy_interpolation(
        image_3d=image,
        original_z_size=pixelsize_in,
        target_z_size=pixelsize_out,
        method=method,
    )


def scipy_interpolation(
    image_3d: np.ndarray,
    original_z_size: float,
    target_z_size: float,
    method: str = "linear",
) -> np.ndarray:
    """
    Interpolate a 3D image array along the z-axis.

    Parameters:
    -----------
    image_3d : ndarray
        Input 3D image array with shape (Z, Y, X)
    original_z_size : float
        Original pixel size in z-axis (e.g., 10 for 10µm)
    target_z_size : float
        Desired pixel size in z-axis (e.g., 5 for 5µm)
    method : str
        Interpolation method ('linear' or 'cubic')

    Returns:
    --------
    ndarray
        Interpolated 3D image with adjusted z-axis resolution

    The name is historical: this used to be scipy's ``zoom``, which is a general
    n-dimensional resampler and walks every axis of the volume even though only
    z changes here (9 s per 33x2048x2048 channel at order 1, 76 s at order 3).
    Only z is resampled, so each output slice is a weighted blend of a few
    source slices, done one slice at a time below in a fraction of the time.
    The slice count, the sample grid and the rounding are zoom's own, so the
    linear result is identical to zoom's to the last bit. The cubic result is
    too on real data; on synthetic data with exact half-way values it can
    differ by one grey level, because the four taps are summed in a different
    order and a tie lands on the other side. The tests hold both paths to
    zoom.
    """
    if method not in INTERPOLATION_METHODS:
        method = "linear"
    new_nz = _zoom_slice_count(image_3d.shape[0], original_z_size / target_z_size)
    if method == "linear":
        return _linear_z_resample(image_3d, new_nz)
    return _cubic_z_resample(image_3d, new_nz)


def _zoom_slice_count(nz: int, scale_factor: float) -> int:
    """How many slices scipy's ``zoom`` makes of ``nz`` at ``scale_factor``."""
    return int(round(nz * scale_factor))


def _zoom_grid(nz: int, new_nz: int) -> np.ndarray:
    """Where scipy's ``zoom`` samples the source for each output slice:
    ``k * (nz - 1) / (new_nz - 1)``, the end slices landing on the end slices."""
    return np.linspace(0.0, nz - 1, new_nz)


def z_resample_scale(nz: int, new_nz: int) -> float:
    """The factor that moves a slice index of an ``nz``-slice volume onto its
    ``new_nz``-slice resampling: source slice ``s`` lands at ``s * scale``.

    The inverse of :func:`_zoom_grid`, whose end slices land on the end slices,
    so the factor is ``(new_nz - 1) / (nz - 1)``, not ``new_nz / nz`` (FIB-1238).
    """
    if nz < 2 or new_nz < 2:
        raise ValueError(
            f"a z resampling needs at least two slices each side, got {nz} -> {new_nz}"
        )
    return (new_nz - 1) / (nz - 1)


def _store_slice(out: np.ndarray, k: int, blend: np.ndarray) -> None:
    """``out[k] = blend`` with zoom's conversion for an integer ``out``: rounded
    half away from zero, in double precision, and clamped to the dtype's range
    (a cubic can overshoot the source values). Round-to-even in float32
    differed from zoom by one grey level on a few hundred voxels per frame."""
    if np.issubdtype(out.dtype, np.unsignedinteger):
        blend += 0.5
        np.floor(blend, out=blend)
    elif np.issubdtype(out.dtype, np.signedinteger):
        blend[:] = np.where(blend >= 0, np.floor(blend + 0.5), np.ceil(blend - 0.5))
    if np.issubdtype(out.dtype, np.integer):
        info = np.iinfo(out.dtype)
        np.clip(blend, info.min, info.max, out=blend)
    out[k] = blend


def _linear_z_resample(image_3d: np.ndarray, new_nz: int) -> np.ndarray:
    """``image_3d`` (ZYX) resampled to ``new_nz`` slices, linearly along z.

    Each output slice is the blend of the two source slices around it, one
    slice at a time, so the working memory is two float64 frames rather than
    a second volume. At order 1 zoom's spline prefilter is the identity and its
    boundary mode is never reached, so this is all zoom does.
    """
    nz = image_3d.shape[0]
    if new_nz < 1:
        return image_3d[:0]
    if nz == 1 or new_nz == 1:
        # nothing to blend between (or a single output plane, which zoom takes
        # from the start of the stack): the first slice, repeated
        return np.repeat(image_3d[:1], new_nz, axis=0)

    source = _zoom_grid(nz, new_nz)
    below = np.floor(source).astype(int)
    above = np.minimum(below + 1, nz - 1)
    weight = source - below

    out = np.empty((new_nz,) + image_3d.shape[1:], dtype=image_3d.dtype)
    # two frames, reused: the blend is memory-bound, so no per-slice temporaries
    blend = np.empty(image_3d.shape[1:], dtype=np.float64)
    other = np.empty_like(blend)
    for k in range(new_nz):
        if weight[k] == 0.0:
            out[k] = image_3d[below[k]]  # an exact source slice, untouched
            continue
        np.multiply(image_3d[below[k]], 1.0 - weight[k], out=blend, casting="unsafe")
        np.multiply(image_3d[above[k]], weight[k], out=other, casting="unsafe")
        blend += other
        _store_slice(out, k, blend)
    return out


def _cubic_z_resample(image_3d: np.ndarray, new_nz: int) -> np.ndarray:
    """``image_3d`` (ZYX) resampled to ``new_nz`` slices with a cubic B-spline
    along z, which is what zoom at order 3 computes.

    The spline prefilter runs along z only (zoom's runs along every axis, but on
    an axis that is not resampled it is undone exactly by the evaluation at the
    knots). Each output slice is then a four-tap blend of coefficient slices,
    with zoom's ``reflect`` boundary for the taps past either end. The
    coefficients are one float64 volume the size of the channel; the blends
    reuse one frame.
    """
    nz = image_3d.shape[0]
    if new_nz < 1:
        return image_3d[:0]
    if nz == 1 or new_nz == 1:
        return np.repeat(image_3d[:1], new_nz, axis=0)

    coefficients = ndimage.spline_filter1d(
        image_3d, order=3, axis=0, mode="reflect", output=np.float64
    )
    source = _zoom_grid(nz, new_nz)
    out = np.empty((new_nz,) + image_3d.shape[1:], dtype=image_3d.dtype)
    blend = np.empty(image_3d.shape[1:], dtype=np.float64)
    for k in range(new_nz):
        base = int(np.floor(source[k]))
        t = source[k] - base
        weights = (
            (1.0 - t) ** 3 / 6.0,
            (3.0 * t**3 - 6.0 * t**2 + 4.0) / 6.0,
            (-3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0) / 6.0,
            t**3 / 6.0,
        )
        blend[:] = 0.0
        for tap, weight in enumerate(weights):
            index = base - 1 + tap
            if index < 0:  # 'reflect': -1 -> 0, -2 -> 1
                index = -index - 1
            elif index >= nz:  # nz -> nz - 1, nz + 1 -> nz - 2
                index = 2 * nz - index - 1
            blend += weight * coefficients[index]
        _store_slice(out, k, blend)
    return out


#### multi-channel interpolation ####

# Channels interpolate side by side; the work is memory-bound numpy, which
# releases the GIL, and four threads take the METEOR stack from 3.0 s to 1.2 s
# (linear). A cubic channel holds a float64 coefficient volume, ~1 GB for a
# 33x2048x2048 channel, so fewer of those may be in flight at once.
_CHANNEL_THREADS = {"linear": 4, "cubic": 2}


def multi_channel_interpolation(
    image: np.ndarray,
    pixelsize_in: float,
    pixelsize_out: float,
    method: str = "linear",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Interpolate a multi-channel z-stack (CZYX) along the z-axis.

    Args:
        image: 4D numpy array (CZYX)
        pixelsize_in: original pixel size in z-axis
        pixelsize_out: desired pixel size in z-axis
        method: one of ``INTERPOLATION_METHODS``
        progress_callback: optional ``fn(channels_done, channels_total)`` invoked
            before the first channel and as each one completes, on the calling
            thread. Kept UI-agnostic (a plain callable, not a Qt object) so the
            algorithm stays testable; a worker passes a callback that emits its
            own progress signal.

    Returns:
        interpolated: 4D numpy array (CZYX) with adjusted z-axis resolution
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = image.shape[0]
    if progress_callback is not None:
        progress_callback(0, n)
    if n == 0:
        return np.empty((0,) + image.shape[1:], dtype=image.dtype)

    def one(i: int) -> np.ndarray:
        logging.info(f"Interpolating channel {i + 1}/{n}")
        return interpolate_z_stack(
            image=image[i],
            pixelsize_in=pixelsize_in,
            pixelsize_out=pixelsize_out,
            method=method,
        )

    # Filled channel by channel as each completes: gathering the channels in a
    # list and stacking them at the end held two copies of the volume at once.
    interpolated = None
    workers = min(n, _CHANNEL_THREADS.get(method, 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, i): i for i in range(n)}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if interpolated is None:
                interpolated = np.empty((n,) + result.shape, dtype=result.dtype)
            interpolated[futures[future]] = result
            if progress_callback is not None:
                progress_callback(done, n)
    return interpolated


def interpolate_fm_volume(
    fm_image,
    target_z_size_m: float,
    method: str = "linear",
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """Resample an FM volume along z, returning a NEW ``FluorescenceImage``.

    Only the z-axis is resampled (XY is untouched). The returned image carries
    metadata consistent with the resampled data: ``pixel_size_z`` and
    ``z_positions`` are recomputed for the new slice count.

    The output slice count is rounded to an integer, and the end slices land on
    the end slices, so the achieved z-scale is :func:`z_resample_scale` —
    ``(new_nz - 1) / (old_nz - 1)`` — not the nominal ``pixel_size_z / target``.
    The effective z pixel size is derived from that actual scale, keeping data,
    metadata, and any caller-side coordinate rescale exactly consistent. Callers
    that must move z-bearing coordinates should scale by
    ``z_resample_scale(old_nz, new_nz)``.

    Args:
        fm_image: source ``FluorescenceImage`` (CZYX, with ``pixel_size_z`` set)
        target_z_size_m: desired z pixel size in metres (e.g. ``pixel_size_x`` for
            an isotropic volume)
        method: one of ``INTERPOLATION_METHODS``
        progress_callback: forwarded to :func:`multi_channel_interpolation`

    Raises:
        ValueError: for a single-plane volume (no ``pixel_size_z``), a
            non-positive target, a non-CZYX array, or a target so coarse the
            volume would resample to fewer than two slices.
    """
    import copy

    from fibsem.fm.structures import FluorescenceImage  # lazy: correlation -> fm

    data = fm_image.data
    if data.ndim != 4:
        raise ValueError(f"expected a CZYX volume, got shape {data.shape}")

    meta = fm_image.metadata
    z_in = getattr(meta, "pixel_size_z", None)
    if not z_in:
        raise ValueError("volume has no z step (single plane) — nothing to interpolate")
    if not target_z_size_m or target_z_size_m <= 0:
        raise ValueError(f"target z pixel size must be positive, got {target_z_size_m}")

    old_nz = data.shape[1]
    interpolated = multi_channel_interpolation(
        data,
        pixelsize_in=z_in,
        pixelsize_out=target_z_size_m,
        method=method,
        progress_callback=progress_callback,
    )
    new_nz = interpolated.shape[1]

    # Effective z pixel size from the ACTUAL resampled grid, not the nominal
    # target — so physical depth (z_index * pixel_size_z) is preserved when the
    # caller rescales coordinates by the same scale.
    new_meta = copy.deepcopy(meta)
    new_meta.pixel_size_z = z_in / z_resample_scale(old_nz, new_nz)

    # z_positions is a per-plane objective ramp; resample it to the new count so
    # its length scales with the volume (convention-agnostic — np.interp on the
    # existing ramp preserves whatever ordering it had).
    positions = getattr(meta, "z_positions", None)
    if positions:
        old = np.asarray(positions, dtype=float)
        new_len = max(1, round(len(old) * new_nz / old_nz))
        if new_len == 1 or len(old) == 1:
            new_meta.z_positions = [float(old[0])] * new_len
        else:
            src = np.linspace(0.0, len(old) - 1, new_len)
            new_meta.z_positions = np.interp(src, np.arange(len(old)), old).tolist()

    # Resampled, but still this file's volume. The correlation UI names an image
    # by its filepath (FIB-509), so dropping it here would blank the header and
    # record a null filename for every correlation run after an interpolation.
    return FluorescenceImage(
        data=interpolated,
        metadata=new_meta,
        # getattr, because this function is exercised with minimal stand-ins that
        # carry only the fields it reads.
        filepath=getattr(fm_image, "filepath", None),
    )


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
    roi = img[:, yc - cutout : yc + cutout, xc - cutout : xc + cutout]
    intensity = np.mean(roi, axis=(1, 2))
    popt_z, _ = fit_guass1d(intensity)
    zi = popt_z[1]

    slc_init = img[int(zi), yc - cutout : yc + cutout, xc - cutout : xc + cutout]
    err = None
    if use_xy_fitting:
        try:
            popt_xy, _ = fit_gauss_2d_mod(slc_init, show=False)
            xopt, yopt = popt_xy[1], popt_xy[2]
        except Exception as e:
            logging.warning(f"Error in initial XY fit: {e}")
            xopt, yopt = cutout, cutout  # fallback to center of cutout if fit fails
            err = e

        if not (0 <= xopt < 2 * cutout and 0 <= yopt < 2 * cutout):
            logging.warning(
                f"XY fit out of bounds, returning original x, y. xopt: {xopt}, yopt: {yopt}, cutout: {cutout}"
            )
            xopt, yopt = cutout, cutout

        xi = xopt + xc - cutout
        yi = yopt + yc - cutout
    else:
        xi, yi = x, y
        xopt, yopt = cutout, cutout  # fit result is the cutout centre for plotting

    # --- confirmation-friendly diagnostic: z (left) + XY hero (right) ---
    z_axis = np.arange(len(intensity))
    diagnostic = FitDiagnostic(
        title="Fluorescence target fit",
        roi_xy=slc_init,
        input_xy=(cutout + (x - xc), cutout + (y - yc)),
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


def apply_refractive_index_correction(
    initial_poi: Tuple[float, float],
    surface_coord: Tuple[float, float],
    correction_factor: float,
) -> Tuple[float, float]:
    """Apply a refractive index correction to the point of interest (POI) coordinates.
    Initial point of interest and surface coordinate are both in image pixels
    Args:
        initial_poi: initial point of interest coordinates (x, y)
        surface_coord: surface coordinate (x, y)
        correction_factor: correction factor for the refractive index
    Returns:
        corrected_poi: corrected point of interest coordinates (x, y)"""

    from fibsem.correlation.structures import scale_about_surface

    corrected_y = scale_about_surface(
        initial_poi[1], surface_coord[1], correction_factor
    )
    logging.info(
        f"Correction Factor: {correction_factor}, "
        f"Depth: {initial_poi[1] - surface_coord[1]}, "
        f"Corrected Depth: {corrected_y - surface_coord[1]}"
    )
    return (initial_poi[0], corrected_y)


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
