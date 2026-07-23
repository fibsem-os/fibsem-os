from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit, leastsq

from fibsem.correlation.fit_diagnostics import FitDiagnostic

# migrated from tdct.beadPos and refactored

### GAUSSIAN FITTING ###
def get_z_gauss(image: np.ndarray, x: int, y: int, show: bool = False) -> Tuple[float, int, float]:
    """Get the best fitting z-value for a 2D point in an ZYX image using a 1D Gaussian fit:
    Args:
        x: x coordinate
        y: y coordinate
        image: 3D numpy array (Z,Y,X)
        show: show the plot of the fit (for debugging)
    Returns:
        z: z coordinate (index)
    """

    # check that img is ndim=3
    if image.ndim != 3:
        raise ValueError(f"img must be a ZYX array, got {image.ndim}")

    # check that x, y are inside the image shape
    if x >= image.shape[-1] or y >= image.shape[-2]:
        raise ValueError(
            f"x and y must be within the image shape, x: {x}, y: {y}, {image.shape}"
        )

    # NOTE: round is important, don't directly cast to int
    if not isinstance(x, int):
        x = round(x)
    if not isinstance(y, int):
        y = round(y)

    # fit the z data for the given x,y
    poptZ, pcov = fit_guass1d(image[:, y, x], show=show)

    return np.array(poptZ)  # zval, zidx, zsigma


def fit_guass1d(data: np.ndarray, show: bool = False, ax=None) -> Tuple[np.ndarray, np.ndarray]:
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

def fit_gauss1d_mod(data: np.ndarray, show: bool = False, ax=None) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 1D Gaussian to the data. Modified for negative hole images
    Args:
        data: 1D numpy array
        show: show the plot of the fit (for debugging)
        ax: optional matplotlib axes to plot on
    Returns:
        popt: optimal parameters
        pcov: covariance matrix
    """

    z=data.copy()
    offset=np.max(z)
    z-=offset # shift data to 0

    popt, pcov = fit_guass1d(z, show=show, ax=ax)

    # popt, pcov = curve_fit(gauss1d_offset, xz, z, p0, maxfev=10000)

    # plot the data and the fit
    # if show:
    #     import matplotlib.pyplot as plt
    #     plt.title("1D Gaussian fit")
    #     plt.plot(data, label="Data")
    #     plt.plot(gauss1d_offset(xz, *popt), label="Gaussian 1D fit")
    #     plt.legend()
    #     plt.show()

    return popt, pcov

def gauss1d_offset(x, a, x0, sigma, offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

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
    return a * np.exp(-(((x - x0)**2) / (2 * sigma_x**2) + ((y - y0)**2) / (2 * sigma_y**2))) + offset

def fit_gauss_2d_mod(slc: np.ndarray, show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    y_indices, x_indices = np.indices(slc.shape)
    # Flatten the coordinate arrays and the slice data for fitting.
    x_data = x_indices.ravel()
    y_data = y_indices.ravel()
    slc_data = slc.ravel()
    
    # define initial guess
    p0 = [float(np.min(slc)) - float(np.max(slc)), slc.shape[1] / 2, slc.shape[0] / 2, 1, 1, np.max(slc)]

    popt, pcov = curve_fit(gauss2d_offset, (x_data, y_data), slc_data, p0=p0, maxfev=10000)
    return popt, pcov

## Gaussian 2D fit from http://scipy.github.io/old-wiki/pages/Cookbook/FittingData
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a Gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the Gaussian parameters of a 2D distribution by calculating its
    moments"""
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Returns (height, x, y, width_x, width_y)
    the Gaussian parameters of a 2D distribution found by a fit"""

    def errorfunction(p):
        return np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)

    try:
        params = moments(data)
    except ValueError:
        return None
    p, success = leastsq(errorfunction, params)
    if np.isnan(p).any():
        return None

    return p

def extract_image_patch(img: np.ndarray, x: int, y:int, z: int, cutout: int) -> np.ndarray:
    # Get image dimensions
    z_max, height, width = img.shape
    
    # Calculate patch bounds
    x_min = int(x - cutout)
    x_max = int(x + cutout)
    y_min = int(y - cutout)
    y_max = int(y + cutout)
    z = int(round(z))

    # Check if patch is within bounds
    is_valid = (
        x_min >= cutout and
        x_max < width and
        y_min >= cutout and
        y_max < height and
        0 <= z < z_max
    )

    if not is_valid:
        logging.warning("Point(s) too close to edge or out of bounds.")
        return None

    # Extract and return the patch
    return np.copy(img[z, y_min:y_max, x_min:x_max])

def threshold_image(data: np.ndarray, threshold_val: float):
    """
    Zero out values below a threshold relative to the data range.
    
    Args:
        data: numpy array of image data
        threshold_percent: float between 0-1, normalized threshold value
    """
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    
    # Calculate threshold value
    threshold_value = data_max - (data_range * threshold_val)
    
    # Zero out values below threshold 
    data[data < threshold_value] = 0
    
    return data

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
    return scipy_interpolation(image_3d=image, original_z_size=pixelsize_in, target_z_size=pixelsize_out, method=method)

def scipy_interpolation(
    image_3d: np.ndarray, original_z_size: float, target_z_size: float, method: str = "linear"
) -> np.ndarray:
    """
    Fast interpolation of a 3D image array along the z-axis using scipy's zoom function.

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
    """
    # Calculate the scaling factor
    scale_factor = original_z_size / target_z_size

    # Create zoom factors for each dimension
    # Only scale the z-axis (first dimension)
    zoom_factors = (scale_factor, 1, 1)

    # Determine the interpolation order
    if method not in INTERPOLATION_METHODS:
        method = "linear"
    order = 1 if method == "linear" else 3

    # Perform the interpolation using scipy's zoom function
    # mode='reflect' to handle edge cases
    # prefilter=True for better quality
    interpolated = ndimage.zoom(
        image_3d, 
        zoom_factors, 
        order=order, 
        mode="reflect", 
        prefilter=True
    )

    return interpolated


#### multi-channel interpolation ####

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
            before the first channel and after each one. Kept UI-agnostic (a plain
            callable, not a Qt object) so the algorithm stays testable; a worker
            passes a callback that emits its own progress signal.

    Returns:
        interpolated: 4D numpy array (CZYX) with adjusted z-axis resolution
    """
    n = image.shape[0]
    if progress_callback is not None:
        progress_callback(0, n)

    ch_interpolated = []
    for i, channel in enumerate(image):
        logging.info(f"Interpolating channel {i + 1}/{n}")
        ch_interpolated.append(
            interpolate_z_stack(
                image=channel,
                pixelsize_in=pixelsize_in,
                pixelsize_out=pixelsize_out,
                method=method,
            )
        )
        if progress_callback is not None:
            progress_callback(i + 1, n)
    return np.array(ch_interpolated)


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

    scipy's ``zoom`` rounds the output slice count to an integer, so the achieved
    z-scale is ``new_nz / old_nz`` — not the nominal ``pixel_size_z / target``.
    The effective z pixel size is derived from that actual ratio, keeping data,
    metadata, and any caller-side coordinate rescale exactly consistent. Callers
    that must move z-bearing coordinates should scale by ``new_nz / old_nz``,
    read from the returned image's shape versus the input's.

    Args:
        fm_image: source ``FluorescenceImage`` (CZYX, with ``pixel_size_z`` set)
        target_z_size_m: desired z pixel size in metres (e.g. ``pixel_size_x`` for
            an isotropic volume)
        method: one of ``INTERPOLATION_METHODS``
        progress_callback: forwarded to :func:`multi_channel_interpolation`

    Raises:
        ValueError: for a single-plane volume (no ``pixel_size_z``), a
            non-positive target, or a non-CZYX array.
    """
    import copy

    from fibsem.fm.structures import FluorescenceImage  # lazy: correlation -> fm

    data = fm_image.data
    if data.ndim != 4:
        raise ValueError(f"expected a CZYX volume, got shape {data.shape}")

    meta = fm_image.metadata
    z_in = getattr(meta, "pixel_size_z", None)
    if not z_in:
        raise ValueError(
            "volume has no z step (single plane) — nothing to interpolate"
        )
    if not target_z_size_m or target_z_size_m <= 0:
        raise ValueError(
            f"target z pixel size must be positive, got {target_z_size_m}"
        )

    old_nz = data.shape[1]
    interpolated = multi_channel_interpolation(
        data,
        pixelsize_in=z_in,
        pixelsize_out=target_z_size_m,
        method=method,
        progress_callback=progress_callback,
    )
    new_nz = interpolated.shape[1]
    if new_nz < 1:
        raise ValueError("interpolation produced an empty volume")

    # Effective z pixel size from the ACTUAL resampled slice count, not the
    # nominal target — so physical depth (z_index * pixel_size_z) is preserved
    # when the caller rescales coordinates by new_nz / old_nz.
    new_meta = copy.deepcopy(meta)
    new_meta.pixel_size_z = z_in * old_nz / new_nz

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
            new_meta.z_positions = np.interp(
                src, np.arange(len(old)), old
            ).tolist()

    return FluorescenceImage(data=interpolated, metadata=new_meta)


def multi_channel_get_z_guass(image: np.ndarray, x: int, y: int, show: bool = False) -> List[float]:
    """Get the best fitting z-value for a 2D point in an multi-channel ZYX image using a 1D Gaussian fit:
    Args:
        x: x coordinate
        y: y coordinate
        image: 4D numpy array (CZYX)
        show: show the plot of the fit (for debugging)
    Returns:
        z: z coordinate (index)
    """
    # shortcut for single channel images
    if image.ndim == 3:
        return get_z_gauss(image, x, y, show=show)

    z_values = []
    for channel in image:
        try:
            # optimisation can fail, fallback to nothing
            zvals = get_z_gauss(channel, x, y, show=show)
        except Exception as e:
            logging.warning(f"Error in channel: {e}")
            zvals = [0, 0, 0]
        z_values.append(zvals)

    # get the channel with the maximum z-value (zval, zidx, zsigma)
    vals = np.array(z_values)
    ch_idx = np.argmax(vals[:, 0])

    return vals[ch_idx] # zval, zidx, zsigma

def hole_fitting_RL(img: np.ndarray,
    x: int,
    y: int,
    z: int,
    cutout: int = 15,
    small_cutout: int = 5,
    apply_threshold: bool = False,
    threshold_val: float = 0,
    iterations: int = 5,
    show: bool = False,
):
    """Refine selection of hole in reflected light image.
    Args:
        img: 3D numpy array (Z,Y,X), interpolated to isotropic pixel size
        x,y,z initial coordinates from the user click
        cutout: size of the cutout around the point in x,y. z uses 3x this value
        small_cutout: size of the cutout for the refined fit. z uses 3x this value
        apply_threshold: apply thresholding to the image
        threshold_val: does nothing
        iterations: does nothing
        show: show the diagnostic figure
    Returns:
        xr, yr, zr: refined x, y, z coordinates
        fig: matplotlib figure with diagnostic plots (or None if show=False)
    """
    import matplotlib.pyplot as plt

    # --- Diagnostic figure ---
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Hole Fitting RL")

    # --- Initial Z fit ---
    roi = img[z-3:z+4, y-cutout:y+cutout+1, x-cutout:x+cutout+1]
    intensity = np.mean(roi, axis=(1, 2))
    err = None
    try:
        popt_z, _ = fit_gauss1d_mod(intensity, ax=axes[0])
        popt_z, pcov = fit_guass1d(intensity, show=True, ax=axes[0] )

        zi = popt_z[1]
    except Exception as e:
        logging.warning(f"Error in initial Z fit: {e}")
        zi = z # fallback to input z if fit fails
        err = e

    if err:
        axes[0].set_title("Initial Z fit failed")
    else:
        axes[0].set_title("Initial Z fit")
        axes[0].axvline(popt_z[1], color='r', label='Fit')
    axes[0].axvline(z, color='k', linestyle='--', label=f'Input Z: {z}')
    axes[0].legend()

    # --- Initial XY fit ---
    slc_init = img[int(zi), y-cutout:y+cutout+1, x-cutout:x+cutout+1]
    err = None
    try:
        popt_xy, _ = fit_gauss_2d_mod(slc_init, show=False)
        xopt, yopt = popt_xy[1], popt_xy[2]
    except Exception as e:
        logging.warning(f"Error in initial XY fit: {e}")
        xopt, yopt = cutout, cutout # fallback to center of cutout if fit fails
        err = e

    # check if xopt, yopt are within the cutout bounds, if not return the original x, y
    if not (0 <= xopt < 2 * cutout and 0 <= yopt < 2 * cutout):
        logging.warning(f"XY fit out of bounds, returning original x, y. xopt: {xopt}, yopt: {yopt}, cutout: {cutout}")
        xopt, yopt = cutout, cutout

    # convert back to original image coordinates
    xi = xopt + x - cutout
    yi = yopt + y - cutout

    im = axes[1].imshow(slc_init, cmap='gray')
    axes[1].scatter(cutout, cutout, color='yellow', label='Input')
    if err:
        axes[1].set_title("Initial XY fit failed")
    else:
        axes[1].set_title(f"Initial XY fit (z={zi:.2f})")
        axes[1].scatter(xopt, yopt, color='r', label='Fit')
    fig.colorbar(im, ax=axes[1], label='Intensity')
    axes[1].legend()
    return xi, yi, zi, fig

    # --- Refined Z fit ---
    zmax = img.shape[0]
    z0 = max(z - small_cutout * 6, 0)
    z1 = min(z + small_cutout * 6, zmax)
    ROI_ref = img[z0:z1,
                  yi-small_cutout:yi+small_cutout,
                  xi-small_cutout:xi+small_cutout]
    I_ref = np.mean(ROI_ref, axis=(1, 2))
    popt_zr, _ = fit_gauss1d_mod(I_ref, ax=axes[1, 0])
    zr = popt_zr[1] + z0
    axes[1, 0].set_title("Refined Z fit")
    axes[1, 0].axvline(popt_zr[1], color='r', label='Refined Fit')
    axes[1, 0].axvline(z - z0, color='k', linestyle='--', label='Input Z')
    axes[1, 0].legend()

    # --- Refined XY fit ---
    slc_ref = img[int(zr),
                  yi-small_cutout:yi+small_cutout,
                  xi-small_cutout:xi+small_cutout]
    popt_xyr, _ = fit_gauss_2d_mod(slc_ref, show=False)
    xr = popt_xyr[1] + xi - small_cutout
    yr = popt_xyr[2] + yi - small_cutout
    axes[1, 1].set_title(f"Refined XY fit (z={int(zr)})")
    axes[1, 1].imshow(slc_ref)
    axes[1, 1].scatter(popt_xyr[1], popt_xyr[2], color='r', label='Fit')
    axes[1, 1].legend()

    plt.tight_layout()

    if show:
        plt.show()

    return xr, yr, zr, fig

def hole_fitting_FIB(img: np.ndarray,
    x: float,
    y: float,
    cutout: int = 15,
):
    """Refine selection of hole in FIB image.
    Args:
        img: 2D numpy array (Y,X)
        x,y initial coordinates from the user click (may be sub-pixel)
        cutout: size of the cutout around the point in x,y
    Returns:
        xr, yr: refined x, y coordinates
        diagnostic: FitDiagnostic for the (XY-only) diagnostic figure
    """
    # The click may be sub-pixel; round for integer slicing but keep the
    # fraction so the input marker is drawn exactly where the user clicked
    # (otherwise a no-change fit shows the markers up to ~1px apart).
    xi, yi = int(round(x)), int(round(y))
    # cut out a box around the point
    roi = img[yi-cutout:yi+cutout, xi-cutout:xi+cutout]
    # fit a 2D gaussian to estimate the hole position
    err = None
    try:
        popt, _ = fit_gauss_2d_mod(roi, show=False)
        xopt, yopt = popt[1], popt[2]
    except Exception as e:
        logging.warning(f"Error in XY fit: {e}")
        xopt, yopt = cutout, cutout # fallback to center of cutout if fit fails
        err = e

    if not (0 <= xopt < 2 * cutout and 0 <= yopt < 2 * cutout):
        logging.warning(f"XY fit out of bounds, returning original x, y. xopt: {xopt}, yopt: {yopt}, cutout: {cutout}")
        xopt, yopt = cutout, cutout

    # get the refined positions in the coordinates of the original image
    xr = xopt + xi - cutout
    yr = yopt + yi - cutout

    # clip the coordinates to the image bounds
    xr = np.clip(xr, cutout, img.shape[1] - cutout)
    yr = np.clip(yr, cutout, img.shape[0] - cutout)

    # --- Diagnostic (XY only — no z for the FIB image) ---
    diagnostic = FitDiagnostic(
        title="FIB hole fit",
        roi_xy=roi,
        input_xy=(cutout + (x - xi), cutout + (y - yi)),
        fitted_xy=None if err is not None else (xopt, yopt),
        xy_title="XY",
        xy_message=None if err is None else "fit failed — using input",
    )
    return xr, yr, diagnostic

def target_fitting_fluorescence(img: np.ndarray,
                                x: float, y: float, z: int,
                                cutout: int = 5,
                                use_xy_fitting: bool = False) -> tuple:
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
    roi = img[:, yc - cutout:yc + cutout, xc - cutout:xc + cutout]
    intensity = np.mean(roi, axis=(1, 2))
    popt_z, _ = fit_guass1d(intensity)
    zi = popt_z[1]

    slc_init = img[int(zi), yc - cutout:yc + cutout, xc - cutout:xc + cutout]
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
            logging.warning(f"XY fit out of bounds, returning original x, y. xopt: {xopt}, yopt: {yopt}, cutout: {cutout}")
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


def zyx_targeting(
    img: np.ndarray,
    x: int,
    y: int,
    cutout: int = 15,
    apply_threshold: bool = False,
    threshold_val: float = 0.1,
    iterations: int = 5,
):
    """Automated ZYX targeting for a 3D image stack. Optimizes the selection of x,y,z coordinates
    for a given 2D point in a 3D image stack using an iterative gaussian fitting approach.
    Args:

        img: 3D numpy array (Z,Y,X)
        x: initial x coordinate
        y: initial y coordinate
        cutout: size of the cutout around the point
        apply_threshold: apply thresholding to the image
        threshold_val: threshold value for thresholding
        iterations: number of iterations
    Returns:
        x, y, (zval, zidx, zsigma): optimized x, y, z coordinates and z-value (max, index, sigma)
    """
    logging.info(f"Starting zyx targeting for {img.shape}, Initial x: {x}, y: {y}")

    # get the initial z position (Note: this can fail, returns None)
    zval, zidx, zsigma = multi_channel_get_z_guass(image=img, x=x, y=y)

    assert img.ndim == 3, "Image must be 3D"

    logging.info(f"Initial z: {zidx}, zval: {zval}, zsigma: {zsigma} for {x}, {y}")

    for i in range(iterations):
        data = extract_image_patch(img, x, y, zidx, cutout)
        if data is None:
            break  # Or handle error case differently

        if apply_threshold:  # apply threshold on normalized data
            data = threshold_image(data, threshold_val)

        # fit a 2d guassian to the 3d cutout
        poptXY = fitgaussian(data)
        if poptXY is None:
            break

        (height, xopt, yopt, width_x, width_y) = poptXY

        # x and y are switched when applying the offset
        x = x - cutout + yopt
        y = y - cutout + xopt
        width, height = img.shape[-1], img.shape[-2]
        if not (0 <= x < width and 0 <= y < height):
            break

        # fit a 1d guassian to the z stack
        zval, zidx, zsigma = get_z_gauss(img, x=x, y=y)
        logging.debug(
            f"iteration: {i}, x: {x}, y: {y}, z: {zidx}, zval: {zval}, zsigma: {zsigma}"
        )

    # TODO: check that xyz are in image bounds, if not return original x, y, z

    return x, y, (zval, zidx, zsigma)

def multi_channel_zyx_targeting(
    image: np.ndarray,
    xinit: int,
    yinit: int,
    zinit: int,
    apply_threshold: bool = False,
    threshold_val: float = 0.1,
    cutout: int = 15,
    iterations: int = 5,
    method: str = "gaussian",
) -> Tuple[int, Tuple[int, int, int]]:
    """ZYX targeting for multi-channel images
    Args:
        image: 4D numpy array (CZYX)
        xinit: initial x coordinate
        yinit: initial y coordinate
        apply_threshold: apply thresholding to the image
        threshold_val: threshold value for thresholding
        cutout: size of the cutout around the point
        iterations: number of iterations
    Returns:
        ch_idx: channel index with the best z-value
        x, y, z: x, y, z coordinates of the best z-value in the best channel
    """

    # shortcut for single channel images
    if image.ndim == 3:
        x1, y1, (zv, z1, zs) = zyx_targeting(
            image,
            xinit,
            yinit,
            cutout=cutout,
            apply_threshold=apply_threshold,
            threshold_val=threshold_val,
            iterations=iterations,
        )
        return 0, (x1, y1, z1)

    if image.ndim != 4:
        raise ValueError(f"image must be a 4D array (CZYX), got {image.ndim}")

    zvalues = []
    xyz_vals = []

    for i in range(image.shape[0]):
        ch_image = image[i]
        try:
            if method not in ["gaussian", "hole"]:
                raise ValueError(f"method must be 'gaussian' or 'hole', got {method}")
            if method == "gaussian":
                x1, y1, (zv, z1, zs) = zyx_targeting(
                    ch_image,
                    xinit,
                    yinit,
                    cutout=cutout,
                    apply_threshold=apply_threshold,
                    threshold_val=threshold_val,
                    iterations=iterations,
                )
            elif method == "hole":
                x1, y1, zv, _ = hole_fitting_RL(
                    ch_image,
                    xinit,
                    yinit,
                    zinit,
                    cutout=cutout,
                    apply_threshold=apply_threshold,
                    threshold_val=threshold_val,
                    iterations=iterations,
                )
        except Exception as e:
            logging.error(f"an error occured during channel {i}: {e}")
            x1, y1, zv, z1, zs = xinit, yinit, 0, None, None
        # zvalues.append((zv, z1, zs))
        # xyz_vals.append((x1, y1, z1))
        break

    return 0, (x1, y1, zv)

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

    corrected_y = scale_about_surface(initial_poi[1], surface_coord[1], correction_factor)
    logging.info(
        f"Correction Factor: {correction_factor}, "
        f"Depth: {initial_poi[1] - surface_coord[1]}, "
        f"Corrected Depth: {corrected_y - surface_coord[1]}"
    )
    return (initial_poi[0], corrected_y)



def hole_fitting_RL_old(img: np.ndarray,
    x: int,
    y: int,
    z: int,
    cutout: int = 11,
    small_cutout: int = 5,
    apply_threshold: bool = False,
    threshold_val: float = 0,
    iterations: int = 5,
    show: bool = False,
):
    import matplotlib.pyplot as plt
    """refine selection of hole in reflected light image
    Args:
        img: 3D numpy array (Z,Y,X), interpolated to isotropic pixel size
        x,y,z initial coordinates from the user click
        cutout: size of the cutout around the point in x,y. z uses 3x this value
        small_cutout: size of the cutout for the refined fit. z uses 3x this value
        apply_threshold: apply thresholding to the image
        threshold_val: does nothing
        iterations: does nothing
    Returns:
        xr, yr, zr: refined x, y, z coordinates
    """
    # cut out the box
    ROI=img[z-cutout*3:z+cutout*3,y-cutout:y+cutout,x-cutout:x+cutout]
    # # fit a gaussian to estimate the in focus plane
    # I=np.mean(ROI,axis=(1,2))
    # popt,popcov=fit_gauss1d_mod(I,show=False)
    # zi=int(popt[1])+z-cutout*3
    # if show:
    #     plt.figure()
    #     plt.plot(I)
    #     plt.scatter(popt[1],popt[2],color='r')
    #     plt.show()
    
    # fit a 2D gaussian to the in focus plane to get rough poition
    # slc=img[zi,y-cutout:y+cutout,x-cutout:x+cutout]
    slc=img[z,y-cutout:y+cutout,x-cutout:x+cutout]
    popt,popcov=fit_gauss_2d_mod(slc,show=False)
    # get the rough position in the coordinates of the original image
    xi=int(popt[1])+x-cutout
    yi=int(popt[2])+y-cutout
    if show:
        plt.figure()
        plt.imshow(slc)
        plt.scatter(popt[1],popt[2],color='r')
        plt.show()
    zi=z
    # refine the estimates
    # cut out a smaller box
    ROI=img[zi-small_cutout*3:zi+small_cutout*3,yi-small_cutout:yi+small_cutout,xi-small_cutout:xi+small_cutout]

    # fit a gaussian to estimate the in focus plane
    I=np.mean(ROI,axis=(1,2))
    print(ROI.shape, I.shape)
    popt,popcov=fit_gauss1d_mod_old(I,show=False)
    zr=popt[1]+zi-small_cutout*3
    if show:
        plt.figure()
        plt.plot(I)
        plt.axvline(popt[1],color='r')
        plt.show()

    # fit a 2D gaussian to the in focus plane to get rough poition
    slc=img[int(zr),yi-small_cutout:yi+small_cutout,xi-small_cutout:xi+small_cutout]
    popt,popcov=fit_gauss_2d_mod(slc,show=False)
    # get the refined positions in the coordinates of the original image
    xr=popt[1]+xi-small_cutout
    yr=popt[2]+yi-small_cutout

    if show:
        plt.figure()
        plt.imshow(slc)
        plt.scatter(popt[1],popt[2],color='r')
        plt.show()

    return xr,yr,zr


def fit_gauss1d_mod_old(data: np.ndarray, show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 1D Gaussian to the data. Modified for negative hole images
    Args:
        data: 1D numpy array
        show: show the plot of the fit (for debugging)
    Returns:
        popt: optimal parameters
        pcov: covariance matrix
    """

    z=data.copy()
    print(z.shape)
    xz=np.arange(z.shape[0])
    offset=np.max(z)
    z-=offset # shift data to 0

    p0=[np.min(z)-offset,np.argmin(z),5,offset]

    popt, pcov = curve_fit(gauss1d_offset, xz, z, p0, maxfev=10000)

    # plot the data and the fit
    # if show:
    #     import matplotlib.pyplot as plt
    #     plt.title("1D Gaussian fit")
    #     plt.plot(data, label="Data")
    #     # plt.plot(gauss1d(x, *popt), label="Gaussian 1D fit")
    #     plt.legend()
    #     plt.show()

    return popt, pcov




def hole_fitting_reflection(da, x, y, z, cutout) -> tuple:

    from scipy.ndimage import gaussian_filter

    # round the (possibly sub-pixel) click for slicing; keep the fraction for
    # the input marker so a no-change fit shows the markers coincident.
    xi, yi = int(round(x)), int(round(y))
    zmin = 10
    zmax = 5
    zmin1 = int(z) - zmin
    zmax1 = int(z) + zmax

    roi = da[zmin1:zmax1, yi - cutout:yi + cutout + 1, xi - cutout:xi + cutout + 1]
    intensity = np.mean(roi, axis=(1, 2))
    intensity = intensity.max() - intensity  # invert: the hole is dark

    popt, _ = fit_guass1d(intensity)
    zopt = popt[1]
    zreal = zopt + zmin1  # back to absolute z

    # xy fitting on the fitted z-slice
    xy_cutout = 15
    roi_fitted = da[round(zreal), yi - xy_cutout:yi + xy_cutout + 1,
                    xi - xy_cutout:xi + xy_cutout + 1]
    popt_xy, _ = fit_gauss_2d_mod(roi_fitted)
    xopt, yopt = popt_xy[1], popt_xy[2]
    xopt_real = xopt + xi - xy_cutout
    yopt_real = yopt + yi - xy_cutout

    # --- confirmation-friendly diagnostic ---
    # Lead with the "did it land on the feature?" view (ROI + input/fitted
    # markers); a compact z panel answers "did z land right?". The old figure
    # had a raw-profile panel with six reference lines and a second z panel in a
    # different (cutout-relative) frame with the same labels — dropped.
    n = roi_fitted.shape[0]
    fit_in_roi = 0 <= xopt < n and 0 <= yopt < n

    # z: signal + gaussian fit (grey). The hole is dark, so the signal is
    # inverted (z_inverted) — the peak is the hole.
    z_axis = np.arange(zmin1, zmin1 + len(intensity))
    gauss_curve = gauss1d(np.arange(len(intensity)), *popt) + intensity.min()

    diagnostic = FitDiagnostic(
        title="Reflection hole fit",
        roi_xy=gaussian_filter(roi_fitted, sigma=1),
        input_xy=(xy_cutout + (x - xi), xy_cutout + (y - yi)),
        # A failed 2D fit lands outside the ROI — say so instead of a marker.
        fitted_xy=(xopt, yopt) if fit_in_roi else None,
        xy_title=f"XY  @ z = {zreal:.1f}",
        xy_message=None if fit_in_roi else "fit fell outside\nthe search region",
        z_axis=z_axis,
        z_signal=intensity,
        z_fit=gauss_curve,
        z_input=z,
        z_fitted=zreal,
        z_inverted=True,
    )
    return xopt_real, yopt_real, zreal, diagnostic
