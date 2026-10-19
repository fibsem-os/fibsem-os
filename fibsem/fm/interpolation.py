"""Resampling a fluorescence z-stack along z.

Only the z-axis is resampled; XY is untouched. This is interpolation between the
planes of one stack, used to make an FM volume isotropic for correlation -- not
z-stacking (combining planes into one image), which lives elsewhere.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from scipy import ndimage

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

    from fibsem.fm.structures import FluorescenceImage  # lazy

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
