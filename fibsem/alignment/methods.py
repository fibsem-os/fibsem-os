from __future__ import annotations

import logging
import os
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from fibsem import utils
from fibsem import config as cfg
from fibsem.imaging import masks
from fibsem.imaging import utils as image_utils
from fibsem.structures import FibsemImage, Point

if TYPE_CHECKING:
    from fibsem.alignment import AlignmentIteration

# ---------------------------------------------------------------------------
# Shift coordinate convention
# ---------------------------------------------------------------------------
# All functions in this module return (dx, dy) as the DISPLACEMENT of the
# new/moving image relative to the reference image, in metres:
#
#   dx > 0  →  feature moved RIGHT  (+x, +column)
#   dy > 0  →  feature moved DOWN   (+y, +row)
#
# This matches the image-plane convention (origin top-left, y increases down).
#
# Note on skimage: phase_cross_correlation returns the CORRECTION (shift to
# apply to the moving image to align it back to the reference), which is the
# negative of the displacement.  shift_from_skimage_phase_correlation negates
# the skimage output so all three methods share the same sign convention.
# ---------------------------------------------------------------------------

USE_SUBPIXEL_PEAK = False        # True → parabolic sub-pixel refinement; False → integer argmax


def _subpixel_peak(xcorr: np.ndarray, row: int, col: int) -> Tuple[float, float]:
    """Refine an integer xcorr peak to sub-pixel precision using a parabolic fit.

    Fits a 1D parabola independently along each axis through the three samples
    centred on the peak and returns the analytic maximum. Falls back to the
    integer position if the peak is on the image border or the denominator is zero.

    Args:
        xcorr: 2D cross-correlation map.
        row: Integer row index of the peak (from argmax).
        col: Integer column index of the peak (from argmax).

    Returns:
        (row_sub, col_sub) — sub-pixel peak position as floats.
    """
    h, w = xcorr.shape
    row_sub, col_sub = float(row), float(col)

    if 0 < row < h - 1:
        denom = 2 * xcorr[row, col] - xcorr[row - 1, col] - xcorr[row + 1, col]
        if denom != 0:
            row_sub = row + 0.5 * (xcorr[row + 1, col] - xcorr[row - 1, col]) / denom

    if 0 < col < w - 1:
        denom = 2 * xcorr[row, col] - xcorr[row, col - 1] - xcorr[row, col + 1]
        if denom != 0:
            col_sub = col + 0.5 * (xcorr[row, col + 1] - xcorr[row, col - 1]) / denom

    return row_sub, col_sub


def shift_from_crosscorrelation(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    lowpass: int = 128,
    highpass: int = 6,
    sigma: int = 6,
    use_rect_mask: bool = False,
    ref_mask: Optional[np.ndarray] = None,
    xcorr_limit: Optional[int] = None,
    save: bool = False,
) -> Tuple[float, float, np.ndarray, float]:
    """Calculates the shift between two images by cross-correlating them and finding the position of maximum correlation.

    Args:
        ref_image (FibsemImage): The reference image.
        new_image (FibsemImage): The new image to align to the reference.
        lowpass (int, optional): The low-pass filter frequency (in pixels) for the bandpass filter used to
            enhance the correlation signal. Defaults to 128.
        highpass (int, optional): The high-pass filter frequency (in pixels) for the bandpass filter used to
            enhance the correlation signal. Defaults to 6.
        sigma (int, optional): The standard deviation (in pixels) of the Gaussian filter used to create the bandpass
            mask. Defaults to 6.
        use_rect_mask (bool, optional): Whether to use a rectangular mask for the correlation. If True, the correlation
            is performed only inside a rectangle that covers most of the image, to reduce the effect of noise at the
            edges. Defaults to False.
        ref_mask (np.ndarray, optional): A mask to apply to the reference image before correlation. If not None,
            it should be a binary array with the same shape as the images. Pixels with value 0 will be ignored in the
            correlation. Defaults to None.
        xcorr_limit (int, optional): If not None, the correlation map will be circularly masked to a square
            with sides of length 2 * xcorr_limit + 1, centred on the maximum correlation peak. This can be used to
            limit the search range and improve the accuracy of the shift. Defaults to None.

    Returns:
        A tuple (x_shift, y_shift, xcorr), where x_shift and y_shift are the shifts along x and y (in meters),
        and xcorr is the cross-correlation map between the images.
    """
    # get pixel_size
    pixelsize_x = new_image.metadata.pixel_size.x
    pixelsize_y = new_image.metadata.pixel_size.y

    if new_image.data.shape != ref_image.data.shape:
        from skimage.transform import resize
        new_image.data = resize(new_image.data, ref_image.data.shape, preserve_range=True).astype(
            np.uint8
        )

    # normalise both images
    ref_data_norm = image_utils.normalise_image(ref_image)
    new_data_norm = image_utils.normalise_image(new_image)

    # cross-correlate normalised images
    if use_rect_mask:
        rect_mask = masks._mask_rectangular(new_data_norm.shape)
        ref_data_norm = rect_mask * ref_data_norm
        new_data_norm = rect_mask * new_data_norm

    if ref_mask is not None:
        ref_data_norm = ref_mask * ref_data_norm  # mask the reference

    # bandpass mask
    bandpass = masks.create_bandpass_mask(
        shape=ref_data_norm.shape, lp=lowpass, hp=highpass, sigma=sigma
    )

    # crosscorrelation
    xcorr = crosscorrelation_v2(ref_data_norm, new_data_norm, bandpass=bandpass)

    # limit xcorr range
    if xcorr_limit:
        xcorr = masks.apply_circular_mask(xcorr, xcorr_limit)

    # normalised cross-correlation peak: bounded [0, 1]
    energy = np.linalg.norm(ref_data_norm) * np.linalg.norm(new_data_norm)
    ncc_peak = xcorr.max() / (energy + 1e-10)
    score = float(np.clip((ncc_peak + 1) / 2, 0.0, 1.0))

    # np.unravel_index returns (row, col); row=Y, col=X
    maxRow, maxCol = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    if USE_SUBPIXEL_PEAK:
        maxRow, maxCol = _subpixel_peak(xcorr, maxRow, maxCol)
    cen = np.asarray(xcorr.shape) / 2
    err = cen - np.array([maxRow, maxCol])  # float when USE_SUBPIXEL_PEAK, int-valued otherwise

    # err[1] = col error → dx;  err[0] = row error → dy
    dx = err[1] * pixelsize_x
    dy = err[0] * pixelsize_y

    msgd = {"msg": "cross-correlation", "pixelsize": (pixelsize_x, pixelsize_y),
        "max": (maxRow, maxCol), "centre": cen, "shift": (err[1], err[0]), "shift_meters": (dx, dy)}
    logging.debug(msgd)

    if save:
        _save_alignment_data(
            ref_image=ref_image,
            new_image=new_image,
            bandpass=bandpass,
            xcorr=xcorr,
            use_rect_mask=use_rect_mask,
            ref_mask=ref_mask,
            xcorr_limit=xcorr_limit,
            lowpass=lowpass,
            highpass=highpass,
            sigma=sigma,
            dx=dx,
            dy=dy,
            pixelsize_x=pixelsize_x,
            pixelsize_y=pixelsize_y,
        )

    return dx, dy, xcorr, score


def shift_from_crosscorrelation_v2(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    minimum_response: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Calculate the shift between two images using cv2 phase correlation.

    Uses ``crosscorrelation_cv2`` (cv2.phaseCorrelate with a Hanning window)
    instead of the custom FFT cross-correlation in ``shift_from_crosscorrelation``.
    Returns a normalised response score (0-1) in place of the raw xcorr map,
    enabling threshold-based quality gating.

    Key differences from ``shift_from_crosscorrelation``:
    - No bandpass filter: phase correlation normalises per-element by magnitude,
      so no frequency dominates regardless of amplitude.
    - No spatial normalisation or masking: phase correlation is already invariant
      to global brightness/contrast; the Hanning window handles edge suppression.
      Asymmetric spatial masking would corrupt the cross-power spectrum.
    - Sub-pixel shift returned natively; no integer-only argmax peak-finding.
    - ``response`` quality metric allows callers to gate on alignment confidence.

    Args:
        ref_image (FibsemImage): Reference image.
        new_image (FibsemImage): New image to align to the reference.
        minimum_response (float, optional): Minimum acceptable response score.
            If ``response < minimum_response``, a warning is logged and
            (0.0, 0.0, response) is returned. Defaults to None (no gate).

    Returns:
        Tuple[float, float, float]: (dx, dy, response)
            dx:       x shift in metres (positive = ref feature is to the right).
            dy:       y shift in metres (positive = ref feature is below).
            response: normalised peak correlation in [0, 1].
    """
    pixelsize_x = new_image.metadata.pixel_size.x
    pixelsize_y = new_image.metadata.pixel_size.y

    shift_x, shift_y, response = crosscorrelation_cv2(ref_image.data, new_image.data)

    logging.debug({
        "msg": "cross-correlation-v2",
        "pixelsize": (pixelsize_x, pixelsize_y),
        "shift_px": (shift_x, shift_y),
        "response": response,
    })

    if minimum_response is not None and response < minimum_response:
        logging.warning(
            f"shift_from_crosscorrelation_v2: response {response:.3f} below "
            f"minimum {minimum_response:.3f} — returning zero shift."
        )
        return 0.0, 0.0, response

    dx = shift_x * pixelsize_x
    dy = shift_y * pixelsize_y

    return dx, dy, response


def crosscorrelation_cv2(
    img1: np.ndarray,
    img2: np.ndarray,
) -> Tuple[float, float, float]:
    """Phase correlation via cv2.phaseCorrelate with a Hanning window.

    Args:
        img1 (np.ndarray): Reference image (any float/int dtype, 2-D).
        img2 (np.ndarray): New image to correlate against the reference.

    Returns:
        Tuple[float, float, float]: (shift_x_px, shift_y_px, response)
            shift_x_px: sub-pixel column shift (positive = img2 shifted right).
            shift_y_px: sub-pixel row shift    (positive = img2 shifted down).
            response:   normalised peak value in [0, 1]; higher means a more
                        confident, higher-quality correlation.
    """
    import cv2 as _cv2

    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    h, w = f1.shape
    win = _cv2.createHanningWindow((w, h), _cv2.CV_32F)
    (shift_x, shift_y), response = _cv2.phaseCorrelate(f1, f2, win)
    return float(shift_x), float(shift_y), float(response)


def crosscorrelation_v2(
    img1: np.ndarray, img2: np.ndarray, bandpass: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Cross-correlate two images using Fourier convolution matching.

    Args:
        img1 (np.ndarray): The reference image.
        img2 (np.ndarray): The new image to be cross-correlated with the reference.
        bandpass (np.ndarray, optional): A bandpass mask to apply to both images before cross-correlation. Defaults to None.

    Returns:
        np.ndarray: The cross-correlation map between the two images.
    """
    if img1.shape != img2.shape:
        err = (
            f"Image 1 {img1.shape} and Image 2 {img2.shape} need to have the same shape"
        )
        logging.error(err)
        raise ValueError(err)

    if bandpass is None:
        bandpass = np.ones_like(img1)

    n_pixels = img1.shape[0] * img1.shape[1]

    img1ft = np.fft.ifftshift(bandpass * np.fft.fftshift(np.fft.fft2(img1)))
    tmp = img1ft * np.conj(img1ft)
    img1ft = n_pixels * img1ft / np.sqrt(tmp.sum())

    img2ft = np.fft.ifftshift(bandpass * np.fft.fftshift(np.fft.fft2(img2)))
    img2ft[0, 0] = 0
    tmp = img2ft * np.conj(img2ft)

    img2ft = n_pixels * img2ft / np.sqrt(tmp.sum())

    xcorr = np.real(np.fft.fftshift(np.fft.ifft2(img1ft * np.conj(img2ft))))

    return xcorr


def _save_alignment_data(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    bandpass: np.ndarray,
    xcorr: np.ndarray,
    ref_mask: np.ndarray = None,
    lowpass: float = None,
    highpass: float = None,
    sigma: float = None,
    xcorr_limit: float = None,
    use_rect_mask: bool = False,
    dx: float = None,
    dy: float = None,
    pixelsize_x: float = None,
    pixelsize_y: float = None,
):
    """Save alignment data to disk."""

    import pandas as pd
    import tifffile as tff

    ts = utils.current_timestamp_v2()
    fname = os.path.join(cfg.DATA_CC_PATH, str(ts))

    # save fibsem images
    ref_image.save(fname + "_ref.tif")
    new_image.save(fname + "_new.tif")

    # convert to tiff , save
    tff.imwrite(fname + "_xcorr.tif", xcorr)
    tff.imwrite(fname + "_bandpass.tif", bandpass)
    if ref_mask is not None:
        tff.imwrite(fname + "_ref_mask.tif", ref_mask)

    info = {
        "lowpass": lowpass, "highpass": highpass, "sigma": sigma,
        "pixelsize_x": pixelsize_x, "pixelsize_y": pixelsize_y,
        "use_rect_mask": use_rect_mask, "xcorr_limit": xcorr_limit, "ref_mask": ref_mask is not None,
        "dx": dx, "dy": dy, "fname": fname, "timestamp": ts }

    df = pd.DataFrame.from_dict(info, orient='index').T

    # save the dataframe to a csv file, append if the file already exists
    DATAFRAME_PATH = os.path.join(cfg.DATA_CC_PATH, "data.csv")
    if os.path.exists(DATAFRAME_PATH):
        df_tmp = pd.read_csv(DATAFRAME_PATH)
        df = pd.concat([df_tmp, df], axis=0, ignore_index=True)

    df.to_csv(DATAFRAME_PATH, index=False)


def shift_from_skimage_phase_correlation(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    upsample_factor: int = 10,
    minimum_response: Optional[float] = None,
) -> AlignmentIteration:
    """Calculate shift using skimage phase_cross_correlation with DFT upsampling.

    Sub-pixel accuracy is achieved by re-evaluating the DFT in a fine grid around
    the integer peak (via ``upsample_factor``), which is more accurate than the
    parabolic fit used by ``shift_from_crosscorrelation``.

    Args:
        ref_image: Reference image.
        new_image: New image to align to the reference.
        upsample_factor: Sub-pixel precision is 1/upsample_factor pixels.
            E.g. 10 → 0.1 px accuracy. Defaults to 10.
        minimum_response: If score < this value, zero the shift and set
            ``success=False``. Defaults to None (no gate).

    Returns:
        AlignmentIteration with shift in metres, score in [0, 1] (higher = better),
        shift_px in pixels, and success flag.
    """
    from skimage.registration import phase_cross_correlation
    from fibsem.alignment import AlignmentIteration  # lazy: avoids circular import at load time

    pixelsize_x = new_image.metadata.pixel_size.x
    pixelsize_y = new_image.metadata.pixel_size.y

    shift, error, _ = phase_cross_correlation(
        ref_image.data.astype(np.float32),
        new_image.data.astype(np.float32),
        upsample_factor=upsample_factor,
        normalization="phase",
    )

    # skimage returns the shift to align moving→reference (correction), but the
    # rest of the alignment pipeline uses displacement (how far new moved from ref).
    # Negate to match cv2 / bandpass convention: positive = moved right/down.
    dy_px, dx_px = -float(shift[0]), -float(shift[1])
    score = float(np.clip(1.0 - error, 0.0, 1.0))

    success = True
    if minimum_response is not None and score < minimum_response:
        logging.warning(
            f"shift_from_skimage_phase_correlation: score {score:.3f} below "
            f"minimum {minimum_response:.3f} — returning zero shift."
        )
        dx_px, dy_px = 0.0, 0.0
        success = False

    dx = dx_px * pixelsize_x
    dy = dy_px * pixelsize_y

    logging.debug({
        "msg": "skimage-phase-cross-correlation",
        "pixelsize": (pixelsize_x, pixelsize_y),
        "shift_px": (dx_px, dy_px),
        "error": error,
        "score": score,
    })

    return AlignmentIteration(
        shift=Point(dx, dy),
        score=score,
        image=new_image,
        success=success,
    )
