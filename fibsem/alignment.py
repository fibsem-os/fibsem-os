import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from fibsem import acquire, utils, validation
from fibsem.exceptions import AlignmentError
from fibsem.config import REFERENCE_FILENAME
from fibsem.constants import DATETIME_DISPLAY
from fibsem.imaging import masks
from fibsem.imaging import utils as image_utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    ImageSettings,
    MicroscopeSettings,
    Point,
    ReferenceImages,
)
from threading import Event as ThreadingEvent


ALIGNMENT_SUBDIR = "Alignment"


@dataclass
class AlignmentResult:
    """Result of a single alignment step using shift_from_crosscorrelation_v2."""
    shift: Point                   # (x, y) shift applied, in metres
    shift_px: Point                # raw sub-pixel (x, y) shift in pixels from cv2
    score: float                   # normalised phase correlation response (0–1); higher = better
    image: FibsemImage             # new image acquired during this alignment step
    success: bool = True           # False if score < minimum_response (shift was zeroed)


@dataclass
class AlignmentStatus:
    """Outcome of align_until_converged."""
    results: "list[AlignmentResult]"
    converged: bool   # True if shift fell below shift_tolerance
    aborted: bool     # True if stopped early due to low score or divergence
    reason: str       # "converged" | "low_score" | "diverging" | "max_steps" | "stop_event"



# TODO: rename to align_with_reference_image as it is not specific to beam shift
def beam_shift_alignment_v2(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    alignment_current: Optional[float] = None,
    use_autocontrast: bool = False,
    use_autofocus: bool = False,
    subsystem: Optional[str] = None,
):
    """Aligns the images by adjusting the beam shift instead of moving the stage.

    This method uses cross-correlation between the reference image and a new image to calculate the
    optimal beam shift for alignment. This approach offers increased precision, but a lower range
    compared to stage movement.

        Args:
        microscope (FibsemMicroscope): An OpenFIBSEM microscope client.
        ref_image (FibsemImage): The reference image to align to.
        alignment_current: The beam current to set before alignment. Defaults to None (no change).
        use_autocontrast (bool): Whether to use autocontrast for the new image. Defaults to False.
        subsystem (Optional[str]): The subsystem to use for alignment. Can be either "stage" or None.
            If "stage", the stage will be moved instead of adjusting the beam shift. Defaults to None.

    Raises:
        ValueError: If `image_settings.beam_type` is not set to `BeamType.ION`.

    """

    # time.sleep(2) # threading is too fast?
    image_settings = ImageSettings.fromFibsemImage(ref_image)
    image_settings.autocontrast = False
    image_settings.save = True

    # save new images in Alignment subdir next to the reference image
    ref_path = image_settings.path if image_settings.path is not None else os.getcwd()
    image_settings.path = os.path.join(ref_path, ALIGNMENT_SUBDIR)
    os.makedirs(image_settings.path, exist_ok=True)

    # use the same named prefix for the filename for traceability (if possible)
    if REFERENCE_FILENAME in image_settings.filename:
        # get everything before "alignment_reference"
        prefix = image_settings.filename.split(REFERENCE_FILENAME)[0]
        image_settings.filename = f"{prefix}beam_shift_alignment_{utils.current_timestamp_v3(timeonly=True)}"
    else:
        image_settings.filename = f"beam_shift_alignment_{utils.current_timestamp_v2()}"

    # set alignment current
    if alignment_current is not None:
        initial_current = microscope.get_beam_current(image_settings.beam_type)
        microscope.set_beam_current(alignment_current, image_settings.beam_type)

    if use_autocontrast:
        microscope.autocontrast(beam_type=image_settings.beam_type,
                                reduced_area=image_settings.reduced_area)

    if use_autofocus:
        microscope.auto_focus(beam_type=image_settings.beam_type, reduced_area=image_settings.reduced_area)

    new_image = acquire.new_image(microscope, settings=image_settings)
    dx, dy, xcorr = shift_from_crosscorrelation(
        ref_image, new_image,
        lowpass=50,
        highpass=4,
        sigma=5,
        use_rect_mask=True
    )

    # adjust beamshift
    if subsystem is None:
        microscope.beam_shift(-dx, dy, image_settings.beam_type)
    elif subsystem == "stage":
        microscope.stable_move(
            dx=dx,
            dy=-dy,
            beam_type=image_settings.beam_type,
        )
    elif subsystem == "stage-vertical":
        if image_settings.beam_type is BeamType.ELECTRON:
            raise AlignmentError(f"Unsupported movement type ({subsystem}) for beam type {image_settings.beam_type}")
        microscope.vertical_move(dy=-dy, dx=dx)

    # reset beam current
    if alignment_current is not None:
        microscope.set_beam_current(initial_current, image_settings.beam_type)
    logging.info(f"Beam Shift Alignment: dx: {dx}, dy: {dy}")
    msgd = {"msg": "beam_shift_alignment", "dx": dx, "dy": dy, "image_settings": image_settings.to_dict()}
    logging.debug(msgd)

    return new_image, xcorr, dx, dy


def correct_stage_drift(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    reference_images: ReferenceImages,
    alignment: Tuple[BeamType, BeamType] = (BeamType.ELECTRON, BeamType.ELECTRON),
    rotate: bool = False,
    ref_mask_rad: int = 512,
    xcorr_limit: Union[Tuple[int, int], None] = None,
    constrain_vertical: bool = False,
    use_beam_shift: bool = False,
) -> bool:
    """Corrects the stage drift by aligning low- and high-resolution reference images
    using cross-correlation.

    Args:
        microscope (FibsemMicroscope): The microscope used for image acquisition.
        settings (MicroscopeSettings): The settings used for image acquisition.
        reference_images (ReferenceImages): A container of low- and high-resolution
            reference images.
        alignment (Tuple[BeamType, BeamType], optional): A tuple of two `BeamType`
            objects, specifying the beam types used for the alignment of low- and
            high-resolution images, respectively. Defaults to (BeamType.ELECTRON,
            BeamType.ELECTRON).
        rotate (bool, optional): Whether to rotate the reference images before
            alignment. Defaults to False.
        ref_mask_rad (int, optional): The radius of the circular mask used for reference
        xcorr_limit (Tuple[int, int] | None, optional): A tuple of two integers that
            represent the minimum and maximum cross-correlation values allowed for the
            alignment. If not specified, the values are set to (None, None), which means
            there are no limits. Defaults to None.
        constrain_vertical (bool, optional): Whether to constrain the alignment to the
            vertical axis. Defaults to False.

    Returns:
        bool: True if the stage drift correction was successful, False otherwise.
    """

    # set reference images
    if alignment[0] is BeamType.ELECTRON:
        ref_lowres, ref_highres = (
            reference_images.low_res_eb,
            reference_images.high_res_eb,
        )
    if alignment[0] is BeamType.ION:
        ref_lowres, ref_highres = (
            reference_images.low_res_ib,
            reference_images.high_res_ib,
        )

    if xcorr_limit is None:
        xcorr_limit = (None, None)

    # rotate reference
    if rotate:
        ref_lowres = image_utils.rotate_image(ref_lowres)
        ref_highres = image_utils.rotate_image(ref_highres)

    # align lowres, then highres
    for i, ref_image in enumerate([ref_lowres, ref_highres]):

        ref_mask = masks.create_circle_mask(ref_image.data.shape, ref_mask_rad)

        # take new images
        # set new image settings (same as reference)
        settings.image = ImageSettings.fromFibsemImage(ref_image)
        settings.image.beam_type = alignment[1]
        new_image = acquire.new_image(microscope, settings.image)

        # crosscorrelation alignment
        ret = align_using_reference_images(
            microscope,
            ref_image,
            new_image,
            ref_mask=ref_mask,
            xcorr_limit=xcorr_limit[i],
            constrain_vertical=constrain_vertical,
            use_beam_shift=use_beam_shift,
        )

        if ret is False:
            break  # cross correlation has failed...

    return ret


def align_using_reference_images(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    new_image: FibsemImage,
    ref_mask: np.ndarray = None,
    xcorr_limit: int = None,
    constrain_vertical: bool = False,
    use_beam_shift: bool = False,
) -> bool:
    """
    Uses cross-correlation to align a new image to a reference image.

    Args:
        microscope: A FibsemMicroscope instance representing the microscope being used.
        ref_image: A FibsemImage instance representing the reference image to which the new image will be aligned.
        new_image: A FibsemImage instance representing the new image that will be aligned to the reference image.
        ref_mask: A numpy array representing a mask to apply to the reference image during alignment. Default is None.
        xcorr_limit: An integer representing the limit for the cross-correlation coefficient. If the coefficient is below
            this limit, alignment will fail. Default is None.
        constrain_vertical: A boolean indicating whether to constrain movement to the vertical axis. If True, movement
            will be restricted to the vertical axis, which is useful for eucentric movement. If False, movement will be
            allowed on both the X and Y axes. Default is False.

    Returns:
        A boolean indicating whether the alignment was successful. True if the alignment was successful, False otherwise.
    """
    # get beam type
    ref_beam_type = BeamType[ref_image.metadata.image_settings.beam_type.name.upper()]
    new_beam_type = BeamType[new_image.metadata.image_settings.beam_type.name.upper()]

    logging.info(
        f"aligning {ref_beam_type.name} reference image to {new_beam_type.name}."
    )
    sigma = 6
    hp_px = 8
    lp_px = 128  # MAGIC_NUMBER

    dx, dy, xcorr = shift_from_crosscorrelation(
        ref_image,
        new_image,
        lowpass=lp_px,
        highpass=hp_px,
        sigma=sigma,
        use_rect_mask=True,
        ref_mask=ref_mask,
        xcorr_limit=xcorr_limit,
    )

    shift_within_tolerance = (
        validation.check_shift_within_tolerance(  # TODO: Abstract validation.py
            dx=dx, dy=dy, ref_image=ref_image, limit=0.5
        )
    )

    if shift_within_tolerance:

        # vertical constraint = eucentric movement
        if constrain_vertical:
            microscope.vertical_move( dx=0, dy=-dy
            )  # FLAG_TEST
        else:
            if use_beam_shift:
                # move the beam shift
                microscope.beam_shift(dx=-dx, dy=-dy, beam_type=new_beam_type)
            else:
                # move the stage
                microscope.stable_move(
                    dx=dx,
                    dy=-dy,
                    beam_type=new_beam_type,
                )

    return shift_within_tolerance


def shift_from_crosscorrelation(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    lowpass: int = 128,
    highpass: int = 6,
    sigma: int = 6,
    use_rect_mask: bool = False,
    ref_mask: np.ndarray = None,
    xcorr_limit: int = None,
) -> Tuple[float, float, np.ndarray]:
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

    # calculate maximum crosscorrelation
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)  # TODO: backwards
    cen = np.asarray(xcorr.shape) / 2
    err = np.array(cen - [maxX, maxY], int)

    # calculate shift in metres
    dx = err[1] * pixelsize_x
    dy = err[0] * pixelsize_y  # this could be the issue?

    msgd = {"msg": "cross-correlation", "pixelsize": (pixelsize_x, pixelsize_y), 
        "max": (maxX, maxY), "centre": cen, "shift": (err[1], err[0]), "shift_meters": (dx, dy)}
    logging.debug(msgd)

    # logging.debug(f"cross-correlation:")
    # logging.debug(f"pixelsize: x: {pixelsize_x:.2e}, y: {pixelsize_y:.2e}")
    # logging.debug(f"maxX: {maxX}, {maxY}, centre: {cen}")
    # logging.debug(f"x: {err[1]}px, y: {err[0]}px")
    # logging.debug(f"x: {dx:.2e}m, y: {dy:.2e} meters")

    # save data
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

    # metres
    return dx, dy, xcorr


def shift_from_crosscorrelation_v2(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    minimum_response: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Calculate the shift between two images using cv2 phase correlation.

    Uses ``crosscorrelation_cv2`` (cv2.phaseCorrelate with a Hanning window)
    instead of the custom FFT cross-correlation in ``shift_from_crosscorrelation``.
    Returns a normalised response score (0–1) in place of the raw xcorr map,
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
    img1: np.ndarray, img2: np.ndarray, bandpass: np.ndarray = None
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
        raise AlignmentError(err)

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

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    # ax[0].imshow(np.fft.ifft2(img1ft).real)
    # ax[1].imshow(np.fft.ifft2(img2ft).real)
    # plt.show()

    # plt.title("Power Spectra")
    # plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img1)))))
    # plt.show()

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
    
    import os

    import pandas as pd
    import tifffile as tff

    from fibsem import config as cfg

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
        # "ref_image": ref_image, "new_image": new_image, "bandpass": bandpass, "xcorr": xcorr, "ref_mask": ref_mask,
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

def multi_step_alignment_v2(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    beam_type: BeamType,
    alignment_current: Optional[float] = None,
    steps: int = 3,
    use_autocontrast: bool = False,
    use_autofocus: bool = False,
    subsystem: Optional[str] = None,
    stop_event: Optional[ThreadingEvent] = None,
    save_plot: bool = True,
    plot_title: Optional[str] = None,
) -> None:
    """Runs the beam shift alignment multiple times. Optionally sets the beam current before alignment."""
    # set alignment current
    if alignment_current is not None:
        initial_current = microscope.get_beam_current(beam_type)
        microscope.set_beam_current(alignment_current, beam_type)

    alignment_results = []
    for i in range(steps):
        if stop_event is not None and stop_event.is_set():
            break
        # only use autocontrast on first step
        use_autocontrast = use_autocontrast if i == 0 else False
        use_autofocus = use_autofocus if i == 0 else False
        new_image, xcorr, dx, dy = beam_shift_alignment_v2(
            microscope=microscope,
            ref_image=ref_image,
            use_autocontrast=use_autocontrast,
            use_autofocus=use_autofocus,
            subsystem=subsystem,
        )
        alignment_results.append((new_image, xcorr, dx, dy))

    # reset beam current
    if alignment_current is not None:
        microscope.set_beam_current(initial_current, beam_type)

    if save_plot:
        plot_multi_step_alignment(ref_image, alignment_results, save=save_plot, title=plot_title)


def align_until_converged(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    beam_type: BeamType,
    max_steps: int = 5,
    shift_tolerance: Optional[float] = None,
    minimum_response: Optional[float] = None,
    detect_divergence: bool = True,
    use_autocontrast: bool = False,
    use_autofocus: bool = False,
    subsystem: Optional[str] = None,
    stop_event: Optional[ThreadingEvent] = None,
    save_plot: bool = False,
    plot_title: Optional[str] = None,
) -> "AlignmentStatus":
    """Align iteratively using cv2 phase correlation, stopping as soon as the
    shift converges, the correlation quality drops, or the alignment diverges.

    Args:
        microscope: The microscope to use.
        ref_image: Reference image to align to.
        beam_type: Beam type for image acquisition.
        max_steps: Hard upper limit on iterations. Defaults to 5.
        shift_tolerance: Stop when |shift| < this value (metres).
            Defaults to None → 1 × pixel size of ref_image (resolution-independent).
        minimum_response: Abort if response drops below this score.
            Defaults to None (no gate).
        detect_divergence: Abort if shift magnitude increases for 2 consecutive
            steps (indicates overcorrection or a failing correlation). Defaults to True.
        use_autocontrast: Run autocontrast before the first acquisition. Defaults to False.
        use_autofocus: Run autofocus before the first acquisition. Defaults to False.
        subsystem: Movement system — None (beam shift), "stage", or "stage-vertical".
        stop_event: Threading event; checked each iteration for external cancellation.
        save_plot: Save a plot of results to disk. Defaults to False.
        plot_title: Optional title for the saved plot.

    Returns:
        AlignmentStatus with all per-step AlignmentResult objects and a reason string:
        "converged" | "low_score" | "diverging" | "max_steps" | "stop_event"
    """
    pixelsize_x = ref_image.metadata.pixel_size.x
    pixelsize_y = ref_image.metadata.pixel_size.y

    if shift_tolerance is None:
        shift_tolerance = pixelsize_x

    # set up image acquisition settings
    image_settings = ImageSettings.fromFibsemImage(ref_image)
    image_settings.autocontrast = False
    image_settings.save = True
    ref_path = image_settings.path if image_settings.path is not None else os.getcwd()
    image_settings.path = os.path.join(ref_path, ALIGNMENT_SUBDIR)
    os.makedirs(image_settings.path, exist_ok=True)

    results: list = []
    prev_shift_magnitude = None
    divergence_count = 0

    for i in range(max_steps):
        if stop_event is not None and stop_event.is_set():
            logging.info("align_until_converged: stop event set, exiting.")
            return AlignmentStatus(results=results, converged=False, aborted=True, reason="stop_event")

        image_settings.filename = f"align_until_converged_step{i+1:02d}_{utils.current_timestamp_v3(timeonly=True)}"

        if i == 0:
            if use_autocontrast:
                microscope.autocontrast(beam_type=beam_type, reduced_area=image_settings.reduced_area)
            if use_autofocus:
                microscope.auto_focus(beam_type=beam_type, reduced_area=image_settings.reduced_area)

        new_image = acquire.new_image(microscope, settings=image_settings)

        dx, dy, score = shift_from_crosscorrelation_v2(ref_image, new_image)
        shift_magnitude = np.sqrt(dx ** 2 + dy ** 2)

        result = AlignmentResult(
            shift=Point(dx, dy),
            shift_px=Point(dx / pixelsize_x, dy / pixelsize_y),
            score=score,
            image=new_image,
            success=(minimum_response is None or score >= minimum_response),
        )
        results.append(result)

        logging.info(
            f"align_until_converged step {i+1}: "
            f"shift={shift_magnitude*1e9:.1f}nm  score={score:.3f}  "
            f"tolerance={shift_tolerance*1e9:.1f}nm"
        )

        # gate: low correlation score
        if minimum_response is not None and score < minimum_response:
            logging.warning(f"align_until_converged: score {score:.3f} < minimum {minimum_response:.3f}, aborting.")
            return AlignmentStatus(results=results, converged=False, aborted=True, reason="low_score")

        # gate: converged
        if shift_magnitude < shift_tolerance:
            logging.info(f"align_until_converged: converged at step {i+1}.")
            return AlignmentStatus(results=results, converged=True, aborted=False, reason="converged")

        # gate: divergence (shift grew for 2 consecutive steps)
        if detect_divergence and prev_shift_magnitude is not None:
            if shift_magnitude > prev_shift_magnitude:
                divergence_count += 1
                if divergence_count >= 2:
                    logging.warning(f"align_until_converged: diverging at step {i+1}, aborting.")
                    return AlignmentStatus(results=results, converged=False, aborted=True, reason="diverging")
            else:
                divergence_count = 0
        prev_shift_magnitude = shift_magnitude

        # apply correction
        if subsystem is None:
            microscope.beam_shift(-dx, dy, beam_type)
        elif subsystem == "stage":
            microscope.stable_move(dx=dx, dy=-dy, beam_type=beam_type)
        elif subsystem == "stage-vertical":
            microscope.vertical_move(dy=-dy, dx=dx)

    logging.info(f"align_until_converged: reached max_steps ({max_steps}).")
    status = AlignmentStatus(results=results, converged=False, aborted=False, reason="max_steps")

    if save_plot:
        plot_multi_step_alignment_v2(ref_image, results, title=plot_title, save=True)

    return status


def _plot_image_with_crosshair(ax, data: np.ndarray, title: str) -> None:
    """Plot an image with a yellow crosshair at the centre."""
    ax.imshow(data, cmap="gray")
    cy, cx = data.shape[0] // 2, data.shape[1] // 2
    ax.axhline(cy, color="yellow", linewidth=2, alpha=0.7)
    ax.axvline(cx, color="yellow", linewidth=2, alpha=0.7)
    ax.set_title(title)
    ax.axis("off")


def _alignment_save_path(ref_image: FibsemImage) -> Tuple[str, str, str]:
    """Return (ref_path, prefix, ts) for saving alignment plots."""
    from datetime import datetime
    ref_settings = ImageSettings.fromFibsemImage(ref_image)
    ref_filename = ref_settings.filename
    ref_path = ref_settings.path if ref_settings.path is not None else os.getcwd()
    ref_path = os.path.join(ref_path, ALIGNMENT_SUBDIR)
    os.makedirs(ref_path, exist_ok=True)
    prefix = ref_filename.split(REFERENCE_FILENAME)[0] if REFERENCE_FILENAME in ref_filename else ref_filename + "_"
    ts = utils.current_timestamp_v2()
    return ref_path, prefix, ts


def plot_multi_step_alignment(
    ref_image: FibsemImage,
    alignment_results: list,
    title: Optional[str] = None,
    save: bool = True,
):
    """Plot the reference image and each alignment step with cross-correlation maps.

    Args:
        ref_image: The reference image used for alignment.
        alignment_results: List of (new_image, xcorr, dx, dy) tuples from multi_step_alignment_v2.
        save: Whether to save the figure to disk. Defaults to True.
        show: Whether to call plt.show(). Defaults to False.

    Returns:
        matplotlib.figure.Figure
    """
    from datetime import datetime

    from matplotlib.figure import Figure

    ref_path, prefix, ts = _alignment_save_path(ref_image)
    ref_filename = ImageSettings.fromFibsemImage(ref_image).filename
    timestamp_str = datetime.now().strftime(DATETIME_DISPLAY)
    if title is None:
        title = f"Multi-Step Alignment — {ref_filename} — {timestamp_str}"
    else:
        title = f"{title} — {timestamp_str}"

    # row 0 = images (reference + each step), row 1 = xcorr for each step
    n_steps = len(alignment_results)
    n_cols = 1 + n_steps
    fig = Figure(figsize=(4 * n_cols, 8))
    axes = fig.subplots(2, n_cols)
    fig.suptitle(title)
    _plot_image_with_crosshair(axes[0, 0], ref_image.data, "Reference")

    # convergence plot in the spare cell
    shifts_dx = [r[2] for r in alignment_results]
    shifts_dy = [r[3] for r in alignment_results]
    step_nums = list(range(1, len(alignment_results) + 1))
    ax_conv = axes[1, 0]
    ax_conv.plot(step_nums, [abs(d) * 1e9 for d in shifts_dx], "o-", label="dx")
    ax_conv.plot(step_nums, [abs(d) * 1e9 for d in shifts_dy], "s-", label="dy")
    ax_conv.set_xlabel("Step")
    ax_conv.set_ylabel("Shift (nm)")
    ax_conv.set_title("Convergence")
    ax_conv.legend(fontsize="small")
    ax_conv.set_xticks(step_nums)

    for i, (new_image, xcorr, dx, dy) in enumerate(alignment_results):
        col = 1 + i
        _plot_image_with_crosshair(axes[0, col], new_image.data, f"Step {i + 1}")
        axes[1, col].imshow(xcorr, cmap="inferno")
        axes[1, col].set_title(f"XCorr {i + 1}\ndx={dx*1e9:.1f}nm, dy={dy*1e9:.1f}nm", fontsize="small")
        axes[1, col].axis("off")

    fig.tight_layout()
    if save:
        save_path = os.path.join(ref_path, f"{prefix}multi_step_alignment_{ts}.png")
        fig.savefig(save_path, dpi=80)
    return fig


def plot_multi_step_alignment_v2(
    ref_image: FibsemImage,
    results: "list[AlignmentResult]",
    title: Optional[str] = None,
    save: bool = True,
    show_convergence: bool = False,
):
    """Plot reference image and each alignment step with response score bars.

    Drop-in companion to ``plot_multi_step_alignment`` for results produced by
    ``shift_from_crosscorrelation_v2``. Replaces xcorr imshow panels with a
    colour-coded response bar (green ≥ 0.5, orange ≥ 0.25, red < 0.25).

    Args:
        ref_image: The reference image used for alignment.
        results: List of AlignmentResult from each alignment step.
        title: Optional figure title. Defaults to auto-generated from ref filename.
        save: Whether to save the figure to disk. Defaults to True.

    Returns:
        matplotlib.figure.Figure
    """
    from datetime import datetime

    from matplotlib.figure import Figure

    ref_path, prefix, ts = _alignment_save_path(ref_image)
    ref_filename = ImageSettings.fromFibsemImage(ref_image).filename
    timestamp_str = datetime.now().strftime(DATETIME_DISPLAY)
    if title is None:
        title = f"Multi-Step Alignment v2 — {ref_filename} — {timestamp_str}"
    else:
        title = f"{title} — {timestamp_str}"

    from matplotlib.gridspec import GridSpec

    n_steps = len(results)
    n_cols = 1 + n_steps
    n_rows = 2 if show_convergence else 1
    fig_height = 8 if show_convergence else 5
    fig = Figure(figsize=(4 * n_cols, fig_height))
    fig.suptitle(title)

    gs = GridSpec(n_rows, n_cols, figure=fig, height_ratios=([2, 1] if show_convergence else [1]))
    image_axes = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
    ax_conv = fig.add_subplot(gs[1, 1:]) if show_convergence else None

    _plot_image_with_crosshair(image_axes[0], ref_image.data, "Reference")

    # annotate each step image with score (coloured) + shift values
    for i, result in enumerate(results):
        label = f"Step {i + 1}" + ("" if result.success else " ✗")
        _plot_image_with_crosshair(image_axes[i + 1], result.image.data, label)

        colour = "lime" if result.score >= 0.5 else ("orange" if result.score >= 0.25 else "red")
        ax_img = image_axes[i + 1]
        ax_img.text(
            0.04, 0.06,
            f"score: {result.score:.2f}",
            transform=ax_img.transAxes,
            color=colour, fontsize=9, fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none"),
        )
        ax_img.text(
            0.04, 0.18,
            f"dx={result.shift.x*1e9:.1f}nm  dy={result.shift.y*1e9:.1f}nm",
            transform=ax_img.transAxes,
            color="white", fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none"),
        )

    # wide convergence chart (optional)
    if show_convergence:
        step_nums = list(range(1, n_steps + 1))
        ax_conv.plot(step_nums, [abs(r.shift.x) * 1e9 for r in results], "o-", label="dx")
        ax_conv.plot(step_nums, [abs(r.shift.y) * 1e9 for r in results], "s-", label="dy")
        ax_conv.set_xlabel("Step")
        ax_conv.set_ylabel("Shift (nm)")
        ax_conv.set_title("Convergence")
        ax_conv.legend(fontsize="small")
        ax_conv.set_xticks(step_nums)

    fig.tight_layout()
    if save:
        save_path = os.path.join(ref_path, f"{prefix}multi_step_alignment_v2_{ts}.png")
        fig.savefig(save_path, dpi=80)
    return fig


def _eucentric_tilt_alignment(microscope: FibsemMicroscope, image_settings: ImageSettings, 
                              target_angle: float, step_size: float, 
                              beam_type: Optional[BeamType] = None, show: bool = False) -> None:
    """Perform eucentric tilt alignment by moving the stage in steps towards the target angle,
    acquiring images at each step, and performing alignment.
    Args:
        microscope (FibsemMicroscope): The microscope to use for alignment.
        image_settings (ImageSettings): The image settings to use for image acquisition.
        target_angle (float): The target tilt angle in degrees.
        step_size (float): The step size in degrees.
        beam_type (Optional[BeamType]): The beam type to use for image acquisition. If None, both beams are used.
        show (bool): Whether to show the images at each step. Defaults to False.
    Returns:
        None
    """
    from fibsem.structures import FibsemStagePosition
    import matplotlib.pyplot as plt

    stage_position = microscope.get_stage_position()
    current_angle = np.degrees(stage_position.t)

    n_steps = int(abs(int(current_angle) - target_angle) // step_size)

    logging.info(f"Current Tilt: {current_angle}, Target Tilt:  {target_angle}, Step Size: {step_size},  Num Steps: {n_steps}")
    steps = np.linspace(current_angle, target_angle, num=n_steps)

    # input()

    # NOTE: this is coincidence + eucentric maintaining tilt
    # for just eucentric, the other beam will drift out of position
    # but it might be required, for systems that don't want to acquire sem, or have no sem.

    # QUERY: should we be updating the ref image as we go?

    image_settings.hfw = 150e-6
    image_settings.save = False
    if beam_type is not None:
        image_settings.beam_type = beam_type
        reference_image = acquire.acquire_image(microscope, image_settings)
    else:
        ref_sem_image, ref_fib_image = acquire.acquire_channels(microscope, image_settings)

    # NOTE: we can skip the first step, its at the current tilt

    fib_images = []
    sem_images = []

    for i, angle in enumerate(steps[1:]):
        microscope.move_stage_absolute(FibsemStagePosition(t=np.radians(angle)))

        if beam_type is not None:
            beam_shift_alignment_v2(microscope, reference_image, subsystem="stage")
        else:
            beam_shift_alignment_v2(microscope, ref_sem_image, subsystem="stage")
            beam_shift_alignment_v2(microscope, ref_fib_image, subsystem="stage-vertical")

        # we prob want to do sem-> stage, fib -> vertical

        sem_image, fib_image = acquire.acquire_channels(microscope, image_settings)

        if show:
            fig, ax = plt.subplots(1, 2, figsize=(10, 7))
            ax[0].imshow(sem_image.data, cmap="gray")
            ax[0].plot(sem_image.data.shape[1]//2, sem_image.data.shape[0]//2, "y+", ms=50)
            ax[1].imshow(fib_image.data, cmap="gray")
            ax[1].plot(fib_image.data.shape[1]//2, fib_image.data.shape[0]//2, "y+", ms=50)
            plt.show()

        sem_images.append(sem_image)
        fib_images.append(fib_image)
        # if i >=5:
            # break
        if beam_type is None:
            ref_sem_image = sem_image
            ref_fib_image = fib_image
        elif beam_type is BeamType.ELECTRON:
            reference_image = sem_image
        elif beam_type is BeamType.ION:
            reference_image = fib_image

    # TODO: have a metric to measure if it failed? how??
    final_position = microscope.get_stage_position()
    diff = stage_position - final_position
    logging.info(f"Start Position: {stage_position.pretty}")
    logging.info(f"Final Position: {final_position.pretty}")
    logging.info(f"Difference: {diff.pretty}")

    return sem_images, fib_images
