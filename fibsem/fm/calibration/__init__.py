"""Focus measurement and calibration utilities for fluorescence microscopy.

This module provides focus measure algorithms used in focus stacking and autofocus
applications. Different algorithms are suitable for different types of samples
and imaging conditions.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

import numpy as np

from fibsem.autofunctions.autofocus import AutoFocusIteration, AutoFocusResult
from fibsem.autofunctions.integration import (
    adaptive_frame_integration,
    frame_integration,
)
from fibsem.autofunctions.metrics import (
    calculate_focus_quality,
    find_best_focus_plane,
    get_focus_measure_function,
    laplacian_focus_measure,
    sobel_focus_measure,
    tenengrad_focus_measure,
    variance_focus_measure,
)
from fibsem.autofunctions.plotting import plot_autofocus_result as plot_autofocus
from fibsem.autofunctions.stacking import (
    block_based_focus_selection,
    create_block_based_focus_stack,
    create_focus_stack_from_selection,
    create_pixel_based_focus_stack,
    pixel_based_focus_selection,
)
from fibsem.cancellation import CANCELLED, COMPLETED, SKIPPED, ended_by
from fibsem.fm.progress import (
    FluorescenceAcquisitionProgress,
    FluorescenceAcquisitionStatus,
)
from fibsem.fm.structures import AutoFocusSettings, ZParameters
from fibsem.structures import FibsemRectangle

__all__ = [
    "get_focus_measure_function",
    "laplacian_focus_measure",
    "sobel_focus_measure",
    "variance_focus_measure",
    "tenengrad_focus_measure",
    "block_based_focus_selection",
    "create_focus_stack_from_selection",
    "pixel_based_focus_selection",
    "create_pixel_based_focus_stack",
    "create_block_based_focus_stack",
    "frame_integration",
    "adaptive_frame_integration",
    "plot_autofocus",
    "find_best_focus_plane",
    "calculate_focus_quality",
    "run_autofocus",
    "run_coarse_fine_autofocus",
]

if TYPE_CHECKING:
    from fibsem.fm.microscope import FluorescenceMicroscope
    from fibsem.fm.structures import ChannelSettings, ZParameters


def run_autofocus(
    microscope: "FluorescenceMicroscope",
    channel_settings: Optional["ChannelSettings"] = None,
    z_parameters: Optional["ZParameters"] = None,
    method: str = "tenengrad",
    stop_event: Optional[threading.Event] = None,
    roi: Optional[FibsemRectangle] = None,
    path: Optional[str] = None,
    name: str = "autofocus",
    pass_index: int = 1,
    total_passes: int = 1,
) -> Optional[AutoFocusResult]:
    """Run autofocus by acquiring images at different z positions and finding the best focus.

    Uses the focus measure functions to evaluate image sharpness at different
    objective positions and moves to the position with the highest focus score.

    Args:
        microscope: The fluorescence microscope instance
        channel_settings: Channel settings to use for autofocus (optional)
        z_parameters: Z-stack parameters defining search range and step size
        method: Focus measure method ('laplacian', 'sobel', 'variance', 'tenengrad')
        stop_event: Threading event to check for cancellation (optional)
        roi: Optional region of interest (0-1 relative coordinates) to measure focus on.
            If provided, only the cropped region is used for focus evaluation.
        path: Folder to save the result in, as ``<path>/<name>/`` (data, probe images
            and the diagnostic plot; see ``AutoFocusResult.save``). Nothing is saved
            when it is None.
        name: Name of the result's folder under ``path``.
        pass_index: Which pass of a multi-pass sweep this is, 1-based. Reported in the
            progress payload so a viewer can say "pass 1/2" rather than restarting an
            apparently identical bar.
        total_passes: How many passes the caller is running in total.

    Returns:
        AutoFocusResult with all acquired data, or None if cancelled

    Example:
        >>> # Run autofocus with default parameters
        >>> best_z = run_autofocus(microscope)
        >>> print(f"Best focus at {best_z*1e6:.1f} μm")

        >>> # Custom autofocus with specific channel and range
        >>> z_params = ZParameters(zmin=-20e-6, zmax=20e-6, zstep=1e-6)
        >>> channel = ChannelSettings(name="DAPI", excitation_wavelength=365,
        ...                          emission_filter=EmissionFilter("450 nm", low=450), power=50, exposure_time=0.1)
        >>> best_z = run_autofocus(microscope, channel, z_params, method='sobel')

        >>> # Autofocus on a specific region
        >>> from fibsem.structures import FibsemRectangle
        >>> roi = FibsemRectangle(left=0.25, top=0.25, width=0.5, height=0.5)
        >>> best_z = run_autofocus(microscope, roi=roi)
    """

    state = microscope.parent.get_device_imaging_state("FM")
    if not state.allows_acquisition:
        raise ValueError(
            "Cannot start autofocus. "
            + microscope.parent.explain_imaging_state("FM", state)
        )

    # Held for the whole sweep, as a tileset holds it for a whole run. Without it
    # every step took and returned the channel on its own -- measured at 43 scopes
    # for one sweep, each a capture, a set and a restore, and each a visible flick
    # of the active view in the microscope's own UI. The depth count is what makes
    # this work: the moves and acquisitions inside open their own scopes and must
    # not restore between steps.
    with microscope.active_channel():
        # Set up default z parameters if not provided
        if z_parameters is None:
            z_parameters = ZParameters(zmin=-10e-6, zmax=10e-6, zstep=1e-6)

        # Generate z positions around current objective position
        z_positions = z_parameters.generate_positions(microscope.objective.position)

        # Validate focus measure method
        get_focus_measure_function(method)  # Will raise error if invalid

        # Apply channel settings if provided
        if channel_settings is not None:
            microscope.set_channel(channel_settings=channel_settings)

        logging.info(
            f"Starting autofocus: {len(z_positions)} positions, method='{method}'"
        )

        initial_z = microscope.objective.position
        iterations: list[AutoFocusIteration] = []

        for i, z_pos in enumerate(z_positions):
            if stop_event and stop_event.is_set():
                logging.info("Autofocus cancelled")
                microscope.objective.move_absolute(initial_z)
                return None

            # The same shape the z-stack and channel loops emit, so a viewer that
            # renders within-tile progress renders this too. Without it the bars froze for
            # the whole sweep -- which on per-tile autofocus is most of the run.
            #
            # Emitted at the top of the loop, before the objective move and the exposure,
            # so it lands microseconds after the sweep begins. That is what made a
            # separate sweep-start announcement redundant: it set "Running Autofocus…"
            # and reset the bar, and this overwrote both almost immediately with
            # something strictly better, having named the channel and the step.
            microscope.acquisition_progress_signal.emit(
                FluorescenceAcquisitionProgress(
                    status=FluorescenceAcquisitionStatus.ACQUIRING_AUTOFOCUS,
                    channel=channel_settings.name
                    if channel_settings is not None
                    else "",
                    zlevel=i + 1,
                    total_zlevels=len(z_positions),
                    pass_index=pass_index,
                    total_passes=total_passes,
                )
            )

            microscope.objective.move_absolute(z_pos)
            image = microscope.acquire_image()
            image_data = image.crop(roi) if roi is not None else image.data
            focus_score = calculate_focus_quality(image_data, method=method)
            iterations.append(
                AutoFocusIteration(
                    working_distance=float(z_pos),
                    focus_score=float(focus_score),
                    pass_index=0,
                    image=image_data,
                )
            )
            logging.debug(
                f"Z[{i + 1}/{len(z_positions)}]: {z_pos * 1e6:.1f} μm, Score: {focus_score:.4f}"
            )

        best_idx = int(np.argmax([it.focus_score for it in iterations]))
        best = iterations[best_idx]

        microscope.objective.move_absolute(best.working_distance)
        logging.info(
            f"Autofocus complete: Best position {best.working_distance * 1e6:.1f} μm (score: {best.focus_score:.4f})"
        )

        result = AutoFocusResult(
            image=best.image,
            working_distance=best.working_distance,
            initial_working_distance=initial_z,
            focus_score=best.focus_score,
            iterations=iterations,
            method=method,
        )

    if path is not None:
        result.save(path=path, name=name)

    return result


def run_coarse_fine_autofocus(
    microscope: "FluorescenceMicroscope",
    autofocus_settings: AutoFocusSettings,
    channel_settings: Optional["ChannelSettings"] = None,
    roi: Optional[FibsemRectangle] = None,
    stop_event=None,
    path: Optional[str] = None,
    name: str = "autofocus",
) -> Optional[AutoFocusResult]:
    """Run a multi-pass autofocus sweep over the objective z-axis.

    Each enabled pass in ``autofocus_settings.passes`` is one sweep centred on
    the best position found by the previous pass (e.g. a coarse pass followed by
    a narrower fine pass). Iterations from all passes are combined into a single
    flat result, tagged by ``pass_index``.

    Records one ``fm_autofocus`` event, however it ends (FIB-1257): when it
    started, and ``completed``; ``cancelled`` (a Stop, with the passes it
    finished); ``failed`` with the error (and re-raises); or ``skipped`` with no
    pass enabled. Given a ``path``, the result is saved to ``<path>/<name>``
    first and the event says where; a save that fails is logged, not raised.

    Args:
        microscope: The fluorescence microscope instance.
        autofocus_settings: Sweep passes + method.
        channel_settings: Channel to use; overrides autofocus_settings.channel_name.
        roi: Optional crop region (0-1 relative) for focus scoring.
            Falls back to ``autofocus_settings.reduced_area``.
        stop_event: Threading event checked for cancellation.
        path: The folder to save the result under, or None not to save it.
        name: The result's folder name under ``path``.

    Returns:
        Combined AutoFocusResult, the partial result if a later pass is
        cancelled, or None if the first pass is cancelled / no passes enabled.
    """
    from fibsem.util.timestamps import now_iso

    initial_position = microscope.objective.position
    roi = roi if roi is not None else autofocus_settings.reduced_area
    active_passes = [p for p in autofocus_settings.passes if p.enabled]
    record = {
        "method": autofocus_settings.method.value,
        "passes": len(active_passes),
        "initial_position": initial_position,
        "started_at": now_iso(),
    }

    if not active_passes:
        logging.warning("No enabled autofocus passes — skipping")
        _record_fm_autofocus(
            microscope,
            {**record, "status": SKIPPED, "reason": "no autofocus pass is enabled"},
        )
        return None

    try:
        last_result, all_iterations = _run_passes(
            microscope,
            active_passes,
            autofocus_settings,
            channel_settings,
            roi,
            stop_event,
        )
    except Exception as error:
        _record_fm_autofocus(microscope, {**record, **ended_by(error)})
        raise
    if last_result is None:
        _record_fm_autofocus(
            microscope, {**record, "status": CANCELLED, "completed_passes": 0}
        )
        return None

    logging.info(
        "Coarse/fine autofocus complete: WD=%.3f mm (adjustment %.1f µm from initial)",
        last_result.working_distance * 1e3,
        (last_result.working_distance - initial_position) * 1e6,
    )
    result = AutoFocusResult(
        image=last_result.image,
        working_distance=last_result.working_distance,
        initial_working_distance=initial_position,
        focus_score=last_result.focus_score,
        iterations=all_iterations,
        settings=autofocus_settings,
        method=last_result.method,
    )
    saved = None
    if path is not None:
        try:
            saved = str(result.save(path=path, name=name))
        except Exception as e:  # noqa: BLE001 - the focus is set either way
            logging.warning(f"Failed to save autofocus result: {e}")
    completed = len({it.pass_index for it in all_iterations})
    _record_fm_autofocus(
        microscope,
        {
            **record,
            "status": COMPLETED if completed == len(active_passes) else CANCELLED,
            "completed_passes": completed,
            "position": last_result.working_distance,
            "focus_score": last_result.focus_score,
            "path": saved,
        },
    )
    return result


def _run_passes(
    microscope: "FluorescenceMicroscope",
    active_passes: list,
    autofocus_settings: AutoFocusSettings,
    channel_settings: Optional["ChannelSettings"],
    roi: Optional[FibsemRectangle],
    stop_event,
) -> tuple:
    """Sweep each pass about the last one's best; the last pass's result and
    every iteration. ``(None, [])`` when the first pass is cancelled; a later
    one cancelled keeps what the passes before it found."""
    all_iterations: list = []
    last_result: Optional[AutoFocusResult] = None

    for pass_index, sweep_pass in enumerate(active_passes):
        # run_autofocus centres on the current objective position and moves to best
        result = run_autofocus(
            microscope=microscope,
            channel_settings=channel_settings,
            z_parameters=ZParameters.from_focus_pass(sweep_pass),
            method=autofocus_settings.method.value,
            roi=roi,
            stop_event=stop_event,
            pass_index=pass_index + 1,
            total_passes=len(active_passes),
        )
        if result is None:
            if last_result is None:
                logging.warning("Autofocus cancelled during first pass")
                return None, []
            logging.warning(
                "Autofocus cancelled at pass %d — keeping previous result", pass_index
            )
            break

        for it in result.iterations:
            it.pass_index = pass_index
        all_iterations.extend(result.iterations)
        last_result = result
    return last_result, all_iterations


def _record_fm_autofocus(microscope: "FluorescenceMicroscope", payload: dict) -> None:
    """For the experiment's record; it never raises. Nothing to record to
    without the microscope it is part of."""
    record = getattr(microscope.parent, "record_event", None)
    if record is not None:
        record("fm_autofocus", payload)
