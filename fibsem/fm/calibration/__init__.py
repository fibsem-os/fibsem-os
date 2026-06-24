"""Focus measurement and calibration utilities for fluorescence microscopy.

This module provides focus measure algorithms used in focus stacking and autofocus
applications. Different algorithms are suitable for different types of samples
and imaging conditions.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from fibsem.autofunctions.autofocus import AutoFocusIteration, AutoFocusResult
from fibsem.fm.structures import AutoFocusSettings, ZParameters
from fibsem.structures import FibsemRectangle
from fibsem.autofunctions.metrics import (
    get_focus_measure_function,
    laplacian_focus_measure,
    sobel_focus_measure,
    variance_focus_measure,
    tenengrad_focus_measure,
    find_best_focus_plane,
    calculate_focus_quality,
)
from fibsem.autofunctions.stacking import (
    block_based_focus_selection,
    create_focus_stack_from_selection,
    pixel_based_focus_selection,
    create_pixel_based_focus_stack,
    create_block_based_focus_stack,
)
from fibsem.autofunctions.integration import frame_integration, adaptive_frame_integration
from fibsem.autofunctions.plotting import plot_autofocus_result as plot_autofocus

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
    "run_multi_position_autofocus",
]

if TYPE_CHECKING:
    from fibsem.fm.microscope import FluorescenceMicroscope
    from fibsem.fm.structures import ChannelSettings, ZParameters
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import FibsemStagePosition


def run_autofocus(
    microscope: "FluorescenceMicroscope",
    channel_settings: Optional["ChannelSettings"] = None,
    z_parameters: Optional["ZParameters"] = None,
    method: str = "tenengrad",
    stop_event: Optional[threading.Event] = None,
    roi: Optional[FibsemRectangle] = None,
    save_plot: bool = False,
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
        save_plot: Whether to save a diagnostic plot of the autofocus results (default: False).

    Returns:
        AutoFocusResult with all acquired data, or None if cancelled

    Example:
        >>> # Run autofocus with default parameters
        >>> best_z = run_autofocus(microscope)
        >>> print(f"Best focus at {best_z*1e6:.1f} μm")

        >>> # Custom autofocus with specific channel and range
        >>> z_params = ZParameters(zmin=-20e-6, zmax=20e-6, zstep=1e-6)
        >>> channel = ChannelSettings(name="DAPI", excitation_wavelength=365,
        ...                          emission_wavelength=450, power=50, exposure_time=0.1)
        >>> best_z = run_autofocus(microscope, channel, z_params, method='sobel')

        >>> # Autofocus on a specific region
        >>> from fibsem.structures import FibsemRectangle
        >>> roi = FibsemRectangle(left=0.25, top=0.25, width=0.5, height=0.5)
        >>> best_z = run_autofocus(microscope, roi=roi)
    """

    if not microscope.has_valid_orientation():
        raise ValueError("Microscope orientation is not valid for autofocus")

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

    microscope.acquisition_progress_signal.emit({"state": "autofocus"})
    logging.info(f"Starting autofocus: {len(z_positions)} positions, method='{method}'")

    initial_z = microscope.objective.position
    iterations: list[AutoFocusIteration] = []

    for i, z_pos in enumerate(z_positions):
        if stop_event and stop_event.is_set():
            logging.info("Autofocus cancelled")
            microscope.objective.move_absolute(initial_z)
            return None

        microscope.objective.move_absolute(z_pos)
        image = microscope.acquire_image()
        image_data = image.crop(roi) if roi is not None else image.data
        focus_score = calculate_focus_quality(image_data, method=method)
        iterations.append(AutoFocusIteration(
            working_distance=float(z_pos),
            focus_score=float(focus_score),
            pass_index=0,
            image=image_data,
        ))
        logging.debug(
            f"Z[{i + 1}/{len(z_positions)}]: {z_pos * 1e6:.1f} μm, Score: {focus_score:.4f}"
        )

    best_idx = int(np.argmax([it.focus_score for it in iterations]))
    best = iterations[best_idx]

    microscope.objective.move_absolute(best.working_distance)
    logging.info(f"Autofocus complete: Best position {best.working_distance * 1e6:.1f} μm (score: {best.focus_score:.4f})")

    result = AutoFocusResult(
        image=best.image,
        working_distance=best.working_distance,
        initial_working_distance=initial_z,
        focus_score=best.focus_score,
        iterations=iterations,
        method=method,
    )

    if save_plot:
        plot_autofocus(result)

    return result


def run_coarse_fine_autofocus(
    microscope: "FluorescenceMicroscope",
    autofocus_settings: AutoFocusSettings,
    channel_settings: Optional["ChannelSettings"] = None,
    roi: Optional[FibsemRectangle] = None,
    stop_event=None,
) -> Optional[AutoFocusResult]:
    """Run a multi-pass autofocus sweep over the objective z-axis.

    Each enabled pass in ``autofocus_settings.passes`` is one sweep centred on
    the best position found by the previous pass (e.g. a coarse pass followed by
    a narrower fine pass). Iterations from all passes are combined into a single
    flat result, tagged by ``pass_index``.

    Args:
        microscope: The fluorescence microscope instance.
        autofocus_settings: Sweep passes + method.
        channel_settings: Channel to use; overrides autofocus_settings.channel_name.
        roi: Optional crop region (0-1 relative) for focus scoring.
            Falls back to ``autofocus_settings.reduced_area``.
        stop_event: Threading event checked for cancellation.

    Returns:
        Combined AutoFocusResult, the partial result if a later pass is
        cancelled, or None if the first pass is cancelled / no passes enabled.
    """
    initial_position = microscope.objective.position
    roi = roi if roi is not None else autofocus_settings.reduced_area
    active_passes = [p for p in autofocus_settings.passes if p.enabled]

    if not active_passes:
        logging.warning("No enabled autofocus passes — skipping")
        return None

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
        )
        if result is None:
            if last_result is None:
                logging.warning("Autofocus cancelled during first pass")
                return None
            logging.warning("Autofocus cancelled at pass %d — keeping previous result", pass_index)
            break

        for it in result.iterations:
            it.pass_index = pass_index
        all_iterations.extend(result.iterations)
        last_result = result

    logging.info(
        "Coarse/fine autofocus complete: WD=%.3f mm (adjustment %.1f µm from initial)",
        last_result.working_distance * 1e3,
        (last_result.working_distance - initial_position) * 1e6,
    )

    return AutoFocusResult(
        image=last_result.image,
        working_distance=last_result.working_distance,
        initial_working_distance=initial_position,
        focus_score=last_result.focus_score,
        iterations=all_iterations,
        settings=autofocus_settings,
        method=last_result.method,
    )


def run_multi_position_autofocus(
    fibsem_microscope: "FibsemMicroscope",
    positions: List["FibsemStagePosition"],
    channel_settings: Optional["ChannelSettings"] = None,
    z_parameters: Optional["ZParameters"] = None,
    method: str = "tenengrad",
    return_to_start: bool = True,
) -> Dict[str, float]:
    """Run autofocus at multiple stage positions and return focus map.

    NOTE: This function requires the main FIBSEM microscope for stage movement
    and uses its fluorescence microscope (fm) for autofocus.

    Useful for characterizing sample tilt or mapping optimal focus across
    a large field of view before starting acquisition.

    Args:
        fibsem_microscope: The main FIBSEM microscope instance (has stage control)
        positions: List of stage positions to test
        channel_settings: Channel settings for autofocus
        z_parameters: Z-stack parameters for search
        method: Focus measure method
        return_to_start: Whether to return to initial position when done

    Returns:
        Dictionary mapping position names to dict with 'focus_z' and 'stage_position'

    Example:
        >>> # Create test positions
        >>> positions = [
        ...     FibsemStagePosition(x=0, y=0, z=0, name="center"),
        ...     FibsemStagePosition(x=100e-6, y=0, z=0, name="right"),
        ...     FibsemStagePosition(x=0, y=100e-6, z=0, name="top")
        ... ]
        >>> focus_map = run_multi_position_autofocus(fibsem_microscope, positions)
        >>> for pos_name, data in focus_map.items():
        ...     print(f"{pos_name}: {data['focus_z']*1e6:.1f} μm at {data['stage_position']}")
    """
    if not positions:
        raise ValueError("At least one position must be provided")

    # Check that fluorescence microscope is available
    if not hasattr(fibsem_microscope, "fm") or fibsem_microscope.fm is None:
        raise ValueError(
            "FIBSEM microscope must have fluorescence microscope (fm) available"
        )

    # Store initial state
    initial_position = fibsem_microscope.get_stage_position()
    initial_objective_position = fibsem_microscope.fm.objective.position

    focus_map = {}

    try:
        logging.info(f"Multi-position autofocus: {len(positions)} positions")

        for i, position in enumerate(positions):
            logging.info(f"Position {i + 1}/{len(positions)}: {position.name}")

            # Move to test position using FIBSEM stage
            fibsem_microscope.safe_absolute_stage_movement(position)

            # Run autofocus using fluorescence microscope
            result = run_autofocus(
                microscope=fibsem_microscope.fm,
                channel_settings=channel_settings,
                z_parameters=z_parameters,
                method=method,
            )

            # Get actual stage position after movement (for verification)
            actual_position = fibsem_microscope.get_stage_position()

            focus_map[position.name] = {
                "focus_z": result.working_distance if result is not None else None,
                "stage_position": actual_position,
            }

        logging.info("Multi-position autofocus complete")

        # Log results summary
        for pos_name, data in focus_map.items():
            focus_z = data["focus_z"]
            stage_pos = data["stage_position"]
            focus_z_str = f"{focus_z * 1e6:.1f} μm" if focus_z is not None else "cancelled"
            logging.info(
                f"  {pos_name}: {focus_z_str} at ({stage_pos.x * 1e6:.1f}, {stage_pos.y * 1e6:.1f}) μm"
            )

    finally:
        if return_to_start:
            logging.info("Restoring initial objective and stage positions")
            # First restore objective to initial position to avoid stage movement issues
            fibsem_microscope.fm.objective.move_absolute(initial_objective_position)
            # Then move stage back to initial position
            fibsem_microscope.safe_absolute_stage_movement(initial_position)

    return focus_map
