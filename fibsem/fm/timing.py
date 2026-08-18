"""
Time estimation utilities for fluorescence microscopy acquisitions.

This module provides functions to calculate the number of images and estimate
acquisition times for various FM acquisition modes including single images,
z-stacks, and tilesets.
"""

import logging
from typing import List, Optional, Tuple, Union

from fibsem.fm.structures import ChannelSettings, ZParameters, ZStackOrder
from fibsem.fm.structures import AutoFocusMode, AutoFocusSettings

# Timing constants for acquisition operations
# Camera readout, processing and save, per image, on top of the exposure. Measured
# against AutoLamella-2026-07-29-18-00-DEV-TEST (Thermo): a 42-image z-stack ran at
# 2.31 s/image net of exposure and z movement, and a 32-image autofocus sweep at
# 1.57 s/image. The earlier 0.5 s was an assumption, ~4x low, and it is what left the
# fluorescence duration estimate at a third of the real figure.
#
# The two figures differ because the z-stack writes each frame to OME-TIFF and the
# autofocus sweep does not, so this is really two costs wearing one name. It takes the
# higher of the two: over-estimating a sweep is the safe direction, and splitting the
# constant would change every caller's shape for a term neither of them separates yet.
DEFAULT_OVERHEAD_PER_IMAGE = 2.3  # seconds - camera readout, processing, save
DEFAULT_Z_MOVE_TIME = 0.1  # seconds - time to move between z-positions
DEFAULT_STAGE_MOVE_TIME = 5.0  # seconds - time to move stage between tiles
DEFAULT_AUTOFOCUS_TIME = 5.0  # seconds - time for each autofocus operation


def calculate_total_images_count(
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    zparams: Optional[ZParameters] = None,
) -> int:
    """Calculate the total number of images that will be taken with the given settings.

    Args:
        channel_settings: Single channel or list of channels to acquire
        zparams: Optional ZParameters for Z-stack acquisition

    Returns:
        int: Total number of images that will be acquired

    Example:
        >>> channel1 = ChannelSettings(name="DAPI", excitation_wavelength=365,
        ...                           emission_wavelength=450, power=0.5, exposure_time=0.1)
        >>> channel2 = ChannelSettings(name="FITC", excitation_wavelength=488,
        ...                           emission_wavelength=525, power=0.3, exposure_time=0.05)
        >>> zparams = ZParameters(zmin=-5e-6, zmax=5e-6, zstep=1e-6)
        >>>
        >>> # Single channel, no z-stack
        >>> count = calculate_total_images_count(channel1)
        >>> print(count)  # 1
        >>>
        >>> # Multiple channels, no z-stack
        >>> count = calculate_total_images_count([channel1, channel2])
        >>> print(count)  # 2
        >>>
        >>> # Single channel with z-stack
        >>> count = calculate_total_images_count(channel1, zparams)
        >>> print(count)  # 11 (11 z-planes)
        >>>
        >>> # Multiple channels with z-stack
        >>> count = calculate_total_images_count([channel1, channel2], zparams)
        >>> print(count)  # 22 (2 channels × 11 z-planes)
    """
    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]

    num_channels = len(channel_settings)

    if zparams is None:
        # No z-stack, one image per channel
        num_z_planes = 1
    else:
        # Z-stack acquisition
        num_z_planes = zparams.num_planes

    return num_channels * num_z_planes


def estimate_autofocus_time(
    autofocus_settings: "AutoFocusSettings",
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
) -> float:
    """Estimate an image-based autofocus sweep from the passes it will actually run.

    Each enabled pass sweeps ``search_range`` in ``step_size`` increments, taking one
    image per step -- ``FocusSweepPass.n_steps`` is exactly that count -- so the sweep
    is arithmetic rather than a constant. With the default coarse/fine pair (50 um at
    5 um, then 10 um at 1 um) that is around twenty images, several times the 5 s
    ``DEFAULT_AUTOFOCUS_TIME`` assumes.

    The channel is resolved the way the acquisition path resolves it: by name when
    ``channel_name`` is set, otherwise the first channel.

    ``DEFAULT_AUTOFOCUS_TIME`` is still used by the tileset estimators below, which
    are handed an autofocus *mode* rather than the settings and so cannot count steps.

    Args:
        autofocus_settings: the sweep to estimate
        channel_settings: the channels available to focus on

    Returns:
        float: estimated autofocus time in seconds, 0.0 if no pass is enabled
    """
    if autofocus_settings is None or not autofocus_settings.enabled:
        return 0.0
    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]
    if not channel_settings:
        return 0.0

    channel = None
    if autofocus_settings.channel_name:
        channel = next(
            (c for c in channel_settings if c.name == autofocus_settings.channel_name), None
        )
    if channel is None:
        channel = channel_settings[0]

    # n_steps + 1, matching the sweep itself: run_autofocus builds its working
    # distances with np.linspace(..., n_steps + 1), so a pass covering n steps takes
    # n + 1 images. The measured run logged "21 positions" for a 20-step pass and
    # "11 positions" for a 10-step one.
    n_images = sum(p.n_steps + 1 for p in autofocus_settings.passes if p.enabled)
    # one objective move between consecutive steps, as in the z-stack above
    n_moves = max(n_images - 1, 0)
    return (
        n_images * (channel.exposure_time + DEFAULT_OVERHEAD_PER_IMAGE)
        + n_moves * DEFAULT_Z_MOVE_TIME
    )


def estimate_acquisition_time(
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    zparams: Optional[ZParameters] = None,
) -> float:
    """Estimate the total time required for image acquisition.

    Args:
        channel_settings: Single channel or list of channels to acquire
        zparams: Optional ZParameters for Z-stack acquisition

    Returns:
        float: Estimated total acquisition time in seconds

    Example:
        >>> channel1 = ChannelSettings(name="DAPI", excitation_wavelength=365,
        ...                           emission_wavelength=450, power=0.5, exposure_time=0.1)
        >>> channel2 = ChannelSettings(name="FITC", excitation_wavelength=488,
        ...                           emission_wavelength=525, power=0.3, exposure_time=0.05)
        >>> zparams = ZParameters(zmin=-5e-6, zmax=5e-6, zstep=1e-6)
        >>>
        >>> # Single channel, no z-stack
        >>> time_s = estimate_acquisition_time(channel1)
        >>> print(f"Estimated time: {time_s:.1f}s")  # ~0.6s (0.1s exposure + 0.5s overhead)
        >>>
        >>> # Multiple channels, no z-stack
        >>> time_s = estimate_acquisition_time([channel1, channel2])
        >>> print(f"Estimated time: {time_s:.1f}s")  # ~1.1s (0.15s exposure + 1.0s overhead)
        >>>
        >>> # Single channel with z-stack
        >>> time_s = estimate_acquisition_time(channel1, zparams)
        >>> print(f"Estimated time: {time_s:.1f}s")  # ~7.1s (1.1s exposure + 5.5s overhead + 1.0s z-moves)
        >>>
        >>> # Multiple channels with z-stack
        >>> time_s = estimate_acquisition_time([channel1, channel2], zparams)
        >>> print(f"Estimated time: {time_s:.1f}s")  # ~12.65s
    """
    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]

    # Calculate total exposure time
    total_exposure_time = 0.0
    for channel in channel_settings:
        total_exposure_time += channel.exposure_time

    # Calculate total number of images
    total_images = calculate_total_images_count(channel_settings, zparams)

    # Calculate overhead time (readout, processing, etc.)
    total_overhead_time = total_images * DEFAULT_OVERHEAD_PER_IMAGE

    # Calculate z-movement time
    z_movement_time = 0.0
    if zparams is not None and zparams.num_planes > 1:
        # Channel-wise: objective traverses full z-range once per channel
        # Z-level-wise: objective traverses full z-range once total
        if zparams.order == ZStackOrder.Z_LEVEL:
            num_z_moves = zparams.num_planes - 1
        else:
            num_z_moves = (zparams.num_planes - 1) * len(channel_settings)
        z_movement_time = num_z_moves * DEFAULT_Z_MOVE_TIME

    # For z-stack: multiply exposure time by number of z-planes
    if zparams is not None:
        total_exposure_time *= zparams.num_planes

    total_time = total_exposure_time + total_overhead_time + z_movement_time

    return total_time


def estimate_tileset_acquisition_time(
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    grid_size: Tuple[int, int],
    zparams: Optional[ZParameters] = None,
    autofocus_mode: AutoFocusMode = AutoFocusMode.NONE,
    tile_mask: Optional[List[List[bool]]] = None,
) -> dict:
    """Estimate the total time required for tileset acquisition.

    This function estimates timing for a complete tileset acquisition, including
    stage movements, optional autofocus operations, and all image acquisitions.

    Args:
        channel_settings: Single channel or list of channels to acquire
        grid_size: Tuple of (rows, cols) defining the grid dimensions
        zparams: Optional ZParameters for Z-stack acquisition
        autofocus_mode: Autofocus mode - AutoFocusMode enum (default: AutoFocusMode.NONE)

    Returns:
        dict: Dictionary containing detailed timing breakdown:
            - total_time: Total estimated acquisition time in seconds
            - image_acquisition_time: Time spent on image acquisition
            - stage_movement_time: Time spent moving the stage
            - autofocus_time: Time spent on autofocus operations
            - total_images: Total number of images to be acquired
            - tiles: Number of tiles in the grid
            - breakdown: Detailed breakdown by category

    Example:
        >>> channel1 = ChannelSettings(name="DAPI", excitation_wavelength=365,
        ...                           emission_wavelength=450, power=0.5, exposure_time=0.1)
        >>> channel2 = ChannelSettings(name="FITC", excitation_wavelength=488,
        ...                           emission_wavelength=525, power=0.3, exposure_time=0.05)
        >>> zparams = ZParameters(zmin=-2e-6, zmax=2e-6, zstep=1e-6)  # 5 z-planes
        >>>
        >>> from fibsem.fm.structures import AutoFocusMode
        >>>
        >>> # 3x3 tileset, multiple channels, z-stack, autofocus each tile
        >>> timing = estimate_tileset_acquisition_time(
        ...     [channel1, channel2],
        ...     grid_size=(3, 3),
        ...     zparams=zparams,
        ...     autofocus_mode=AutoFocusMode.EACH_TILE
        ... )
        >>> print(f"Total time: {timing['total_time']/60:.1f} minutes")
        >>> print(f"Total images: {timing['total_images']}")
    """
    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]

    rows, cols = grid_size

    # Validate grid dimensions
    if rows <= 0 or cols <= 0:
        raise ValueError(
            f"Grid dimensions must be positive integers, got rows={rows}, cols={cols}"
        )

    # A masked run visits only the enabled tiles, and only the rows that contain one.
    # Estimating the full grid instead reports roughly three times the real duration
    # for a typical sparse selection -- and the estimate is what the confirmation
    # dialog and the remaining-time readout are built on.
    if tile_mask is not None:
        total_tiles = sum(bool(v) for row in tile_mask for v in row)
        rows = sum(1 for row in tile_mask if any(row))
    else:
        total_tiles = rows * cols

    # Calculate image acquisition time per tile
    tile_acquisition_time = estimate_acquisition_time(channel_settings, zparams)
    total_image_acquisition_time = tile_acquisition_time * total_tiles

    # Calculate total images
    total_images = calculate_total_images_count(channel_settings, zparams) * total_tiles

    # Calculate stage movement time. One absolute move per tile: the runner projects
    # every tile position up front and drives straight to each one, so there are no
    # separate row-advance or row-reset moves to count. (The previous model counted
    # those, and also assumed a full rectangular sweep, which a mask breaks.)
    total_stage_moves = max(0, total_tiles - 1)

    total_stage_movement_time = total_stage_moves * DEFAULT_STAGE_MOVE_TIME

    # Calculate autofocus time
    total_autofocus_time = 0.0
    autofocus_operations = 0

    # Handle AutoFocusMode enum
    if autofocus_mode == AutoFocusMode.ONCE:
        autofocus_operations = 1
    elif autofocus_mode == AutoFocusMode.EACH_ROW:
        autofocus_operations = rows
    elif autofocus_mode == AutoFocusMode.EACH_TILE:
        autofocus_operations = total_tiles
    elif autofocus_mode == AutoFocusMode.NONE:
        autofocus_operations = 0
    else:
        # Handle unknown enum value
        logging.warning(
            f"Unknown autofocus mode: {autofocus_mode}, defaulting to no autofocus"
        )
        autofocus_operations = 0

    total_autofocus_time = autofocus_operations * DEFAULT_AUTOFOCUS_TIME

    # Calculate total time
    total_time = (
        total_image_acquisition_time + total_stage_movement_time + total_autofocus_time
    )

    # Create detailed breakdown
    breakdown = {
        "image_acquisition": {
            "time_per_tile": tile_acquisition_time,
            "total_time": total_image_acquisition_time,
            "images_per_tile": calculate_total_images_count(channel_settings, zparams),
            "percentage": (total_image_acquisition_time / total_time * 100)
            if total_time > 0
            else 0,
        },
        "stage_movement": {
            "moves_per_tile": total_stage_moves / total_tiles if total_tiles > 0 else 0,
            "total_moves": total_stage_moves,
            "time_per_move": DEFAULT_STAGE_MOVE_TIME,
            "total_time": total_stage_movement_time,
            "percentage": (total_stage_movement_time / total_time * 100)
            if total_time > 0
            else 0,
        },
        "autofocus": {
            "mode": autofocus_mode.name,
            "operations": autofocus_operations,
            "time_per_operation": DEFAULT_AUTOFOCUS_TIME,
            "total_time": total_autofocus_time,
            "percentage": (total_autofocus_time / total_time * 100)
            if total_time > 0
            else 0,
        },
        "grid_info": {
            "rows": rows,
            "cols": cols,
            "total_tiles": total_tiles,
            "channels": len(channel_settings),
            "z_planes": 1 if zparams is None else zparams.num_planes,
        },
    }

    return {
        "total_time": total_time,
        "image_acquisition_time": total_image_acquisition_time,
        "stage_movement_time": total_stage_movement_time,
        "autofocus_time": total_autofocus_time,
        "total_images": total_images,
        "tiles": total_tiles,
        "breakdown": breakdown,
    }


def estimate_positions_acquisition_time(
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    num_positions: int,
    zparams: Optional[ZParameters] = None,
    use_autofocus: bool = False,
) -> dict:
    """Estimate the total time required for multi-position acquisition.

    This function estimates timing for acquiring images at multiple positions,
    including stage movements, optional autofocus operations, and all image acquisitions.

    Args:
        channel_settings: Single channel or list of channels to acquire
        num_positions: Number of positions to acquire at
        zparams: Optional ZParameters for Z-stack acquisition
        use_autofocus: Whether autofocus will be performed at each position

    Returns:
        dict: Dictionary containing detailed timing breakdown:
            - total_time: Total estimated acquisition time in seconds
            - image_acquisition_time: Time spent on image acquisition
            - stage_movement_time: Time spent moving the stage
            - autofocus_time: Time spent on autofocus operations
            - total_images: Total number of images to be acquired
            - positions: Number of positions
            - breakdown: Detailed breakdown by category

    Example:
        >>> channel1 = ChannelSettings(name="DAPI", excitation_wavelength=365,
        ...                           emission_wavelength=450, power=0.5, exposure_time=0.1)
        >>> channel2 = ChannelSettings(name="FITC", excitation_wavelength=488,
        ...                           emission_wavelength=525, power=0.3, exposure_time=0.05)
        >>> zparams = ZParameters(zmin=-2e-6, zmax=2e-6, zstep=1e-6)  # 5 z-planes
        >>>
        >>> # 5 positions, multiple channels, z-stack, autofocus at each position
        >>> timing = estimate_positions_acquisition_time(
        ...     [channel1, channel2],
        ...     num_positions=5,
        ...     zparams=zparams,
        ...     use_autofocus=True
        ... )
        >>> print(f"Total time: {timing['total_time']/60:.1f} minutes")
        >>> print(f"Total images: {timing['total_images']}")
    """
    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]

    # Validate inputs
    if num_positions <= 0:
        raise ValueError(f"Number of positions must be positive, got {num_positions}")

    # Calculate image acquisition time per position
    position_acquisition_time = estimate_acquisition_time(channel_settings, zparams)
    total_image_acquisition_time = position_acquisition_time * num_positions

    # Calculate total images
    total_images = calculate_total_images_count(channel_settings, zparams) * num_positions

    # Calculate stage movement time (assuming we move between each position)
    # For n positions, we need (n-1) moves
    total_stage_moves = max(0, num_positions - 1)
    total_stage_movement_time = total_stage_moves * DEFAULT_STAGE_MOVE_TIME # TODO: actually calculate the distance covered.

    # Calculate autofocus time
    autofocus_operations = num_positions if use_autofocus else 0
    total_autofocus_time = autofocus_operations * DEFAULT_AUTOFOCUS_TIME

    # Calculate total time
    total_time = (
        total_image_acquisition_time + total_stage_movement_time + total_autofocus_time
    )

    # Create detailed breakdown
    breakdown = {
        "image_acquisition": {
            "time_per_position": position_acquisition_time,
            "total_time": total_image_acquisition_time,
            "images_per_position": calculate_total_images_count(channel_settings, zparams),
            "percentage": (total_image_acquisition_time / total_time * 100)
            if total_time > 0
            else 0,
        },
        "stage_movement": {
            "moves_per_position": total_stage_moves / num_positions if num_positions > 0 else 0,
            "total_moves": total_stage_moves,
            "time_per_move": DEFAULT_STAGE_MOVE_TIME,
            "total_time": total_stage_movement_time,
            "percentage": (total_stage_movement_time / total_time * 100)
            if total_time > 0
            else 0,
        },
        "autofocus": {
            "use_autofocus": use_autofocus,
            "operations": autofocus_operations,
            "time_per_operation": DEFAULT_AUTOFOCUS_TIME,
            "total_time": total_autofocus_time,
            "percentage": (total_autofocus_time / total_time * 100)
            if total_time > 0
            else 0,
        },
        "position_info": {
            "num_positions": num_positions,
            "channels": len(channel_settings),
            "z_planes": 1 if zparams is None else zparams.num_planes,
        },
    }

    return {
        "total_time": total_time,
        "image_acquisition_time": total_image_acquisition_time,
        "stage_movement_time": total_stage_movement_time,
        "autofocus_time": total_autofocus_time,
        "total_images": total_images,
        "positions": num_positions,
        "breakdown": breakdown,
    }
