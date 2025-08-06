"""
Time estimation utilities for fluorescence microscopy acquisitions.

This module provides functions to calculate the number of images and estimate
acquisition times for various FM acquisition modes including single images,
z-stacks, and tilesets.
"""

import logging
from typing import List, Optional, Tuple, Union

from fibsem.fm.structures import ChannelSettings, ZParameters
from fibsem.fm.structures import AutoFocusMode

# Timing constants for acquisition operations
DEFAULT_OVERHEAD_PER_IMAGE = 0.5  # seconds - camera readout, processing overhead
DEFAULT_Z_MOVE_TIME = 0.1  # seconds - time to move between z-positions
DEFAULT_STAGE_MOVE_TIME = 2.0  # seconds - time to move stage between tiles
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
        # Number of z-movements needed
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

    total_tiles = rows * cols

    # Calculate image acquisition time per tile
    tile_acquisition_time = estimate_acquisition_time(channel_settings, zparams)
    total_image_acquisition_time = tile_acquisition_time * total_tiles

    # Calculate total images
    total_images = calculate_total_images_count(channel_settings, zparams) * total_tiles

    # Calculate stage movement time
    # For a grid pattern: (cols-1) horizontal moves per row + (rows-1) vertical moves + row resets
    if total_tiles <= 1:
        # No movement needed for 0 or 1 tiles
        total_stage_moves = 0
    else:
        horizontal_moves = (cols - 1) * rows  # Moves within each row
        vertical_moves = rows - 1  # Moves to next row
        # Row resets only needed when there are multiple columns
        row_resets = (rows - 1) if cols > 1 else 0  # Return to first column of next row
        total_stage_moves = horizontal_moves + vertical_moves + row_resets

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
