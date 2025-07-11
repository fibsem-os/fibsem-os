import logging
from copy import deepcopy
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fibsem.fm.calibration import run_autofocus
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings, FluorescenceImage, ZParameters
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, FibsemStagePosition


class AutofocusMode(Enum):
    """Auto-focus modes for tileset acquisition."""
    NONE = "none"
    ONCE = "once"
    EACH_ROW = "each_row"
    EACH_TILE = "each_tile"


def acquire_channels(
    microscope: FluorescenceMicroscope, 
    channel_settings: Union[ChannelSettings, List[ChannelSettings]]
) -> FluorescenceImage:
    """Acquire images for multiple channels."""
    
    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]  # Ensure settings is a list

    images: List[FluorescenceImage] = []
    for channel in channel_settings:
        image = microscope.acquire_image(channel)
        images.append(image)
    return FluorescenceImage.create_multi_channel_image(images)

def acquire_z_stack(
    microscope: FluorescenceMicroscope,
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    zparams: ZParameters,
) -> FluorescenceImage:
    """Acquire a Z-stack of images for a given channel."""

    z_init = microscope.objective.position  # initial z position of the objective
    z_positions = zparams.generate_positions(z_init=z_init)
    images: List[FluorescenceImage] = []

    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]

    for ch in channel_settings:

        ch_images: List[FluorescenceImage] = []
        for z in z_positions:
            # Move objective to the specified z position
            microscope.objective.move_absolute(z)
            # Acquire image at the current z position
            image = microscope.acquire_image(channel_settings=ch)
            ch_images.append(image)

        # stack the images along the z-axis
        zstack = FluorescenceImage.create_z_stack(ch_images)
        images.append(zstack)

    # restore objective to initial position
    microscope.objective.move_absolute(z_init)

    return FluorescenceImage.create_multi_channel_image(images)

def acquire_image(microscope: FluorescenceMicroscope,
                  channel_settings: Union[ChannelSettings, List[ChannelSettings]], 
                  zparams: Optional[ZParameters] = None) -> FluorescenceImage:
    """Acquire a fluroescence image for a single channel or multiple channels.
    If zparams is provided, a Z-stack will be acquired instead.
    Args:
        microscope: The fluorescence microscope instance
        channel_settings: Single channel or list of channels to acquire
        zparams: ZParameters for Z-stack acquisition (optional)
    Returns:
            FluorescenceImage object containing the acquired image(s)"""
    
    if zparams is not None:
        # Acquire Z-stack if zparams is provided
        return acquire_z_stack(microscope, channel_settings, zparams)
    
    # Acquire single image(s) for specified channel(s)
    return acquire_channels(microscope, channel_settings)

# Autofocus functions have been moved to fibsem.fm.calibration

def acquire_at_positions(
    microscope: FibsemMicroscope,
    positions: List[FibsemStagePosition],
    channel_settings: Union[ChannelSettings, List[ChannelSettings]], 
    zparams: Optional[ZParameters] = None,
    use_autofocus: bool = False,
    ) -> List[FluorescenceImage]:
    """Acquire fluorescence images at specified stage positions.
    This function moves the stage to each specified position and acquires images
    for the given channel settings. If zparams is provided, a Z-stack will be acquired
    at each position.
    Args:
        microscope: The fluorescence microscope instance
        positions: List of FibsemStagePosition objects defining where to acquire images
        channel_settings: Single channel or list of channels to acquire
        zparams: ZParameters for Z-stack acquisition (optional)
        use_autofocus: Whether to run autofocus at each position (default: False)
    Returns:
        List of FluorescenceImage objects containing the acquired images
    Raises:
        ValueError: If positions is empty or contains invalid stage positions
    Example:
        >>> positions = [FibsemStagePosition(x=0, y=0, z=0),
        ...             FibsemStagePosition(x=10e-6, y=0, z=0)]
        >>> channel = ChannelSettings(name="DAPI", excitation_wavelength=365, 
        ...                          emission_wavelength=450, power=50, exposure_time=0.1)
        >>> images = acquire_at_positions(microscope, positions, channel)
    """ 
    if not positions:
        raise ValueError("Positions list cannot be empty")
    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]

    images: List[FluorescenceImage] = []
    for i, pos in enumerate(positions):
        logging.info(f"Moving to position {i}/{len(positions)}: {pos}")
        microscope.safe_absolute_stage_movement(pos)
        
        # Run autofocus if requested
        if use_autofocus:
            logging.info(f"Running autofocus at position: {pos}")
            run_autofocus(microscope.fm, channel_settings[0])
            logging.info("Running autofocus")

        # Acquire image(s) at the current position
        image = acquire_image(microscope.fm, channel_settings, zparams)
        images.append(image)
        
        logging.info(f"Acquired image at position: {pos}")

    return images

# TODO: handle multiple channels properly
def acquire_tileset(
    microscope: FibsemMicroscope,
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    grid_size: Tuple[int, int],
    tile_overlap: float = 0.1,
    zparams: Optional[ZParameters] = None,
    beam_type: BeamType = BeamType.ELECTRON,
    autofocus_mode: AutofocusMode = AutofocusMode.NONE,
    autofocus_channel: Optional[ChannelSettings] = None,
    autofocus_zparams: Optional[ZParameters] = None,
) -> List[List[FluorescenceImage]]:
    """Acquire a tileset of fluorescence images across a grid pattern.
    
    This function moves the stage in a grid pattern to acquire multiple overlapping
    fluorescence images that can later be stitched together to create a larger
    field of view mosaic. The stage is moved using stable_move to maintain proper
    focus and positioning.
    
    Args:
        microscope: The fluorescence microscope instance
        channel_settings: Single channel or list of channels to acquire
        grid_size: Tuple of (rows, cols) defining the grid dimensions
        tile_overlap: Fraction of overlap between adjacent tiles (0.0 to 1.0)
        zparams: Optional Z parameters for z-stack acquisition
        beam_type: Beam type to use for stage movements (default: ELECTRON)
        autofocus_mode: Auto-focus mode for tileset acquisition (default: NONE)
                       - NONE: No auto-focus
                       - ONCE: Auto-focus once before acquisition
                       - EACH_ROW: Auto-focus at start of each row
                       - EACH_TILE: Auto-focus at each tile position
        autofocus_channel: Channel settings to use for auto-focus (uses first channel if None)
        autofocus_zparams: Z parameters for auto-focus search range (uses zparams if None)
        
    Returns:
        List of lists containing FluorescenceImage objects organized as [row][col]
        
    Raises:
        ValueError: If grid_size contains non-positive values or overlap is invalid
        
    Example:
        >>> # Acquire a 3x3 grid with 10% overlap
        >>> channel = ChannelSettings(name="DAPI", excitation_wavelength=365, 
        ...                          emission_wavelength=450, power=50, exposure_time=0.1)
        >>> tileset = acquire_tileset(microscope, channel, grid_size=(3, 3), tile_overlap=0.1)
        >>> print(f"Acquired {len(tileset)}x{len(tileset[0])} tiles")
    """
    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]

    rows, cols = grid_size
    if rows <= 0 or cols <= 0:
        raise ValueError("Grid size must contain positive values")

    if not 0.0 <= tile_overlap < 1.0:
        raise ValueError("Tile overlap must be between 0.0 and 1.0 (exclusive)")

    logging.info(f"Starting tileset acquisition: {rows}x{cols} grid with {tile_overlap:.1%} overlap")

    # Store initial position to return to later
    initial_position = microscope.get_stage_position()

    # Calculate the physical field of view from metadata
    pixel_size_x, pixel_size_y = microscope.fm.camera.pixel_size
    image_width, image_height = microscope.fm.camera.resolution
    fov_x = image_width * pixel_size_x
    fov_y = image_height * pixel_size_y

    # Calculate step size accounting for overlap
    step_x = fov_x * (1.0 - tile_overlap)
    step_y = fov_y * (1.0 - tile_overlap)

    logging.info(f"Field of view: {fov_x*1e6:.1f} x {fov_y*1e6:.1f} μm")
    logging.info(f"Step size: {step_x*1e6:.1f} x {step_y*1e6:.1f} μm")

    # Set up auto-focus parameters
    if autofocus_mode != AutofocusMode.NONE:
        if autofocus_channel is None:
            autofocus_channel = channel_settings[0] if isinstance(channel_settings, list) else channel_settings
        if autofocus_zparams is None:
            autofocus_zparams = zparams
        logging.info(f"Auto-focus mode: {autofocus_mode.value}")
        
        # Perform initial auto-focus if mode is ONCE (before moving to starting position)
        if autofocus_mode == AutofocusMode.ONCE:
            logging.info("Performing initial auto-focus at current position")
            try:
                run_autofocus(
                    microscope=microscope.fm,
                    channel_settings=autofocus_channel,
                    z_parameters=autofocus_zparams,
                    method='laplacian'
                )
                logging.info("Initial auto-focus completed")
            except Exception as e:
                logging.warning(f"Initial auto-focus failed: {e}")

    # Calculate starting position (top-left corner of grid)
    start_offset_x = -(cols - 1) * step_x / 2
    start_offset_y = -(rows - 1) * step_y / 2

    # Move to starting position
    logging.info("Moving to grid starting position")
    microscope.stable_move(dx=start_offset_x, dy=start_offset_y, beam_type=beam_type)

    # Initialize results array
    tileset = []

    # TODO: migrate to generate_grid_positions
    try:
        for row in range(rows):
            row_images = []
            
            # Perform auto-focus at start of each row
            if autofocus_mode == AutofocusMode.EACH_ROW:
                logging.info(f"Performing auto-focus for row {row+1}")
                try:
                    run_autofocus(
                        microscope=microscope.fm,
                        channel_settings=autofocus_channel,
                        z_parameters=autofocus_zparams,
                        method='laplacian'
                    )
                    logging.info(f"Auto-focus completed for row {row+1}")
                except Exception as e:
                    logging.warning(f"Auto-focus failed for row {row+1}: {e}")

            for col in range(cols):
                # Perform auto-focus at each tile
                if autofocus_mode == AutofocusMode.EACH_TILE:
                    logging.info(f"Performing auto-focus for tile [{row+1}/{rows}][{col+1}/{cols}]")
                    try:
                        run_autofocus(
                            microscope=microscope.fm,
                            channel_settings=autofocus_channel,
                            z_parameters=autofocus_zparams,
                            method='laplacian'
                        )
                        logging.info(f"Auto-focus completed for tile [{row+1}/{rows}][{col+1}/{cols}]")
                    except Exception as e:
                        logging.warning(f"Auto-focus failed for tile [{row+1}/{rows}][{col+1}/{cols}]: {e}")
                
                logging.info(f"Acquiring tile [{row+1}/{rows}][{col+1}/{cols}]")

                # Acquire all channels at current position
                tile_image = acquire_image(microscope.fm, channel_settings, zparams)
                logging.info(f"Stage position for tile [{row+1}/{rows}][{col+1}/{cols}]: {tile_image.metadata.stage_position}")

                if zparams is not None:
                    tile_image = tile_image.max_intensity_projection()
                row_images.append(tile_image)

                # Move to next column position (except for last column)
                if col < cols - 1:
                    microscope.stable_move(dx=step_x, dy=0, beam_type=beam_type)

            tileset.append(row_images)

            # Move to next row (except for last row)
            if row < rows - 1:
                # Return to first column of next row
                microscope.stable_move(dx=-(cols - 1) * step_x, dy=step_y, beam_type=beam_type)

    except Exception as e:
        logging.error(f"Error during tileset acquisition: {e}")
        raise

    finally:
        # Return to initial position
        logging.info("Returning to initial position")
        microscope.safe_absolute_stage_movement(initial_position)

    logging.info(f"Tileset acquisition complete: {rows}x{cols} tiles acquired")
    return tileset


def stitch_tileset(tileset: List[List[FluorescenceImage]], 
                   tile_overlap: float = 0.1) -> FluorescenceImage:
    """Stitch a tileset of 2D fluorescence images into a single mosaic image.
    
    Simple concatenation stitching for 2D images (YX format). Overlapping regions
    are handled by taking pixels from the rightmost/bottommost tile.
    
    Args:
        tileset: List of lists containing FluorescenceImage objects [row][col]
        tile_overlap: Fraction of overlap between adjacent tiles (0.0 to 1.0)
            
    Returns:
        Single FluorescenceImage containing the stitched mosaic
        
    Raises:
        ValueError: If tileset is empty or irregular
        
    Example:
        >>> tileset = acquire_tileset(microscope, channel, grid_size=(3, 3))
        >>> mosaic = stitch_tileset(tileset, tile_overlap=0.1)
        >>> print(f"Mosaic size: {mosaic.data.shape}")
    """
    if not tileset or not tileset[0]:
        raise ValueError("Tileset cannot be empty")

    rows = len(tileset)
    cols = len(tileset[0])

    # Validate tileset is rectangular
    for row in tileset:
        if len(row) != cols:
            raise ValueError("Tileset must be rectangular (all rows same length)")

    logging.info(f"Stitching {rows}x{cols} tileset with {tile_overlap:.1%} overlap")

    # Get reference tile for dimensions
    ref_tile = tileset[0][0]
    tile_height, tile_width = ref_tile.data.shape[-2:]

    # Calculate overlap in pixels
    overlap_pixels_x = int(tile_width * tile_overlap)
    overlap_pixels_y = int(tile_height * tile_overlap)

    # Calculate final mosaic dimensions
    effective_tile_width = tile_width - overlap_pixels_x
    effective_tile_height = tile_height - overlap_pixels_y

    mosaic_width = effective_tile_width * (cols - 1) + tile_width
    mosaic_height = effective_tile_height * (rows - 1) + tile_height

    logging.info(f"Tile size: {tile_height}x{tile_width}")
    logging.info(f"Mosaic size: {mosaic_height}x{mosaic_width}")

    # Initialize mosaic array
    mosaic_data = np.zeros((mosaic_height, mosaic_width), dtype=ref_tile.data.dtype)

    # Place each tile
    for row in range(rows):
        for col in range(cols):
            tile = tileset[row][col]
            
            # Calculate position in mosaic
            y_start = row * effective_tile_height
            x_start = col * effective_tile_width
            y_end = y_start + tile_height
            x_end = x_start + tile_width
            
            # Ensure we don't exceed mosaic boundaries
            y_end = min(y_end, mosaic_height)
            x_end = min(x_end, mosaic_width)
            
            # Calculate actual tile region to use
            tile_y_end = y_end - y_start
            tile_x_end = x_end - x_start
            
            # Place tile data (assumes 2D YX format)
            tile_data = tile.data[:tile_y_end, :tile_x_end]
            mosaic_data[y_start:y_end, x_start:x_end] = tile_data
    
    # Create updated metadata for stitched image
    stitched_metadata = deepcopy(ref_tile.metadata)

    # Update resolution to reflect new mosaic size
    stitched_metadata.resolution = (mosaic_width, mosaic_height)

    # Calculate the center position as average of all tile positions
    if any(hasattr(tileset[row][col].metadata, 'stage_position') and 
           tileset[row][col].metadata.stage_position is not None 
           for row in range(rows) for col in range(cols)):
        
        # Collect all stage positions from tiles
        x_positions = []
        y_positions = []
        z_positions = []
        r_positions = []
        t_positions = []
        coordinate_systems = []
        
        for row in range(rows):
            for col in range(cols):
                tile_metadata = tileset[row][col].metadata
                if hasattr(tile_metadata, 'stage_position') and tile_metadata.stage_position is not None:
                    pos = tile_metadata.stage_position
                    x_positions.append(pos.x)
                    y_positions.append(pos.y)
                    z_positions.append(pos.z)
                    r_positions.append(pos.r)
                    t_positions.append(pos.t)
                    coordinate_systems.append(pos.coordinate_system)
        
        if x_positions:  # If we found any positions
            # Calculate average position (center of mosaic)
            avg_x = sum(x_positions) / len(x_positions)
            avg_y = sum(y_positions) / len(y_positions)
            avg_z = sum(z_positions) / len(z_positions)
            avg_r = sum(r_positions) / len(r_positions)
            avg_t = sum(t_positions) / len(t_positions)
            
            # Use coordinate system from first tile
            coord_system = coordinate_systems[0] if coordinate_systems else "Unknown"
            
            # Update stage position to mosaic center
            stitched_metadata.stage_position = FibsemStagePosition(
                x=avg_x,
                y=avg_y,
                z=avg_z,
                r=avg_r,
                t=avg_t,
                coordinate_system=coord_system,
                name=f"stitched_mosaic_{rows}x{cols}"
            )
    
    # Create stitched FluorescenceImage
    stitched_image = FluorescenceImage(
        data=mosaic_data,
        metadata=stitched_metadata
    )
    
    logging.info(f"Stitching complete: {mosaic_height}x{mosaic_width} mosaic created")
    logging.info(f"Updated stage position to mosaic center: {stitched_metadata.stage_position}")
    return stitched_image

def acquire_and_stitch_tileset(
    microscope: FibsemMicroscope,
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    grid_size: Tuple[int, int],
    tile_overlap: float = 0.1,
    zparams: Optional[ZParameters] = None,
    beam_type: BeamType = BeamType.ELECTRON,
    autofocus_mode: AutofocusMode = AutofocusMode.NONE,
    autofocus_channel: Optional[ChannelSettings] = None,
    autofocus_zparams: Optional[ZParameters] = None,
) -> FluorescenceImage:
    """Acquire a tileset and stitch it into a single mosaic image.
    
    Args:
        microscope: The fluorescence microscope instance
        channel_settings: Single channel settings to acquire (multi-channel not yet supported)
        grid_size: Tuple of (rows, cols) defining the grid dimensions
        tile_overlap: Fraction of overlap between adjacent tiles (0.0 to 1.0)
        zparams: Optional Z parameters for z-stack acquisition
        beam_type: Beam type to use for stage movements (default: ELECTRON)
        autofocus_mode: Auto-focus mode for tileset acquisition (default: NONE)
        autofocus_channel: Channel settings to use for auto-focus (uses main channel if None)
        autofocus_zparams: Z parameters for auto-focus search range (uses zparams if None)
        
    Returns:
        Single stitched FluorescenceImage
    """

    # TODO: support multi-channel tilesets
    # TODO: support different projection methods (e.g. max intensity, focus stacking)

    if isinstance(channel_settings, list):
        raise ValueError("Channel settings must be a single ChannelSettings instance for tileset acquisition")

    tileset = acquire_tileset(
        microscope=microscope,
        channel_settings=channel_settings,
        grid_size=grid_size,
        tile_overlap=tile_overlap,
        beam_type=beam_type,
        zparams=zparams,
        autofocus_mode=autofocus_mode,
        autofocus_channel=autofocus_channel,
        autofocus_zparams=autofocus_zparams,
    )
    
    return stitch_tileset(tileset, tile_overlap)


def generate_grid_positions(ncols: int, nrows: int, fov_x: float, fov_y: float, overlap: float = 0.1) -> List[Tuple[float, float]]:
    """Generate a grid of positions, centered around the origin, for acquiring tiles.
    
    Creates a regular grid of (x, y) positions that are properly centered around the origin
    (0, 0) for both odd and even numbers of columns and rows. The spacing between positions
    accounts for the specified field of view and tile overlap.
    
    Args:
        ncols: Number of columns in the grid (must be positive)
        nrows: Number of rows in the grid (must be positive)
        fov_x: Horizontal field of view size in meters (physical dimension of each tile)
        fov_y: Vertical field of view size in meters (physical dimension of each tile)
        overlap: Fraction of overlap between adjacent tiles (0.0 to 1.0)
    
    Returns:
        List of (x, y) tuples representing grid positions in meters, centered around origin
    
    Example:
        >>> # 3x3 grid with 10μm x 8μm FOV and 10% overlap
        >>> positions = generate_grid_positions(3, 3, 10e-6, 8e-6, 0.1)
        >>> len(positions)
        9
        >>> positions[4]  # Center position
        (0.0, 0.0)
        
        >>> # 4x4 grid is also properly centered
        >>> positions = generate_grid_positions(4, 4, 10e-6, 10e-6, 0.0)
        >>> import numpy as np
        >>> np.mean([pos[0] for pos in positions])  # Mean x should be ~0
        0.0
    """
    positions = []
    for i in range(ncols):
        for j in range(nrows):
            x = (i - (ncols - 1) / 2) * (fov_x * (1 - overlap))
            y = (j - (nrows - 1) / 2) * (fov_y * (1 - overlap))
            positions.append((x, y))

    return positions


def convert_grid_positions_to_stage_positions(
    microscope: FibsemMicroscope,
    positions: List[Tuple[float, float]],
    beam_type: BeamType = BeamType.ELECTRON,
    base_position: Optional[FibsemStagePosition] = None
) -> List[FibsemStagePosition]:
    """Convert grid positions to stage positions using microscope projection.
    
    Takes a list of (x, y) grid positions and converts them to FibsemStagePosition
    objects using the microscope's project_stable_move method. This accounts for
    the microscope's coordinate system and current stage configuration.
    
    Args:
        microscope: The FibsemMicroscope instance to use for projection
        positions: List of (x, y) tuples representing grid positions in meters
        beam_type: Beam type to use for projection (default: ELECTRON)
        base_position: Base stage position to project from (default: current position)
    
    Returns:
        List of FibsemStagePosition objects representing the projected stage positions
    
    Example:
        >>> # Generate grid positions
        >>> positions = generate_grid_positions(3, 3, 10e-6, 0.1)
        >>> # Convert to stage positions
        >>> stage_positions = convert_grid_positions_to_stage_positions(
        ...     microscope, positions, BeamType.ELECTRON
        ... )
        >>> len(stage_positions)
        9
    """
    if base_position is None:
        base_position = microscope.get_stage_position()
    
    stage_positions = []
    for pos in positions:
        x, y = pos
        stage_position = microscope.project_stable_move(
            dx=x, dy=y, beam_type=beam_type,
            base_position=base_position
        )
        stage_positions.append(stage_position)
    return stage_positions


def calculate_grid_size_for_area(
    area_width: float,
    area_height: float,
    fov_x: float,
    fov_y: float,
    overlap: float = 0.1
) -> Tuple[int, int]:
    """Calculate the number of rows and columns needed to cover a given area.
    
    Determines the minimum grid dimensions required to fully cover a rectangular area
    with the specified field of view and overlap between adjacent tiles.
    
    Args:
        area_width: Width of the area to cover in meters
        area_height: Height of the area to cover in meters
        fov_x: Horizontal field of view size in meters
        fov_y: Vertical field of view size in meters
        overlap: Fraction of overlap between adjacent tiles (0.0 to 1.0)
        
    Returns:
        Tuple of (ncols, nrows) representing the minimum grid dimensions needed
        
    Raises:
        ValueError: If area dimensions, FOV, or overlap are invalid
        
    Example:
        >>> # Cover a 100×80 μm area with 10×8 μm FOV and 10% overlap
        >>> ncols, nrows = calculate_grid_size_for_area(100e-6, 80e-6, 10e-6, 8e-6, 0.1)
        >>> print(f"Need {ncols}×{nrows} grid")
        Need 12×11 grid
        
        >>> # Cover a 50×50 μm area with 20×20 μm FOV and no overlap
        >>> ncols, nrows = calculate_grid_size_for_area(50e-6, 50e-6, 20e-6, 20e-6, 0.0)
        >>> print(f"Need {ncols}×{nrows} grid")
        Need 3×3 grid
    """
    # Validate inputs
    if area_width <= 0 or area_height <= 0:
        raise ValueError("Area dimensions must be positive")
    if fov_x <= 0 or fov_y <= 0:
        raise ValueError("FOV dimensions must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Overlap must be between 0.0 and 1.0 (exclusive)")
    
    # Calculate effective step size (distance between tile centers)
    step_x = fov_x * (1.0 - overlap)
    step_y = fov_y * (1.0 - overlap)
    
    # Calculate number of tiles needed
    # For n tiles, we need: (n-1) * step + fov >= area
    # Solving for n: n >= (area - fov) / step + 1
    
    if area_width <= fov_x:
        # Area fits in single tile horizontally
        ncols = 1
    else:
        # Need multiple tiles
        ncols = int(np.ceil((area_width - fov_x) / step_x)) + 1
    
    if area_height <= fov_y:
        # Area fits in single tile vertically
        nrows = 1
    else:
        # Need multiple tiles
        nrows = int(np.ceil((area_height - fov_y) / step_y)) + 1
    
    return ncols, nrows


def calculate_grid_coverage_area(
    ncols: int,
    nrows: int,
    fov_x: float,
    fov_y: float,
    overlap: float = 0.1
) -> Tuple[float, float]:
    """Calculate the total area covered by a grid of tiles.
    
    Determines the total width and height of the area covered by a grid of tiles
    with specified dimensions, field of view, and overlap between adjacent tiles.
    
    Args:
        ncols: Number of columns in the grid (must be positive)
        nrows: Number of rows in the grid (must be positive)
        fov_x: Horizontal field of view size in meters
        fov_y: Vertical field of view size in meters
        overlap: Fraction of overlap between adjacent tiles (0.0 to 1.0)
        
    Returns:
        Tuple of (total_width, total_height) in meters representing the covered area
        
    Raises:
        ValueError: If grid dimensions, FOV, or overlap are invalid
        
    Example:
        >>> # Calculate area covered by 3×4 grid with 10×8 μm FOV and 10% overlap
        >>> width, height = calculate_grid_coverage_area(3, 4, 10e-6, 8e-6, 0.1)
        >>> print(f"Covers {width*1e6:.1f}×{height*1e6:.1f} μm")
        Covers 28.0×30.4 μm
        
        >>> # Calculate area covered by 2×2 grid with 20×20 μm FOV and no overlap
        >>> width, height = calculate_grid_coverage_area(2, 2, 20e-6, 20e-6, 0.0)
        >>> print(f"Covers {width*1e6:.1f}×{height*1e6:.1f} μm")
        Covers 40.0×40.0 μm
    """
    # Validate inputs
    if ncols <= 0 or nrows <= 0:
        raise ValueError("Grid dimensions must be positive")
    if fov_x <= 0 or fov_y <= 0:
        raise ValueError("FOV dimensions must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Overlap must be between 0.0 and 1.0 (exclusive)")
    
    # Calculate step size (distance between tile centers)
    step_x = fov_x * (1.0 - overlap)
    step_y = fov_y * (1.0 - overlap)
    
    # Calculate total coverage area
    # For n tiles: total_coverage = (n-1) * step + fov
    if ncols == 1:
        total_width = fov_x
    else:
        total_width = (ncols - 1) * step_x + fov_x
    
    if nrows == 1:
        total_height = fov_y
    else:
        total_height = (nrows - 1) * step_y + fov_y
    
    return total_width, total_height


def calculate_grid_dimensions(positions: List[Tuple[float, float]]) -> Tuple[int, int]:
    """Calculate the number of rows and columns from grid positions.
    
    Analyzes the grid positions to determine the number of unique rows and columns
    in the grid layout. Works with regular grids where positions are arranged in
    a rectangular pattern.
    
    Args:
        positions: List of (x, y) tuples representing grid positions in meters
        
    Returns:
        Tuple of (ncols, nrows) representing the grid dimensions
        Returns (0, 0) if positions is empty
        
    Example:
        >>> positions = generate_grid_positions(3, 4, 10e-6, 8e-6, 0.1)
        >>> ncols, nrows = calculate_grid_dimensions(positions)
        >>> print(f"Grid: {ncols}×{nrows}")
        Grid: 3×4
    """
    if not positions:
        return 0, 0
    
    # Extract unique x and y coordinates
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    # Count unique coordinates with tolerance for floating point precision
    tolerance = 1e-10
    
    # Find unique x coordinates (columns)
    unique_x = []
    for x in x_coords:
        if not any(abs(x - ux) < tolerance for ux in unique_x):
            unique_x.append(x)
    
    # Find unique y coordinates (rows)
    unique_y = []
    for y in y_coords:
        if not any(abs(y - uy) < tolerance for uy in unique_y):
            unique_y.append(y)
    
    ncols = len(unique_x)
    nrows = len(unique_y)
    
    return ncols, nrows


def calculate_grid_overlap(
    positions: List[Tuple[float, float]],
    fov_x: float,
    fov_y: float
) -> Tuple[float, float]:
    """Calculate the overlap between grid positions given the FOV dimensions.
    
    Analyzes the spacing between adjacent grid positions to determine the overlap
    fraction in both horizontal and vertical directions.
    
    Args:
        positions: List of (x, y) tuples representing grid positions in meters
        fov_x: Horizontal field of view size in meters
        fov_y: Vertical field of view size in meters
        
    Returns:
        Tuple of (horizontal_overlap, vertical_overlap) as fractions (0.0 to 1.0)
        Returns (0.0, 0.0) if overlap cannot be determined
        
    Example:
        >>> positions = generate_grid_positions(3, 3, 10e-6, 8e-6, 0.1)
        >>> overlap_x, overlap_y = calculate_grid_overlap(positions, 10e-6, 8e-6)
        >>> print(f"Horizontal overlap: {overlap_x:.1%}, Vertical overlap: {overlap_y:.1%}")
        Horizontal overlap: 10.0%, Vertical overlap: 10.0%
    """
    if len(positions) < 2:
        return 0.0, 0.0
    
    # We'll analyze all pairs of positions to find minimum steps
    
    # Find minimum horizontal and vertical step distances
    min_x_step = float('inf')
    min_y_step = float('inf')
    
    # Check all pairs of positions to find minimum steps
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            if i >= j:
                continue
                
            x1, y1 = pos1
            x2, y2 = pos2
            
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            # Look for horizontal steps (same y, different x)
            if abs(dy) < 1e-10 and dx > 1e-10:  # Same y coordinate
                min_x_step = min(min_x_step, dx)
                
            # Look for vertical steps (same x, different y)
            if abs(dx) < 1e-10 and dy > 1e-10:  # Same x coordinate
                min_y_step = min(min_y_step, dy)
    
    # Calculate overlaps
    overlap_x = 0.0
    overlap_y = 0.0
    
    if min_x_step != float('inf') and min_x_step > 0:
        overlap_x = max(0.0, min(1.0, (fov_x - min_x_step) / fov_x))
    
    if min_y_step != float('inf') and min_y_step > 0:
        overlap_y = max(0.0, min(1.0, (fov_y - min_y_step) / fov_y))
    
    return overlap_x, overlap_y


def plot_grid_positions(
    positions: List[Tuple[float, float]],
    fov_x: float,
    fov_y: float,
    title: str = "Grid Positions with FOV",
    figsize: Tuple[float, float] = (8, 8),
    show_fov_boxes: bool = True,
    show_grid_lines: bool = True,
    show_center_lines: bool = True,
    show_overlap_info: bool = True
) -> None:
    """Plot grid positions with field of view bounding boxes.
    
    Creates a visualization of the grid positions showing:
    - Grid positions as red circles
    - FOV bounding boxes as dashed rectangles around each position
    - Center lines (optional)
    - Grid lines (optional)
    - Calculated overlap information (optional)
    
    Args:
        positions: List of (x, y) tuples representing grid positions in meters
        fov_x: Horizontal field of view size in meters
        fov_y: Vertical field of view size in meters
        title: Plot title
        figsize: Figure size tuple (width, height)
        show_fov_boxes: Whether to show FOV bounding boxes around each position
        show_grid_lines: Whether to show grid lines
        show_center_lines: Whether to show center axis lines
        show_overlap_info: Whether to calculate and display overlap information
    
    Example:
        >>> positions = generate_grid_positions(3, 3, 10e-6, 8e-6, 0.1)
        >>> plot_grid_positions(positions, 10e-6, 8e-6)
    """
    _, ax = plt.subplots(figsize=figsize)
    
    # Plot grid positions as red circles
    for pos in positions:
        x, y = pos
        ax.plot(x, y, 'ro', markersize=8, label='Grid Position' if pos == positions[0] else "")
        
        # Draw FOV bounding box around each position
        if show_fov_boxes:
            # Create rectangle centered at position
            rect = patches.Rectangle(
                (x - fov_x/2, y - fov_y/2),  # Bottom-left corner
                fov_x, fov_y,  # Width and height
                linewidth=1,
                edgecolor='blue',
                facecolor='none',
                linestyle='--',
                alpha=0.7,
                label='FOV Boundary' if pos == positions[0] else ""
            )
            ax.add_patch(rect)
    
    # Calculate plot limits from positions and FOV dimensions
    if positions:
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Find the extent of positions
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add FOV/2 to account for the bounding boxes around each position
        # Plus some padding for better visualization
        padding_x = fov_x * 0.25
        padding_y = fov_y * 0.25
        
        x_extent_min = x_min - fov_x/2 - padding_x
        x_extent_max = x_max + fov_x/2 + padding_x
        y_extent_min = y_min - fov_y/2 - padding_y
        y_extent_max = y_max + fov_y/2 + padding_y
        
        ax.set_xlim(x_extent_min, x_extent_max)
        ax.set_ylim(y_extent_min, y_extent_max)
    else:
        # Fallback for empty positions
        ax.set_xlim(-fov_x, fov_x)
        ax.set_ylim(-fov_y, fov_y)
    
    # Add center lines
    if show_center_lines:
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    
    # Add grid lines
    if show_grid_lines:
        ax.grid(True, alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Equal aspect ratio for proper visualization
    ax.set_aspect('equal', adjustable='box')
    
    # Add text annotation with grid info
    ncols, nrows = calculate_grid_dimensions(positions)
    if ncols > 0 and nrows > 0:
        info_text = f"Grid: {ncols}×{nrows}\nFOV: {fov_x*1e6:.1f}×{fov_y*1e6:.1f} μm"
        
        # Calculate and add total grid area
        overlap_x, overlap_y = calculate_grid_overlap(positions, fov_x, fov_y) if show_overlap_info and len(positions) > 1 else (0.0, 0.0)
        overlap = max(overlap_x, overlap_y)
        total_width, total_height = calculate_grid_coverage_area(ncols, nrows, fov_x, fov_y, overlap)
        info_text += f"\nArea: {total_width*1e6:.1f}×{total_height*1e6:.1f} μm"
    else:
        info_text = f"Positions: {len(positions)}\nFOV: {fov_x*1e6:.1f}×{fov_y*1e6:.1f} μm"
    
    # Optionally add overlap information
    if show_overlap_info and len(positions) > 1:
        overlap_x, overlap_y = calculate_grid_overlap(positions, fov_x, fov_y)
        if overlap_x > 0 or overlap_y > 0:
            # Use the maximum overlap value (they should be the same for regular grids)
            overlap = max(overlap_x, overlap_y)
            info_text += f"\nOverlap: {overlap:.1%}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()