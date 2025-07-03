import logging
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import scipy.ndimage
from skimage import img_as_float
from skimage.filters import laplace
from skimage.filters.rank import gradient
from skimage.morphology import disk

from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings, FluorescenceImage, ZParameters
from fibsem.structures import BeamType, FibsemStagePosition
from fibsem.microscope import FibsemMicroscope

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

        # TODO: properly handle metadata + image structure
        images.append(zstack)

    # restore objective to initial position
    microscope.objective.move_absolute(z_init)

    return FluorescenceImage.create_multi_channel_image(images)


########## CALIBRATION FUNCTIONS ##########


def get_sharpness(img: np.ndarray, **kwargs) -> float:
    """Calculate sharpness (accutance) of an image.
    (Acutance: https://en.wikipedia.org/wiki/Acutance)
    """
    disk_size = kwargs.get("disk_size", 5)
    logging.debug(f"calculating sharpness (accutance) of image {img.shape}: {img.dtype}")
    # normalise and convert to uint8, for faster processing
    img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    filtered = scipy.ndimage.median_filter(img_norm, 3)
    return np.mean(gradient(filtered, disk(disk_size)))

def get_variance(img: np.ndarray, **kwargs) -> float:
    """Get variance of the Laplacian of an image.
    (Faster than get_sharpness, but more sensitive to noise)"""
    # Convert to float to avoid precision issues
    img_float = img_as_float(img)
    laplacian = laplace(img_float)
    return np.var(laplacian)


FOCUS_FN_MAP = {"sharpness": get_sharpness, "variance": get_variance}
DEFAULT_FOCUS_METHOD = "variance"
DEFAULT_FOCUS_FN = FOCUS_FN_MAP[DEFAULT_FOCUS_METHOD]

def run_auto_focus(
    microscope: FluorescenceMicroscope,
    channel_settings: Optional[ChannelSettings] = None,
    method: str = DEFAULT_FOCUS_METHOD,
) -> float:
    """Run autofocus by acquiring images at different z positions and finding the best focus."""

    # TODO:
    # - allow user to specify focus function
    # - allow user to specify z parameters (defaults for coarse/fine focus?)

    # set up z parameters
    zparams = ZParameters(zmin=-10e-6, zmax=10e-6, zstep=2.5e-6)
    z_positions = zparams.generate_positions(microscope.objective.position)

    # get the focus metric function
    focus_fn = FOCUS_FN_MAP.get(method, FOCUS_FN_MAP[DEFAULT_FOCUS_METHOD])

    # apply the channel settings
    if channel_settings is not None:
        microscope.set_channel(channel_settings=channel_settings)

    scores = []
    for pos in z_positions:
        # set objective position
        microscope.objective.move_absolute(pos)

        # acquire image
        img = microscope.acquire_image()

        # calculate sharpness of image
        score = focus_fn(img.data)
        logging.info(f"Z Position: {pos*1e3:.2e} mm, Score: {score:.2f}")
        scores.append(score)

    # get best focus position
    idx = np.argmax(scores)
    best_focus = z_positions[idx]

    logging.info(f"Best focus at pos: {best_focus:.2e} with score: {scores[idx]:.2f}")
    # move objective to best focus position
    microscope.objective.move_absolute(best_focus)

    return best_focus



# TODO: auto-focus, z-stack + mip
def acquire_tileset(
    microscope: FibsemMicroscope,
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    grid_size: Tuple[int, int],
    tile_overlap: float = 0.1,
    beam_type: BeamType = BeamType.ELECTRON,
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
        beam_type: Beam type to use for stage movements (default: ELECTRON)
        
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
    
    # Acquire a reference image to determine field of view and pixel size
    reference_image = acquire_channels(microscope.fm, channel_settings)
    
    # Calculate the physical field of view from image metadata
    pixel_size_x, pixel_size_y = reference_image.metadata.pixel_size_x, reference_image.metadata.pixel_size_y
    image_height, image_width = reference_image.data.shape[-2:]
    fov_x = image_width * pixel_size_x
    fov_y = image_height * pixel_size_y

    # Calculate step size accounting for overlap
    step_x = fov_x * (1.0 - tile_overlap)
    step_y = fov_y * (1.0 - tile_overlap)
    
    logging.info(f"Field of view: {fov_x*1e6:.1f} x {fov_y*1e6:.1f} μm")
    logging.info(f"Step size: {step_x*1e6:.1f} x {step_y*1e6:.1f} μm")
    
    # Calculate starting position (top-left corner of grid)
    start_offset_x = -(cols - 1) * step_x / 2
    start_offset_y = -(rows - 1) * step_y / 2
    
    # Move to starting position
    logging.info("Moving to grid starting position")
    microscope.stable_move(dx=start_offset_x, dy=start_offset_y, beam_type=beam_type)
    
    # Initialize results array
    tileset = []
    
    try:
        for row in range(rows):
            row_images = []
            
            for col in range(cols):
                logging.info(f"Acquiring tile [{row+1}/{rows}][{col+1}/{cols}]")
                
                # Acquire all channels at current position
                tile_image = acquire_channels(microscope.fm, channel_settings)
                logging.info(f"Stage position for tile [{row+1}/{rows}][{col+1}/{cols}]: {tile_image.metadata.stage_position}")
                
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
    from copy import deepcopy
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


