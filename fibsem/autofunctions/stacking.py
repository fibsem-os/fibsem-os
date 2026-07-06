from __future__ import annotations

from fibsem.autofunctions.metrics import get_focus_measure_function
import numpy as np

def block_based_focus_selection(
    focus_stack: np.ndarray, method: str = "laplacian", block_size: int = 128
) -> np.ndarray:
    """Create focus map using block-based focus measure analysis.

    Divides the image into blocks and selects the best z-plane for each block
    based on the average focus measure within that block. This approach is
    more robust to noise and provides smoother focus transitions than per-pixel
    selection.

    Args:
        focus_stack: 3D numpy array (Z, Y, X) representing multiple focal planes
        method: Focus measure method to use
        block_size: Size of square blocks for focus analysis (default: 128)

    Returns:
        2D numpy array (Y, X) with the selected z-plane index for each pixel

    Raises:
        ValueError: If focus_stack is not 3D or method is not supported

    Note:
        All pixels within a block will have the same selected z-plane.
        Edge blocks that don't fit exactly will be handled appropriately.
    """
    if focus_stack.ndim != 3:
        raise ValueError("focus_stack must be 3D (Z, Y, X)")

    nz, ny, nx = focus_stack.shape
    focus_func = get_focus_measure_function(method)

    # Initialize z-plane selection map
    z_selection = np.zeros((ny, nx), dtype=np.int32)

    # Process image in blocks
    for y_start in range(0, ny, block_size):
        for x_start in range(0, nx, block_size):
            # Define block boundaries
            y_end = min(y_start + block_size, ny)
            x_end = min(x_start + block_size, nx)

            # Calculate average focus measure for each z-plane in this block
            block_focus_scores = []

            for z in range(nz):
                block_data = focus_stack[z, y_start:y_end, x_start:x_end]
                focus_measure = focus_func(block_data)
                # Use mean focus measure for the block
                block_focus_scores.append(np.mean(focus_measure))

            # Select z-plane with highest average focus for this block
            best_z = np.argmax(block_focus_scores)

            # Assign the same z-plane to all pixels in this block
            z_selection[y_start:y_end, x_start:x_end] = best_z

    return z_selection


def create_focus_stack_from_selection(
    image_stack: np.ndarray, z_selection: np.ndarray
) -> np.ndarray:
    """Create a focus-stacked image from a z-plane selection map.

    Takes the pixel values from the selected z-plane for each spatial location.

    Args:
        image_stack: 3D numpy array (Z, Y, X) of image data
        z_selection: 2D numpy array (Y, X) with z-plane indices for each pixel

    Returns:
        2D numpy array (Y, X) with the focus-stacked result

    Raises:
        ValueError: If dimensions don't match
    """
    if image_stack.ndim != 3:
        raise ValueError("image_stack must be 3D (Z, Y, X)")

    if z_selection.ndim != 2:
        raise ValueError("z_selection must be 2D (Y, X)")

    nz, ny, nx = image_stack.shape

    if z_selection.shape != (ny, nx):
        raise ValueError("z_selection shape must match image_stack spatial dimensions")

    y_idx, x_idx = np.indices((ny, nx))
    return image_stack[z_selection, y_idx, x_idx]


def pixel_based_focus_selection(
    focus_stack: np.ndarray, method: str = "laplacian"
) -> np.ndarray:
    """Create focus map using per-pixel focus measure analysis.

    Calculates focus measure for each pixel across all z-planes and selects
    the z-plane with the highest focus measure for each pixel location.

    Args:
        focus_stack: 3D numpy array (Z, Y, X) representing multiple focal planes
        method: Focus measure method to use

    Returns:
        2D numpy array (Y, X) with the selected z-plane index for each pixel

    Raises:
        ValueError: If focus_stack is not 3D or method is not supported

    Note:
        Per-pixel selection provides maximum spatial accuracy but can be noisy.
        Consider using block-based selection for noisy images.
    """
    if focus_stack.ndim != 3:
        raise ValueError("focus_stack must be 3D (Z, Y, X)")

    nz, ny, nx = focus_stack.shape
    focus_func = get_focus_measure_function(method)

    # Calculate focus measure for each z-plane
    focus_measures = np.zeros((nz, ny, nx), dtype=np.float32)

    for z_idx in range(nz):
        z_plane = focus_stack[z_idx, :, :]
        focus_measures[z_idx] = focus_func(z_plane)

    # Find the z-plane with maximum focus measure at each pixel
    best_z_indices = np.argmax(focus_measures, axis=0)  # Shape: (Y, X)

    return best_z_indices.astype(np.int32)


def create_pixel_based_focus_stack(
    image_stack: np.ndarray, method: str = "laplacian"
) -> np.ndarray:
    """Create a focus-stacked image using per-pixel focus selection.

    Convenience function that combines pixel_based_focus_selection and
    create_focus_stack_from_selection into a single operation.

    Args:
        image_stack: 3D numpy array (Z, Y, X) representing multiple focal planes
        method: Focus measure method to use

    Returns:
        2D numpy array (Y, X) with the focus-stacked result

    Raises:
        ValueError: If image_stack is not 3D or method is not supported

    Note:
        Per-pixel selection provides maximum spatial accuracy but can be noisy.
        For noisy images, consider using create_block_based_focus_stack instead.
    """
    z_selection = pixel_based_focus_selection(image_stack, method=method)
    return create_focus_stack_from_selection(image_stack, z_selection)


def create_block_based_focus_stack(
    image_stack: np.ndarray,
    method: str = "laplacian",
    block_size: int = 128,
    smooth_transitions: bool = True,
    sigma: float = 1.0,
) -> np.ndarray:
    """Create a focus-stacked image using block-based focus selection.

    Convenience function that combines block_based_focus_selection and
    create_focus_stack_from_selection into a single operation, with optional
    smoothing to reduce sharp block boundary artifacts.

    Args:
        image_stack: 3D numpy array (Z, Y, X) representing multiple focal planes
        method: Focus measure method to use
        block_size: Size of square blocks for focus analysis (default: 128)
        smooth_transitions: If True, apply Gaussian blur to reduce block artifacts (default: True)
        sigma: Standard deviation for Gaussian smoothing (default: 1.0)

    Returns:
        2D numpy array (Y, X) with the focus-stacked result

    Raises:
        ValueError: If image_stack is not 3D or method is not supported

    Note:
        Block-based focus stacking can create sharp boundaries between blocks.
        Setting smooth_transitions=True applies a gentle Gaussian blur to reduce
        these artifacts while preserving the overall focus structure.

        Block size considerations:
        - 64x64: Good for detailed structures, but more noise-sensitive
        - 128x128 (default): Optimal balance - good noise reduction, fewer artifacts,
          efficient computation, suitable for most biological structures
        - 256x256+: Maximum noise reduction but may miss fine focus variations
    """
    z_selection = block_based_focus_selection(
        image_stack, method=method, block_size=block_size
    )
    stacked_image = create_focus_stack_from_selection(image_stack, z_selection)

    if smooth_transitions:
        from scipy import ndimage

        # Apply gentle Gaussian blur to smooth block boundaries
        stacked_image = ndimage.gaussian_filter(
            stacked_image.astype(np.float32), sigma=sigma
        )
        # Convert back to original dtype
        stacked_image = stacked_image.astype(image_stack.dtype)

    return stacked_image
