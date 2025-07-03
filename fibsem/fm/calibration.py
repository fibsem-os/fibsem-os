"""Focus measurement and calibration utilities for fluorescence microscopy.

This module provides focus measure algorithms used in focus stacking and autofocus
applications. Different algorithms are suitable for different types of samples
and imaging conditions.
"""

import numpy as np
from typing import Optional


def laplacian_focus_measure(image: np.ndarray) -> np.ndarray:
    """Calculate focus measure using Laplacian variance.
    
    The Laplacian operator measures local intensity variation by computing
    the second derivative. Higher values indicate sharper regions.
    Good general-purpose focus measure for most fluorescence samples.
    
    Args:
        image: 2D numpy array representing a single image plane
        
    Returns:
        2D numpy array of focus measure values (same shape as input)
        
    Note:
        Uses a discrete Laplacian kernel: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    """
    from scipy import ndimage
    
    # Convert to float32 for numerical stability
    image_f = image.astype(np.float32)
    
    # Discrete Laplacian kernel
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    
    # Apply Laplacian filter
    filtered = ndimage.convolve(image_f, laplacian_kernel, mode='reflect')
    
    # Return absolute value to measure variation magnitude
    return np.abs(filtered)


def sobel_focus_measure(image: np.ndarray) -> np.ndarray:
    """Calculate focus measure using Sobel gradient magnitude.
    
    The Sobel operator measures edge strength by computing gradients in
    x and y directions. Higher values indicate stronger edges.
    Good for samples with clear structural features and edges.
    
    Args:
        image: 2D numpy array representing a single image plane
        
    Returns:
        2D numpy array of focus measure values (same shape as input)
        
    Note:
        Computes gradient magnitude: sqrt(sobel_x^2 + sobel_y^2)
    """
    from scipy import ndimage
    
    # Convert to float32 for numerical stability
    image_f = image.astype(np.float32)
    
    # Compute Sobel gradients in x and y directions
    sobel_x = ndimage.sobel(image_f, axis=1)
    sobel_y = ndimage.sobel(image_f, axis=0)
    
    # Compute gradient magnitude
    return np.sqrt(sobel_x**2 + sobel_y**2)


def variance_focus_measure(image: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Calculate focus measure using local variance.
    
    Computes the variance within a local window around each pixel.
    Higher variance indicates more local intensity variation (sharper regions).
    Fast and simple, good for uniform samples with moderate noise.
    
    Args:
        image: 2D numpy array representing a single image plane
        window_size: Size of the local window for variance calculation (default: 3)
        
    Returns:
        2D numpy array of focus measure values (same shape as input)
        
    Note:
        Uses local variance formula: E[X^2] - E[X]^2
    """
    from scipy import ndimage
    
    # Convert to float32 for numerical stability
    image_f = image.astype(np.float32)
    
    # Create averaging kernel
    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)
    
    # Compute local mean and mean of squares
    local_mean = ndimage.convolve(image_f, kernel, mode='reflect')
    local_mean_sq = ndimage.convolve(image_f**2, kernel, mode='reflect')
    
    # Compute local variance: E[X^2] - E[X]^2
    local_variance = local_mean_sq - local_mean**2
    
    # Ensure non-negative values (handle numerical precision issues)
    return np.maximum(local_variance, 0)


def tenengrad_focus_measure(image: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """Calculate focus measure using Tenengrad (thresholded Sobel).
    
    Similar to Sobel gradient but applies an optional threshold to reduce
    noise influence. Only pixels with gradient magnitude above the threshold
    contribute to the focus measure.
    
    Args:
        image: 2D numpy array representing a single image plane
        threshold: Optional threshold value. If None, uses mean + std of gradients
        
    Returns:
        2D numpy array of focus measure values (same shape as input)
        
    Note:
        Good for noisy images where weak gradients should be ignored.
    """
    # Compute Sobel gradient magnitude
    gradient_mag = sobel_focus_measure(image)
    
    # Apply threshold if specified
    if threshold is not None:
        gradient_mag = np.where(gradient_mag > threshold, gradient_mag, 0)
    else:
        # Auto-threshold: mean + standard deviation
        auto_threshold = np.mean(gradient_mag) + np.std(gradient_mag)
        gradient_mag = np.where(gradient_mag > auto_threshold, gradient_mag, 0)
    
    return gradient_mag


def get_focus_measure_function(method: str):
    """Get focus measure function by name.
    
    Args:
        method: Name of the focus measure method
        
    Returns:
        Function that takes an image and returns focus measure
        
    Raises:
        ValueError: If method is not supported
    """
    focus_measures = {
        'laplacian': laplacian_focus_measure,
        'sobel': sobel_focus_measure,
        'variance': variance_focus_measure,
        'tenengrad': tenengrad_focus_measure,
    }
    
    if method not in focus_measures:
        available = list(focus_measures.keys())
        raise ValueError(f"Method '{method}' not supported. Available methods: {available}")
    
    return focus_measures[method]


def find_best_focus_plane(focus_stack: np.ndarray, method: str = 'laplacian') -> int:
    """Find the z-plane with the best overall focus in a stack.
    
    Useful for autofocus applications where you want to identify the single
    best focal plane rather than creating a focus-stacked composite.
    
    Args:
        focus_stack: 3D numpy array (Z, Y, X) representing multiple focal planes
        method: Focus measure method to use
        
    Returns:
        Index of the z-plane with the best focus
        
    Raises:
        ValueError: If focus_stack is not 3D or method is not supported
    """
    if focus_stack.ndim != 3:
        raise ValueError("focus_stack must be 3D (Z, Y, X)")
    
    focus_func = get_focus_measure_function(method)
    
    # Calculate mean focus measure for each z-plane
    focus_scores = []
    for z in range(focus_stack.shape[0]):
        focus_measure = focus_func(focus_stack[z, :, :])
        focus_scores.append(np.mean(focus_measure))
    
    # Return index of plane with highest average focus
    return int(np.argmax(focus_scores))


def calculate_focus_quality(image: np.ndarray, method: str = 'laplacian') -> float:
    """Calculate overall focus quality score for an image.
    
    Useful for comparing focus quality between different images or
    monitoring focus drift during acquisition.
    
    Args:
        image: 2D numpy array representing a single image
        method: Focus measure method to use
        
    Returns:
        Single scalar value representing overall focus quality
        
    Raises:
        ValueError: If method is not supported
    """
    focus_func = get_focus_measure_function(method)
    focus_measure = focus_func(image)
    
    # Return mean focus measure as overall quality score
    return float(np.mean(focus_measure))


def block_based_focus_selection(focus_stack: np.ndarray, method: str = 'laplacian', 
                               block_size: int = 128) -> np.ndarray:
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


def create_focus_stack_from_selection(image_stack: np.ndarray, z_selection: np.ndarray) -> np.ndarray:
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
    
    # Create output image
    stacked_image = np.zeros((ny, nx), dtype=image_stack.dtype)
    
    # Select pixels from appropriate z-planes
    for y in range(ny):
        for x in range(nx):
            selected_z = z_selection[y, x]
            stacked_image[y, x] = image_stack[selected_z, y, x]
    
    return stacked_image


def pixel_based_focus_selection(focus_stack: np.ndarray, method: str = 'laplacian') -> np.ndarray:
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


def create_pixel_based_focus_stack(image_stack: np.ndarray, method: str = 'laplacian') -> np.ndarray:
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


def create_block_based_focus_stack(image_stack: np.ndarray, method: str = 'laplacian', 
                                 block_size: int = 128, smooth_transitions: bool = True,
                                 sigma: float = 1.0) -> np.ndarray:
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
    z_selection = block_based_focus_selection(image_stack, method=method, block_size=block_size)
    stacked_image = create_focus_stack_from_selection(image_stack, z_selection)
    
    if smooth_transitions:
        from scipy import ndimage
        # Apply gentle Gaussian blur to smooth block boundaries
        stacked_image = ndimage.gaussian_filter(stacked_image.astype(np.float32), sigma=sigma)
        # Convert back to original dtype
        stacked_image = stacked_image.astype(image_stack.dtype)
    
    return stacked_image


def frame_integration(image_stack: np.ndarray, method: str = 'mean', 
                     axis: int = 0, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Integrate multiple frames from a stack of images to improve signal-to-noise ratio.
    
    Frame integration combines multiple frames through averaging, summing, or other
    statistical operations. This is commonly used to reduce noise in fluorescence
    microscopy by combining multiple acquisitions of the same field of view.
    
    Args:
        image_stack: N-dimensional numpy array where one axis represents frames
        method: Integration method to use:
            - 'mean': Average frames (default, preserves original intensity range)
            - 'sum': Sum frames (increases signal intensity)
            - 'median': Median of frames (robust to outliers)
            - 'max': Maximum intensity projection
            - 'min': Minimum intensity projection
        axis: Axis along which to integrate frames (default: 0)
        dtype: Output data type. If None, preserves input dtype. For 'sum' method,
               values are clipped to the dtype's min/max range to prevent overflow
    
    Returns:
        Integrated image with one less dimension than input
        
    Raises:
        ValueError: If method is not supported or axis is invalid
        
    Examples:
        >>> # Time series integration (frames along axis 0)
        >>> time_series = np.random.rand(10, 512, 512)  # 10 frames
        >>> integrated = frame_integration(time_series, method='mean')
        >>> print(integrated.shape)  # (512, 512)
        
        >>> # Z-stack maximum intensity projection
        >>> z_stack = np.random.rand(256, 256, 20)  # 20 z-planes
        >>> mip = frame_integration(z_stack, method='max', axis=2)
        >>> print(mip.shape)  # (256, 256)
    """
    if image_stack.ndim < 2:
        raise ValueError("image_stack must have at least 2 dimensions")
    
    if axis < 0:
        axis = image_stack.ndim + axis
    
    if axis >= image_stack.ndim or axis < 0:
        raise ValueError(f"axis {axis} is out of bounds for array with {image_stack.ndim} dimensions")
    
    # Determine output dtype
    if dtype is None:
        dtype = image_stack.dtype
    
    # Get dtype info for clipping
    dtype_info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)
    
    # Apply integration method
    if method == 'mean':
        result = np.mean(image_stack, axis=axis)
    elif method == 'sum':
        # Sum with higher precision then clip to original dtype range
        result = np.sum(image_stack.astype(np.float64), axis=axis)
        result = np.clip(result, dtype_info.min, dtype_info.max)
    elif method == 'median':
        result = np.median(image_stack, axis=axis)
    elif method == 'max':
        result = np.max(image_stack, axis=axis)
    elif method == 'min':
        result = np.min(image_stack, axis=axis)
    else:
        supported_methods = ['mean', 'sum', 'median', 'max', 'min']
        raise ValueError(f"Method '{method}' not supported. Available methods: {supported_methods}")
    
    # Convert to desired dtype
    if result.dtype != dtype:
        result = result.astype(dtype)
    
    return result


def adaptive_frame_integration(image_stack: np.ndarray, method: str = 'mean', 
                              axis: int = 0, outlier_threshold: float = 2.0,
                              min_frames: int = 3) -> np.ndarray:
    """Integrate frames with adaptive outlier rejection for improved robustness.
    
    Similar to frame_integration but excludes outlier frames that deviate
    significantly from the median. This is useful when some frames may be
    corrupted by motion artifacts, photobleaching, or other acquisition issues.
    
    Args:
        image_stack: N-dimensional numpy array where one axis represents frames
        method: Integration method to use after outlier rejection:
            - 'mean': Average remaining frames (default)
            - 'sum': Sum remaining frames
            - 'median': Median of remaining frames
        axis: Axis along which to integrate frames (default: 0)
        outlier_threshold: Number of standard deviations from median to consider outlier (default: 2.0)
        min_frames: Minimum number of frames to keep (default: 3)
    
    Returns:
        Integrated image with outlier frames excluded
        
    Raises:
        ValueError: If method is not supported, axis is invalid, or insufficient frames remain
        
    Note:
        Outlier detection is performed on per-pixel basis using median absolute deviation.
        Frames are excluded if their pixel values deviate by more than outlier_threshold
        standard deviations from the median across frames.
    """
    if image_stack.ndim < 2:
        raise ValueError("image_stack must have at least 2 dimensions")
    
    if axis < 0:
        axis = image_stack.ndim + axis
    
    if axis >= image_stack.ndim or axis < 0:
        raise ValueError(f"axis {axis} is out of bounds for array with {image_stack.ndim} dimensions")
    
    num_frames = image_stack.shape[axis]
    
    if num_frames < min_frames:
        raise ValueError(f"Need at least {min_frames} frames, got {num_frames}")
    
    # Move integration axis to the end for easier processing
    stack_moved = np.moveaxis(image_stack, axis, -1)
    original_shape = stack_moved.shape
    
    # Reshape to 2D for easier processing: (all_pixels, frames)
    pixels_flat = stack_moved.reshape(-1, num_frames)
    
    # Calculate median and MAD for each pixel across frames
    median_values = np.median(pixels_flat, axis=1, keepdims=True)
    mad_values = np.median(np.abs(pixels_flat - median_values), axis=1, keepdims=True)
    
    # Convert MAD to standard deviation equivalent (for normal distribution)
    mad_to_std = 1.4826
    std_equiv = mad_values * mad_to_std
    
    # Create outlier mask: True for outliers
    outlier_mask = np.abs(pixels_flat - median_values) > (outlier_threshold * std_equiv)
    
    # Ensure minimum number of frames per pixel
    valid_frames_per_pixel = np.sum(~outlier_mask, axis=1)
    insufficient_mask = valid_frames_per_pixel < min_frames
    
    # For pixels with insufficient valid frames, keep the best frames
    if np.any(insufficient_mask):
        for pixel_idx in np.where(insufficient_mask)[0]:
            pixel_values = pixels_flat[pixel_idx]
            
            # Calculate distances from median
            distances = np.abs(pixel_values - median_values[pixel_idx, 0])
            
            # Keep the min_frames closest to median
            closest_indices = np.argsort(distances)[:min_frames]
            
            # Reset outlier mask for this pixel
            outlier_mask[pixel_idx] = True
            outlier_mask[pixel_idx, closest_indices] = False
    
    # Apply integration method with outlier masking
    integrated_pixels = np.zeros(pixels_flat.shape[0], dtype=np.float32)
    
    for pixel_idx in range(pixels_flat.shape[0]):
        pixel_values = pixels_flat[pixel_idx]
        valid_mask = ~outlier_mask[pixel_idx]
        valid_values = pixel_values[valid_mask]
        
        if method == 'mean':
            integrated_pixels[pixel_idx] = np.mean(valid_values)
        elif method == 'sum':
            integrated_pixels[pixel_idx] = np.sum(valid_values)
        elif method == 'median':
            integrated_pixels[pixel_idx] = np.median(valid_values)
        else:
            supported_methods = ['mean', 'sum', 'median']
            raise ValueError(f"Method '{method}' not supported. Available methods: {supported_methods}")
    
    # Reshape back to original spatial dimensions
    result_shape = original_shape[:-1]  # Remove frame dimension
    integrated_image = integrated_pixels.reshape(result_shape)
    
    # Convert to appropriate dtype and clip if needed
    if method == 'sum':
        # Get dtype info for clipping
        dtype_info = np.iinfo(image_stack.dtype) if np.issubdtype(image_stack.dtype, np.integer) else np.finfo(image_stack.dtype)
        # Clip to original dtype range
        integrated_image = np.clip(integrated_image, dtype_info.min, dtype_info.max)
    
    integrated_image = integrated_image.astype(image_stack.dtype)
    
    return integrated_image
