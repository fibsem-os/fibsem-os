from __future__ import annotations
from typing import Optional

import numpy as np


def frame_integration(
    image_stack: np.ndarray,
    method: str = "mean",
    axis: int = 0,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
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
        raise ValueError(
            f"axis {axis} is out of bounds for array with {image_stack.ndim} dimensions"
        )

    # Determine output dtype
    if dtype is None:
        dtype = image_stack.dtype

    # Get dtype info for clipping
    dtype_info = (
        np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)
    )

    # Apply integration method
    if method == "mean":
        result = np.mean(image_stack, axis=axis)
    elif method == "sum":
        # Sum with higher precision then clip to original dtype range
        result = np.sum(image_stack.astype(np.float64), axis=axis)
        result = np.clip(result, dtype_info.min, dtype_info.max)
    elif method == "median":
        result = np.median(image_stack, axis=axis)
    elif method == "max":
        result = np.max(image_stack, axis=axis)
    elif method == "min":
        result = np.min(image_stack, axis=axis)
    else:
        supported_methods = ["mean", "sum", "median", "max", "min"]
        raise ValueError(
            f"Method '{method}' not supported. Available methods: {supported_methods}"
        )

    # Convert to desired dtype
    if result.dtype != dtype:
        result = result.astype(dtype)

    return result


def adaptive_frame_integration(
    image_stack: np.ndarray,
    method: str = "mean",
    axis: int = 0,
    outlier_threshold: float = 2.0,
    min_frames: int = 3,
) -> np.ndarray:
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
        raise ValueError(
            f"axis {axis} is out of bounds for array with {image_stack.ndim} dimensions"
        )

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

    # Create outlier mask: True for outliers.
    # When std_equiv==0 (all frames identical), no frame is an outlier regardless of threshold.
    outlier_mask = (std_equiv > 0) & (np.abs(pixels_flat - median_values) > (outlier_threshold * std_equiv))

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

        if method == "mean":
            integrated_pixels[pixel_idx] = np.mean(valid_values)
        elif method == "sum":
            integrated_pixels[pixel_idx] = np.sum(valid_values)
        elif method == "median":
            integrated_pixels[pixel_idx] = np.median(valid_values)
        else:
            supported_methods = ["mean", "sum", "median"]
            raise ValueError(
                f"Method '{method}' not supported. Available methods: {supported_methods}"
            )

    # Reshape back to original spatial dimensions
    result_shape = original_shape[:-1]  # Remove frame dimension
    integrated_image = integrated_pixels.reshape(result_shape)

    # Convert to appropriate dtype and clip if needed
    if method == "sum":
        # Get dtype info for clipping
        dtype_info = (
            np.iinfo(image_stack.dtype)
            if np.issubdtype(image_stack.dtype, np.integer)
            else np.finfo(image_stack.dtype)
        )
        # Clip to original dtype range
        integrated_image = np.clip(integrated_image, dtype_info.min, dtype_info.max)

    integrated_image = integrated_image.astype(image_stack.dtype)

    return integrated_image
