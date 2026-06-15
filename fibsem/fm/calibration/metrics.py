from __future__ import annotations

from typing import Optional

import numpy as np

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
    filtered = ndimage.convolve(image_f, laplacian_kernel, mode="reflect")

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
    kernel = np.ones((window_size, window_size), dtype=np.float32) / (
        window_size * window_size
    )

    # Compute local mean and mean of squares
    local_mean = ndimage.convolve(image_f, kernel, mode="reflect")
    local_mean_sq = ndimage.convolve(image_f**2, kernel, mode="reflect")

    # Compute local variance: E[X^2] - E[X]^2
    local_variance = local_mean_sq - local_mean**2

    # Ensure non-negative values (handle numerical precision issues)
    return np.maximum(local_variance, 0)


def tenengrad_focus_measure(
    image: np.ndarray, threshold: Optional[float] = None
) -> np.ndarray:
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
        "laplacian": laplacian_focus_measure,
        "sobel": sobel_focus_measure,
        "variance": variance_focus_measure,
        "tenengrad": tenengrad_focus_measure,
    }

    if method not in focus_measures:
        available = list(focus_measures.keys())
        raise ValueError(
            f"Method '{method}' not supported. Available methods: {available}"
        )

    return focus_measures[method]


def find_best_focus_plane(focus_stack: np.ndarray, method: str = "laplacian") -> int:
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

    focus_scores = []
    for z in range(focus_stack.shape[0]):
        focus_measure = focus_func(focus_stack[z, :, :])
        focus_scores.append(np.mean(focus_measure))

    return int(np.argmax(focus_scores))


def calculate_focus_quality(image: np.ndarray, method: str = "laplacian") -> float:
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

    return float(np.mean(focus_measure))