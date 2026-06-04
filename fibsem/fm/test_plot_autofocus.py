"""
Quick visual test for plot_autofocus with the updated layout.

Generates a synthetic AutoFocusResult with a Gaussian focus curve and random
images, then saves the plot to /tmp/test_autofocus_plot.png and opens it.

Run with:
    conda run -n fibsem-os python fibsem/fm/test_plot_autofocus.py
"""

import numpy as np

from fibsem.fm.calibration import plot_autofocus
from fibsem.fm.structures import AutoFocusResult


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def make_synthetic_result(n_positions=25, image_shape=(256, 256)):
    rng = np.random.default_rng(42)

    z_positions = np.linspace(-12e-6, 12e-6, n_positions).tolist()
    best_idx = n_positions // 2

    # Gaussian focus curve with a little noise
    scores = [
        gaussian(i, best_idx, n_positions / 6) + rng.normal(0, 0.02)
        for i in range(n_positions)
    ]
    best_score = max(scores)
    best_z = z_positions[best_idx]

    # Synthetic images: blurred near ends, sharp at centre
    images = []
    for i in range(n_positions):
        sharpness = gaussian(i, best_idx, n_positions / 5)
        # Base image: random noise
        img = rng.integers(0, 256, image_shape, dtype=np.uint8)
        # Blur with a box filter proportional to distance from best
        blur_radius = int((1 - sharpness) * 8) * 2 + 1
        from scipy.ndimage import uniform_filter
        blurred = uniform_filter(img.astype(float), size=blur_radius)
        images.append(blurred.astype(np.uint8))

    return AutoFocusResult(
        best_z=best_z,
        best_idx=best_idx,
        best_score=best_score,
        z_positions=z_positions,
        scores=scores,
        images=images,
        method="laplacian",
    )


if __name__ == "__main__":
    import os

    # Single-stage (25 positions, 2-row thumbnails)
    result = make_synthetic_result(n_positions=25)
    save_path = "/tmp/test_autofocus_plot.png"
    plot_autofocus(result, save_path=save_path)
    print(f"Saved single-stage to {save_path}")

    # Single-stage small (< 10 positions, 1-row thumbnails)
    result_small = make_synthetic_result(n_positions=5)
    save_path_small = "/tmp/test_autofocus_plot_small.png"
    plot_autofocus(result_small, save_path=save_path_small)
    print(f"Saved small to {save_path_small}")

    # Multi-stage: 3 iterations with halving range (simulates IterativeAutoFocusStrategy)
    stages = []
    for i, n_pos in enumerate([20, 12, 8]):
        r = make_synthetic_result(n_positions=n_pos)
        stages.append(r)
    final = stages[-1]
    final.iterations = stages
    save_path_iter = "/tmp/test_autofocus_plot_iterative.png"
    plot_autofocus(final, save_path=save_path_iter)
    print(f"Saved iterative to {save_path_iter}")

    # Multi-stage: 2 stages (simulates CoarseFineAutoFocusStrategy)
    coarse = make_synthetic_result(n_positions=20)
    fine = make_synthetic_result(n_positions=10)
    fine.iterations = [coarse, fine]
    save_path_cf = "/tmp/test_autofocus_plot_coarsefine.png"
    plot_autofocus(fine, save_path=save_path_cf)
    print(f"Saved coarse-fine to {save_path_cf}")

    os.system(f"open {save_path} {save_path_small} {save_path_iter} {save_path_cf}")
