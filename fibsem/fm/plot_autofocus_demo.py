"""
Quick visual test for plot_autofocus_result with the updated layout.

Generates a synthetic AutoFocusResult with a Gaussian focus curve and random
images, then saves the plot to /tmp/test_autofocus_plot.png and opens it.

Run with:
    conda run -n fibsem-os python fibsem/fm/test_plot_autofocus.py
"""

import numpy as np

from fibsem.autofunctions.autofocus import AutoFocusIteration, AutoFocusResult
from fibsem.autofunctions.plotting import plot_autofocus_result


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def make_synthetic_result(n_positions=25, image_shape=(256, 256), pass_index=0):
    rng = np.random.default_rng(42)

    z_positions = np.linspace(-12e-6, 12e-6, n_positions)
    best_idx = n_positions // 2
    best_z = float(z_positions[best_idx])

    iterations = []
    for i, z in enumerate(z_positions):
        sharpness = gaussian(i, best_idx, n_positions / 5)
        score = gaussian(i, best_idx, n_positions / 6) + rng.normal(0, 0.02)
        img = rng.integers(0, 256, image_shape, dtype=np.uint8)
        blur_radius = int((1 - sharpness) * 8) * 2 + 1
        from scipy.ndimage import uniform_filter
        blurred = uniform_filter(img.astype(float), size=blur_radius).astype(np.uint8)
        iterations.append(AutoFocusIteration(
            working_distance=float(z),
            focus_score=float(score),
            pass_index=pass_index,
            image=blurred,
        ))

    best = iterations[best_idx]
    return AutoFocusResult(
        image=best.image,
        working_distance=best.working_distance,
        initial_working_distance=float(z_positions[0]),
        focus_score=best.focus_score,
        iterations=iterations,
        method="laplacian",
    )


if __name__ == "__main__":
    import os

    # Single-pass
    result = make_synthetic_result(n_positions=25)
    save_path = "/tmp/test_autofocus_plot.png"
    plot_autofocus_result(result, save_path=save_path)
    print(f"Saved single-pass to {save_path}")

    # Coarse + fine (pass_index 0 and 1)
    coarse_iters = make_synthetic_result(n_positions=20, pass_index=0).iterations
    fine_iters = make_synthetic_result(n_positions=10, pass_index=1).iterations
    all_iters = coarse_iters + fine_iters
    best = max(all_iters, key=lambda it: it.focus_score)
    result_cf = AutoFocusResult(
        image=best.image,
        working_distance=best.working_distance,
        initial_working_distance=coarse_iters[0].working_distance,
        focus_score=best.focus_score,
        iterations=all_iters,
        method="laplacian",
    )
    save_path_cf = "/tmp/test_autofocus_plot_coarsefine.png"
    plot_autofocus_result(result_cf, save_path=save_path_cf)
    print(f"Saved coarse-fine to {save_path_cf}")

    os.system(f"open {save_path} {save_path_cf}")
