"""Test script for auto-contrast/brightness: generates synthetic data and shows plots.

Run from the repo root:
    PYTHONPATH=. python scripts/test_acb.py
"""
import numpy as np
import matplotlib.pyplot as plt

from fibsem.structures import FibsemImage
from fibsem.imaging.utils import percentile_stretch
from fibsem.autofunctions.acb import (
    AutoContrastBrightnessIteration,
    AutoContrastBrightnessResult,
    AutoContrastBrightnessSettings,
)
from fibsem.autofunctions.plotting import plot_acb_result


def make_probe_image(mean_frac: float, noise_frac: float = 0.05) -> FibsemImage:
    """Synthetic 16-bit probe image with a given mean fraction of dtype max."""
    dtype_max = 65535
    base = int(dtype_max * mean_frac)
    noise = int(dtype_max * noise_frac)
    data = np.clip(
        np.random.randint(base - noise, base + noise + 1, (128, 96)),
        0, dtype_max,
    ).astype(np.uint16)
    return FibsemImage(data=data, metadata=None)


def make_synthetic_result(n_iters: int = 7) -> AutoContrastBrightnessResult:
    """Simulate a converging ACB run: mean ramps from 0.1 to 0.5 over n_iters."""
    settings = AutoContrastBrightnessSettings()
    iterations = []

    for i in range(n_iters):
        t = i / max(n_iters - 1, 1)
        mean_frac = 0.1 + 0.42 * t           # ramps toward 0.52
        brightness = 0.25 + 0.3 * t
        contrast = 0.5
        img = make_probe_image(mean_frac)
        stats = img.compute_stats()
        iterations.append(
            AutoContrastBrightnessIteration(
                brightness=brightness,
                contrast=contrast,
                stats=stats,
                image=img,
            )
        )

    final = iterations[-1]
    return AutoContrastBrightnessResult(
        image=final.image,
        stats=final.stats,
        converged=final.stats.converged(settings.mean_target, settings.mean_tolerance, settings.saturation_limit),
        iterations=iterations,
    )


def demo_percentile_stretch():
    """Show before/after of FibsemImage.auto_contrast_brightness() on a dark image."""
    dark = make_probe_image(mean_frac=0.08, noise_frac=0.03)
    stretched = dark.auto_contrast_brightness()

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle("FibsemImage.auto_contrast_brightness() — percentile stretch", fontsize=10)

    axes[0].imshow(dark.data, cmap="gray")
    axes[0].set_title(f"before\nmean={dark.data.mean()/65535:.3f}", fontsize=8)
    axes[0].axis("off")

    axes[1].hist(dark.data.ravel(), bins=128, color="steelblue")
    axes[1].set_title("histogram before", fontsize=8)
    axes[1].set_xlabel("pixel value")

    axes[2].imshow(stretched.data, cmap="gray")
    axes[2].set_title(f"after\nmean={stretched.data.mean()/65535:.3f}", fontsize=8)
    axes[2].axis("off")

    axes[3].hist(stretched.data.ravel(), bins=128, color="darkorange")
    axes[3].set_title("histogram after", fontsize=8)
    axes[3].set_xlabel("pixel value")

    plt.tight_layout()
    plt.savefig("/tmp/acb_stretch.png", dpi=120)
    print("stretch plot saved to /tmp/acb_stretch.png")
    plt.show()


def demo_run_result():
    """Show the full ACB run diagnostic plot."""
    result = make_synthetic_result(n_iters=7)
    print(f"Converged: {result.converged}, iterations: {result.n_iterations}")
    for i, it in enumerate(result.iterations):
        print(f"  iter {i}: b={it.brightness:.2f} c={it.contrast:.2f} {it.stats}")

    plot_acb_result(result, save_path="/tmp/acb_result.png")
    print("result plot saved to /tmp/acb_result.png")

    img = plt.imread("/tmp/acb_result.png")
    plt.figure(figsize=(14, 7))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_percentile_stretch()
    demo_run_result()
