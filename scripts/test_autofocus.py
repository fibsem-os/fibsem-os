"""Test script for image-based auto-focus: generates synthetic data and shows plots.

Run from the repo root:
    PYTHONPATH=. python scripts/test_autofocus.py
"""
import numpy as np

from fibsem.structures import FibsemImage
from fibsem.autofunctions.autofocus import (
    AutoFocusIteration,
    AutoFocusResult,
    AutoFocusSettings,
    FocusSweepPass,
)
from fibsem.autofunctions.plotting import plot_autofocus_result
from fibsem.autofunctions.metrics import get_focus_measure_function


def make_probe_image(wd: float, best_wd: float, noise: float = 0.05) -> FibsemImage:
    """Synthetic 16-bit probe image whose sharpness peaks at best_wd."""
    dtype_max = 65535
    sharpness = max(0.01, 1.0 - abs(wd - best_wd) / 0.003)
    size = 128
    x = np.linspace(0, 8 * np.pi, size)
    base = np.outer(np.sin(x * sharpness * 3), np.cos(x * sharpness * 3))
    data = np.clip(
        (base + 1) / 2 * dtype_max * 0.8 + np.random.randn(size, size) * dtype_max * noise,
        0, dtype_max,
    ).astype(np.uint16)
    return FibsemImage(data=data, metadata=None)


def make_synthetic_result(
    settings: AutoFocusSettings,
    best_wd: float = 4.5e-3,
    initial_wd: float = 7.0e-3,
) -> AutoFocusResult:
    """Simulate a multi-pass autofocus run with a known best WD."""
    focus_fn = get_focus_measure_function(settings.method)
    iterations = []
    centre_wd = initial_wd

    for pass_index, sweep_pass in enumerate(settings.passes):
        half = sweep_pass.n_steps / 2 * sweep_pass.step_size
        wds = np.linspace(centre_wd - half, centre_wd + half, sweep_pass.n_steps + 1)
        pass_scores = []
        for wd in wds:
            img = make_probe_image(wd, best_wd)
            score = float(np.mean(focus_fn(img.data.astype(np.float32))))
            iterations.append(AutoFocusIteration(
                pass_index=pass_index,
                working_distance=float(wd),
                focus_score=score,
                image=img,
            ))
            pass_scores.append(score)
        best_in_pass = int(np.argmax(pass_scores))
        centre_wd = float(wds[best_in_pass])

    best_idx = int(np.argmax([it.focus_score for it in iterations]))
    best = iterations[best_idx]
    return AutoFocusResult(
        image=best.image,
        working_distance=best.working_distance,
        initial_working_distance=initial_wd,
        focus_score=best.focus_score,
        iterations=iterations,
        settings=settings,
    )


def demo_single_pass():
    settings = AutoFocusSettings(
        method="laplacian",
        passes=[FocusSweepPass(n_steps=20, step_size=0.5e-3)],
    )
    result = make_synthetic_result(settings)
    print(f"Single pass — {result.n_iterations} steps, best WD={result.working_distance*1e3:.3f} mm")
    plot_autofocus_result(result, save_path="/tmp/autofocus_single.png")
    print("saved to /tmp/autofocus_single.png")


def demo_multi_pass():
    settings = AutoFocusSettings(
        method="laplacian",
        passes=[
            FocusSweepPass(n_steps=10, step_size=2e-3),
            FocusSweepPass(n_steps=10, step_size=0.5e-3),
            FocusSweepPass(n_steps=10, step_size=0.05e-3),
        ],
    )
    result = make_synthetic_result(settings)
    print(f"3-pass — {result.n_iterations} total steps, best WD={result.working_distance*1e3:.3f} mm")
    for it in result.iterations:
        print(f"  pass {it.pass_index}  wd={it.working_distance*1e3:.3f} mm  score={it.focus_score:.1f}")
    plot_autofocus_result(result, save_path="/tmp/autofocus_multi.png")
    print("saved to /tmp/autofocus_multi.png")


if __name__ == "__main__":
    demo_single_pass()
    demo_multi_pass()
