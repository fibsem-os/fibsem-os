"""Tests for fibsem/autofunctions: ACB, autofocus, and plotting."""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fibsem.structures import BeamType, FibsemImage, FibsemRectangle
from fibsem.autofunctions.acb import (
    AutoContrastBrightnessSettings,
    AutoContrastBrightnessIteration,
    AutoContrastBrightnessResult,
    run_auto_contrast_brightness,
)
from fibsem.autofunctions.autofocus import (
    AutoFocusSettings,
    AutoFocusIteration,
    AutoFocusResult,
    FocusSweepPass,
    run_auto_focus,
)
from fibsem.autofunctions.plotting import plot_acb_result, plot_autofocus_result
from fibsem.autofunctions.charge_neutralisation import auto_charge_neutralisation


# ── helpers ───────────────────────────────────────────────────────────────────

def make_image(mean_frac: float = 0.5, size: int = 64, dtype=np.uint16) -> FibsemImage:
    dtype_max = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    base = int(dtype_max * mean_frac)
    noise = int(dtype_max * 0.02)
    rng = np.random.default_rng(42)
    data = np.clip(
        rng.integers(base - noise, base + noise + 1, size=(size, size), dtype=np.int64),
        0, dtype_max,
    ).astype(dtype)
    return FibsemImage(data=data, metadata=None)


def _sharpness_score(wd: float, best_wd: float) -> float:
    """Deterministic focus score that peaks at best_wd."""
    return max(0.01, 1.0 - abs(wd - best_wd) / 0.003)


def make_sharp_image(wd: float, best_wd: float, size: int = 64) -> FibsemImage:
    """Synthetic image. The score is embedded as the mean pixel value so the
    mock focus function (which reads mean) peaks reliably at best_wd."""
    score = _sharpness_score(wd, best_wd)
    data = np.full((size, size), int(score * 65535), dtype=np.uint16)
    return FibsemImage(data=data, metadata=None)


def mock_microscope(best_wd: float = 4.5e-3, initial_wd: float = 7.0e-3):
    m = MagicMock()
    m.get_working_distance.return_value = initial_wd
    m.get_detector_brightness.return_value = 0.3
    m.get_detector_contrast.return_value = 0.5
    current_wd = [initial_wd]

    def set_wd(wd, beam_type=None):
        current_wd[0] = wd

    def acquire_image(image_settings=None):
        return make_sharp_image(current_wd[0], best_wd)

    m.set_working_distance.side_effect = set_wd
    m.acquire_image.side_effect = acquire_image
    return m


def make_acb_result(n_iters: int = 3) -> AutoContrastBrightnessResult:
    from fibsem.structures import ImageStats
    iters = []
    for i in range(n_iters):
        mean = 0.3 + i * 0.07
        stats = ImageStats(
            mean=mean, std=0.05, median=mean, p01=mean - 0.1, p99=mean + 0.1,
            saturation_lo=0.0, saturation_hi=0.0, contrast_ratio=0.1,
            range_utilisation=0.2, snr=10.0, entropy=4.0,
        )
        iters.append(AutoContrastBrightnessIteration(
            brightness=0.3 + i * 0.05,
            contrast=0.5,
            stats=stats,
            image=make_image(mean_frac=mean),
        ))
    final_stats = iters[-1].stats
    return AutoContrastBrightnessResult(
        image=iters[-1].image,
        stats=final_stats,
        converged=True,
        iterations=iters,
        settings=AutoContrastBrightnessSettings(),
    )


def make_autofocus_result(n_passes: int = 1) -> AutoFocusResult:
    best_wd = 4.5e-3
    initial_wd = 7.0e-3
    passes = [FocusSweepPass(n_steps=5, step_size=0.5e-3) for _ in range(n_passes)]
    settings = AutoFocusSettings(method="laplacian", passes=passes)
    iters = []
    centre_wd = initial_wd
    for pi, sp in enumerate(passes):
        half = sp.n_steps / 2 * sp.step_size
        wds = np.linspace(centre_wd - half, centre_wd + half, sp.n_steps + 1)
        scores = []
        for wd in wds:
            img = make_sharp_image(wd, best_wd)
            from fibsem.autofunctions.metrics import get_focus_measure_function
            fn = get_focus_measure_function("laplacian")
            score = float(np.mean(fn(img.data.astype(np.float32))))
            iters.append(AutoFocusIteration(
                pass_index=pi, working_distance=float(wd),
                focus_score=score, image=img,
            ))
            scores.append(score)
        centre_wd = float(wds[int(np.argmax(scores))])
    best_idx = int(np.argmax([it.focus_score for it in iters]))
    best = iters[best_idx]
    return AutoFocusResult(
        image=best.image,
        working_distance=best.working_distance,
        initial_working_distance=initial_wd,
        focus_score=best.focus_score,
        iterations=iters,
        settings=settings,
    )


# ── ImageStats / compute_stats ────────────────────────────────────────────────

def test_compute_stats_mid_exposure():
    img = make_image(mean_frac=0.5)
    stats = img.compute_stats()
    assert abs(stats.mean - 0.5) < 0.05
    assert stats.saturation_lo == pytest.approx(0.0)
    assert stats.saturation_hi == pytest.approx(0.0)


def test_compute_stats_all_zeros():
    img = FibsemImage(data=np.zeros((64, 64), dtype=np.uint16), metadata=None)
    stats = img.compute_stats()
    assert stats.mean == pytest.approx(0.0)
    assert stats.saturation_lo == pytest.approx(1.0)


def test_compute_stats_all_max():
    img = FibsemImage(data=np.full((64, 64), 65535, dtype=np.uint16), metadata=None)
    stats = img.compute_stats()
    assert stats.mean == pytest.approx(1.0)
    assert stats.saturation_hi == pytest.approx(1.0)


def test_compute_stats_uint8():
    img = FibsemImage(data=np.full((64, 64), 128, dtype=np.uint8), metadata=None)
    stats = img.compute_stats()
    assert abs(stats.mean - 128 / 255) < 0.01


def test_image_stats_converged():
    from fibsem.structures import ImageStats
    stats = ImageStats(
        mean=0.5, std=0.05, median=0.5, p01=0.4, p99=0.6,
        saturation_lo=0.0, saturation_hi=0.0, contrast_ratio=0.1,
        range_utilisation=0.2, snr=10.0, entropy=4.0,
    )
    assert stats.converged(mean_target=0.5, mean_tolerance=0.05, saturation_limit=0.005)
    assert not stats.converged(mean_target=0.8, mean_tolerance=0.05, saturation_limit=0.005)


# ── ACB settings serialisation ────────────────────────────────────────────────

def test_acb_settings_round_trip():
    s = AutoContrastBrightnessSettings(
        n_iterations=8, brightness_step=0.03, mean_target=0.45,
    )
    assert AutoContrastBrightnessSettings.from_dict(s.to_dict()) == s


# ── run_auto_contrast_brightness ──────────────────────────────────────────────

def test_run_acb_basic():
    m = MagicMock()
    m.get_detector_brightness.return_value = 0.3
    m.get_detector_contrast.return_value = 0.5
    call_count = [0]

    def acquire(image_settings=None):
        mean = min(0.3 + call_count[0] * 0.1, 0.5)
        call_count[0] += 1
        return make_image(mean_frac=mean)

    m.acquire_image.side_effect = acquire

    settings = AutoContrastBrightnessSettings(n_iterations=6)
    result = run_auto_contrast_brightness(m, settings=settings)

    assert result.settings is settings
    assert len(result.iterations) <= settings.n_iterations
    assert result.image is not None


def test_run_acb_preserves_initial_settings():
    m = MagicMock()
    m.get_detector_brightness.return_value = 0.7
    m.get_detector_contrast.return_value = 0.9
    m.acquire_image.return_value = make_image(mean_frac=0.5)

    result = run_auto_contrast_brightness(m, settings=AutoContrastBrightnessSettings(n_iterations=2))
    assert result.iterations[0].brightness == pytest.approx(0.7)
    assert result.iterations[0].contrast == pytest.approx(0.9)


# ── ACB result serialisation ──────────────────────────────────────────────────

def test_acb_result_save_load(tmp_path):
    result = make_acb_result(n_iters=3)
    saved_dir = result.save(path=str(tmp_path))
    loaded = AutoContrastBrightnessResult.load(str(saved_dir))

    assert loaded.converged == result.converged
    assert len(loaded.iterations) == len(result.iterations)
    assert loaded.settings.n_iterations == result.settings.n_iterations
    assert loaded.stats.mean == pytest.approx(result.stats.mean, abs=1e-6)


# ── Autofocus settings serialisation ─────────────────────────────────────────

def test_autofocus_settings_round_trip():
    s = AutoFocusSettings(
        method="sobel",
        passes=[FocusSweepPass(8, 1e-3), FocusSweepPass(10, 0.1e-3)],
    )
    assert AutoFocusSettings.from_dict(s.to_dict()).passes[0].step_size == pytest.approx(1e-3)
    assert AutoFocusSettings.from_dict(s.to_dict()).method == "sobel"


def test_autofocus_settings_default_pass():
    s = AutoFocusSettings.from_dict({"method": "laplacian", "passes": [],
                                      "probe_resolution": [768, 512],
                                      "probe_dwell_time": 0.5e-6, "reduced_area": None})
    assert len(s.passes) == 1


def test_autofocus_settings_reduced_area_round_trip():
    ra = FibsemRectangle(left=0.1, top=0.2, width=0.5, height=0.4)
    s = AutoFocusSettings(reduced_area=ra)
    loaded = AutoFocusSettings.from_dict(s.to_dict())
    assert loaded.reduced_area.left == pytest.approx(0.1)


# ── run_auto_focus ────────────────────────────────────────────────────────────

def _mean_focus_fn(arr):
    """Focus function that returns mean — peaks at best_wd in make_sharp_image."""
    return np.array([np.mean(arr)])


def test_run_auto_focus_single_pass():
    best_wd = 4.5e-3
    m = mock_microscope(best_wd=best_wd, initial_wd=4.5e-3)
    settings = AutoFocusSettings(
        method="laplacian",
        passes=[FocusSweepPass(n_steps=10, step_size=0.5e-3)],
    )
    with patch("fibsem.autofunctions.metrics.get_focus_measure_function", return_value=_mean_focus_fn):
        result = run_auto_focus(m, settings=settings)

    assert result.working_distance == pytest.approx(best_wd, abs=settings.passes[0].step_size)
    assert result.initial_working_distance == pytest.approx(4.5e-3)
    assert result.settings is settings
    assert len(result.iterations) == settings.passes[0].n_steps + 1


def test_run_auto_focus_multi_pass_converges():
    best_wd = 4.5e-3
    single = AutoFocusSettings(passes=[FocusSweepPass(10, 2e-3)])
    multi = AutoFocusSettings(passes=[
        FocusSweepPass(10, 2e-3),
        FocusSweepPass(10, 0.2e-3),
        FocusSweepPass(10, 0.02e-3),
    ])

    m_single = mock_microscope(best_wd=best_wd, initial_wd=7.0e-3)
    m_multi = mock_microscope(best_wd=best_wd, initial_wd=7.0e-3)

    with patch("fibsem.autofunctions.metrics.get_focus_measure_function", return_value=_mean_focus_fn):
        r_single = run_auto_focus(m_single, settings=single)
        r_multi = run_auto_focus(m_multi, settings=multi)

    assert abs(r_multi.working_distance - best_wd) <= abs(r_single.working_distance - best_wd)


def test_run_auto_focus_iteration_count():
    m = mock_microscope()
    passes = [FocusSweepPass(5, 1e-3), FocusSweepPass(4, 0.1e-3)]
    settings = AutoFocusSettings(passes=passes)
    with patch("fibsem.autofunctions.metrics.get_focus_measure_function", return_value=_mean_focus_fn):
        result = run_auto_focus(m, settings=settings)
    expected = sum(p.n_steps + 1 for p in passes)
    assert len(result.iterations) == expected


def test_run_auto_focus_restores_wd_on_error():
    initial_wd = 7.0e-3
    m = mock_microscope(initial_wd=initial_wd)
    m.acquire_image.side_effect = RuntimeError("hardware error")

    settings = AutoFocusSettings(passes=[FocusSweepPass(5, 0.5e-3)])
    with pytest.raises(RuntimeError):
        run_auto_focus(m, settings=settings)

    last_set = m.set_working_distance.call_args[0][0]
    assert last_set == pytest.approx(initial_wd)


# ── AutoFocusResult serialisation ─────────────────────────────────────────────

def test_autofocus_result_save_load(tmp_path):
    result = make_autofocus_result(n_passes=2)
    saved_dir = result.save(path=str(tmp_path))
    loaded = AutoFocusResult.load(str(saved_dir))

    assert loaded.working_distance == pytest.approx(result.working_distance)
    assert loaded.initial_working_distance == pytest.approx(result.initial_working_distance)
    assert loaded.focus_score == pytest.approx(result.focus_score)
    assert len(loaded.iterations) == len(result.iterations)
    assert loaded.settings.method == result.settings.method


# ── Plotting ──────────────────────────────────────────────────────────────────

def test_plot_acb_minimal(tmp_path):
    result = make_acb_result(n_iters=1)
    out = str(tmp_path / "acb.png")
    plot_acb_result(result, save_path=out)
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_plot_acb_multi_iter(tmp_path):
    result = make_acb_result(n_iters=5)
    out = str(tmp_path / "acb.png")
    plot_acb_result(result, save_path=out)
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_plot_autofocus_single_pass(tmp_path):
    result = make_autofocus_result(n_passes=1)
    out = str(tmp_path / "af.png")
    plot_autofocus_result(result, save_path=out)
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_plot_autofocus_multi_pass(tmp_path):
    result = make_autofocus_result(n_passes=3)
    out = str(tmp_path / "af.png")
    plot_autofocus_result(result, save_path=out)
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_plot_autofocus_threadsafe(tmp_path):
    result = make_autofocus_result(n_passes=2)
    errors = []

    def run(i):
        try:
            plot_autofocus_result(result, save_path=str(tmp_path / f"af_{i}.png"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert all((tmp_path / f"af_{i}.png").exists() for i in range(4))


# ── Charge neutralisation ─────────────────────────────────────────────────────

def test_auto_charge_neutralisation():
    from fibsem.structures import ImageSettings
    m = MagicMock()
    image_settings = MagicMock(spec=ImageSettings)
    image_settings.hfw = 150e-6

    with patch("fibsem.acquire.new_image", return_value=make_image()) as mock_new_image:
        auto_charge_neutralisation(m, image_settings, n_iterations=3)
        assert mock_new_image.call_count == 4  # 3 discharge + 1 final
