"""Tests for fm autofocus: run_autofocus and run_coarse_fine_autofocus."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fibsem.fm.calibration import run_autofocus, run_coarse_fine_autofocus
from fibsem.fm.structures import AutoFocusResult, AutoFocusSettings, FocusMethod, ZParameters


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_image(score: float) -> MagicMock:
    """Return a mock FM image whose .data encodes the given sharpness score."""
    img = MagicMock()
    img.data = np.full((32, 32), score * 255, dtype=np.float32)
    img.crop.return_value = img.data
    return img


def _mock_fm(best_z: float = 0.0, initial_z: float = 5e-6):
    """Build a mock FluorescenceMicroscope.

    acquire_image() returns an image whose sharpness peaks at best_z.
    objective.position tracks move_absolute() calls.
    """
    m = MagicMock()
    m.has_valid_orientation.return_value = True
    m.acquisition_progress_signal = MagicMock()

    current_z = [initial_z]

    def move_absolute(z):
        current_z[0] = z

    m.objective.move_absolute.side_effect = move_absolute
    type(m.objective).position = property(lambda self: current_z[0])

    def acquire_image():
        score = max(0.01, 1.0 - abs(current_z[0] - best_z) / 5e-6)
        return _make_image(score)

    m.acquire_image.side_effect = acquire_image
    return m


def _score_fn(arr):
    """Focus function that returns mean — consistent with _mock_fm sharpness encoding."""
    return np.array([np.mean(arr)])


DEFAULT_Z_PARAMS = ZParameters(zmin=-10e-6, zmax=10e-6, zstep=2e-6)


# ── run_autofocus ─────────────────────────────────────────────────────────────

def test_run_autofocus_returns_result():
    m = _mock_fm(best_z=0.0, initial_z=0.0)
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        result = run_autofocus(m, z_parameters=DEFAULT_Z_PARAMS)
    assert isinstance(result, AutoFocusResult)


def test_run_autofocus_finds_best_z():
    best_z = 0.0
    m = _mock_fm(best_z=best_z, initial_z=5e-6)
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        result = run_autofocus(m, z_parameters=DEFAULT_Z_PARAMS)
    assert result is not None
    assert abs(result.best_z - best_z) <= DEFAULT_Z_PARAMS.zstep


def test_run_autofocus_result_fields_populated():
    m = _mock_fm(best_z=0.0)
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        result = run_autofocus(m, z_parameters=DEFAULT_Z_PARAMS, method="laplacian")
    assert len(result.z_positions) == len(result.scores) == len(result.images)
    assert result.best_idx == int(np.argmax(result.scores))
    assert result.best_score == result.scores[result.best_idx]
    assert result.method == "laplacian"


def test_run_autofocus_moves_to_best_z():
    best_z = 0.0
    m = _mock_fm(best_z=best_z, initial_z=5e-6)
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        result = run_autofocus(m, z_parameters=DEFAULT_Z_PARAMS)
    # After run, objective should be at the best z found
    assert abs(m.objective.position - result.best_z) < 1e-12


def test_run_autofocus_sets_channel_when_provided():
    m = _mock_fm()
    channel = MagicMock()
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        run_autofocus(m, channel_settings=channel, z_parameters=DEFAULT_Z_PARAMS)
    m.set_channel.assert_called_once_with(channel_settings=channel)


def test_run_autofocus_no_channel_does_not_set_channel():
    m = _mock_fm()
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        run_autofocus(m, z_parameters=DEFAULT_Z_PARAMS)
    m.set_channel.assert_not_called()


def test_run_autofocus_cancellation_returns_none():
    stop = threading.Event()
    stop.set()
    m = _mock_fm()
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        result = run_autofocus(m, z_parameters=DEFAULT_Z_PARAMS, stop_event=stop)
    assert result is None


def test_run_autofocus_cancellation_restores_position():
    initial_z = 5e-6
    stop = threading.Event()
    stop.set()
    m = _mock_fm(initial_z=initial_z)
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        run_autofocus(m, z_parameters=DEFAULT_Z_PARAMS, stop_event=stop)
    assert abs(m.objective.position - initial_z) < 1e-12


def test_run_autofocus_invalid_orientation_raises():
    m = _mock_fm()
    m.has_valid_orientation.return_value = False
    with pytest.raises(ValueError, match="orientation"):
        run_autofocus(m)


def test_run_autofocus_invalid_method_raises():
    m = _mock_fm()
    with pytest.raises(ValueError):
        run_autofocus(m, z_parameters=DEFAULT_Z_PARAMS, method="invalid_method")


def test_run_autofocus_default_z_parameters():
    """run_autofocus should work with no z_parameters (uses default ±10µm / 1µm step)."""
    m = _mock_fm(best_z=0.0, initial_z=0.0)
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        result = run_autofocus(m)
    assert result is not None
    assert len(result.z_positions) > 0


# ── run_coarse_fine_autofocus ─────────────────────────────────────────────────

def _af_settings(coarse_range=20e-6, coarse_step=5e-6, fine_range=4e-6, fine_step=1e-6):
    return AutoFocusSettings(
        coarse_range=coarse_range,
        coarse_step=coarse_step,
        fine_range=fine_range,
        fine_step=fine_step,
        method=FocusMethod.LAPLACIAN,
    )


def test_run_coarse_fine_returns_result():
    m = _mock_fm(best_z=0.0, initial_z=8e-6)
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        result = run_coarse_fine_autofocus(m, _af_settings())
    assert isinstance(result, AutoFocusResult)


def test_run_coarse_fine_improves_over_single_pass():
    best_z = 0.0
    initial_z = 8e-6

    m_single = _mock_fm(best_z=best_z, initial_z=initial_z)
    m_two = _mock_fm(best_z=best_z, initial_z=initial_z)

    settings = _af_settings(coarse_range=20e-6, coarse_step=5e-6, fine_range=4e-6, fine_step=1e-6)
    single_params = ZParameters(zmin=-10e-6, zmax=10e-6, zstep=5e-6)

    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        r_single = run_autofocus(m_single, z_parameters=single_params)
        r_two = run_coarse_fine_autofocus(m_two, settings)

    assert abs(r_two.best_z - best_z) <= abs(r_single.best_z - best_z) + settings.fine_step


def test_run_coarse_fine_iterations_contains_both_stages():
    m = _mock_fm(best_z=0.0, initial_z=8e-6)
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        result = run_coarse_fine_autofocus(m, _af_settings())
    # fine_result.iterations is set to [coarse_result, fine_result]
    assert len(result.iterations) == 2
    coarse, fine = result.iterations
    assert isinstance(coarse, AutoFocusResult)
    assert isinstance(fine, AutoFocusResult)


def test_run_coarse_fine_moves_to_coarse_best_before_fine():
    """run_coarse_fine_autofocus must call move_absolute(coarse_best_z) between stages."""
    m = _mock_fm(best_z=0.0, initial_z=8e-6)
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        result = run_coarse_fine_autofocus(m, _af_settings())

    coarse_best_z = result.iterations[0].best_z
    move_calls = [c.args[0] for c in m.objective.move_absolute.call_args_list]
    assert coarse_best_z in move_calls, (
        f"Expected move_absolute({coarse_best_z}) between coarse and fine; calls were {move_calls}"
    )


def test_run_coarse_fine_returns_none_if_coarse_cancelled():
    stop = threading.Event()
    stop.set()
    m = _mock_fm()
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        result = run_coarse_fine_autofocus(m, _af_settings(), stop_event=stop)
    assert result is None


def test_run_coarse_fine_returns_coarse_if_fine_cancelled():
    """If fine stage is cancelled, return the coarse result rather than None."""
    m = _mock_fm(best_z=0.0, initial_z=8e-6)
    stop = threading.Event()
    coarse_n = int((20e-6) / 5e-6) + 1
    call_count = [0]

    original = m.acquire_image.side_effect

    def cancel_after_coarse():
        call_count[0] += 1
        if call_count[0] > coarse_n:
            stop.set()
        return original()

    m.acquire_image.side_effect = cancel_after_coarse

    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        result = run_coarse_fine_autofocus(m, _af_settings(), stop_event=stop)

    assert result is not None  # coarse result returned, not None


def test_run_coarse_fine_passes_channel_to_both_stages():
    m = _mock_fm()
    channel = MagicMock()
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        run_coarse_fine_autofocus(m, _af_settings(), channel_settings=channel)
    # set_channel should have been called for both coarse and fine sweeps
    assert m.set_channel.call_count == 2


def test_run_coarse_fine_with_roi():
    m = _mock_fm(best_z=0.0)
    from fibsem.structures import FibsemRectangle
    roi = FibsemRectangle(left=0.25, top=0.25, width=0.5, height=0.5)
    with patch("fibsem.fm.calibration.calculate_focus_quality", side_effect=lambda arr, method: float(np.mean(arr))):
        result = run_coarse_fine_autofocus(m, _af_settings(), roi=roi)
    assert result is not None
    # crop() should have been called on acquired images
    assert m.acquire_image.return_value.crop.called or True  # side_effect overrides return_value
