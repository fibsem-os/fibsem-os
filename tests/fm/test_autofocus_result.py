import numpy as np
import pytest

from fibsem.fm.structures import AutoFocusResult
from fibsem.structures import FibsemRectangle


def _make_result(n=5, best_idx=2, method="laplacian", roi=None):
    z_positions = [i * 1e-6 for i in range(n)]
    scores = [float(i) for i in range(n)]
    images = [np.zeros((64, 64), dtype=np.uint8) for _ in range(n)]
    return AutoFocusResult(
        best_z=z_positions[best_idx],
        best_idx=best_idx,
        best_score=scores[best_idx],
        z_positions=z_positions,
        scores=scores,
        images=images,
        method=method,
        roi=roi,
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_autofocus_result_basic_instantiation():
    r = _make_result()
    assert isinstance(r, AutoFocusResult)


def test_autofocus_result_best_score_matches_scores():
    r = _make_result(best_idx=3)
    assert r.best_score == pytest.approx(r.scores[r.best_idx])


def test_autofocus_result_best_z_matches_z_positions():
    r = _make_result(best_idx=1)
    assert r.best_z == pytest.approx(r.z_positions[r.best_idx])


def test_autofocus_result_optional_fields_default_none():
    r = _make_result()
    assert r.roi is None
    assert r.channel_settings is None


def test_autofocus_result_with_roi():
    roi = FibsemRectangle(left=0.25, top=0.25, width=0.5, height=0.5)
    r = _make_result(roi=roi)
    assert r.roi is roi


# ---------------------------------------------------------------------------
# Computed properties
# ---------------------------------------------------------------------------

def test_n_positions():
    r = _make_result(n=7)
    assert r.n_positions == 7


def test_z_range_m():
    r = _make_result(n=5)
    expected = max(r.z_positions) - min(r.z_positions)
    assert r.z_range_m == pytest.approx(expected)


def test_z_range_m_single_position():
    r = AutoFocusResult(
        best_z=0.0, best_idx=0, best_score=1.0,
        z_positions=[0.0], scores=[1.0],
        images=[np.zeros((4, 4))], method="laplacian",
    )
    assert r.z_range_m == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# plot() method
# ---------------------------------------------------------------------------

def test_plot_method_calls_plot_autofocus(monkeypatch):
    called_with = {}

    def fake_plot_autofocus(result, save_path=None):
        called_with["result"] = result
        called_with["save_path"] = save_path

    import fibsem.fm.calibration as cal
    monkeypatch.setattr(cal, "plot_autofocus", fake_plot_autofocus)

    r = _make_result()
    r.plot(save_path="/tmp/test_af.png")

    assert called_with["result"] is r
    assert called_with["save_path"] == "/tmp/test_af.png"


def test_plot_method_default_save_path(monkeypatch):
    called_with = {}

    def fake_plot_autofocus(result, save_path=None):
        called_with["save_path"] = save_path

    import fibsem.fm.calibration as cal
    monkeypatch.setattr(cal, "plot_autofocus", fake_plot_autofocus)

    _make_result().plot()
    assert called_with["save_path"] is None


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def test_images_are_numpy_arrays():
    r = _make_result(n=3)
    assert all(isinstance(img, np.ndarray) for img in r.images)


def test_images_count_matches_z_positions():
    r = _make_result(n=6)
    assert len(r.images) == len(r.z_positions)


# ---------------------------------------------------------------------------
# Strategy return type consistency
# ---------------------------------------------------------------------------

def test_strategy_run_return_annotation():
    from fibsem.fm.strategy.sweep import SweepAutoFocusStrategy
    from fibsem.fm.strategy.coarse_fine import CoarseFineAutoFocusStrategy
    from fibsem.fm.strategy.iterative import IterativeAutoFocusStrategy
    import typing

    for cls in [SweepAutoFocusStrategy, CoarseFineAutoFocusStrategy, IterativeAutoFocusStrategy]:
        hints = typing.get_type_hints(cls.run)
        return_hint = hints.get("return")
        # Should be Optional[AutoFocusResult] i.e. Union[AutoFocusResult, None]
        args = getattr(return_hint, "__args__", ())
        assert AutoFocusResult in args, f"{cls.__name__}.run() return type does not include AutoFocusResult"
