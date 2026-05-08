import pytest

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig, MillingTaskAcquisitionSettings
from fibsem.structures import FibsemMillingSettings, ImageSettings


# ── MillingTaskAcquisitionSettings.estimated_time ────────────────────────────

def test_acquisition_estimated_time_disabled():
    acq = MillingTaskAcquisitionSettings(acquire_sem=False, acquire_fib=False)
    assert acq.estimated_time == 0.0


def test_acquisition_estimated_time_sem_only():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6)
    acq = MillingTaskAcquisitionSettings(acquire_sem=True, acquire_fib=False, imaging=img)
    assert acq.estimated_time == pytest.approx(img.estimated_time * 1)


def test_acquisition_estimated_time_both_beams():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6)
    acq = MillingTaskAcquisitionSettings(acquire_sem=True, acquire_fib=True, imaging=img)
    assert acq.estimated_time == pytest.approx(img.estimated_time * 2)


# ── FibsemMillingTaskConfig.estimated_time ────────────────────────────────────

def test_milling_task_config_estimated_time_empty():
    cfg = FibsemMillingTaskConfig()
    assert cfg.estimated_time == 0.0


def test_milling_task_config_estimated_time_milling_only():
    stage = FibsemMillingStage(
        milling=FibsemMillingSettings(milling_current=2e-9),
        pattern=RectanglePattern(width=10e-6, height=5e-6, depth=1e-6),
    )
    cfg = FibsemMillingTaskConfig(stages=[stage])
    assert cfg.estimated_time == pytest.approx(stage.estimated_time)


def test_milling_task_config_estimated_time_includes_acquisition():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6)
    stage = FibsemMillingStage(
        milling=FibsemMillingSettings(milling_current=2e-9),
        pattern=RectanglePattern(width=10e-6, height=5e-6, depth=1e-6),
    )
    acq = MillingTaskAcquisitionSettings(acquire_sem=True, acquire_fib=False, imaging=img)
    cfg = FibsemMillingTaskConfig(stages=[stage], acquisition=acq)
    assert cfg.estimated_time == pytest.approx(stage.estimated_time + acq.estimated_time)


def test_milling_task_config_estimated_time_multiple_stages():
    stage1 = FibsemMillingStage(
        milling=FibsemMillingSettings(milling_current=2e-9),
        pattern=RectanglePattern(width=10e-6, height=5e-6, depth=1e-6),
    )
    stage2 = FibsemMillingStage(
        milling=FibsemMillingSettings(milling_current=7.6e-9),
        pattern=RectanglePattern(width=20e-6, height=10e-6, depth=2e-6),
    )
    cfg = FibsemMillingTaskConfig(stages=[stage1, stage2])
    assert cfg.estimated_time == pytest.approx(stage1.estimated_time + stage2.estimated_time)
