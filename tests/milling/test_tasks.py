import pytest

from fibsem import utils
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.milling.tasks import (
    FibsemMillingTask,
    FibsemMillingTaskConfig,
    MillingTaskAcquisitionSettings,
)
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


# ── the post-task image refresh is opt-out ───────────────────────────────────
#
# A milling task ends by acquiring one FIB image to refresh the view, when the task
# didn't already acquire one of its own. That is the right default, but it has never
# been switchable — and for a low-kV polish, imaging the lamella with a higher-voltage
# beam undoes the polish.

def _spy(obj, name):
    """Replace ``obj.name`` with a wrapper that records calls, returns the recorder."""
    calls = []
    orig = getattr(obj, name)

    def wrapper(*a, **k):
        calls.append((a, k))
        return orig(*a, **k)

    setattr(obj, name, wrapper)
    return calls


def _task(tmp_path, **acquisition):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    cfg = FibsemMillingTaskConfig.from_stages(stages=[FibsemMillingStage(name="s")], name="t")
    cfg.acquisition.imaging.path = str(tmp_path)
    # drift correction acquires its own images; switch it off so the assertions below
    # are about the post-task refresh and nothing else
    cfg.alignment.enabled = False
    for key, value in acquisition.items():
        setattr(cfg.acquisition, key, value)
    return microscope, FibsemMillingTask(microscope, cfg)


def test_final_image_is_acquired_by_default(tmp_path):
    microscope, task = _task(tmp_path)
    assert task.config.acquisition.acquire_final_image is True

    acquired = _spy(microscope, "acquire_image")
    task.run()

    assert acquired, "the post-task FIB refresh should fire by default"


def test_final_image_can_be_suppressed(tmp_path):
    microscope, task = _task(tmp_path, acquire_final_image=False)

    acquired = _spy(microscope, "acquire_image")
    task.run()

    assert acquired == [], "no image should be acquired when the final refresh is off"


def test_final_image_defaults_to_true_for_protocols_predating_the_flag():
    assert MillingTaskAcquisitionSettings.from_dict({}).acquire_final_image is True


def test_final_image_flag_round_trips():
    settings = MillingTaskAcquisitionSettings(acquire_final_image=False)
    restored = MillingTaskAcquisitionSettings.from_dict(settings.to_dict())
    assert restored.acquire_final_image is False
