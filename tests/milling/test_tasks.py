from pathlib import Path

import pytest

from fibsem import utils
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.milling.tasks import (
    FibsemMillingTask,
    FibsemMillingTaskConfig,
    MillingTaskAcquisitionSettings,
)
from fibsem.structures import BeamType, FibsemMillingSettings, ImageSettings


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


def test_milling_task_config_estimated_time_ignores_disabled_stages():
    """A disabled stage is not milled, so it must not be quoted either.

    estimated_time summed self.stages while run() uses enabled_stages, so
    switching a stage off left the quoted time unchanged.
    """
    stage1 = FibsemMillingStage(
        milling=FibsemMillingSettings(milling_current=2e-9),
        pattern=RectanglePattern(width=10e-6, height=5e-6, depth=1e-6),
    )
    stage2 = FibsemMillingStage(
        milling=FibsemMillingSettings(milling_current=7.6e-9),
        pattern=RectanglePattern(width=20e-6, height=10e-6, depth=2e-6),
        enabled=False,
    )
    cfg = FibsemMillingTaskConfig(stages=[stage1, stage2])
    assert cfg.estimated_time == pytest.approx(stage1.estimated_time)
    # and strictly less than the same task with the stage switched back on
    stage2.enabled = True
    assert cfg.estimated_time > stage1.estimated_time


def test_milling_task_config_estimated_time_all_stages_disabled():
    """Only the acquisition remains: nothing is milled."""
    stage = FibsemMillingStage(
        milling=FibsemMillingSettings(milling_current=2e-9),
        pattern=RectanglePattern(width=10e-6, height=5e-6, depth=1e-6),
        enabled=False,
    )
    cfg = FibsemMillingTaskConfig(stages=[stage])
    assert cfg.estimated_time == pytest.approx(cfg.acquisition.estimated_time)


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


def _task(tmp_path, stages=None, **acquisition):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    stages = stages if stages is not None else [FibsemMillingStage(name="s")]
    cfg = FibsemMillingTaskConfig.from_stages(stages=stages, name="t")
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


# ── restores go back to what the user was imaging at ─────────────────────────
#
# Nothing shipped has ever milled at a voltage other than 30 kV, so the restore
# paths have never been exercised at a second voltage. Low-kV polishing steps
# 30 -> 8 -> 2 kV across stages, which is where the two paths disagreeing matters.

def test_every_restore_uses_the_captured_voltage(tmp_path):
    """Regression: _acquire_milling_task_images restored system.ion.beam.voltage — the
    config default — while run()'s finally correctly used the captured value. A task
    milling at 8 kV therefore bounced the column to 30 kV between stages."""
    polish_stage = FibsemMillingStage(
        name="polish", milling=FibsemMillingSettings(milling_voltage=8e3)
    )
    microscope, task = _task(tmp_path, stages=[polish_stage], acquire_fib=True)

    # image at a voltage that is neither the config default nor the milling voltage,
    # so a wrong restore is unambiguous
    microscope.set_beam_voltage(voltage=2e3, beam_type=BeamType.ION)
    assert microscope.system.ion.beam.voltage != 2e3, "guard: config default must differ"

    restored = _spy(microscope, "finish_milling")
    task.run()

    assert restored, "finish_milling should have run"
    voltages = [call[1]["imaging_voltage"] for call in restored]
    assert all(v == 2e3 for v in voltages), f"expected every restore at 2 kV, got {voltages}"
    assert microscope.get_beam_voltage(BeamType.ION) == 2e3


# ── an unset save path is not a directory called "None" ──────────────────────

def test_unset_path_does_not_resolve_under_a_none_directory(tmp_path, monkeypatch):
    """Regression: _configure_path stringified the imaging path before joining it, so a
    task with no path configured (the default, and what the milling widget hands over
    when its Path field is left empty) wrote its reference image and its whole
    drift-correction run to ./None/Milling/<task>/ beside the cwd."""
    monkeypatch.chdir(tmp_path)
    microscope, _ = utils.setup_session(manufacturer="Demo")
    cfg = FibsemMillingTaskConfig.from_stages(stages=[FibsemMillingStage(name="s")])
    assert cfg.acquisition.imaging.path is None, "guard: the path starts out unset"

    task = FibsemMillingTask(microscope, cfg)
    task._configure_path()

    configured = str(task.config.acquisition.imaging.path)
    assert Path(configured).is_absolute(), f"unset path resolved to a relative {configured!r}"
    assert "None" not in Path(configured).parts, configured
    assert Path(configured).parts[-2:] == ("Milling", "Milling-Task")


def test_configured_path_is_preserved(tmp_path):
    """The fallback only applies when nothing is configured."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    cfg = FibsemMillingTaskConfig.from_stages(stages=[FibsemMillingStage(name="s")], name="t")
    cfg.acquisition.imaging.path = str(tmp_path)

    task = FibsemMillingTask(microscope, cfg)
    task._configure_path()

    assert task.config.acquisition.imaging.path == str(tmp_path / "Milling" / "t")


def test_imaging_conditions_falls_back_before_capture(tmp_path):
    """If the task failed before it could capture the live conditions, fall back to the
    system defaults rather than passing None to the column."""
    microscope, task = _task(tmp_path)
    assert task.initial_imaging_current is None and task.initial_imaging_voltage is None

    current, voltage = task._imaging_conditions()

    assert current == microscope.system.ion.beam.beam_current
    assert voltage == microscope.system.ion.beam.voltage
