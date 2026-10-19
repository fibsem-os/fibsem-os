import logging
import os
import threading
from pathlib import Path

import pytest

from fibsem import utils
from fibsem.milling.base import FibsemMillingStage, MillingStrategyResult
from fibsem.milling.patterning.patterns import RectanglePattern
from fibsem.milling.progress import MillingProgressStatus
from fibsem.milling.strategy.standard import StandardMillingStrategy
from fibsem.milling.tasks import (
    FibsemMillingTask,
    FibsemMillingTaskConfig,
    MillingTaskAcquisitionSettings,
    record_milling_skipped,
    run_milling_task,
)
from fibsem.structures import (
    BeamType,
    FibsemMillingSettings,
    ImageSettings,
    MillingState,
)

# ── MillingTaskAcquisitionSettings.estimated_time ────────────────────────────


def test_acquisition_estimated_time_disabled():
    acq = MillingTaskAcquisitionSettings(acquire_sem=False, acquire_fib=False)
    assert acq.estimated_time == 0.0


def test_acquisition_estimated_time_sem_only():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6)
    acq = MillingTaskAcquisitionSettings(
        acquire_sem=True, acquire_fib=False, imaging=img
    )
    assert acq.estimated_time == pytest.approx(img.estimated_time * 1)


def test_acquisition_estimated_time_both_beams():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6)
    acq = MillingTaskAcquisitionSettings(
        acquire_sem=True, acquire_fib=True, imaging=img
    )
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
    acq = MillingTaskAcquisitionSettings(
        acquire_sem=True, acquire_fib=False, imaging=img
    )
    cfg = FibsemMillingTaskConfig(stages=[stage], acquisition=acq)
    assert cfg.estimated_time == pytest.approx(
        stage.estimated_time + acq.estimated_time
    )


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
    assert cfg.estimated_time == pytest.approx(
        stage1.estimated_time + stage2.estimated_time
    )


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
    milling at 8 kV therefore bounced the column to 30 kV between stages. Every
    restore now puts back what setup_milling found."""
    polish_stage = FibsemMillingStage(
        name="polish", milling=FibsemMillingSettings(milling_voltage=8e3)
    )
    microscope, task = _task(tmp_path, stages=[polish_stage], acquire_fib=True)

    # image at a voltage that is neither the config default nor the milling voltage,
    # so a wrong restore is unambiguous
    microscope.set_beam_voltage(voltage=2e3, beam_type=BeamType.ION)
    assert microscope.system.ion.beam.voltage != 2e3, (
        "guard: config default must differ"
    )

    voltages = []
    finish = microscope.finish_milling

    def restore_and_read():
        finish()
        voltages.append(microscope.get_beam_voltage(BeamType.ION))

    microscope.finish_milling = restore_and_read
    task.run()

    assert voltages, "finish_milling should have run"
    assert all(v == 2e3 for v in voltages), (
        f"expected every restore at 2 kV, got {voltages}"
    )


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
    assert Path(configured).is_absolute(), (
        f"unset path resolved to a relative {configured!r}"
    )
    assert "None" not in Path(configured).parts, configured
    assert Path(configured).parts[-2:] == ("Milling", "Milling-Task")


def test_configured_path_is_preserved(tmp_path):
    """The fallback only applies when nothing is configured."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    cfg = FibsemMillingTaskConfig.from_stages(
        stages=[FibsemMillingStage(name="s")], name="t"
    )
    cfg.acquisition.imaging.path = str(tmp_path)

    task = FibsemMillingTask(microscope, cfg)
    task._configure_path()

    assert task.config.acquisition.imaging.path == str(tmp_path / "Milling" / "t")


# ── the run's folder is the caller's, and every stage is told it ─────────────
#
# Where a milling run writes used to be read off the imaging path -- an imaging
# setting that also meant "where to save" -- and strategies worked out the
# lamella from its shape. The folder is now an argument to the run and a
# runtime field on each stage (FIB-1253).


def test_a_folder_passed_to_the_run_wins_over_the_imaging_path(tmp_path):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    cfg = FibsemMillingTaskConfig.from_stages(
        stages=[FibsemMillingStage(name="s")], name="t"
    )
    cfg.acquisition.imaging.path = str(tmp_path / "imaging")

    task = FibsemMillingTask(microscope, cfg, output_dir=str(tmp_path / "run"))
    task._configure_path()

    assert task.output_dir == str(tmp_path / "run" / "Milling" / "t")
    # still set, for strategies written before stage.output_dir existed
    assert task.config.acquisition.imaging.path == task.output_dir


def test_every_stage_writes_under_the_run_folder(tmp_path, monkeypatch):
    """A real run on Demo with drift correction and per-stage images on: the
    reference image, each stage's drift alignment and its images all land under
    <output_dir>/Milling/<name>, and the imaging path is never written to."""
    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo")
    cfg = FibsemMillingTaskConfig.from_stages(
        stages=[FibsemMillingStage(name="one"), FibsemMillingStage(name="two")],
        name="t",
    )
    cfg.alignment.enabled = True
    cfg.acquisition.acquire_fib = True
    cfg.acquisition.imaging.path = str(tmp_path / "imaging")

    task = run_milling_task(microscope, cfg, output_dir=str(tmp_path / "run"))

    assert task.outcome is MillingProgressStatus.TASK_FINISHED, task.error
    run_folder = Path(task.output_dir)
    assert [stage.output_dir for stage in task.stages] == [task.output_dir] * 2
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written, "the run saved nothing, so this proves nothing"
    assert all(p.is_relative_to(run_folder) for p in written), [
        str(p.relative_to(tmp_path)) for p in written
    ]
    drift = [p for p in written if "Alignment" in p.relative_to(run_folder).parts]
    assert drift, "each stage's drift alignment is saved in the run folder"
    assert not (tmp_path / "imaging").exists()


def test_the_run_folder_is_not_saved_with_the_stage():
    """It belongs to one run, not to the protocol the stage is saved in."""
    stage = FibsemMillingStage(name="s")
    stage.output_dir = "/a/run/folder"

    data = stage.to_dict()

    assert "output_dir" not in data
    assert FibsemMillingStage.from_dict(data).output_dir is None


# ── a stage that raises fails the task (FIB-1112) ────────────────────────────
#
# A stage's error used to be logged and the task reported finished. On an Aquilos
# every pattern call was rejected and AutoLamella recorded the rough mill Completed
# with nothing milled.


class _Rejected(RuntimeError):
    pass


def _reject_milling(microscope):
    """Make the microscope reject every mill, as the Aquilos did; return the calls."""
    calls = []

    def run_milling(*a, **k):
        calls.append((a, k))
        raise _Rejected("could not load type")

    microscope.run_milling = run_milling
    return calls


def test_a_failed_stage_fails_the_task_and_skips_the_rest(tmp_path):
    stages = [FibsemMillingStage(name="rough"), FibsemMillingStage(name="polish")]
    microscope, task = _task(tmp_path, stages=stages, acquire_final_image=False)
    milled = _reject_milling(microscope)
    progress = []
    microscope.milling.progress.changed.connect(progress.append)
    restored = _spy(microscope, "finish_milling")

    task.run()

    assert len(milled) == 1, "polish must not mill on a rough that failed"
    statuses = [p.status for p in progress]
    assert MillingProgressStatus.STAGE_FINISHED not in statuses
    assert statuses[-1] is MillingProgressStatus.TASK_FAILED
    assert progress[-1].error == "could not load type"
    assert task.outcome is MillingProgressStatus.TASK_FAILED
    assert isinstance(task.error, _Rejected)
    assert restored, "cleanup still runs"


def test_a_finished_task_says_so(tmp_path):
    _, task = _task(tmp_path, acquire_final_image=False)

    task.run()

    assert task.outcome is MillingProgressStatus.TASK_FINISHED
    assert task.error is None


def test_run_milling_task_raises_a_failure_when_asked(tmp_path):
    microscope, task = _task(tmp_path, acquire_final_image=False)
    _reject_milling(microscope)
    restored = _spy(microscope, "finish_milling")

    with pytest.raises(_Rejected):
        run_milling_task(microscope, task.config, raise_on_failure=True)
    assert restored, "raised after cleanup, not instead of it"

    returned = run_milling_task(microscope, task.config)
    assert returned.outcome is MillingProgressStatus.TASK_FAILED


# ── how a run ended, on the record (FIB-1257) ────────────────────────────────
#
# Each run records one ``milling`` event with how each stage ended. Before, a
# stage recorded only that it started: a stage that failed or was stopped, and
# the stages after it, left nothing to say so.


def _milling_events(microscope):
    events = []
    microscope.session.record_signal.connect(
        lambda kind, payload: events.append(payload) if kind == "milling" else None
    )
    return events


def test_a_run_records_each_stage_in_the_protocol_s_order(tmp_path):
    stages = [
        FibsemMillingStage(name="rough"),
        FibsemMillingStage(name="off"),
        FibsemMillingStage(name="polish"),
    ]
    stages[1].enabled = False
    microscope, task = _task(tmp_path, stages=stages, acquire_final_image=False)
    events = _milling_events(microscope)

    task.run()

    (run,) = events
    assert run["status"] == "completed"
    assert run["milling_task_name"] == "t"
    assert run["milling_run_id"] == task.task_id
    assert run["started_at"] and run["path"] == task.output_dir
    assert [(s["name"], s["status"]) for s in run["stages"]] == [
        ("rough", "completed"),
        ("off", "skipped"),
        ("polish", "completed"),
    ]
    rough, off, polish = run["stages"]
    assert off["reason"] == "turned off in the protocol"
    # the index milling_stage_started gives it: its place among the enabled
    assert (rough["index"], polish["index"]) == (0, 1)
    assert rough["started_at"] <= rough["ended_at"] <= polish["started_at"]
    assert rough["milling_current"] == stages[0].milling.milling_current


def test_a_failed_stage_is_recorded_and_the_stages_after_it_skipped(tmp_path):
    stages = [FibsemMillingStage(name="rough"), FibsemMillingStage(name="polish")]
    microscope, task = _task(tmp_path, stages=stages, acquire_final_image=False)
    _reject_milling(microscope)
    events = _milling_events(microscope)

    task.run()

    (run,) = events
    assert run["status"] == "failed"
    assert run["error"] == "_Rejected: could not load type"
    rough, polish = run["stages"]
    assert rough["status"] == "failed" and rough["error"] == run["error"]
    assert rough["ended_at"]
    assert polish["status"] == "skipped"
    assert polish["reason"] == "the run failed before this stage"


def test_a_stop_between_stages_is_recorded_as_cancelled(tmp_path):
    stages = [FibsemMillingStage(name="rough"), FibsemMillingStage(name="polish")]
    microscope, task = _task(tmp_path, stages=stages, acquire_final_image=False)
    task._stop_event = threading.Event()
    microscope.milling.progress.changed.connect(
        lambda p: (
            task._stop_event.set()
            if p.status is MillingProgressStatus.STAGE_FINISHED
            else None
        )
    )
    events = _milling_events(microscope)

    task.run()

    (run,) = events
    assert run["status"] == "cancelled" and "error" not in run
    rough, polish = run["stages"]
    assert rough["status"] == "completed"
    assert polish["status"] == "skipped"
    assert polish["reason"] == "the run was stopped before this stage"


def test_the_milling_service_s_stop_ends_the_whole_task(tmp_path):
    """An agent's stop_milling, or Stop agent, stops the milling service. That
    ended only the stage milling then, and the next started (FIB-1398): it must
    end the task."""
    stages = [FibsemMillingStage(name=n) for n in ("one", "two", "three")]
    microscope, task = _task(tmp_path, stages=stages, acquire_final_image=False)
    milled = []

    def run_milling(*a, **k):
        milled.append(k.get("stop_event"))
        microscope.milling.stop()  # from elsewhere, while the stage mills

    microscope.run_milling = run_milling
    events = _milling_events(microscope)

    task.run()

    assert len(milled) == 1, "no stage starts after a stop"
    assert task.outcome is MillingProgressStatus.TASK_CANCELLED
    (run,) = events
    assert [s["status"] for s in run["stages"]] == ["cancelled", "skipped", "skipped"]


def test_the_microscope_s_stop_milling_stops_only_the_pattern(tmp_path):
    """A strategy that ends a mill on purpose (coincidence milling) stops the
    pattern through the microscope; the task goes on to its next stage."""
    stages = [FibsemMillingStage(name=n) for n in ("one", "two")]
    microscope, task = _task(tmp_path, stages=stages, acquire_final_image=False)
    milled = []

    def run_milling(*a, **k):
        milled.append(1)
        microscope.stop_milling()

    microscope.run_milling = run_milling

    task.run()

    assert len(milled) == 2
    assert task.outcome is MillingProgressStatus.TASK_FINISHED


def test_a_run_with_every_stage_turned_off_is_skipped(tmp_path):
    stage = FibsemMillingStage(name="rough")
    stage.enabled = False
    microscope, task = _task(tmp_path, stages=[stage], acquire_final_image=False)
    events = _milling_events(microscope)

    task.run()

    (run,) = events
    assert run["status"] == "skipped"
    assert run["reason"] == "turned off in the protocol"
    assert [s["status"] for s in run["stages"]] == ["skipped"]


def test_each_run_is_its_own_record(tmp_path):
    microscope, task = _task(tmp_path, acquire_final_image=False)
    events = _milling_events(microscope)

    task.run()
    task.run()

    assert [run["status"] for run in events] == ["completed", "completed"]
    assert events[0]["started_at"] <= events[1]["started_at"]


def test_a_task_that_milled_nothing_records_why(tmp_path):
    microscope, task = _task(tmp_path, acquire_final_image=False)
    events = _milling_events(microscope)

    record_milling_skipped(microscope, task.config, "Continue was pressed")

    (skipped,) = events
    assert skipped["status"] == "skipped"
    assert skipped["reason"] == "Continue was pressed"
    assert skipped["milling_task_name"] == "t"
    assert skipped["stages"] == [{"name": "s", "status": "skipped"}]


# ── what a strategy says about its run (FIB-1257) ────────────────────────────


class _PluginStrategy(StandardMillingStrategy):
    """A strategy as a plugin writes one: mills as Standard does, then returns
    whatever it is told to."""

    name = "Plugin"
    returns = None

    def run(self, microscope, stage, stop_event=None):
        super().run(microscope, stage, stop_event)
        return self.returns


def _plugin_stage(returns):
    strategy = _PluginStrategy()
    strategy.returns = returns
    return FibsemMillingStage(name="s", strategy=strategy)


def test_stages_with_different_strategies_of_one_config_class_differ():
    plugin = FibsemMillingStage(name="s", strategy=_PluginStrategy())
    standard = FibsemMillingStage(name="s")

    assert plugin != standard
    assert not plugin.is_compatible_with(standard)
    assert plugin == FibsemMillingStage(name="s", strategy=_PluginStrategy())


def test_the_standard_strategy_says_what_it_milled(tmp_path):
    stage = FibsemMillingStage(name="rough")
    microscope, task = _task(tmp_path, stages=[stage], acquire_final_image=False)
    events = _milling_events(microscope)

    task.run()

    (run,) = events
    strategy = run["stages"][0]["strategy"]
    assert strategy["name"] == "Standard"
    assert strategy["facts"] == {"Pattern": stage.pattern.name}
    assert "end_reason" not in strategy and "figure" not in strategy


def test_a_strategy_that_returns_nothing_is_still_named(tmp_path):
    microscope, task = _task(
        tmp_path, stages=[_plugin_stage(None)], acquire_final_image=False
    )
    events = _milling_events(microscope)

    task.run()

    (run,) = events
    assert run["status"] == "completed"
    assert run["stages"][0]["strategy"] == {"name": "Plugin"}


def test_a_strategy_s_result_is_recorded_as_plain_values(tmp_path):
    import numpy as np

    result = MillingStrategyResult(
        end_reason="intensity drop",
        facts={"Peak": np.float32(0.5), "Steps": 3, "Seen": True, "Mode": object()},
        figure=str(tmp_path / "summary.png"),
    )
    microscope, task = _task(
        tmp_path, stages=[_plugin_stage(result)], acquire_final_image=False
    )
    events = _milling_events(microscope)

    task.run()

    (run,) = events
    strategy = run["stages"][0]["strategy"]
    assert strategy["end_reason"] == "intensity drop"
    assert strategy["figure"] == str(tmp_path / "summary.png")
    facts = strategy["facts"]
    assert facts["Peak"] == 0.5 and type(facts["Peak"]) is float
    assert facts["Steps"] == 3 and facts["Seen"] is True
    assert isinstance(facts["Mode"], str)


def test_a_strategy_that_returns_the_wrong_thing_does_not_fail_the_mill(
    tmp_path, caplog
):
    microscope, task = _task(
        tmp_path, stages=[_plugin_stage({"end": "drop"})], acquire_final_image=False
    )
    events = _milling_events(microscope)

    # setup_session replaced the root logger's handlers, pytest's among them
    logging.getLogger().addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING):
            task.run()
    finally:
        logging.getLogger().removeHandler(caplog.handler)

    (run,) = events
    assert run["status"] == "completed"
    assert run["stages"][0]["strategy"] == {"name": "Plugin"}
    assert "not a MillingStrategyResult" in caplog.text


def test_a_strategy_that_raised_is_named_on_its_failed_stage(tmp_path):
    microscope, task = _task(tmp_path, acquire_final_image=False)
    _reject_milling(microscope)
    events = _milling_events(microscope)

    task.run()

    (run,) = events
    stage = run["stages"][0]
    assert stage["status"] == "failed"
    assert stage["strategy"] == {"name": "Standard"}


# ── stage runs and how they end (FIB-1480) ───────────────────────────────────


def test_a_stage_records_how_it_ended_and_its_steps(tmp_path):
    stage = FibsemMillingStage(name="rough")
    microscope, task = _task(tmp_path, stages=[stage], acquire_final_image=False)
    events = _milling_events(microscope)

    task.run()

    (run,) = events
    rough = run["stages"][0]
    assert rough["status"] == "completed"
    finished = {"kind": "finished", "by": "instrument", "reason": None}
    assert rough["end"] == finished
    (step,) = rough["steps"]
    assert step["index"] == 0 and step["end"] == finished
    assert rough["started_at"] <= step["started_at"] <= step["ended_at"]


class _TwoSteps(StandardMillingStrategy):
    """Mills twice, as Overtilt mills once a pattern; the first step is ended."""

    name = "TwoSteps"

    def run(self, microscope, stage, stop_event=None):
        def end_once(progress):
            if progress.milling_state is MillingState.RUNNING:
                microscope.milling.progress.changed.disconnect(end_once)
                microscope.milling.end("drop seen", by="agent")

        microscope.milling.progress.changed.connect(end_once)
        super().run(microscope, stage, stop_event)
        super().run(microscope, stage, stop_event)
        return MillingStrategyResult(facts={"Steps": 2})


def test_ending_a_stage_completes_it_and_the_task_goes_on(tmp_path):
    stages = [
        FibsemMillingStage(name="rough", strategy=_TwoSteps()),
        FibsemMillingStage(name="polish"),
    ]
    microscope, task = _task(tmp_path, stages=stages, acquire_final_image=False)
    events = _milling_events(microscope)

    task.run()

    assert task.outcome is MillingProgressStatus.TASK_FINISHED
    (run,) = events
    rough, polish = run["stages"]
    assert rough["status"] == "completed"
    assert rough["end"] == {"kind": "ended", "by": "agent", "reason": "drop seen"}
    assert len(rough["steps"]) == 1, "the ended stage mills no second step"
    assert rough["strategy"]["facts"] == {"Steps": 2}
    assert polish["status"] == "completed" and polish["end"]["kind"] == "finished"


class _KeepsMilling(StandardMillingStrategy):
    """Mills without the task's stop event, as coincidence does, and says so."""

    name = "KeepsMilling"

    def run(self, microscope, stage, stop_event=None):
        super().run(microscope, stage, None)
        return MillingStrategyResult(end_reason="stopped", facts={"Seen": True})


def test_a_stopped_stage_keeps_what_its_strategy_said(tmp_path):
    stage = FibsemMillingStage(name="rough", strategy=_KeepsMilling())
    microscope, task = _task(tmp_path, stages=[stage], acquire_final_image=False)

    def stop_once(progress):
        if progress.milling_state is MillingState.RUNNING:
            microscope.milling.progress.changed.disconnect(stop_once)
            microscope.milling.stop()

    microscope.milling.progress.changed.connect(stop_once)
    events = _milling_events(microscope)

    task.run()

    assert task.outcome is MillingProgressStatus.TASK_CANCELLED
    (run,) = events
    rough = run["stages"][0]
    assert rough["status"] == "cancelled"
    assert rough["strategy"]["end_reason"] == "stopped"
    assert rough["end"] == {
        "kind": "stopped",
        "by": None,
        "reason": "Milling stopped.",
    }
    assert rough["steps"][0]["end"]["kind"] == "stopped"


def test_a_failed_stage_records_its_end(tmp_path):
    microscope, task = _task(tmp_path, acquire_final_image=False)
    _reject_milling(microscope)
    events = _milling_events(microscope)

    task.run()

    (run,) = events
    stage = run["stages"][0]
    assert stage["end"] == {
        "kind": "failed",
        "by": None,
        "reason": "_Rejected: could not load type",
    }
    assert stage["steps"] == []


# ── monitors on a stage's mill steps (FIB-1481) ──────────────────────────────


def test_a_stage_s_monitors_are_recorded_on_its_steps(tmp_path, monkeypatch):
    from fibsem.milling.monitors import (
        MONITOR_PLUGINS,
        MillingMonitor,
        MillingMonitorConfig,
        MillingMonitorResult,
    )

    class Counts(MillingMonitor[MillingMonitorConfig]):
        name = "Counts"

        def on_start(self):
            self.report({"started": True}, notable=True)

        def result(self):
            return MillingMonitorResult(facts={"Seen": True}, files=self.folder)

    monkeypatch.setitem(MONITOR_PLUGINS.registered, Counts.name, Counts)
    stage = FibsemMillingStage(name="rough")
    stage.monitors = [MillingMonitorConfig().made_for(Counts)]
    microscope, task = _task(tmp_path, stages=[stage], acquire_final_image=False)
    events = _milling_events(microscope)

    task.run()

    (run,) = events
    (step,) = run["stages"][0]["steps"]
    (monitor,) = step["monitors"]
    assert monitor["name"] == "Counts" and monitor["facts"] == {"Seen": True}
    assert monitor["files"].endswith("monitors/Counts".replace("/", os.sep))
    assert step["notable"][0]["reading"] == {"started": True}
    assert step["pauses"] == []
