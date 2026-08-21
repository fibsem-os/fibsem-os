from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pytest

from fibsem import utils
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks.trench import (
    MillTrenchTask,
    MillTrenchTaskConfig,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemImage, FibsemStagePosition


def _make_trench_task(microscope: FibsemMicroscope, tmp_path: Path) -> MillTrenchTask:
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    config = MillTrenchTaskConfig()
    return MillTrenchTask(microscope=microscope, config=config, lamella=lamella)


@pytest.fixture
def compustage_microscope() -> FibsemMicroscope:
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope.system.stage.shuttle_pre_tilt = 0
    microscope._update_orientations()
    return microscope


def test_get_stage_position_for_orientation_none_returns_same_position(
    compustage_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """When orientation is None the stage position is returned unchanged."""
    task = _make_trench_task(compustage_microscope, tmp_path)
    pos = FibsemStagePosition(
        x=1e-3, y=2e-3, z=3e-3, r=np.radians(0), t=np.radians(-23)
    )
    result = task._get_stage_position_for_orientation(pos, None)
    assert result is pos


def test_get_stage_position_for_orientation_delegates_to_get_target_position(
    compustage_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """When orientation is set the result comes from get_target_position."""
    task = _make_trench_task(compustage_microscope, tmp_path)
    pos = FibsemStagePosition(r=np.radians(0), t=np.radians(-23))
    result = task._get_stage_position_for_orientation(pos, "SEM")
    sem = compustage_microscope.get_orientation("SEM")
    assert sem.t is not None and result.t is not None
    assert np.isclose(result.t, sem.t, atol=1e-6)


@pytest.mark.parametrize("orientation", ["SEM", "FIB", "MILLING", None])
def test_mill_trench_task_config_orientation_accepts_none_and_strings(
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]],
) -> None:
    """MillTrenchTaskConfig accepts all valid orientation values including None."""
    config = MillTrenchTaskConfig(orientation=orientation)
    assert config.orientation == orientation


# ── output recording ──────────────────────────────────────────────────────────


def _image_written_to(path: Path) -> FibsemImage:
    """A FibsemImage that reports having been written to `path`."""
    image = FibsemImage(data=np.zeros((4, 4), dtype=np.uint8))
    image.filepath = str(path)
    return image


def test_record_output_stores_paths_relative_to_the_lamella(
    compustage_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """Absolute paths would break the moment an experiment is copied off the
    microscope, which they routinely are."""
    task = _make_trench_task(compustage_microscope, tmp_path)
    written = Path(task.lamella.path) / "ref_MillTrench_final_res_01_eb.tif"

    task._record_output("final_sem", _image_written_to(written))

    assert task.lamella.task_state.outputs == {
        "final_sem": ["ref_MillTrench_final_res_01_eb.tif"]
    }


def test_record_output_skips_an_image_that_was_never_written(
    compustage_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """save() sets filepath only after a successful write, so an image with none
    was not written. The FM task relies on this: its acquire swallows save errors."""
    task = _make_trench_task(compustage_microscope, tmp_path)

    task._record_output(
        "fluorescence", FibsemImage(data=np.zeros((4, 4), dtype=np.uint8))
    )
    task._record_output("fluorescence", None)

    assert task.lamella.task_state.outputs == {}


def test_pre_task_clears_the_previous_runs_outputs_and_end_timestamp(
    compustage_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """task_state is one object reused across runs. Without an explicit reset the
    next run's history entry accumulates the previous run's paths, and reports the
    previous run's completion time while it is still in progress."""
    task = _make_trench_task(compustage_microscope, tmp_path)

    # stand in for a completed previous run
    task.lamella.task_state.outputs = {
        "final_sem": ["ref_Previous_final_res_01_eb.tif"]
    }
    task.lamella.task_state.end_timestamp = 1234.0

    task.pre_task()

    assert task.lamella.task_state.outputs == {}
    assert task.lamella.task_state.end_timestamp is None


def test_pre_task_keeps_the_same_task_state_object(
    compustage_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """The lamella list and card widgets connect psygnal handlers to this instance
    and use them as their only refresh trigger. Replacing the object orphans those
    connections and they silently stop updating mid-workflow."""
    task = _make_trench_task(compustage_microscope, tmp_path)
    before = task.lamella.task_state

    task.pre_task()

    assert task.lamella.task_state is before


def test_default_filename_is_recorded_as_the_final_reference_set(
    monkeypatch, compustage_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    from fibsem import acquire
    from fibsem.structures import ImageSettings

    task = _make_trench_task(compustage_microscope, tmp_path)
    lam = Path(task.lamella.path)
    pair = (
        _image_written_to(lam / "ref_MillTrench_final_res_01_eb.tif"),
        _image_written_to(lam / "ref_MillTrench_final_res_01_ib.tif"),
    )
    monkeypatch.setattr(acquire, "acquire_set_of_channels", lambda *a, **k: [pair])

    task._acquire_set_of_channels(ImageSettings(), field_of_views=(100e-6,))

    assert set(task.lamella.task_state.outputs) == {"final_sem", "final_fib"}


def test_a_custom_filename_is_not_recorded_as_a_final_reference_set(
    monkeypatch, compustage_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """Undercut alignment and one-off reference acquisitions pass their own name and
    do not match the review panel's convention today. Recording them as `final` would
    start showing them there."""
    from fibsem import acquire
    from fibsem.structures import ImageSettings

    task = _make_trench_task(compustage_microscope, tmp_path)
    lam = Path(task.lamella.path)
    pair = (
        _image_written_to(lam / "ref_MillTrench_undercut_res_01_eb.tif"),
        _image_written_to(lam / "ref_MillTrench_undercut_res_01_ib.tif"),
    )
    monkeypatch.setattr(acquire, "acquire_set_of_channels", lambda *a, **k: [pair])

    task._acquire_set_of_channels(
        ImageSettings(), field_of_views=(100e-6,), filename="ref_MillTrench_undercut"
    )

    assert set(task.lamella.task_state.outputs) == {"other_sem", "other_fib"}


def test_record_output_does_not_record_the_same_file_twice(
    compustage_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """Acquiring the same set twice in one run overwrites the same files; the record
    describes files, not writes."""
    task = _make_trench_task(compustage_microscope, tmp_path)
    written = Path(task.lamella.path) / "ref_MillTrench_final_res_01_eb.tif"

    task._record_output("final_sem", _image_written_to(written))
    task._record_output("final_sem", _image_written_to(written))

    assert task.lamella.task_state.outputs == {
        "final_sem": ["ref_MillTrench_final_res_01_eb.tif"]
    }


def test_task_is_cancellation_classifies_stop_vs_failure(tmp_path: Path):
    """AutoLamellaTask._is_cancellation decides which hook event a raise produces: a user
    Stop (cancel exc / stopping manager) fires task_cancelled, a real failure fires
    task_failed.

    Driven off a real TaskManager: this asks the manager whether the task should be
    unwinding, and a stand-in would only confirm whatever shape the test assumed.
    """
    import types

    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
    from fibsem.cancellation import OperationCancelledError

    class _NoMicroscope:
        fm = None

    def _manager() -> TaskManager:
        return TaskManager(
            microscope=_NoMicroscope(),
            experiment=Experiment(path=tmp_path, name="test-exp"),
            parent_ui=None,
        )

    def _isc(exc, manager=None):
        s = types.SimpleNamespace(task_manager=manager)
        return AutoLamellaTask._is_cancellation(s, exc)

    running = _manager()
    assert _isc(OperationCancelledError("x"), running) is True
    assert _isc(InterruptedError("Workflow aborted by user."), running) is True
    assert _isc(ValueError("real bug"), running) is False

    # Either kind of stop makes an exception thrown on the way out a cancellation,
    # not a failure — abandoning one task is as much a user Stop as ending the run.
    stopped_run = _manager()
    stopped_run.stop()
    assert _isc(ValueError("real bug"), stopped_run) is True

    stopped_task = _manager()
    stopped_task.stop_task()
    assert _isc(ValueError("real bug"), stopped_task) is True
    assert stopped_task.is_stopped is False  # ... and the run is untouched


# ── which role a task's reference set is recorded under (FIB-579) ──────────────
#
# The default rule -- default filename means "final", a caller-supplied one means
# "other" -- is already pinned in both directions above, by
# test_default_filename_is_recorded_as_the_final_reference_set and
# test_a_custom_filename_is_not_recorded_as_a_final_reference_set. Those still pass
# unchanged, which is the point: this adds a way for a task to opt out of the rule
# rather than altering it.


def _fake_acquire_set_of_channels(image_settings, hfws, filename):
    """Stand in for a real acquisition: write the files it would have written.

    The reader requires the files to exist, so touching them is what makes the
    round trip through `final_reference_images` meaningful rather than a check on
    a dict of strings.
    """
    images = []
    for i, _ in enumerate(hfws, start=1):
        pair = []
        for beam in ("eb", "ib"):
            path = Path(image_settings.path) / f"{filename}_res_{i:02d}_{beam}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            pair.append(_image_written_to(path))
        images.append(tuple(pair))
    return images


@pytest.fixture
def stub_acquisition(monkeypatch):
    """Patch out the hardware, keeping the filename convention the real one uses."""
    from fibsem import acquire

    monkeypatch.setattr(
        acquire,
        "acquire_set_of_channels",
        lambda microscope, image_settings, hfws, filename="ref_image", **kwargs: (
            _fake_acquire_set_of_channels(image_settings, hfws, filename)
        ),
    )


def _make_reference_image_task(microscope: FibsemMicroscope, tmp_path: Path):
    from fibsem.applications.autolamella.workflows.tasks.reference_image import (
        AcquireReferenceImageConfig,
        AcquireReferenceImageTask,
    )

    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.milling_pose = microscope.get_microscope_state()
    return AcquireReferenceImageTask(
        microscope=microscope, config=AcquireReferenceImageConfig(), lamella=lamella
    )


def test_acquire_reference_image_task_reaches_the_review_panel(
    compustage_microscope: FibsemMicroscope, tmp_path: Path, stub_acquisition
) -> None:
    """The task names its own files, and the role used to be read off that name.

    A named set means "an extra one this task happened to grab", which is right for
    a task acquiring mid-run and wrong for the one whose entire product is a
    reference set -- it landed in `other_*` and the review panel, which asks for
    `final_*`, showed nothing for it on every run.
    """
    from fibsem.applications.autolamella.task_outputs import final_reference_images

    task = _make_reference_image_task(compustage_microscope, tmp_path)

    task._run()

    outputs = task.lamella.task_state.outputs
    assert set(outputs) == {"final_sem", "final_fib"}
    found = final_reference_images(task.lamella, task.lamella.task_state)
    assert len(found) == 4, "both resolutions, both beams"
    assert all(Path(path).is_file() for path in found)
