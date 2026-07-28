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
    pos = FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3, r=np.radians(0), t=np.radians(-23))
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

    task._record_output("fluorescence", FibsemImage(data=np.zeros((4, 4), dtype=np.uint8)))
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
    task.lamella.task_state.outputs = {"final_sem": ["ref_Previous_final_res_01_eb.tif"]}
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
