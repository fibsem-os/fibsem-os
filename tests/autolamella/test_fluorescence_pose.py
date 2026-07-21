"""Tests for objective (focus) position handling in the fluorescence workflow tasks.

Regression coverage for the bug where a lamella's configured objective focus
position was lost (reset to None or overwritten with the live objective
position), which made the Selected Lamella objective control disappear.
"""
import os
import types
from pathlib import Path

import pytest

import fibsem.config as fconfig
from fibsem import utils
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (
    AcquireFluorescenceImageConfig,
    AcquireFluorescenceImageTask,
)
from fibsem.applications.autolamella.workflows.tasks.mill_coincident import (
    MillCoincidentTask,
    MillCoincidentTaskConfig,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition, MicroscopeState

SIM_ARCTIS_CONFIG_PATH = os.path.join(
    fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml"
)

CONFIGURED_OBJECTIVE = 0.006  # 6 mm — the user's configured focus position
CLIP_LIMIT = 0.004            # 4 mm — forces the live objective to diverge on move


@pytest.fixture
def fm_microscope() -> FibsemMicroscope:
    """A simulated microscope with a fluorescence module attached."""
    microscope, _ = utils.setup_session(
        config_path=SIM_ARCTIS_CONFIG_PATH, setup_logging=False
    )
    if microscope.fm is None:
        pytest.skip("Fluorescence microscope not available in simulator")
    microscope.fm._allow_unknown_orientations = True
    return microscope


def _make_lamella(tmp_path: Path, objective_position, with_fluorescence_pose: bool = True) -> Lamella:
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    stage = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0, coordinate_system="RAW")
    lamella.milling_pose = MicroscopeState(stage_position=stage)  # provides lamella.stage_position
    if with_fluorescence_pose:
        fp = MicroscopeState(stage_position=stage)
        fp.objective_position = objective_position
        lamella.fluorescence_pose = fp
    return lamella


def _acquire_task(microscope: FibsemMicroscope, lamella: Lamella) -> AcquireFluorescenceImageTask:
    return AcquireFluorescenceImageTask(
        microscope=microscope, config=AcquireFluorescenceImageConfig(), lamella=lamella
    )


def test_update_fluorescence_pose_preserves_configured_objective(
    fm_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """Refreshing the pose keeps the configured objective position, even when the
    live objective is at a different position."""
    # clip limit below the configured value so moving the objective diverges from it
    fm_microscope.fm.objective._limit_position = CLIP_LIMIT
    lamella = _make_lamella(tmp_path, CONFIGURED_OBJECTIVE)
    task = _acquire_task(fm_microscope, lamella)

    # move the live objective to a position that differs from the configured one
    fm_microscope.fm.objective.move_absolute(CONFIGURED_OBJECTIVE)  # clips to CLIP_LIMIT
    assert fm_microscope.fm.objective.position != pytest.approx(CONFIGURED_OBJECTIVE)

    task._update_fluorescence_pose()

    assert lamella.fluorescence_pose is not None
    assert lamella.fluorescence_pose.objective_position == pytest.approx(CONFIGURED_OBJECTIVE)


def test_update_fluorescence_pose_refreshes_stage_position(
    fm_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """The pose's stage position is refreshed from the current microscope state."""
    lamella = _make_lamella(tmp_path, CONFIGURED_OBJECTIVE)
    task = _acquire_task(fm_microscope, lamella)

    task._update_fluorescence_pose()

    assert lamella.fluorescence_pose.stage_position is not None
    assert lamella.fluorescence_pose.objective_position == pytest.approx(CONFIGURED_OBJECTIVE)


def test_update_fluorescence_pose_without_existing_pose_does_not_crash(
    fm_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """When no fluorescence pose exists yet, the helper sets one with a None objective
    (rather than raising)."""
    lamella = _make_lamella(tmp_path, None, with_fluorescence_pose=False)
    assert lamella.fluorescence_pose is None
    task = _acquire_task(fm_microscope, lamella)

    task._update_fluorescence_pose()

    assert lamella.fluorescence_pose is not None
    assert lamella.fluorescence_pose.objective_position is None


def test_acquire_fluorescence_persists_autofocus_result(
    fm_microscope: FibsemMicroscope, tmp_path: Path, monkeypatch
) -> None:
    """When autofocus runs, its refined objective (working_distance) is saved to the
    pose, overriding the pre-run configured value (and surviving the pose refresh)."""
    import fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence as af

    refined = 0.0055  # differs from the configured value
    assert refined != pytest.approx(CONFIGURED_OBJECTIVE)

    lamella = _make_lamella(tmp_path, CONFIGURED_OBJECTIVE)
    task = _acquire_task(fm_microscope, lamella)
    assert task.config.autofocus_settings.enabled  # default; drives the autofocus branch

    # stub the heavy work: autofocus returns a known result, image acquisition is a no-op.
    # working_distance mirrors the real AutoFocusResult field the task reads.
    monkeypatch.setattr(task, "_run_autofocus", lambda: types.SimpleNamespace(working_distance=refined))
    monkeypatch.setattr(af, "acquire_image", lambda **kwargs: None)

    task._run()

    assert lamella.fluorescence_pose.objective_position == pytest.approx(refined)


def test_mill_coincident_requires_objective_position(
    fm_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """MillCoincidentTask raises a clear ValueError (not AttributeError) when the
    lamella has no configured objective position."""
    lamella = _make_lamella(tmp_path, None, with_fluorescence_pose=False)
    task = MillCoincidentTask(
        microscope=fm_microscope, config=MillCoincidentTaskConfig(), lamella=lamella
    )
    with pytest.raises(ValueError, match="objective position"):
        task._run()
