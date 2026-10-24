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
from fibsem.fm.structures import ChannelSettings
from fibsem.microscope import FibsemMicroscope
from fibsem.structures.settings import FibsemStagePosition, MicroscopeState
from tests.autolamella._on_the_record import manager_for

SIM_ARCTIS_CONFIG_PATH = os.path.join(
    fconfig.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
)

CONFIGURED_OBJECTIVE = 0.006  # 6 mm — the user's configured focus position
CLIP_LIMIT = 0.004  # 4 mm — forces the live objective to diverge on move


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


def _make_lamella(
    tmp_path: Path, objective_position, with_fluorescence_pose: bool = True
) -> Lamella:
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    stage = FibsemStagePosition(
        x=0.0, y=0.0, z=0.0, r=0.0, t=0.0, coordinate_system="RAW"
    )
    lamella.milling_pose = MicroscopeState(
        stage_position=stage
    )  # provides lamella.stage_position
    if with_fluorescence_pose:
        fp = MicroscopeState(stage_position=stage)
        fp.objective_position = objective_position
        lamella.fluorescence_pose = fp
    return lamella


def _acquire_task(
    microscope: FibsemMicroscope, lamella: Lamella
) -> AcquireFluorescenceImageTask:
    return AcquireFluorescenceImageTask(
        microscope=microscope,
        # one channel: the task refuses to run without any (FIB-1067)
        config=AcquireFluorescenceImageConfig(
            channel_settings=[ChannelSettings(name="GFP")]
        ),
        lamella=lamella,
        task_manager=manager_for(microscope, lamella),
    )


def test_update_fluorescence_pose_preserves_configured_objective(
    fm_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """Refreshing the pose keeps the configured objective position, even when the
    live objective is at a different position."""
    # clip limit below the configured value so moving the objective diverges from it
    fm_microscope.fm.objective.limit_position = CLIP_LIMIT
    lamella = _make_lamella(tmp_path, CONFIGURED_OBJECTIVE)
    task = _acquire_task(fm_microscope, lamella)

    # move the live objective to a position that differs from the configured one
    fm_microscope.fm.objective.move_absolute(
        CONFIGURED_OBJECTIVE
    )  # clips to CLIP_LIMIT
    assert fm_microscope.fm.objective.position != pytest.approx(CONFIGURED_OBJECTIVE)

    task._update_fluorescence_pose()

    assert lamella.fluorescence_pose is not None
    assert lamella.fluorescence_pose.objective_position == pytest.approx(
        CONFIGURED_OBJECTIVE
    )


def test_update_fluorescence_pose_refreshes_stage_position(
    fm_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """The pose's stage position is refreshed from the current microscope state."""
    lamella = _make_lamella(tmp_path, CONFIGURED_OBJECTIVE)
    task = _acquire_task(fm_microscope, lamella)

    task._update_fluorescence_pose()

    assert lamella.fluorescence_pose.stage_position is not None
    assert lamella.fluorescence_pose.objective_position == pytest.approx(
        CONFIGURED_OBJECTIVE
    )


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
    assert (
        task.config.autofocus_settings.enabled
    )  # default; drives the autofocus branch

    # stub the heavy work: autofocus returns a known result, image acquisition is a no-op.
    # working_distance mirrors the real AutoFocusResult field the task reads.
    monkeypatch.setattr(
        task,
        "_run_autofocus",
        lambda: types.SimpleNamespace(working_distance=refined, found=True),
    )
    monkeypatch.setattr(af, "acquire_image", lambda **kwargs: None)

    task._run()

    assert lamella.fluorescence_pose.objective_position == pytest.approx(refined)


def test_the_autofocus_saves_under_the_task_s_folder_and_records_where(
    fm_microscope: FibsemMicroscope, tmp_path: Path, monkeypatch
) -> None:
    """The sweep saves its result itself, so its ``fm_autofocus`` event says where
    and the History tab can open it (FIB-1257). The task used to save it
    afterwards, outside the record."""
    from fibsem.autofunctions.autofocus import (
        AutoFocusSettings,
        FocusMethod,
        FocusSweepPass,
    )

    lamella = _make_lamella(tmp_path, None, with_fluorescence_pose=False)
    lamella.path.mkdir(parents=True, exist_ok=True)
    task = _acquire_task(fm_microscope, lamella)
    task.config.autofocus_settings = AutoFocusSettings(
        passes=[
            FocusSweepPass(search_range=4e-6, step_size=2e-6),
            FocusSweepPass(search_range=2e-6, step_size=1e-6),
        ],
        method=FocusMethod.LAPLACIAN,
    )
    task.config.autofocus_settings.channel_name = "GFP"
    events = []
    fm_microscope.session.record_signal.connect(
        lambda kind, payload: events.append(payload) if kind == "fm_autofocus" else None
    )
    # the sim's frames score flat, which finds no focus (FIB-1519)
    monkeypatch.setattr("fibsem.fm.calibration.has_peak", lambda scores: True)

    result = task._run_autofocus()

    (event,) = events
    assert event["status"] == "completed"
    saved = Path(event["path"])
    assert saved.parent == Path(task.output_dir) / "autofunctions"
    assert saved.name.startswith(f"{task.task_name}_autofocus_")
    assert (saved / "plot.png").is_file() and (saved / "data.json").is_file()
    assert event["position"] == pytest.approx(result.working_distance)


def test_the_z_stack_is_saved_in_the_run_s_folder(
    fm_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """The run's own folder, not the lamella's (FIB-1275): a re-run keeps the
    earlier stack beside its other images."""
    lamella = _make_lamella(tmp_path, CONFIGURED_OBJECTIVE)
    lamella.path.mkdir(parents=True, exist_ok=True)
    task = _acquire_task(fm_microscope, lamella)
    task.config.autofocus_settings.passes = []  # no autofocus: the stack is the point
    task.lamella.task_state.outputs = {}

    task._run()

    (stack,) = lamella.task_state.outputs["fluorescence"]
    assert Path(lamella.path, stack).parent == Path(task.output_dir)
    assert Path(lamella.path, stack).is_file()


def test_mill_coincident_requires_objective_position(
    fm_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """MillCoincidentTask raises a clear ValueError (not AttributeError) when the
    lamella's coincidence setup has no recorded objective position.

    Since FIB-910 the objective height lives on the Setup Coincidence Milling
    record rather than the fluorescence pose; a lamella whose setup never ran is
    refused before anything moves."""
    from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
        SetupCoincidenceMillingTaskConfig,
    )

    lamella = _make_lamella(tmp_path, None, with_fluorescence_pose=False)
    lamella.task_config["Setup Coincidence Milling"] = (
        SetupCoincidenceMillingTaskConfig(task_name="Setup Coincidence Milling")
    )
    task = MillCoincidentTask(
        microscope=fm_microscope, config=MillCoincidentTaskConfig(), lamella=lamella
    )
    with pytest.raises(ValueError, match="objective position"):
        task._run()
