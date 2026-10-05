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
from fibsem.structures import FibsemStagePosition, MicroscopeState

SIM_ARCTIS_CONFIG_PATH = os.path.join(
    fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml"
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
        task, "_run_autofocus", lambda: types.SimpleNamespace(working_distance=refined)
    )
    monkeypatch.setattr(af, "acquire_image", lambda **kwargs: None)

    task._run()

    assert lamella.fluorescence_pose.objective_position == pytest.approx(refined)


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


# ── what a task's re-record says about where the pose came from ─────────────


def _marked(microscope: FibsemMicroscope, tmp_path: Path, orientation: str) -> Lamella:
    """A lamella marked the way the app marks one, from a position in *orientation*."""
    from copy import deepcopy

    from fibsem.applications.autolamella.poses import build_lamella_poses

    pose = microscope.get_orientation(orientation)
    position = deepcopy(microscope.get_stage_position())
    position.x, position.y, position.r, position.t = 100e-6, 50e-6, pose.r, pose.t
    lamella = Lamella(path=tmp_path / orientation, number=0, petname=orientation)
    build_lamella_poses(microscope, position=position).write_to(lamella)
    return lamella


def _moved(position: FibsemStagePosition, dx: float) -> FibsemStagePosition:
    from copy import deepcopy

    moved = deepcopy(position)
    moved.x += dx
    return moved


def test_acquiring_leaves_a_derived_pose_derived(
    fm_microscope: FibsemMicroscope, tmp_path: Path, monkeypatch
) -> None:
    """The acquire task drives to the pose and re-reads the stage. Nobody looked, so a
    derived pose is still a guess -- and still follows when its milling pose moves.

    It used to be recorded as observed, after which the move kept it, pointing at where
    the lamella had been (FIB-954).
    """
    import fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence as af
    from fibsem.applications.autolamella.poses import (
        FLUORESCENCE_POSE,
        MILLING_POSE,
        Followed,
        PoseProvenance,
        move_pose,
    )

    lamella = _marked(fm_microscope, tmp_path, "MILLING")
    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.DERIVED
    task = _acquire_task(fm_microscope, lamella)
    monkeypatch.setattr(task, "_run_autofocus", lambda: None)
    monkeypatch.setattr(af, "acquire_image", lambda **kwargs: None)

    task._run()

    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.DERIVED

    moved = _moved(lamella.milling_pose.stage_position, 20e-6)
    followed = move_pose(fm_microscope, lamella, MILLING_POSE, position=moved)

    assert followed is Followed.DERIVED
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(moved.x)


def test_acquiring_leaves_an_observed_pose_observed(
    fm_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    from fibsem.applications.autolamella.poses import FLUORESCENCE_POSE, PoseProvenance

    lamella = _marked(fm_microscope, tmp_path, "FM")
    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.OBSERVED
    task = _acquire_task(fm_microscope, lamella)

    task._update_fluorescence_pose()

    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.OBSERVED


def test_a_pose_a_person_confirmed_is_observed_and_its_milling_pose_follows(
    fm_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """Select Fluorescence Position, supervised: the operator can re-centre while the
    task waits, so what it records is an observation. A milling pose derived from the
    old place follows it there, as any other move of the fluorescence pose does."""
    from fibsem.applications.autolamella.poses import (
        FLUORESCENCE_POSE,
        MILLING_POSE,
        PoseProvenance,
    )

    lamella = _marked(fm_microscope, tmp_path, "FM")
    assert lamella.provenance_of(MILLING_POSE) is PoseProvenance.DERIVED
    recentred = _moved(lamella.fluorescence_pose.stage_position, 20e-6)
    fm_microscope.move_stage_absolute(recentred)
    task = _acquire_task(fm_microscope, lamella)

    task._update_fluorescence_pose(observed=True)

    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.OBSERVED
    assert lamella.fluorescence_pose.objective_position is not None
    assert lamella.milling_pose.stage_position.x == pytest.approx(recentred.x)
    assert lamella.provenance_of(MILLING_POSE) is PoseProvenance.DERIVED


@pytest.mark.parametrize("supervised", [False, True])
def test_select_fluorescence_position_records_what_it_saw(
    fm_microscope: FibsemMicroscope, tmp_path: Path, monkeypatch, supervised: bool
) -> None:
    """The task itself: unattended it keeps a derived pose derived; supervised, the
    operator had the chance to centre it, so it is observed."""
    from fibsem.applications.autolamella.poses import FLUORESCENCE_POSE, PoseProvenance
    from fibsem.applications.autolamella.workflows.tasks.select_fluorescence_position import (
        SelectFluorescencePositionConfig,
        SelectFluorescencePositionTask,
    )

    monkeypatch.setattr(
        SelectFluorescencePositionTask, "validate", property(lambda self: supervised)
    )
    lamella = _marked(fm_microscope, tmp_path, "MILLING")
    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.DERIVED
    task = SelectFluorescencePositionTask(
        microscope=fm_microscope,
        config=SelectFluorescencePositionConfig(),
        lamella=lamella,
    )

    task._run()

    expected = PoseProvenance.OBSERVED if supervised else PoseProvenance.DERIVED
    assert lamella.provenance_of(FLUORESCENCE_POSE) is expected


# ── back to back: the objective and a change of pose ────────────────────────


def _acquire_quietly(task, monkeypatch) -> None:
    import fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence as af

    monkeypatch.setattr(task, "_run_autofocus", lambda: None)
    monkeypatch.setattr(af, "acquire_image", lambda **kwargs: None)


def _objective_during_moves(microscope: FibsemMicroscope) -> list:
    """The objective's state at every absolute stage move, and the tilt moved to."""
    seen = []
    real = microscope.safe_absolute_stage_movement

    def record(position, *args, **kwargs):
        seen.append((microscope.fm.objective.state, position.t))
        return real(position, *args, **kwargs)

    microscope.safe_absolute_stage_movement = record
    return seen


def test_back_to_back_the_objective_is_out_before_the_stage_tilts(
    fm_microscope: FibsemMicroscope, tmp_path: Path, monkeypatch
) -> None:
    """With the objective left in between lamellae, a lamella whose fluorescence
    pose is in another orientation used to be reached by tilting under it."""
    flipped = _marked(fm_microscope, tmp_path, "FIB")
    at_milling = _marked(fm_microscope, tmp_path, "MILLING")
    assert (
        flipped.fluorescence_pose.stage_position.t
        != at_milling.fluorescence_pose.stage_position.t
    )
    config = AcquireFluorescenceImageConfig(
        channel_settings=[ChannelSettings(name="GFP")], retract_objective=False
    )
    seen = _objective_during_moves(fm_microscope)

    for lamella in (flipped, at_milling):
        task = AcquireFluorescenceImageTask(
            microscope=fm_microscope, config=config, lamella=lamella
        )
        _acquire_quietly(task, monkeypatch)
        task._run()

    state, tilt = seen[-1]
    assert tilt == pytest.approx(at_milling.fluorescence_pose.stage_position.t)
    assert state != "Inserted"
    assert fm_microscope.fm.objective.state == "Inserted"  # and back in to acquire


def test_back_to_back_in_one_pose_the_objective_stays_in(
    fm_microscope: FibsemMicroscope, tmp_path: Path, monkeypatch
) -> None:
    """The usual step to the next lamella: same pose, no reason to retract."""
    first = _marked(fm_microscope, tmp_path, "FIB")
    second = _marked(fm_microscope, tmp_path / "b", "FIB")
    second.fluorescence_pose.stage_position.x += 50e-6
    config = AcquireFluorescenceImageConfig(
        channel_settings=[ChannelSettings(name="GFP")], retract_objective=False
    )
    seen = _objective_during_moves(fm_microscope)

    for lamella in (first, second):
        task = AcquireFluorescenceImageTask(
            microscope=fm_microscope, config=config, lamella=lamella
        )
        _acquire_quietly(task, monkeypatch)
        task._run()

    assert seen[-1][0] == "Inserted"


def test_a_pose_the_fm_cannot_acquire_from_goes_to_where_it_can(
    tmp_path: Path,
) -> None:
    """On an offset mount a fluorescence pose standing at the beams is not somewhere
    the objective sees the sample. The task used to re-pose it to SEM, still at the
    beams; it goes to the FM instead."""
    from fibsem.structures import DeviceImagingState

    microscope, _ = utils.setup_session(
        config_path=os.path.join(fconfig.CONFIG_PATH, "sim-iflm-configuration.yaml"),
        setup_logging=False,
    )
    fib = microscope.get_orientation("FIB")
    at_the_beams = FibsemStagePosition(x=100e-6, y=50e-6, z=0.0, r=fib.r, t=fib.t)
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.milling_pose = MicroscopeState(stage_position=at_the_beams)
    pose = MicroscopeState(stage_position=at_the_beams)
    pose.objective_position = CONFIGURED_OBJECTIVE
    lamella.fluorescence_pose = pose
    task = _acquire_task(microscope, lamella)

    task._move_to_stage_position()

    assert microscope.is_at_device("FM")
    assert microscope.get_device_imaging_state("FM") is DeviceImagingState.READY
