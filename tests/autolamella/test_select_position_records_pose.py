"""SelectMillingPositionTask records the milling pose and the alignment
reference whether or not coincidence alignment is on.

Regression: the task's tail -- the position confirmation, the point of
interest, the alignment area, the reference images and the milling pose --
had ended up inside the coincidence helper, which only runs with
``auto_milling_alignment`` on and returns early when the stage is already at
the milling angle. With the default configuration the task moved, imaged,
and recorded nothing.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
)
from fibsem.workflows.task import ALIGNMENT_REFERENCE_IMAGE_FILENAME
from tests.answering import answering
from tests.autolamella._on_the_record import manager_for
from tests.fixtures.microscopes import demo_microscope

# Tens of seconds each on the simulator, so pull requests run these on one CI leg
# only; the nightly and release runs take them everywhere (python-package.yml).
pytestmark = pytest.mark.slow


microscope = demo_microscope("microscope-configuration.yaml", scope="function")


@pytest.mark.parametrize("auto_milling_alignment", [False, True])
def test_the_task_records_pose_and_reference_either_way(
    microscope, tmp_path, auto_milling_alignment
):
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    assert lamella.milling_angle is None
    config = SelectMillingPositionTaskConfig(
        auto_milling_alignment=auto_milling_alignment,
        use_autofocus=False,
        select_poi=True,
    )
    task = SelectMillingPositionTask(
        microscope=microscope,
        config=config,
        lamella=lamella,
        task_manager=manager_for(microscope, lamella),
    )

    with answering(task.task_manager.experiment):
        task._run()

    assert lamella.milling_angle is not None, "milling pose and angle were stored"
    assert os.path.exists(
        os.path.join(lamella.path, ALIGNMENT_REFERENCE_IMAGE_FILENAME)
    ), "the alignment reference was acquired"


@pytest.mark.parametrize("sync", [True, False])
def test_the_fluorescence_pose_follows_the_milling_pose(tmp_path, sync):
    """On an instrument with a fluorescence microscope, the task leaves the
    fluorescence pose describing the position it recorded the milling pose
    at, not the one the lamella was marked at (FIB-954), when asked to. Off,
    the default, leaves the fluorescence pose where it was."""
    from fibsem.applications.autolamella.poses import (
        _to_fluorescence,
        build_lamella_poses,
    )
    from fibsem.structures import FibsemStagePosition

    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(
            cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
        ),
    )
    try:
        marked = microscope.get_stage_position()
        poses = build_lamella_poses(microscope=microscope, position=marked)
        lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
        lamella.path.mkdir(parents=True, exist_ok=True)
        lamella.milling_pose = poses.beam
        lamella.fluorescence_pose = poses.fluorescence
        assert lamella.fluorescence_pose is not None
        stale = lamella.fluorescence_pose.stage_position

        # the operator centres the site 20 um away before the task records it
        moved = FibsemStagePosition(
            x=marked.x + 20e-6,
            y=marked.y,
            z=marked.z,
            r=marked.r,
            t=marked.t,
            coordinate_system=marked.coordinate_system,
        )
        microscope.move_stage_absolute(moved)
        lamella.milling_pose = microscope.get_microscope_state()
        config = SelectMillingPositionTaskConfig(
            auto_milling_alignment=False,
            use_autofocus=False,
            select_poi=False,
            sync_fluorescence_pose=sync,
        )
        SelectMillingPositionTask(
            microscope=microscope,
            config=config,
            lamella=lamella,
            task_manager=manager_for(microscope, lamella),
        )._run()

        got = lamella.fluorescence_pose.stage_position
        if sync:
            expected = _to_fluorescence(microscope, lamella.milling_pose.stage_position)
            assert got.x == pytest.approx(expected.x, abs=1e-9)
            assert got.y == pytest.approx(expected.y, abs=1e-9)
            assert abs(got.x - stale.x) > 10e-6, "the pose moved with the lamella"
        else:
            assert got.x == pytest.approx(stale.x, abs=1e-9)
            assert got.y == pytest.approx(stale.y, abs=1e-9)
        assert lamella.fluorescence_pose.objective_position is not None
        assert SelectMillingPositionTaskConfig().sync_fluorescence_pose is False
    finally:
        microscope.disconnect()


def test_the_alignment_reference_is_lamella_state_not_the_run_s(
    microscope, tmp_path, monkeypatch
):
    """Later tasks read it from the lamella's folder (FIB-1258), so it is written
    there even when the run saves everything else in its own folder (FIB-1275)."""
    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask

    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    run_folder = lamella.path / "Setup" / "20261014-093012"
    monkeypatch.setattr(
        AutoLamellaTask, "output_dir", property(lambda self: str(run_folder))
    )
    task = SelectMillingPositionTask(
        microscope=microscope,
        config=SelectMillingPositionTaskConfig(use_autofocus=False),
        lamella=lamella,
        task_manager=manager_for(microscope, lamella),
    )

    task._run()

    assert (lamella.path / ALIGNMENT_REFERENCE_IMAGE_FILENAME).is_file()
    assert not (run_folder / ALIGNMENT_REFERENCE_IMAGE_FILENAME).exists()
    finals = lamella.task_state.outputs.get("final_fib", [])
    assert finals and all(
        p.startswith(os.path.join("Setup", "20261014-093012")) for p in finals
    )
