"""SelectMillingPositionTask with auto_milling_alignment: coincidence in the
workflow (FIB-899, initial scope).

Two supported starts: at the SEM orientation (align, tilt coincidently,
undo the walk) and at the milling angle (align only). Runs on the Demo's
projection-true scene with a non-eucentric stage, so the geometry is real.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.alignment import ALIGNMENT_SUBDIR
from fibsem.alignment.coincidence import check_coincidence
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
)
from fibsem.microscopes.sim_scene import SampleScene
from fibsem.projection import BeamStageProjection
from fibsem.structures import BeamType

TFS_SHUTTLE_CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")
MILLING_ANGLE = 15.0  # deg -> stage tilt 12 deg on the 35 deg shuttle
NON_EUCENTRIC = 200e-6  # m


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=TFS_SHUTTLE_CONFIG
    )
    microscope.system.sim["coincidence_projection"] = True
    microscope._setup_sample_scene()
    yield microscope
    microscope.disconnect()


def _anchor_scene(microscope, coincidence_offset: float) -> None:
    scene = SampleScene(
        coincidence_offset=coincidence_offset,
        tilt_axis_offset=NON_EUCENTRIC,
        fiducial=True,
        noise_sigma=0.0,
        noise_fraction=0.0,
    )
    scene.anchor(microscope.get_stage_position())
    microscope._sample_scene = scene


def _anchor_in_fib_view(microscope) -> tuple:
    """Where the world anchor projects into the FIB view (m). The FIB view is
    the invariant: with reference=ION nothing the task does may move what
    the operator centred there - not the alignment, not the tilt."""
    projection = BeamStageProjection.from_microscope(microscope, BeamType.ION)
    scene = microscope._sample_scene
    pose = microscope.get_stage_position()
    return projection.to_plane(scene._non_eucentric_reference(pose, projection), pose)


def _make_task(microscope, tmp_path: Path) -> SelectMillingPositionTask:
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    config = SelectMillingPositionTaskConfig(
        milling_angle=MILLING_ANGLE,
        auto_milling_alignment=True,
        use_autofocus=False,
        select_poi=False,
    )
    return SelectMillingPositionTask(
        microscope=microscope, config=config, lamella=lamella
    )


def test_from_the_sem_orientation_the_task_ends_coincident_at_the_milling_angle(
    microscope, tmp_path
):
    _anchor_scene(microscope, coincidence_offset=8e-6)
    task = _make_task(microscope, tmp_path)
    before = _anchor_in_fib_view(microscope)

    task._run()

    from fibsem.transformations import get_stage_tilt_from_milling_angle

    target = get_stage_tilt_from_milling_angle(microscope, np.radians(MILLING_ANGLE))
    assert microscope.get_stage_position().t == pytest.approx(target, abs=1e-6)
    m = check_coincidence(microscope)
    assert m.is_reliable and abs(m.dz) < 1e-6, (m.refusal_reason, m.dz)
    # the site the operator chose is still centred: the walk was undone
    drift = np.hypot(*np.subtract(_anchor_in_fib_view(microscope), before))
    assert drift < 2e-6, f"site drifted {drift * 1e6:.1f} um"
    # diagnostics for the pre-tilt alignment and the tilt landed with the lamella
    saved = os.listdir(task.lamella.path / ALIGNMENT_SUBDIR)
    assert any(f.startswith("start_") for f in saved)
    assert any(f.startswith("tilt01_") for f in saved)


def test_at_the_milling_angle_the_task_only_aligns(microscope, tmp_path):
    from fibsem.transformations import get_stage_tilt_from_milling_angle

    pose = microscope.get_stage_position()
    pose.t = get_stage_tilt_from_milling_angle(microscope, np.radians(MILLING_ANGLE))
    microscope.move_stage_absolute(pose)
    _anchor_scene(microscope, coincidence_offset=8e-6)
    task = _make_task(microscope, tmp_path)

    task._run()

    assert microscope.get_stage_position().t == pytest.approx(pose.t, abs=1e-6)
    m = check_coincidence(microscope)
    assert m.is_reliable and abs(m.dz) < 1e-6, (m.refusal_reason, m.dz)
    saved = os.listdir(task.lamella.path / ALIGNMENT_SUBDIR)
    assert any(f.startswith("start_") for f in saved)
    assert not any(f.startswith("tilt") for f in saved)
