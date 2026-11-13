"""tilt_coincident: tilt the stage and restore coincidence there (FIB-899).

The simulator's scene is eucentric by construction - the projection tilts
the world about the point the stage reports - so these tests give it a
`tilt_axis_offset`: the surface sits that far above the tilt axis, and a
tilt change then costs coincidence the way a real stage does.
"""

import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.alignment.coincidence import (
    REASON_CONVERGED,
    check_coincidence,
    tilt_coincident,
)
from tests.fixtures.sim_scenes import mammalian_scene

# Tens of seconds each on the simulator, so pull requests run these on one CI leg
# only; the nightly and release runs take them everywhere (python-package.yml).
pytestmark = pytest.mark.slow

# The geometry under test is a pre-tilted TFS shuttle; pin it rather than
# inherit whatever configuration an earlier test left as the default
TFS_SHUTTLE_CONFIG = os.path.join(
    cfg.SHIPPED_CONFIG_PATH, "microscope-configuration.yaml"
)

SEM_TILT = np.deg2rad(35.0)  # the boot pose: SEM orientation
MILLING_TILT = np.deg2rad(12.0)
NON_EUCENTRIC = (
    200e-6  # m, surface above the tilt axis: ~16 um of sag over the 23 deg tilt
)


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=TFS_SHUTTLE_CONFIG
    )
    microscope.system.sim["coincidence_projection"] = True
    microscope._connection.setup_sample_scene()
    yield microscope
    microscope.disconnect()


def _anchor_scene(microscope, tilt_axis_offset: float) -> None:
    """A noise-free scene, coincident at the current (SEM) pose."""
    scene = mammalian_scene(
        coincidence_offset=0.0,
        tilt_axis_offset=tilt_axis_offset,
        fiducial=True,
        noise_sigma=0.0,
        noise_fraction=0.0,
    )
    pose = microscope.get_stage_position()
    assert pose.t == pytest.approx(SEM_TILT)
    scene.anchor(pose)
    microscope._connection.sample_scene = scene


def _tilt_to(microscope, tilt: float) -> None:
    pose = microscope.get_stage_position()
    pose.t = tilt
    microscope.move_stage_absolute(pose)


def test_eucentric_stage_stays_coincident_through_a_tilt(microscope):
    _anchor_scene(microscope, tilt_axis_offset=0.0)
    _tilt_to(microscope, MILLING_TILT)

    m = check_coincidence(microscope)
    assert m.is_reliable, m.refusal_reason
    assert abs(m.dz) < 1e-6


def test_tilt_coincident_restores_coincidence_at_the_target(microscope):
    _anchor_scene(microscope, tilt_axis_offset=NON_EUCENTRIC)

    result = tilt_coincident(microscope, MILLING_TILT, tolerance=1e-6)

    assert result.converged, result.reason
    assert result.reason == REASON_CONVERGED
    assert result.tilts == [MILLING_TILT]  # straight there, no stepping
    assert result.moves_applied >= 1
    assert abs(result.alignments[-1].final.dz) <= 1e-6
    assert microscope.get_stage_position().t == pytest.approx(MILLING_TILT)


def test_tilt_coincident_on_a_eucentric_stage_moves_nothing(microscope):
    _anchor_scene(microscope, tilt_axis_offset=0.0)

    result = tilt_coincident(microscope, MILLING_TILT, tolerance=1e-6)

    assert result.converged
    assert result.moves_applied == 0
    assert len(result.alignments) == 1


def test_out_of_splits_reports_the_refusal_and_stays_at_the_target(microscope):
    _anchor_scene(microscope, tilt_axis_offset=NON_EUCENTRIC)

    result = tilt_coincident(
        microscope,
        MILLING_TILT,
        tolerance=1e-6,
        max_splits=0,
        capture_range=8e-6,
        coarse_hfw=None,
    )

    assert not result.converged
    assert result.reason != REASON_CONVERGED
    assert result.moves_applied == 0  # refused: nothing was moved
    assert microscope.get_stage_position().t == pytest.approx(MILLING_TILT)


def _anchor_in_sem_view(microscope) -> tuple:
    """Where the world anchor projects into the SEM view (m) - the sim's
    own projection, so exact and alias-free."""
    from fibsem.geometry.projection import BeamStageProjection
    from fibsem.structures.settings import BeamType

    projection = BeamStageProjection.from_microscope(microscope, BeamType.ELECTRON)
    scene = microscope._connection.sample_scene
    pose = microscope.get_stage_position()
    reference = scene._non_eucentric_reference(pose, projection)
    return projection.to_plane(reference, pose)


def test_walk_undo_can_be_declined(microscope):
    _anchor_scene(microscope, tilt_axis_offset=NON_EUCENTRIC)
    before = _anchor_in_sem_view(microscope)

    result = tilt_coincident(microscope, MILLING_TILT, tolerance=1e-6, undo_walk=False)

    assert result.converged and not result.walk_undone
    drift = np.hypot(*np.subtract(_anchor_in_sem_view(microscope), before))
    assert drift > 10e-6  # the walk is real and was left alone
