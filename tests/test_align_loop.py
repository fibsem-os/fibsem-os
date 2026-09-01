"""ensure_coincident: the measure -> correct -> re-measure loop (FIB-868).

Runs against the simulator's projection-true scene: every layer below the
loop (measurement geometry, vertical_move decomposition, scene rendering)
shares BeamStageProjection, so convergence here is a real end-to-end fact.
"""

import numpy as np
import pytest

from fibsem import utils
from fibsem.alignment.coincidence import (
    REASON_CONVERGED,
    REASON_MAX_ITERATIONS,
    ensure_coincident,
)
from fibsem.structures import FibsemStagePosition


@pytest.fixture
def microscope():
    microscope, settings = utils.setup_session(manufacturer="Demo")
    microscope.system.sim["coincidence_projection"] = True
    microscope.system.sim["coincidence_offset"] = 8e-6
    microscope._setup_coincidence_projection()
    from fibsem.microscopes.sim_scene import CoincidenceScene

    # deterministic scene: the loop is about control flow and geometry
    scene = CoincidenceScene(
        coincidence_offset=8e-6, noise_sigma=0.0, noise_fraction=0.0
    )
    scene.anchor(microscope.get_stage_position())
    microscope._coincidence_scene = scene
    microscope.move_stage_relative(
        FibsemStagePosition(x=0, y=0, z=0, r=0, t=np.deg2rad(12.0))
    )
    yield microscope
    microscope.disconnect()


def test_converges_from_the_boot_offset(microscope):
    result = ensure_coincident(microscope, tolerance=1e-6)

    assert result.converged, result.reason
    assert result.reason == REASON_CONVERGED
    assert abs(result.final.dz) <= 1e-6
    # with exact geometry underneath, one corrective move suffices
    assert result.moves_applied == 1
    assert result.measurements[0].is_reliable
    assert abs(result.measurements[0].dz) == pytest.approx(8e-6, abs=1e-6)


def test_already_coincident_moves_nothing(microscope):
    first = ensure_coincident(microscope, tolerance=1e-6)
    assert first.converged

    again = ensure_coincident(microscope, tolerance=1e-6)
    assert again.converged
    assert again.moves_applied == 0
    assert len(again.measurements) == 1


def test_refusal_stops_the_loop(microscope):
    # an absurdly small capture range forces a window-edge/agreement refusal
    result = ensure_coincident(microscope, tolerance=1e-6, capture_range=0.3e-6)

    assert not result.converged
    assert result.reason not in (REASON_CONVERGED, REASON_MAX_ITERATIONS)
    assert result.moves_applied == 0  # refuse means do not move


def test_under_relaxed_loop_hits_the_iteration_limit(microscope):
    # heavy damping cannot close 8 um in one move; with one allowed
    # iteration the loop must report max-iterations, not success
    result = ensure_coincident(
        microscope, tolerance=0.5e-6, max_iterations=1, relaxation=0.3
    )

    assert not result.converged
    assert result.reason == REASON_MAX_ITERATIONS
    assert result.moves_applied == 1
    # but it did make progress, honestly recorded
    assert abs(result.final.dz) < abs(result.measurements[0].dz)
