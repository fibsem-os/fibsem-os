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
    # an absurdly small capture range forces a window-edge/agreement refusal;
    # coarse escalation disabled so the refusal is terminal
    result = ensure_coincident(
        microscope, tolerance=1e-6, capture_range=0.3e-6, coarse_hfw=None
    )

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


@pytest.mark.parametrize("offset", [40e-6, 80e-6])
def test_coarse_pass_rescues_errors_beyond_the_fine_window(microscope, offset):
    """Errors past the fine capture range refuse at fine FOV, escalate to
    the coarse pass, and still converge - up to ~100 um of height error."""
    microscope._coincidence_scene.coincidence_offset = offset
    microscope._coincidence_scene.reference_position = None
    microscope._coincidence_scene.anchor(microscope.get_stage_position())

    result = ensure_coincident(microscope, tolerance=1e-6)

    assert result.converged, result.reason
    assert result.coarse_used
    assert abs(result.final.dz) <= 1e-6
    assert not result.measurements[0].is_reliable  # fine pass refused first


@pytest.mark.xfail(
    reason="on a purely periodic mesh, correlation cannot distinguish true "
    "coincidence from a grid-pitch alias: with the error beyond ~one pitch "
    "both bands agree on the alias and the loop converges onto it (FIB-711 "
    "taken to its logical end - real physics, same on hardware over bare "
    "mesh). Defences are contextual: bound the plausible error from 'how "
    "far since last aligned', or anchor on aperiodic features. FIB-868.",
    strict=False,
)
def test_error_beyond_the_coarse_window_does_not_claim_success(microscope):
    """The desired property - never claim success on an aliased lock -
    cannot be guaranteed by correlation alone on a periodic scene."""
    microscope._coincidence_scene.coincidence_offset = 250e-6
    microscope._coincidence_scene.reference_position = None
    microscope._coincidence_scene.anchor(microscope.get_stage_position())

    result = ensure_coincident(microscope, tolerance=1e-6)

    assert not result.converged
    assert result.coarse_used  # it tried


def test_progress_reports_every_step_and_moves_are_counted(microscope, tmp_path):
    from fibsem.alignment.coincidence import (
        PROGRESS_MEASURED,
        PROGRESS_MEASURING,
        PROGRESS_MOVING,
    )
    from fibsem.alignment.plotting import save_coincidence_diagnostics

    steps = []
    result = ensure_coincident(microscope, tolerance=1e-6, on_progress=steps.append)

    assert result.converged
    assert result.moves_applied == 1
    # measure, report, move, measure, report - and every step describes itself
    assert [s.stage for s in steps] == [
        PROGRESS_MEASURING,
        PROGRESS_MEASURED,
        PROGRESS_MOVING,
        PROGRESS_MEASURING,
        PROGRESS_MEASURED,
    ]
    assert steps[2].measurement is result.measurements[0]
    assert steps[2].iteration == 0 and steps[3].iteration == 1
    assert all(s.describe() for s in steps)

    # each measurement keeps its pair, so the run is fully re-plottable
    saved = save_coincidence_diagnostics(result, str(tmp_path))
    assert len(saved) == len(result.measurements) == 2
    assert saved[0].endswith("_01_fine.png")


def test_coarse_measurements_do_not_count_as_moves(microscope):
    microscope._coincidence_scene.coincidence_offset = 40e-6
    microscope._coincidence_scene.reference_position = None
    microscope._coincidence_scene.anchor(microscope.get_stage_position())

    result = ensure_coincident(microscope, tolerance=1e-6)

    assert result.converged and result.coarse_used
    assert result.measurements[1].coarse
    assert not result.measurements[0].coarse
    # refused fine + coarse + post-move fine (+ maybe a fine refinement):
    # the history is longer than the moves by the extra measurements
    assert result.moves_applied == len(result.measurements) - 2
