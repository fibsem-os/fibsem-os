"""ensure_coincident: the measure -> correct -> re-measure loop (FIB-868).

Runs against the simulator's projection-true scene: every layer below the
loop (measurement geometry, vertical_move decomposition, scene rendering)
shares BeamStageProjection, so convergence here is a real end-to-end fact.
"""

import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.alignment.coincidence import (
    REASON_CONVERGED,
    REASON_MAX_ITERATIONS,
    ensure_coincident,
)
from fibsem.structures import BeamType, FibsemStagePosition

# The geometry under test is a pre-tilted TFS shuttle; pin it rather than
# inherit whatever configuration an earlier test left as the default
TFS_SHUTTLE_CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, settings = utils.setup_session(
        manufacturer="Demo", config_path=TFS_SHUTTLE_CONFIG
    )
    microscope.system.sim["coincidence_projection"] = True
    microscope.system.sim["coincidence_offset"] = 8e-6
    microscope._setup_sample_scene()
    from fibsem.microscopes.sim_scene import SampleScene

    # deterministic scene: the loop is about control flow and geometry
    scene = SampleScene(coincidence_offset=8e-6, noise_sigma=0.0, noise_fraction=0.0)
    scene.anchor(microscope.get_stage_position())
    microscope._sample_scene = scene
    pose = microscope.get_stage_position()
    pose.t = np.deg2rad(12.0)
    microscope.move_stage_absolute(pose)
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


@pytest.mark.parametrize("offset", [40e-6, 50e-6])
def test_coarse_pass_rescues_errors_beyond_the_fine_window(microscope, offset):
    """Errors past the fine capture range refuse at fine FOV, escalate to
    the coarse pass, and still converge. The honest reach on this mesh is
    ~50 um - about half a grid pitch, where rival peaks start to compete;
    an 80 um "rescue" this test once claimed was a false-lock chain."""
    microscope._sample_scene.coincidence_offset = offset
    microscope._sample_scene.reference_position = None
    microscope._sample_scene.anchor(microscope.get_stage_position())

    result = ensure_coincident(microscope, tolerance=1e-6)

    assert result.converged, result.reason
    assert result.coarse_used
    assert abs(result.final.dz) <= 1e-6
    assert not result.measurements[0].is_reliable  # fine pass refused first


@pytest.mark.parametrize("offset", [80e-6, 250e-6])
def test_error_beyond_the_coarse_window_does_not_claim_success(microscope, offset):
    """Beyond ~one grid pitch the periodic mesh offers an alias both bands
    agree on, and the loop used to converge onto it (this test was an xfail
    recording that). The alias is caught by its lateral offset instead: a
    height error cannot move x, and on the rotated mesh the rival peak sits
    well off-axis - so the lock is refused rather than acted on."""
    microscope._sample_scene.coincidence_offset = offset
    microscope._sample_scene.reference_position = None
    microscope._sample_scene.anchor(microscope.get_stage_position())

    result = ensure_coincident(microscope, tolerance=1e-6)

    assert not result.converged
    assert result.coarse_used  # it tried
    assert result.moves_applied == 0  # and did not act on the alias


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

    # each measurement keeps its pair, so the run is saved as a replayable
    # case: every pair as tif with metadata, the numbers, one summary figure
    import os

    from fibsem.alignment.plotting import load_coincidence_run

    run_dir = save_coincidence_diagnostics(result, str(tmp_path), prefix="start_")
    assert os.path.basename(run_dir).startswith("start_coincidence_")
    saved = sorted(os.listdir(run_dir))
    assert saved == [
        "01_fine_fib.tif",
        "01_fine_sem.tif",
        "02_fine_fib.tif",
        "02_fine_sem.tif",
        "run.json",
        "summary.png",
    ]
    replayed = load_coincidence_run(run_dir)
    assert len(replayed) == 2
    for record, _, _, fresh in replayed:
        assert fresh.is_reliable == record["is_reliable"]
        assert fresh.dz == pytest.approx(record["dz"], abs=0.5e-6)


def test_a_coarse_run_replays_with_its_own_parameters(microscope, tmp_path):
    """A coarse pair re-measured with the fine window reads differently;
    the run records what each measurement was run with, and the replay
    uses it, so the replay reproduces the run."""
    from fibsem.alignment.plotting import (
        load_coincidence_run,
        save_coincidence_diagnostics,
    )

    microscope._sample_scene.coincidence_offset = 40e-6
    microscope._sample_scene.reference_position = None
    microscope._sample_scene.anchor(microscope.get_stage_position())
    result = ensure_coincident(microscope, tolerance=1e-6)
    assert result.coarse_used

    replayed = load_coincidence_run(save_coincidence_diagnostics(result, str(tmp_path)))
    assert [r["pass"] for r, *_ in replayed] == ["fine", "coarse", "fine"]
    for record, _, _, fresh in replayed:
        assert fresh.is_reliable == record["is_reliable"]
        assert fresh.refusal_reason == record["refusal_reason"]
        assert fresh.dz == pytest.approx(record["dz"], abs=0.5e-6)


def test_coarse_measurements_do_not_count_as_moves(microscope):
    microscope._sample_scene.coincidence_offset = 40e-6
    microscope._sample_scene.reference_position = None
    microscope._sample_scene.anchor(microscope.get_stage_position())

    result = ensure_coincident(microscope, tolerance=1e-6)

    assert result.converged and result.coarse_used
    assert result.measurements[1].coarse
    assert not result.measurements[0].coarse
    # refused fine + coarse + post-move fine (+ maybe a fine refinement):
    # the history is longer than the moves by the extra measurements
    assert result.moves_applied == len(result.measurements) - 2


def _anchor_in_view(microscope, beam_type) -> tuple:
    """Where the scene's world anchor projects into one beam's view (m).

    Read straight off the projection the scene renders with, so it is
    exact and immune to the periodic mesh's correlation aliases.
    """
    from fibsem.projection import BeamStageProjection

    projection = BeamStageProjection.from_microscope(microscope, beam_type)
    return projection.to_plane(
        microscope._sample_scene.reference_position,
        microscope.get_stage_position(),
    )


@pytest.mark.parametrize("reference", [BeamType.ELECTRON, BeamType.ION])
def test_reference_view_keeps_its_centre(microscope, reference):
    """Whichever view is the reference sees the same scene after the
    alignment; the other view is the one that moved onto it."""
    before = {b: _anchor_in_view(microscope, b) for b in BeamType}

    result = ensure_coincident(microscope, tolerance=1e-6, reference=reference)
    assert result.converged, result.reason
    assert result.moves_applied == 1

    after = {b: _anchor_in_view(microscope, b) for b in BeamType}
    other = BeamType.ION if reference is BeamType.ELECTRON else BeamType.ELECTRON
    kept = np.hypot(*(np.subtract(after[reference], before[reference])))
    moved = np.hypot(*(np.subtract(after[other], before[other])))
    assert kept < 1e-6, f"{reference.name} view moved by {kept * 1e6:.2f} um"
    assert moved > 3e-6, f"{other.name} view only moved {moved * 1e6:.2f} um"


@pytest.mark.parametrize("reference", [BeamType.ELECTRON, BeamType.ION])
def test_converges_at_the_fib_orientation(microscope, reference):
    """At the FIB orientation (half a turn round, the FIB looking at the
    surface face-on) the projections carry the flip; the loop converges
    exactly. Validated here only - the guard is a logged warning, not a
    refusal, until hardware confirms it."""
    fib = microscope.get_orientation("FIB")
    pose = microscope.get_stage_position()
    pose.r, pose.t = fib.r, fib.t
    microscope.move_stage_absolute(pose)
    microscope._sample_scene.reference_position = None
    microscope._sample_scene.anchor(microscope.get_stage_position())

    result = ensure_coincident(microscope, tolerance=1e-6, reference=reference)

    assert result.converged, result.reason
    assert result.moves_applied == 1
    assert abs(result.measurements[0].dz) == pytest.approx(8e-6, abs=1e-6)
    assert result.measurements[0].y_stretch < 1  # the FIB is the face-on view here
