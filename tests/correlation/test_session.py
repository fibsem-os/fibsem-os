"""The correlation session without Qt: the run gate and what it says, a run
and the result it keeps, the transform it starts from, the refractive-index
correction, the points it tells of changes, and the run folder it opens and
keeps current.

Real images and a real fit throughout; nothing here imports Qt.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from fibsem.correlation import points as point_rules
from fibsem.correlation.fit import fit_transform, seed_for
from fibsem.correlation.geometry import nominal_transform
from fibsem.correlation.history import CORRELATION_FILENAME, CorrelationRun
from fibsem.correlation.refractive_index import ZetaParams
from fibsem.correlation.session import RUNNING, CorrelationSession, RunOutcome
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationResult,
    CorrelationState,
    PointStatus,
    PointType,
    PointXYZ,
)
from fibsem.fm.structures import (
    EmissionFilter,
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import FibsemImage

from .test_nominal_transform import ENTRIES, _input_data

ARCTIS = next(
    e
    for e in ENTRIES
    if e["system"] == "arctis" and e["fit"]["fm_slices_in_fit"] != 106
)


def _fm_image():
    meta = FluorescenceImageMetadata(
        acquisition_datetime=datetime(2026, 7, 24).isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        pixel_size_z=300e-9,
        resolution=(200, 200),
        channels=[
            FluorescenceChannelMetadata(
                name="GFP",
                excitation_wavelength=488.0,
                emission_filter=EmissionFilter.band("520 nm", 520.0),
                power=0.3,
                exposure_time=0.05,
                gain=1.5,
                offset=50.0,
            )
        ],
    )
    return FluorescenceImage(data=np.zeros((1, 20, 200, 200), np.uint16), metadata=meta)


def _pairs():
    """FM points and their FIB images under a rotation and scale, with click noise."""
    fm = np.array(
        [[40, 50, 4], [150, 40, 9], [160, 150, 14], [50, 160, 6], [100, 100, 11]],
        float,
    )
    a = np.deg2rad(20)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    fib = (rot @ fm[:, :2].T).T * 1.5 + [60.0, 30.0]
    fib += np.random.default_rng(0).normal(0, 0.5, fib.shape)
    return (
        [Coordinate(PointXYZ(x, y, 0.0), PointType.FIB) for x, y in fib],
        [Coordinate(PointXYZ(x, y, z), PointType.FM) for x, y, z in fm],
    )


def _target():
    return Coordinate(PointXYZ(90, 90, 8), PointType.POI)


@pytest.fixture
def session():
    session = CorrelationSession()
    session.fib_image = FibsemImage.generate_blank_image(resolution=(400, 300))
    session.fm_image = _fm_image()
    return session


@pytest.fixture
def ready(session):
    fib, fm = _pairs()
    session.set_data(
        CorrelationInputData(
            fib_coordinates=fib, fm_coordinates=fm, poi_coordinates=[_target()]
        )
    )
    return session


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_without_images_it_asks_for_them():
    session = CorrelationSession()

    assert not session.can_run()
    assert session.next_step() == "Load the FIB and FM images on the Setup tab."


def test_it_names_each_missing_step_in_order(session):
    fib, fm = _pairs()
    assert session.next_step().startswith("Seed the spot burns")

    session.set_data(CorrelationInputData(fib_coordinates=fib))
    assert session.next_step().startswith("Project FM from FIB")

    session.set_data(CorrelationInputData(fib_coordinates=fib, fm_coordinates=fm[:3]))
    assert session.next_step() == "3 of 4 pairs placed."

    session.set_data(CorrelationInputData(fib_coordinates=fib[:4], fm_coordinates=fm))
    assert session.next_step() == "FIB and FM fiducial counts differ (4 and 5)."

    session.set_data(CorrelationInputData(fib_coordinates=fib, fm_coordinates=fm))
    assert session.next_step() == "Place the target on the FM image."
    assert not session.can_run()

    session.points.add(_target())
    assert session.can_run()
    assert session.next_step() == "Ready to run with 5 pairs."


def test_predictions_are_not_pairs_until_placed(ready):
    fm = ready.points.of_type(PointType.FM)
    for coord in fm[1:]:
        coord.status = PointStatus.PREDICTED

    assert not ready.can_run()
    assert ready.next_step() == (
        "Drag the predicted rings onto their burns: 1 of 4 pairs placed."
    )


def test_a_rejected_pair_leaves_the_fit_and_its_row_is_kept(ready):
    ready.points.toggle_rejected(ready.points.of_type(PointType.FIB)[1])

    assert len(ready.fit_data.fib_coordinates) == 4
    assert len(ready.fit_data.fm_coordinates) == 4
    assert len(ready.data.fib_coordinates) == 5  # the record keeps it
    assert ready.fit_rows == [0, 2, 3, 4]
    assert ready.next_step() == "Ready to run with 4 pairs."


def test_accepted_predictions_are_counted_apart_from_placed_pairs(ready):
    for coord in ready.points.of_type(PointType.FM)[:2]:
        coord.status = PointStatus.ACCEPTED

    assert ready.next_step() == (
        "Ready to run with 5 pairs. 3 pairs placed by you, 2 accepted from the "
        "projection."
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def test_a_run_keeps_its_result_and_says_when_the_points_move(ready):
    result = ready.run()

    assert ready.result is result
    assert result.rms_error is not None and result.rms_error < 5.0
    assert result.matches_inputs(ready.fit_data)
    assert ready.next_step() == "Ready to run with 5 pairs."

    ready.points.move(ready.points.of_type(PointType.FM)[0], 45.0, 50.0)
    assert ready.next_step() == (
        "The points changed since the last run; run again with 5 pairs."
    )


def test_a_run_that_cannot_start_says_why(session):
    with pytest.raises(ValueError, match="Seed the spot burns"):
        session.run()
    assert session.result is None
    assert not session.running


def test_a_run_in_flight_holds_the_gate(ready):
    pending = ready.start_run()

    assert ready.running
    assert not ready.can_run()
    assert ready.next_step() == RUNNING
    with pytest.raises(RuntimeError):
        ready.start_run()

    assert ready.finish_run(pending.compute()) is RunOutcome.LIVE
    assert not ready.running and ready.result is not None


def test_a_run_finishing_after_the_result_is_discarded_is_dropped(ready):
    """FIB-1242: what it was started on was cleared while it ran."""
    pending = ready.start_run()
    ready.discard_result()

    assert ready.finish_run(pending.compute()) is RunOutcome.SUPERSEDED
    assert ready.result is None
    assert not ready.running


def test_a_failed_run_lets_the_next_one_start(ready):
    ready.start_run()
    ready.fail_run()

    assert not ready.running
    assert ready.can_run()


def test_an_abandoned_run_opens_the_gate_and_delivers_nothing(ready):
    """Whoever computed it has gone (a closed window): the session must not
    stay running forever."""
    pending = ready.start_run()
    ready.abandon_run(pending)

    assert not ready.running
    assert ready.can_run()
    assert ready.result is None


def test_abandoning_a_finished_run_leaves_the_next_one_alone(ready):
    first = ready.start_run()
    ready.finish_run(first.compute())
    second = ready.start_run()

    ready.abandon_run(first)

    assert ready.running
    assert ready.finish_run(second.compute()) is RunOutcome.LIVE


def test_the_run_reads_a_copy_of_the_points(ready):
    pending = ready.start_run()
    ready.points.of_type(PointType.FM)[0].point.x += 20.0

    outcome = ready.finish_run(pending.compute())

    assert outcome is RunOutcome.STALE  # the result is for the points as they were
    assert pending.data.fm_coordinates[0].point.x == 40.0


# ---------------------------------------------------------------------------
# The seed
# ---------------------------------------------------------------------------


@pytest.fixture
def geometric():
    """A session on images that record their geometry, with a saved run's picks."""
    data = _input_data(ARCTIS)
    session = CorrelationSession()
    session.fib_image, session.fm_image = data.fib_image, data.fm_image
    session.set_data(data)
    return session


def _previous_run(result):
    run = CorrelationRun(path="/x/r", name="r", state=CorrelationState(result=result))
    return [("02-other, run r", run)]


def _fitted_run(rotation, geometry):
    """A previous run that fitted ``rotation`` at the geometry's scale."""
    return _previous_run(
        CorrelationResult(
            scale=geometry.scale,
            rotation_matrix=rotation.tolist(),
            translation=[0.0, 0.0, 0.0],
            fm_z_scale=geometry.z_anisotropy,
            input_data=CorrelationInputData(
                stored_fib_image_pixel_size=ARCTIS["fib"]["pixel_size"]
            ),
        )
    )


def _offset_run(offset_um):
    """A previous run that measured the stage metadata's placement miss."""
    px = ARCTIS["fib"]["pixel_size"]
    return _previous_run(
        CorrelationResult(
            placement_offset=list(offset_um),
            rms_error=0.5e-6 / px,
            updated_at=time.time(),
            input_data=CorrelationInputData(stored_fib_image_pixel_size=px),
        )
    )


def test_a_run_starts_from_the_images_geometry(geometric):
    geometry = nominal_transform(geometric.fib_image, geometric.fm_image)

    seed = geometric.seed()
    assert np.allclose(seed.transform.projection, geometry.projection)
    assert seed.prior is None and seed.note == ""

    result = geometric.run()
    assert np.allclose(geometric.last_seed.transform.projection, geometry.projection)
    assert result.seed is not None
    assert result.branch_check["selected"] == "nominal"


def test_images_without_geometry_run_unseeded_and_say_why(ready):
    ready.run()

    assert ready.last_seed.transform is None
    assert ready.last_seed.note  # the status line's "unseeded fit: ..."


def test_a_previous_run_gives_the_rotation_unless_far_from_the_geometry(geometric):
    geometry = nominal_transform(geometric.fib_image, geometric.fm_image)

    geometric.prior_runs = _fitted_run(geometry.rotation, geometry)
    seed = geometric.seed()
    assert seed.prior.source == "02-other, run r" and seed.prior_rejected is None

    a = np.radians(60.0)
    about_x = np.array(
        [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    )
    geometric.prior_runs = _fitted_run(about_x @ geometry.rotation, geometry)
    seed = geometric.seed()
    assert seed.prior is None
    assert seed.prior_rejected[0] == "02-other, run r"
    assert seed.prior_rejected[1] == pytest.approx(60.0, abs=1.0)
    assert np.allclose(seed.transform.projection, geometry.projection)


def test_a_previous_runs_placement_offset_can_be_ignored(geometric):
    bare = nominal_transform(geometric.fib_image, geometric.fm_image).translation
    px_um = ARCTIS["fib"]["pixel_size"] * 1e6

    geometric.prior_runs = _offset_run([3.0, -4.0])
    assert geometric.placement_offset().source == "02-other, run r"
    shifted = geometric.seed().transform.translation
    assert np.allclose(shifted, bare + np.array([3.0, -4.0]) / px_um)

    geometric.ignore_placement_offset = True
    assert np.allclose(geometric.seed().transform.translation, bare)


@pytest.mark.parametrize("ignore", [False, True])
def test_a_run_with_no_seed_given_starts_where_the_tab_did(geometric, ignore):
    """The tab worked the seed out itself and passed it in; the session's own
    seed is that one, from the same previous runs and choices."""
    geometric.prior_runs = _offset_run([3.0, -4.0])
    geometric.ignore_placement_offset = ignore
    seed = seed_for(
        geometric.fib_image,
        geometric.fm_image,
        geometric.prior_runs,
        assumed_fm_transform=None,
        use_placement_offset=not ignore,
    )
    expected = fit_transform(geometric.fit_data, seed, rows=geometric.fit_rows)

    result = geometric.run()

    assert result.seed == expected.seed
    assert result.rms_error == pytest.approx(expected.rms_error)
    assert result.placement_offset == pytest.approx(expected.placement_offset)
    assert result.poi[0].image_px.x == pytest.approx(expected.poi[0].image_px.x)
    assert result.poi[0].image_px.y == pytest.approx(expected.poi[0].image_px.y)


def test_the_last_seed_goes_with_the_result_it_described(ready, tmp_path):
    """A loaded or discarded result is not the last run's: the status line
    must not describe it with that run's seed."""
    ready.run()
    assert ready.last_seed is not None
    ready.discard_result()
    assert ready.last_seed is None

    ready.run()
    path = tmp_path / "saved.json"
    ready.save(str(path))
    ready.load(str(path))
    assert ready.last_seed is None


def test_points_differing_from_the_last_seed_are_manual_edits(ready):
    assert ready.has_manual_edits()  # placed, never seeded

    ready.mark_seeded()
    assert not ready.has_manual_edits()

    ready.points.move(ready.points.of_type(PointType.FIB)[0], 1.0, 1.0)
    assert ready.has_manual_edits()


# ---------------------------------------------------------------------------
# The refractive-index correction
# ---------------------------------------------------------------------------

_OPTICS = ZetaParams(tilt_deg=0.0, depth_um=5.0, NA=0.85, n2=1.33, wavelength_um=0.52)


def test_an_armed_pre_correction_is_recorded_run_and_cleared(ready, tmp_path):
    ready.points.add(Coordinate(PointXYZ(80.0, 80.0, 2.0), PointType.SURFACE_FM))
    ready.set_project_dir(str(tmp_path))

    ready.arm_pre_correction(1.5, _OPTICS, computed=False)
    assert ready.auto_save_due()  # arming is an edit
    assert ready.data.ri_pre_correction_factor == 1.5
    assert ready.data.ri_pre_correction_params == {
        "tilt_deg": 0.0,
        "depth_um": 5.0,
        "na": 0.85,
        "n2": 1.33,
        "wavelength_um": 0.52,
        "computed": False,
    }

    result = ready.run()
    assert result.refractive_index_correction_mode == "pre"
    assert result.refractive_index_correction_factor == pytest.approx(1.5)

    ready.clear_pre_correction()
    assert ready.data.ri_pre_correction_factor is None
    assert ready.data.ri_pre_correction_params is None


def test_a_factor_alone_is_armed_with_the_experiments_optics(ready):
    """From a script the factor is enough: the record is the experiment's RI
    settings along the optical axis, and the factor is recorded as entered."""
    ready.points.add(Coordinate(PointXYZ(80.0, 80.0, 2.0), PointType.SURFACE_FM))
    ri = ready.config.ri

    ready.arm_pre_correction(1.3)

    assert ready.data.ri_pre_correction_factor == 1.3
    assert ready.data.ri_pre_correction_params == {
        "tilt_deg": 0.0,
        "depth_um": ri.depth_um,
        "na": ri.na,
        "n2": ri.n2,
        "wavelength_um": ri.wavelength_um,
        "computed": False,
    }
    assert ready.run().refractive_index_correction_factor == pytest.approx(1.3)


def test_a_refused_second_correction_keeps_the_result_stale(ready):
    """FIB-315: Apply, drag the FIB surface (the result is stale), Apply again.
    The second is refused, and must not record the dragged surface on the
    result first, which made the stale result read as current."""
    result = ready.run()
    surface = Coordinate(PointXYZ(10.0, 100.0, 0.0), PointType.SURFACE)
    ready.points.add(surface)
    assert ready.apply_post_correction(1.5) is True
    ready.points.move(surface, 10.0, 120.0)
    assert not result.matches_inputs(ready.fit_data)

    with pytest.raises(ValueError, match="already been applied"):
        ready.apply_post_correction(1.5)

    assert result.input_data.surface_coordinate.point.y == 100.0
    assert not result.matches_inputs(ready.fit_data)
    assert "points changed" in ready.next_step()


def test_a_post_correction_moves_poi_1_about_the_fib_surface(ready):
    with pytest.raises(ValueError):
        ready.apply_post_correction(1.5)  # no result yet
    result = ready.run()
    with pytest.raises(ValueError):
        ready.apply_post_correction(1.5)  # no FIB surface point
    surface = Coordinate(PointXYZ(10.0, 100.0, 0.0), PointType.SURFACE)
    ready.points.add(surface)
    y = result.poi[0].image_px.y

    assert ready.apply_post_correction(1.5) is True  # the surface makes it current
    assert ready.result is result
    assert result.poi[0].image_px.y == pytest.approx(100.0 + (y - 100.0) * 1.5)
    assert result.poi_uncorrected[0].image_px.y == pytest.approx(y)
    # the record holds a copy, so a later drag cannot rewrite it (FIB-315)
    assert result.input_data.surface_coordinate is not surface
    with pytest.raises(ValueError, match="already been applied"):
        ready.apply_post_correction(1.5)


# ---------------------------------------------------------------------------
# The points
# ---------------------------------------------------------------------------


def test_changes_to_the_points_are_told_until_disposed(session):
    told = []
    dispose = session.points.subscribe(lambda change, coords: told.append(change))
    coord = Coordinate(PointXYZ(1.0, 2.0, 0.0), PointType.FIB)

    session.points.add(coord)
    session.points.move(coord, 3.0, 4.0)
    assert told == [point_rules.STRUCTURE, point_rules.SELECTION, point_rules.VALUES]
    assert (coord.point.x, coord.point.y) == (3.0, 4.0)

    dispose()
    session.points.remove(coord)
    assert len(told) == 3
    assert session.data.fib_coordinates == []


def test_one_surface_point_and_no_factor_without_the_fm_surface(session):
    fib_surface = Coordinate(PointXYZ(1.0, 1.0, 0.0), PointType.SURFACE)
    fm_surface = Coordinate(PointXYZ(2.0, 2.0, 3.0), PointType.SURFACE_FM)

    session.set_data(
        CorrelationInputData(
            surface_coordinate=fib_surface,
            fm_surface_coordinate=fm_surface,
            ri_pre_correction_factor=1.4,
        )
    )
    assert session.data.surface_coordinate is None
    assert session.data.fm_surface_coordinate is fm_surface
    assert session.ri_pre_correction_factor == 1.4

    session.set_data(
        CorrelationInputData(
            surface_coordinate=fib_surface, ri_pre_correction_factor=1.4
        )
    )
    assert session.ri_pre_correction_factor is None
    assert session.ri_pre_correction_params is None


def test_removing_the_fm_surface_through_the_points_disarms_the_factor(ready):
    """The factor goes with the FM surface however the surface goes, not only
    through the tab: a script removing it must not run with the factor."""
    fm_surface = Coordinate(PointXYZ(100.0, 100.0, 5.0), PointType.SURFACE_FM)
    ready.points.add(fm_surface)
    ready.ri_pre_correction_factor = 1.3
    ready.ri_pre_correction_params = {"tilt_deg": 0.0, "computed": False}

    ready.points.remove(fm_surface)
    result = ready.run()

    assert ready.data.ri_pre_correction_factor is None
    assert ready.data.ri_pre_correction_params is None
    assert result.input_data.ri_pre_correction_factor is None


def test_moving_the_fm_surface_keeps_the_factor(ready):
    fm_surface = Coordinate(PointXYZ(100.0, 100.0, 5.0), PointType.SURFACE_FM)
    ready.points.add(fm_surface)
    ready.ri_pre_correction_factor = 1.3

    ready.points.move(fm_surface, 110.0, 100.0)
    ready.points.add(Coordinate(PointXYZ(1.0, 1.0, 0.0), PointType.FIB))

    assert ready.ri_pre_correction_factor == 1.3


def test_a_removal_policy_can_be_uninstalled_while_it_is_the_one_installed(session):
    fib = Coordinate(PointXYZ(1.0, 1.0, 0.0), PointType.FIB)
    session.points.add(fib)
    asked = []

    def ask(coords):
        asked.append(coords)
        return None  # cancelled

    uninstall = session.points.set_removal_policy(ask)
    assert not session.points.remove(fib)
    uninstall()
    assert session.points.remove(fib)
    assert len(asked) == 1

    keep = session.points.set_removal_policy(ask)
    stale = session.points.set_removal_policy(lambda coords: coords)
    keep()  # no longer installed: the newer policy stays
    session.points.add(fib)
    assert session.points.remove(fib)
    assert len(asked) == 1
    stale()


# ---------------------------------------------------------------------------
# Saving and opening a run
# ---------------------------------------------------------------------------


def _points(session):
    return [
        (c.point_type, c.point.x, c.point.y, c.point.z)
        for c in session.points.coordinates
    ]


def _with_images():
    session = CorrelationSession()
    session.fib_image = FibsemImage.generate_blank_image(resolution=(400, 300))
    session.fm_image = _fm_image()
    return session


def test_a_saved_run_opens_with_the_same_points_and_result(ready, tmp_path):
    run = tmp_path / "2026-07-27_09-00-00"
    ready.set_project_dir(str(run))
    result = ready.run()

    assert ready.auto_save() is True
    assert (run / CORRELATION_FILENAME).is_file()

    opened = _with_images()
    state = opened.load(str(run / CORRELATION_FILENAME))

    assert _points(opened) == _points(ready)
    assert state.result is opened.result
    assert opened.result.rms_error == pytest.approx(result.rms_error)
    assert opened.result.matches_inputs(opened.fit_data)
    # an open is not an edit: it writes nothing until something arms it
    opened.set_project_dir(str(run))
    assert opened.auto_save() is None


def test_opening_either_legacy_file_takes_its_points(ready, tmp_path):
    result = ready.run()
    data_file = tmp_path / "correlation_data.json"
    result_file = tmp_path / "correlation_result.json"
    ready.data.save(str(data_file))
    result.save(str(result_file))

    from_data = _with_images()
    from_data.load(str(data_file))
    assert _points(from_data) == _points(ready)
    assert from_data.result is None

    from_result = _with_images()
    from_result.load(str(result_file))
    assert _points(from_result) == _points(ready)
    # the points are a copy of the result's snapshot, so an edit makes it stale
    assert from_result.result.matches_inputs(from_result.fit_data)
    from_result.points.move(from_result.points.of_type(PointType.FIB)[0], 1.0, 1.0)
    assert not from_result.result.matches_inputs(from_result.fit_data)


def test_the_legacy_loaders_read_before_they_change_anything(ready, tmp_path):
    result = ready.run()
    data_file = tmp_path / "correlation_data.json"
    result_file = tmp_path / "correlation_result.json"
    ready.data.save(str(data_file))
    result.save(str(result_file))

    from_data = _with_images()
    state = from_data.load_data(str(data_file))
    assert _points(from_data) == _points(ready)
    assert state.result is None and from_data.result is None

    from_result = _with_images()
    loaded, live = from_result.load_result(str(result_file))
    assert live and from_result.result is loaded
    assert _points(from_result) == _points(ready)

    kept = _points(from_data)
    with pytest.raises(ValueError):
        from_data.load_result(str(data_file))  # a coordinates file
    assert _points(from_data) == kept


def test_a_file_that_cannot_be_read_leaves_the_armed_save(ready, tmp_path):
    ready.set_project_dir(str(tmp_path))
    ready.points.move(ready.points.of_type(PointType.FIB)[0], 1.0, 1.0)
    assert ready.auto_save_due()

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{ not json")
    for load in (ready.load, ready.load_data, ready.load_result):
        with pytest.raises(ValueError):
            load(str(corrupt))
        assert ready.auto_save_due()


def test_an_explicit_save_makes_its_folder(ready, tmp_path):
    ready.run()
    path = tmp_path / "not" / "yet" / "here.json"

    assert ready.save(str(path)) == str(path)

    opened = _with_images()
    opened.load(str(path))
    assert _points(opened) == _points(ready)
    assert opened.result is not None


def test_a_new_folder_can_keep_the_save_that_is_due(ready, tmp_path):
    ready.run()
    ready.set_project_dir(str(tmp_path / "a"))
    ready.arm_auto_save()

    ready.set_project_dir(str(tmp_path / "b"), keep_armed=True)
    assert ready.auto_save_due()
    ready.set_project_dir(str(tmp_path / "c"))
    assert not ready.auto_save_due()


def test_a_state_without_a_result_drops_the_previous_one(ready):
    ready.run()
    ready.adopt_state(CorrelationState(input_data=CorrelationInputData()))
    assert ready.result is None
    assert ready.data.fib_coordinates == []


def test_only_a_load_an_edit_or_a_result_arms_the_auto_save(session, tmp_path):
    session.set_project_dir(str(tmp_path))
    assert not session.auto_save_due()  # an image load alone never writes

    coord = Coordinate(PointXYZ(1.0, 2.0, 0.0), PointType.FIB)
    session.points.add(coord)
    assert session.auto_save_due()

    session.set_project_dir(str(tmp_path))  # a new folder disarms
    session.points.select(coord)
    assert not session.auto_save_due()  # a selection is not an edit
    session.points.move(coord, 3.0, 4.0)
    assert session.auto_save_due()

    session.arm_auto_save()
    with session.reading():
        with session.reading():
            pass
        assert not session.auto_save_due()  # still reading
    assert not session.auto_save_due()  # and the read disarmed it
    session.arm_auto_save()
    assert session.auto_save() is True


def test_a_run_folder_is_made_only_for_a_result(ready, tmp_path):
    run = tmp_path / "2026-07-27_09-00-00"
    ready.set_project_dir(str(run))
    ready.arm_auto_save()
    assert ready.auto_save() is None
    assert not run.exists()

    ready.run()  # a delivered result arms it, and makes the folder worth having
    assert ready.auto_save() is True
    assert run.is_dir()


def test_a_failed_save_is_kept_until_one_succeeds(ready, tmp_path):
    blocked = tmp_path / "a_file"
    blocked.write_text("")
    ready.set_project_dir(str(blocked))
    ready.run()

    assert ready.auto_save() is False
    assert ready.save_error

    ready.set_project_dir(str(tmp_path))
    ready.arm_auto_save()
    assert ready.auto_save() is True
    assert ready.save_error is None


# ---------------------------------------------------------------------------
# The app's session
# ---------------------------------------------------------------------------


def test_the_app_session_makes_one_correlation_when_asked():
    from fibsem.workflows.session import AppSession

    app = AppSession()

    assert isinstance(app.correlation, CorrelationSession)
    assert app.correlation is app.correlation


def test_threads_asking_at_once_get_the_same_correlation(monkeypatch):
    import threading
    import time

    from fibsem.workflows.session import AppSession

    made = []
    real_init = CorrelationSession.__init__

    def slow_init(self):
        made.append(self)
        time.sleep(0.05)  # a window wide enough for a second thread to enter
        real_init(self)

    monkeypatch.setattr(CorrelationSession, "__init__", slow_init)
    app = AppSession()
    got = []
    threads = [
        threading.Thread(target=lambda: got.append(app.correlation)) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(made) == 1
    assert all(c is made[0] for c in got)


def test_a_new_experiment_starts_a_new_correlation(tmp_path):
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.workflows.session import AppSession

    app = AppSession()
    first = app.correlation
    first.points.add(_target())

    app.adopt(Experiment(path=tmp_path, name="next"))

    assert app.correlation is not first
    assert len(app.correlation.points) == 0


def test_the_app_session_does_not_import_the_correlation_until_asked():
    code = (
        "import sys, fibsem.workflows.session; "
        "assert 'fibsem.correlation.session' not in sys.modules"
    )
    root = str(Path(__file__).resolve().parents[2])  # this checkout's fibsem
    env = {**os.environ, "PYTHONPATH": root}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_a_reset_starts_a_new_correlation(ready, tmp_path):
    from fibsem.correlation.config import CorrelationConfig

    ready.run()
    ready.set_project_dir(str(tmp_path / "run"))
    ready.config.load_spot_burns = False
    ready.ignore_placement_offset = True
    ready.mark_seeded()

    ready.reset()

    assert ready.points.coordinates == []
    assert ready.result is None and ready.last_seed is None
    assert ready.fib_image is None and ready.fm_image is None
    assert ready.ri_pre_correction_factor is None
    assert ready.seeded_positions is None and not ready.ignore_placement_offset
    assert ready.prior_runs == [] and ready.project_dir is None
    assert ready.config == CorrelationConfig()
    # Emptying the points was not an edit: given a run folder again, nothing
    # is due until the user changes something.
    folder = tmp_path / "next"
    folder.mkdir()
    ready.set_project_dir(str(folder))
    assert not ready.auto_save_due()
    ready.points.add(_target())
    assert ready.auto_save_due()


def test_a_run_in_flight_at_a_reset_delivers_nothing(ready):
    pending = ready.start_run()

    ready.reset()

    assert ready.finish_run(pending.compute()) is RunOutcome.SUPERSEDED
    assert ready.result is None and not ready.running


def test_what_listens_to_the_points_hears_a_reset(ready):
    told = []
    ready.points.subscribe(lambda change, _coords: told.append(change))

    ready.reset()

    assert point_rules.STRUCTURE in told
