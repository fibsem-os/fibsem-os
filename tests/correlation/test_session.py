"""The correlation session without Qt: the run gate and what it says, a run
and the result it keeps, and the points it tells of changes.

Real images and a real fit throughout; nothing here imports Qt.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from fibsem.correlation import points as point_rules
from fibsem.correlation.session import RUNNING, CorrelationSession
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
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


def _fm_image():
    meta = FluorescenceImageMetadata(
        acquisition_date=datetime(2026, 7, 24).isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        pixel_size_z=300e-9,
        resolution=(200, 200),
        channels=[
            FluorescenceChannelMetadata(
                name="GFP",
                excitation_wavelength=488.0,
                emission_filter=EmissionFilter("520 nm", low=520.0),
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
    assert ready.fit_rows() == [0, 2, 3, 4]
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

    assert ready.finish_run(pending.compute()) is True
    assert not ready.running and ready.result is not None


def test_a_run_finishing_after_the_result_is_discarded_is_dropped(ready):
    """FIB-1242: what it was started on was cleared while it ran."""
    pending = ready.start_run()
    ready.discard_result()

    assert ready.finish_run(pending.compute()) is None
    assert ready.result is None
    assert not ready.running


def test_a_failed_run_lets_the_next_one_start(ready):
    ready.start_run()
    ready.fail_run()

    assert not ready.running
    assert ready.can_run()


def test_the_run_reads_a_copy_of_the_points(ready):
    pending = ready.start_run()
    ready.points.of_type(PointType.FM)[0].point.x += 20.0

    live = ready.finish_run(pending.compute())

    assert live is False  # the result is for the points as they were
    assert pending.data.fm_coordinates[0].point.x == 40.0


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


# ---------------------------------------------------------------------------
# The app's session
# ---------------------------------------------------------------------------


def test_the_app_session_makes_one_correlation_when_asked():
    from fibsem.workflows.session import AppSession

    app = AppSession()

    assert isinstance(app.correlation, CorrelationSession)
    assert app.correlation is app.correlation


def test_the_app_session_does_not_import_the_correlation_until_asked():
    code = (
        "import sys, fibsem.workflows.session; "
        "assert 'fibsem.correlation.session' not in sys.modules"
    )
    root = str(Path(__file__).resolve().parents[2])  # this checkout's fibsem
    env = {**os.environ, "PYTHONPATH": root}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
