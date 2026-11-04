"""The spot burn service on the Demo, and `run_spot_burn` over it.

The Demo burns with the shared point-by-point burn through `microscope.spot_burn`:
blank, park the beam on the point, unblank (which burns the spot into the sample
scene), wait, and back to full frame at the burn current's start value. A
microscope with no service refuses to burn.
"""

import threading

import pytest

from fibsem import config as cfg
from fibsem import utils
from fibsem.devices.core import Device
from fibsem.drivers.demo.services import DemoSpotBurn
from fibsem.imaging.spot import SpotBurnSettings, SpotBurnStatus, run_spot_burn
from fibsem.services import Service
from fibsem.structures.settings import BeamType, Point, ScanMode
from tests.fixtures.microscopes import demo_microscope

IMAGING_CURRENT = 2e-11
BURN_CURRENT = 9e-11


microscope = demo_microscope(scope="function")


@pytest.fixture(autouse=True)
def _imaging_current(microscope):
    microscope.set_beam_current(IMAGING_CURRENT, BeamType.ION)


@pytest.fixture
def reports(microscope):
    seen = []
    microscope.spot_burn.progress.changed.connect(seen.append)
    return seen


@pytest.fixture
def burns(microscope, monkeypatch):
    """The points the Demo burned into its scene, in order."""
    burned = []
    burn = microscope._connection.burn_into_sample_scene

    def record(beam_type):
        burned.append(microscope.beams[beam_type].sim_scanning_mode_value)
        burn(beam_type)

    monkeypatch.setattr(microscope._connection, "burn_into_sample_scene", record)
    return burned


def _settings(*points, exposure_time=2.0, milling_current=BURN_CURRENT, align=False):
    # The burn alone unless a test aligns: the alignment has its own tests below.
    return SpotBurnSettings(
        coordinates=list(points),
        exposure_time=exposure_time,
        milling_current=milling_current,
        align=align,
    )


def _ion(microscope):
    return microscope.beams[BeamType.ION]


def test_the_demo_has_a_spot_burn_service_that_is_not_a_device(microscope):
    spot_burn = microscope.spot_burn
    assert isinstance(spot_burn, DemoSpotBurn)
    assert isinstance(spot_burn, Service) and not isinstance(spot_burn, Device)
    assert spot_burn.ion is microscope.beams[BeamType.ION]
    assert "electron" not in spot_burn.declared_roles()


def test_run_spot_burn_goes_to_the_service(microscope, burns):
    points = [Point(0.2, 0.3), Point(0.5, 0.5)]
    run_spot_burn(microscope, _settings(*points))
    assert burns == points


def test_a_burn_reports_every_step_and_finishes(microscope, reports):
    status = microscope.spot_burn.run(_settings(Point(0.2, 0.3), Point(0.5, 0.5)))

    assert status is SpotBurnStatus.FINISHED
    assert [(r.status, r.current_point, r.remaining_time) for r in reports] == [
        (SpotBurnStatus.BURNING, 0, 2.0),
        (SpotBurnStatus.BURNING, 1, 1.0),
        (SpotBurnStatus.BURNING, 1, 0.0),
        (SpotBurnStatus.BURNING, 2, 1.0),
        (SpotBurnStatus.BURNING, 2, 0.0),
        (SpotBurnStatus.FINISHED, 2, None),
    ]
    assert reports[0].total_estimated_time == 4.0
    assert microscope.spot_burn.progress.get_value() == reports[-1]


def test_an_exposure_that_is_not_whole_seconds_is_timed_exactly(
    microscope, monkeypatch
):
    waits = []
    monkeypatch.setattr(microscope.spot_burn, "_wait", waits.append)
    microscope.spot_burn.run(_settings(Point(0.5, 0.5), exposure_time=2.5))
    assert waits == [1.0, 1.0, 0.5]


def test_the_beam_is_put_back_after_a_burn(microscope):
    microscope.spot_burn.run(_settings(Point(0.5, 0.5)))
    ion = _ion(microscope)
    assert ion.current.get_value() == IMAGING_CURRENT
    assert ion.scanning_mode.get_value() is ScanMode.FULL_FRAME


def test_it_burns_at_the_burn_current(microscope, monkeypatch):
    currents = []
    burn = microscope._connection.burn_into_sample_scene
    monkeypatch.setattr(
        microscope._connection,
        "burn_into_sample_scene",
        lambda bt: (currents.append(_ion(microscope).current.get_value()), burn(bt)),
    )
    microscope.spot_burn.run(_settings(Point(0.5, 0.5)))
    assert currents == [BURN_CURRENT]


def test_points_outside_the_image_are_dropped_and_strings_are_numbers(
    microscope, burns, reports
):
    microscope.spot_burn.run(
        _settings(
            Point(-0.1, 0.5),
            Point(0.0, 1.0),
            Point(1.2, 0.5),
            exposure_time="1",
            milling_current="9e-11",
        )
    )
    assert burns == [Point(0.0, 1.0)]
    assert reports[0].total_points == 1


def test_the_start_is_recorded(microscope):
    recorded = {}
    microscope.session.record_signal.connect(
        lambda kind, payload: recorded.setdefault(kind, payload)
    )
    microscope.spot_burn.run(_settings(Point(0.5, 0.5), Point(2.0, 2.0)))
    started = recorded["spot_burn_started"]
    assert started["coordinates"] == [[0.5, 0.5]]
    assert started["milling_current"] == BURN_CURRENT
    assert started["dropped"] == 1


def test_no_points_finishes_and_leaves_the_beam_alone(microscope, reports):
    ion = _ion(microscope)
    ion.spot(Point(0.5, 0.5))  # left parked: a run with nothing to burn keeps it

    status = microscope.spot_burn.run(_settings(Point(3.0, 3.0)))

    assert status is SpotBurnStatus.FINISHED
    assert reports[-1].status is SpotBurnStatus.FINISHED
    assert reports[-1].total_points == 0
    assert ion.scanning_mode.get_value() is ScanMode.SPOT


def test_a_stop_before_the_first_point_burns_nothing(microscope, burns, reports):
    stop_event = threading.Event()
    stop_event.set()

    status = microscope.spot_burn.run(_settings(Point(0.5, 0.5)), stop_event=stop_event)

    assert status is SpotBurnStatus.CANCELLED
    assert burns == []
    assert reports[-1].status is SpotBurnStatus.CANCELLED
    assert _ion(microscope).current.get_value() == IMAGING_CURRENT


def test_a_stop_during_a_point_blanks_and_ends_cancelled(
    microscope, burns, monkeypatch
):
    spot_burn = microscope.spot_burn
    monkeypatch.setattr(spot_burn, "_wait", lambda seconds: spot_burn.stop())

    status = spot_burn.run(_settings(Point(0.2, 0.2), Point(0.8, 0.8)))

    assert status is SpotBurnStatus.CANCELLED
    assert burns == [Point(0.2, 0.2)]
    assert _ion(microscope).blanked.get_value() is True
    assert _ion(microscope).scanning_mode.get_value() is ScanMode.FULL_FRAME


def test_a_cancelled_run_spot_burn_returns_without_raising(microscope):
    stop_event = threading.Event()
    stop_event.set()
    assert (
        microscope.run_spot_burn(_settings(Point(0.5, 0.5)), stop_event=stop_event)
        is None
    )


def test_a_failure_is_reported_raised_and_the_beam_put_back(
    microscope, reports, monkeypatch
):
    ion = _ion(microscope)

    def refuse(point):
        raise RuntimeError("beam refused")

    monkeypatch.setattr(ion, "_spot", refuse)

    with pytest.raises(RuntimeError, match="beam refused"):
        microscope.run_spot_burn(_settings(Point(0.5, 0.5)))

    assert reports[-1].status is SpotBurnStatus.FAILED
    assert reports[-1].error == "beam refused"
    assert ion.current.get_value() == IMAGING_CURRENT


def test_a_failing_restore_does_not_hide_the_error(microscope, monkeypatch):
    ion = _ion(microscope)

    def refuse(*args):
        raise RuntimeError("beam refused")

    def restore_fails():
        raise RuntimeError("restore also failed")

    monkeypatch.setattr(ion, "_spot", refuse)
    monkeypatch.setattr(ion, "_full_frame", restore_fails)

    with pytest.raises(RuntimeError, match="beam refused"):
        microscope.spot_burn.run(_settings(Point(0.5, 0.5)))
    # the current is restored even though the scan restore failed
    assert ion.current.get_value() == IMAGING_CURRENT


def test_the_estimate_counts_only_the_points_it_burns(microscope):
    settings = _settings(Point(0.5, 0.5), Point(0.1, 0.1), Point(5, 5))
    assert microscope.spot_burn.estimate(settings) == 4.0


def test_the_supported_settings_carry_the_beam_current_choices(microscope):
    supported = microscope.spot_burn.supported_settings()
    assert set(supported) == {
        "coordinates",
        "exposure_time",
        "milling_current",
        "align",
        "align_steps",
    }
    assert supported["milling_current"] == _ion(microscope).current.metadata


def test_only_the_ion_beam_burns(microscope, burns):
    with pytest.raises(ValueError, match="only supported on the ion beam"):
        microscope.run_spot_burn(
            _settings(Point(0.5, 0.5)), beam_type=BeamType.ELECTRON
        )
    assert burns == []


def test_a_burn_on_demo_burns_the_points_in_order_and_puts_the_beam_back(
    monkeypatch,
):
    """Through ``run_spot_burn`` on a Demo session: the points inside the image
    burned in order, a report per second of exposure plus the first and last, and
    the beam back at its imaging current, scanning full frame."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    settings = _settings(Point(0.2, 0.3), Point(1.5, 0.5), Point(0.7, 0.4))
    microscope, _ = utils.setup_session(
        config_path=cfg.DEFAULT_CONFIGURATION_PATH,
        manufacturer="Demo",
        setup_logging=False,
    )
    microscope.set_beam_current(IMAGING_CURRENT, BeamType.ION)
    reports, burned = [], []
    microscope.spot_burn.progress.changed.connect(reports.append)
    scene_burn = microscope._connection.burn_into_sample_scene
    monkeypatch.setattr(
        microscope._connection,
        "burn_into_sample_scene",
        lambda bt: (
            burned.append(microscope.beams[bt].sim_scanning_mode_value),
            scene_burn(bt),
        ),
    )

    microscope.run_spot_burn(settings)

    ion = microscope.beams[BeamType.ION]
    assert len(reports) == 6 and burned == [Point(0.2, 0.3), Point(0.7, 0.4)]
    assert reports[-1].status is SpotBurnStatus.FINISHED
    assert ion.current.get_value() == IMAGING_CURRENT
    assert ion.scanning_mode.get_value() is ScanMode.FULL_FRAME
    assert ion.blanked.get_value() is False


def test_a_microscope_with_no_spot_burn_service_refuses_to_burn(microscope, reports):
    microscope.spot_burn = None
    with pytest.raises(ValueError, match="no spot burn"):
        microscope.run_spot_burn(_settings(Point(0.5, 0.5)))
    assert reports == []


# --- the alignment at the burn current ------------------------------------------


@pytest.fixture
def alignments(microscope, monkeypatch):
    """Each alignment a run makes: the ion current it aligned at, its reference
    and its arguments."""
    from fibsem import alignment

    seen = []

    def record(**kwargs):
        seen.append({"current": _ion(microscope).current.get_value(), **kwargs})

    monkeypatch.setattr(alignment, "multi_step_alignment_v2", record)
    return seen


def test_it_aligns_at_the_burn_current_on_a_full_frame_reference(
    microscope, alignments, tmp_path
):
    _ion(microscope).hfw.write_through(80e-6)
    microscope.spot_burn.run(_settings(Point(0.5, 0.5), align=True), path=str(tmp_path))

    [aligned] = alignments
    assert aligned["current"] == BURN_CURRENT
    assert aligned["steps"] == 1
    reference = aligned["ref_image"].metadata.image_settings
    assert reference.reduced_area is None
    assert reference.hfw == pytest.approx(80e-6)
    # taken at the imaging current, before the burn current was set
    assert aligned["ref_image"].metadata.microscope_state.ion_beam.beam_current == (
        pytest.approx(IMAGING_CURRENT)
    )
    assert reference.path == str(tmp_path)
    assert aligned["path"] == str(tmp_path / "Alignment")


def test_it_aligns_before_the_first_point(microscope, alignments, burns, monkeypatch):
    order = []
    monkeypatch.setattr(
        microscope.spot_burn,
        "_align",
        lambda *a: order.append(("align", list(burns))),
    )
    microscope.spot_burn.run(_settings(Point(0.2, 0.2), Point(0.8, 0.8), align=True))
    assert order == [("align", [])]
    assert burns == [Point(0.2, 0.2), Point(0.8, 0.8)]


def test_the_steps_are_the_settings(microscope, alignments):
    settings = _settings(Point(0.5, 0.5), align=True)
    settings.align_steps = 3
    microscope.spot_burn.run(settings)
    assert alignments[0]["steps"] == 3


def test_it_aligns_unless_told_not_to():
    assert SpotBurnSettings().align is True


def test_without_align_it_burns_at_once(microscope, alignments, burns):
    settings = _settings(Point(0.5, 0.5), align=False)
    microscope.spot_burn.run(settings)
    assert alignments == []
    assert burns == [Point(0.5, 0.5)]


def test_the_beam_shift_is_put_back_after_an_aligned_burn(microscope, monkeypatch):
    from fibsem import alignment

    ion = _ion(microscope)
    before = ion.shift.get_value()
    monkeypatch.setattr(
        alignment,
        "multi_step_alignment_v2",
        lambda **_: ion.shift.write_through(Point(before.x + 1e-6, before.y)),
    )
    microscope.spot_burn.run(_settings(Point(0.5, 0.5), align=True))
    after = ion.shift.get_value()
    assert (after.x, after.y) == pytest.approx((before.x, before.y))
    assert ion.current.get_value() == IMAGING_CURRENT


def test_a_stop_during_the_alignment_burns_nothing(microscope, burns, monkeypatch):
    from fibsem import alignment

    spot_burn = microscope.spot_burn
    stops = []

    def stopped_mid_alignment(**kwargs):
        spot_burn.stop()
        stops.append(kwargs["stop_event"].is_set())

    monkeypatch.setattr(alignment, "multi_step_alignment_v2", stopped_mid_alignment)
    status = spot_burn.run(_settings(Point(0.5, 0.5), align=True))

    assert stops == [True]  # the alignment sees the service's stop
    assert status is SpotBurnStatus.CANCELLED
    assert burns == []
    assert _ion(microscope).current.get_value() == IMAGING_CURRENT


def test_a_failed_alignment_fails_the_burn_and_puts_the_beam_back(
    microscope, burns, reports, monkeypatch
):
    from fibsem import alignment

    def fails(**_):
        raise RuntimeError("no features")

    monkeypatch.setattr(alignment, "multi_step_alignment_v2", fails)
    with pytest.raises(RuntimeError, match="no features"):
        microscope.spot_burn.run(_settings(Point(0.5, 0.5), align=True))
    assert burns == []
    assert reports[-1].status is SpotBurnStatus.FAILED
    assert _ion(microscope).current.get_value() == IMAGING_CURRENT


def test_the_aligned_burn_on_the_demo_lands_where_it_was_placed(microscope, burns):
    """The real alignment on the simulated scene: it runs, and the spots go where
    they were placed."""
    points = [Point(0.3, 0.4)]
    assert (
        microscope.spot_burn.run(_settings(*points, align=True))
        is SpotBurnStatus.FINISHED
    )
    assert burns == points
