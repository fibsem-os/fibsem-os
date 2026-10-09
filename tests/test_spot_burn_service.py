"""The spot burn service on the Demo, and `run_spot_burn` over it.

The Demo burns with the shared point-by-point burn through `microscope.spot_burn`:
blank, park the beam on the point, unblank (which burns the spot into the sample
scene), wait, and back to full frame at the burn current's start value. The
reference for what it must do is the microscope's own implementation, which still
runs where a driver builds no service: the parity test runs both on a Demo.
"""

import threading

import pytest

from fibsem import config as cfg
from fibsem import utils
from fibsem.devices.core import Device
from fibsem.drivers.demo.services import DemoSpotBurn
from fibsem.imaging.spot import SpotBurnSettings, SpotBurnStatus, run_spot_burn
from fibsem.services import Service
from fibsem.structures import BeamType, Point, ScanMode

IMAGING_CURRENT = 2e-11
BURN_CURRENT = 9e-11


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        config_path=cfg.DEFAULT_CONFIGURATION_PATH,
        manufacturer="Demo",
        setup_logging=False,
    )
    microscope.set_beam_current(IMAGING_CURRENT, BeamType.ION)
    return microscope


@pytest.fixture
def reports(microscope):
    seen = []
    microscope.spot_burn_progress_signal.connect(seen.append)
    return seen


@pytest.fixture
def burns(microscope, monkeypatch):
    """The points the Demo burned into its scene, in order."""
    burned = []
    burn = microscope._burn_into_sample_scene

    def record(beam_type):
        burned.append(microscope.beams[beam_type].sim_scanning_mode_value)
        burn(beam_type)

    monkeypatch.setattr(microscope, "_burn_into_sample_scene", record)
    return burned


def _settings(*points, exposure_time=2.0, milling_current=BURN_CURRENT):
    return SpotBurnSettings(
        coordinates=list(points),
        exposure_time=exposure_time,
        milling_current=milling_current,
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
    burn = microscope._burn_into_sample_scene
    monkeypatch.setattr(
        microscope,
        "_burn_into_sample_scene",
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
    recorded = []
    microscope.record_signal.connect(lambda kind, payload: recorded.append(payload))
    microscope.spot_burn.run(_settings(Point(0.5, 0.5), Point(2.0, 2.0)))
    assert recorded[0]["coordinates"] == [[0.5, 0.5]]
    assert recorded[0]["milling_current"] == BURN_CURRENT
    assert recorded[0]["dropped"] == 1


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
    assert set(supported) == {"coordinates", "exposure_time", "milling_current"}
    assert supported["milling_current"] == _ion(microscope).current.metadata


def test_only_the_ion_beam_burns(microscope, burns):
    with pytest.raises(ValueError, match="only supported on the ion beam"):
        microscope.run_spot_burn(
            _settings(Point(0.5, 0.5)), beam_type=BeamType.ELECTRON
        )
    assert burns == []


def test_the_service_matches_the_microscope_s_own_burn(monkeypatch):
    """Parity with the implementation it replaces, on whole-second exposures (the
    old one counted down in 1 s steps): the same points burned in order, the same
    reports, and the beam left the same way."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    settings = _settings(Point(0.2, 0.3), Point(1.5, 0.5), Point(0.7, 0.4))

    def burn(use_service):
        microscope, _ = utils.setup_session(
            config_path=cfg.DEFAULT_CONFIGURATION_PATH,
            manufacturer="Demo",
            setup_logging=False,
        )
        microscope.set_beam_current(IMAGING_CURRENT, BeamType.ION)
        if not use_service:
            microscope.spot_burn = None
        reports, burned = [], []
        microscope.spot_burn_progress_signal.connect(reports.append)
        scene_burn = microscope._burn_into_sample_scene
        monkeypatch.setattr(
            microscope,
            "_burn_into_sample_scene",
            lambda bt: (
                burned.append(microscope.beams[bt].sim_scanning_mode_value),
                scene_burn(bt),
            ),
        )
        microscope.run_spot_burn(settings)
        ion = microscope.beams[BeamType.ION]
        state = (
            ion.current.get_value(),
            ion.scanning_mode.get_value(),
            ion.blanked.get_value(),
        )
        return reports, burned, state

    service = burn(use_service=True)
    reports, burned, _ = service
    assert len(reports) == 6 and burned == [Point(0.2, 0.3), Point(0.7, 0.4)]
    assert service == burn(use_service=False)
