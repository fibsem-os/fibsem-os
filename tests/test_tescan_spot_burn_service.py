"""Tescan spot burn through the spot burn service, which drives the milling service.

Each case runs over the fake SDK (tests/fixtures/tescan_sdk.py) with the DrawBeam of
test_tescan_milling_service.py. The reference is the microscope's own DrawBeam spot
burn, which still runs when the service is not built (``spot_burn = None``): the same
layer, the same dots in the same order. What the service changes on purpose is asserted
on its own: the preset and field of view are put back, and the milling progress signal
stays quiet while it burns.
"""

import threading
import types

import pytest

from fibsem.drivers.tescan import microscope as tescan_module
from fibsem.drivers.tescan.services import TescanSpotBurn
from fibsem.imaging.spot import SpotBurnSettings, SpotBurnStatus
from fibsem.services import Service
from fibsem.structures import BeamType, Point
from tests.test_tescan_milling_service import (  # noqa: F401  (fixtures)
    IMAGING_PRESET,
    DBStatus,
    connect,
    tescan_system,
)

SPOT_BURN_PRESET = tescan_module.SPOT_BURN_PRESET


@pytest.fixture
def tescan(connect, monkeypatch):
    """A connected fake Tescan that can draw dots, and whose expositions end."""
    monkeypatch.setattr(
        tescan_module,
        "DepthUnit",
        types.SimpleNamespace(Second="Second"),
        raising=False,
    )
    monkeypatch.setattr(tescan_module.time, "sleep", lambda s: None)

    def make(service=True, running_looks=2):
        microscope, fake = connect()
        microscope.milling.poll_interval = 0
        if not service:
            microscope.spot_burn = None
        _finishing(fake, running_looks)
        # the resolution of the last image, which the points are scaled to
        microscope.beams[BeamType.ION]._cache.resolution = (1536, 1024)
        microscope.set_preset(IMAGING_PRESET, BeamType.ION)
        fake.log.clear()
        return microscope, fake

    return make


def _finishing(fake, running_looks):
    """GetStatus reports the exposition running for *running_looks* looks after it
    starts, then over."""
    looks = []

    def get_status():
        fake.DrawBeam._call("GetStatus")
        if fake.DrawBeam.status == DBStatus.ProjectLoadedExpositionInProgress:
            looks.append(1)
            if len(looks) > running_looks:
                fake.DrawBeam.status = DBStatus.ProjectLoadedExpositionIdle
        return (fake.DrawBeam.status, 4.0, 1.0 * len(looks))

    fake.DrawBeam.GetStatus = get_status


def _settings(*points, exposure_time=2.0, milling_current=1e-9):
    return SpotBurnSettings(
        coordinates=list(points),
        exposure_time=exposure_time,
        milling_current=milling_current,
    )


def layer(fake):
    """The layer's settings and the dots drawn on it, in order. (The layer's name is
    left out: milling names its layers "Layer1", the old burn "SpotBurn".)"""
    return [(p, k) for p, _, k in fake.log if p in ("DrawBeam.Layer", "Layer.addDot")]


def drawbeam(fake):
    return [
        p for p, _, _ in fake.log if p.startswith("DrawBeam.") and "Status" not in p
    ]


def activated(fake):
    return [a[0] for p, a, _ in fake.log if p == "FIB.Preset.Activate"]


TWO = (Point(0.25, 0.25), Point(0.75, 0.5))


def test_tescan_builds_its_spot_burn_service(tescan):
    microscope, _ = tescan()
    spot_burn = microscope.spot_burn
    assert isinstance(spot_burn, TescanSpotBurn) and isinstance(spot_burn, Service)
    assert spot_burn.ion is microscope.beams[BeamType.ION]
    # the preset sets the current, so the request's isn't one it burns with
    assert set(spot_burn.supported_settings()) == {"coordinates", "exposure_time"}


@pytest.mark.parametrize(
    "points",
    [TWO, (Point(0.5, 0.5),), (Point(1.5, 0.5), Point(0.5, 0.5), Point(-0.1, 0.2))],
)
def test_the_service_draws_the_same_layer_as_the_microscope_s_own_burn(tescan, points):
    old, old_fake = tescan(service=False)
    old.run_spot_burn(_settings(*points))
    new, new_fake = tescan()
    new.run_spot_burn(_settings(*points))

    assert layer(new_fake) == layer(old_fake)
    assert len([p for p, _ in layer(new_fake) if p == "Layer.addDot"]) == len(
        [p for p in points if 0 <= p.x <= 1 and 0 <= p.y <= 1]
    )


def test_the_dots_are_timed_at_the_exposure(tescan):
    microscope, fake = tescan()
    microscope.run_spot_burn(_settings(*TWO, exposure_time=3.5))
    dots = [k for p, k in layer(fake) if p == "Layer.addDot"]
    assert [(d["Depth"], d["DepthUnit"]) for d in dots] == [(3.5, "Second")] * 2
    assert dots[0]["CenterX"] < dots[1]["CenterX"]


def test_the_layer_is_loaded_run_and_unloaded(tescan):
    microscope, fake = tescan()
    status = microscope.spot_burn.run(_settings(*TWO))
    assert status is SpotBurnStatus.FINISHED
    calls = drawbeam(fake)
    assert calls[0] == "DrawBeam.UnloadLayer"  # a layer left loaded is cleared
    assert calls.index("DrawBeam.Layer") < calls.index("DrawBeam.LoadLayer")
    assert calls.index("DrawBeam.LoadLayer") < calls.index("DrawBeam.Start")
    assert calls[-1] == "DrawBeam.UnloadLayer"


def test_it_burns_at_the_spot_burn_preset_and_puts_the_preset_back(tescan):
    microscope, fake = tescan()
    fake.FIB.Optics.viewfield = 0.15  # mm
    microscope.run_spot_burn(_settings(*TWO))
    assert activated(fake) == [SPOT_BURN_PRESET, IMAGING_PRESET]
    settings = layer(fake)[0][1]
    assert settings["preset"] == SPOT_BURN_PRESET
    assert settings["writeFieldSize"] == pytest.approx(0.15e-3)
    assert settings["parallel"] is False


def test_the_old_burn_never_put_the_preset_back(tescan):
    """What the service changes: the microscope's own burn left the preset to the
    layer, and activated none afterwards."""
    microscope, fake = tescan(service=False)
    microscope.run_spot_burn(_settings(*TWO))
    assert activated(fake) == []


def test_a_burn_reports_spot_burn_progress_not_milling(tescan):
    microscope, _ = tescan()
    burns, mills = [], []
    microscope.spot_burn_progress_signal.connect(burns.append)
    microscope.milling_progress_signal.connect(mills.append)

    microscope.run_spot_burn(_settings(*TWO))

    assert mills == [], mills
    assert burns[0].status is SpotBurnStatus.BURNING
    assert (burns[0].current_point, burns[0].total_estimated_time) == (0, 4.0)
    for update in burns[1:-1]:
        assert update.status is SpotBurnStatus.BURNING
        assert 1 <= update.current_point <= 2
    assert burns[-1].status is SpotBurnStatus.FINISHED
    assert burns[-1].current_point == 2


def test_milling_reports_again_after_a_burn(tescan):
    from fibsem.services.milling import progress_update
    from fibsem.structures import MillingState

    microscope, _ = tescan()
    microscope.run_spot_burn(_settings(*TWO))
    mills = []
    microscope.milling_progress_signal.connect(mills.append)
    microscope.milling.progress.report(progress_update(MillingState.RUNNING))
    assert len(mills) == 1


def test_a_stop_event_cancels_without_raising_and_puts_the_preset_back(tescan):
    microscope, fake = tescan(running_looks=10)
    stop_event = threading.Event()
    burns = []
    microscope.spot_burn_progress_signal.connect(burns.append)
    status = fake.DrawBeam.GetStatus

    def stop_on_the_second_look():
        result = status()
        if fake.DrawBeam.status == DBStatus.ProjectLoadedExpositionInProgress:
            stop_event.set()
        return result

    fake.DrawBeam.GetStatus = stop_on_the_second_look

    assert microscope.run_spot_burn(_settings(*TWO), stop_event=stop_event) is None

    assert burns[-1].status is SpotBurnStatus.CANCELLED
    assert "DrawBeam.Stop" in drawbeam(fake)
    assert drawbeam(fake)[-1] == "DrawBeam.UnloadLayer"
    assert activated(fake)[-1] == IMAGING_PRESET


def test_stop_ends_a_run_from_another_thread(tescan):
    microscope, fake = tescan(running_looks=10)
    status = fake.DrawBeam.GetStatus

    def stop_while_running():
        result = status()
        if fake.DrawBeam.status == DBStatus.ProjectLoadedExpositionInProgress:
            microscope.spot_burn.stop()
        return result

    fake.DrawBeam.GetStatus = stop_while_running
    assert microscope.spot_burn.run(_settings(*TWO)) is SpotBurnStatus.CANCELLED


def test_no_points_touches_nothing(tescan):
    microscope, fake = tescan()
    status = microscope.spot_burn.run(_settings(Point(1.5, 0.5)))
    assert status is SpotBurnStatus.FINISHED
    assert drawbeam(fake) == [] and activated(fake) == []


def test_a_failure_is_raised_and_the_preset_put_back(tescan):
    microscope, fake = tescan()
    burns = []
    microscope.spot_burn_progress_signal.connect(burns.append)

    def lost():
        raise RuntimeError("connection lost")

    fake.DrawBeam.Start = lost
    with pytest.raises(RuntimeError, match="connection lost"):
        microscope.run_spot_burn(_settings(*TWO))
    assert burns[-1].status is SpotBurnStatus.FAILED
    assert drawbeam(fake)[-1] == "DrawBeam.UnloadLayer"
    assert activated(fake)[-1] == IMAGING_PRESET


def test_the_burn_current_may_be_none(tescan):
    microscope, fake = tescan()
    recorded = []
    microscope.record_signal.connect(lambda kind, payload: recorded.append(payload))
    microscope.run_spot_burn(_settings(*TWO, milling_current=None))
    assert recorded[0]["milling_current"] is None
    assert "DrawBeam.Start" in drawbeam(fake)


def test_only_the_ion_beam_burns(tescan):
    microscope, fake = tescan()
    with pytest.raises(ValueError, match="ion beam"):
        microscope.run_spot_burn(
            _settings(Point(0.5, 0.5)), beam_type=BeamType.ELECTRON
        )
    assert drawbeam(fake) == []


@pytest.mark.parametrize("exposure_time", [0.0, -1.0])
def test_a_non_positive_exposure_is_refused_before_anything_is_touched(
    tescan, exposure_time
):
    microscope, fake = tescan()
    with pytest.raises(ValueError, match="exposure_time"):
        microscope.run_spot_burn(
            _settings(Point(0.5, 0.5), exposure_time=exposure_time)
        )
    assert drawbeam(fake) == [] and activated(fake) == []
