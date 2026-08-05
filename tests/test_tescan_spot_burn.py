"""Tests for the Tescan DrawBeam spot burn implementation.

TESCAN cannot do the blank -> park -> unblank sequence the other backends use: FIB.Scan is a
strict subset of SEM.Scan, missing exactly SetBlanker, GetBlanker and SetBeamPosition. The
driver instead submits a single DrawBeam layer holding one dot per point, with DepthUnit.Second
turning each dot's "depth" into an exposure time.

No hardware or Tescan SDK required: the microscope object is created without __init__, the
connection is stubbed, and the SDK names the driver imports (IEtching, DepthUnit) are
monkeypatched in as fakes.
"""

import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pytest

from fibsem.microscopes import tescan as tescan_module
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import BeamType, MillingState, Point

HFW = 100e-6
RESOLUTION = (1536, 1024)  # (width, height)


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

@dataclass
class FakeDot:
    x: float
    y: float
    depth: float
    depth_unit: object


@dataclass
class FakeLayer:
    name: str
    settings: object
    dots: List[FakeDot] = field(default_factory=list)

    def addDot(self, CenterX, CenterY, Depth, DepthUnit=None, **kwargs):
        self.dots.append(FakeDot(CenterX, CenterY, Depth, DepthUnit))


class FakeDrawBeam:
    """Records the DrawBeam call sequence and reports a configurable number of busy polls."""

    def __init__(self, busy_polls: int = 1):
        self.calls: List[str] = []
        self.layers: List[FakeLayer] = []
        self._busy_polls = busy_polls
        self._polls_remaining = 0

    def Layer(self, name, settings):
        layer = FakeLayer(name, settings)
        self.layers.append(layer)
        return layer

    def LoadLayer(self, layer):
        self.calls.append("LoadLayer")

    def Start(self):
        self.calls.append("Start")
        self._polls_remaining = self._busy_polls

    def UnloadLayer(self):
        self.calls.append("UnloadLayer")

    def Stop(self):
        self.calls.append("Stop")
        self._polls_remaining = 0

    def GetStatus(self):
        # (status, total, elapsed); only the driver's get_milling_state reads [0]
        return ("RUNNING" if self._polls_remaining > 0 else "IDLE", 1.0, 0.0)

    def consume_poll(self):
        if self._polls_remaining > 0:
            self._polls_remaining -= 1


class FakeConnection:
    def __init__(self, busy_polls: int = 1):
        self.DrawBeam = FakeDrawBeam(busy_polls=busy_polls)


def make_microscope(monkeypatch, busy_polls: int = 1, resolution: Tuple[int, int] = RESOLUTION):
    """Create a TescanMicroscope with the SDK and connection stubbed out."""
    microscope = object.__new__(TescanMicroscope)
    microscope.connection = FakeConnection(busy_polls=busy_polls)
    microscope.milling_channel = BeamType.ION

    state = {"prepared": []}
    microscope._prepare_beam = lambda beam_type: state["prepared"].append(beam_type)
    microscope._test_state = state

    def fake_get(key, beam_type=None):
        return {"hfw": HFW, "resolution": resolution, "current": 1e-9}[key]

    microscope.get = fake_get

    # milling state follows the fake DrawBeam, consuming one poll per query so the
    # exposure loop terminates deterministically
    def fake_get_milling_state():
        db = microscope.connection.DrawBeam
        active = db.GetStatus()[0] == "RUNNING"
        db.consume_poll()
        return MillingState.RUNNING if active else MillingState.IDLE

    microscope.get_milling_state = fake_get_milling_state
    microscope.stop_milling = lambda: microscope.connection.DrawBeam.Stop()

    # the SDK names the driver imports are absent without the SDK installed
    monkeypatch.setattr(tescan_module, "IEtching", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(tescan_module, "DepthUnit", type("DepthUnit", (), {"Second": "Second"}), raising=False)
    monkeypatch.setattr(tescan_module.time, "sleep", lambda s: None)

    return microscope


def collect_progress(microscope) -> List[dict]:
    events: List[dict] = []
    microscope.milling_progress_signal.connect(lambda d: events.append(d))
    return events


# --------------------------------------------------------------------------
# coordinate conversion
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "normalised, expected",
    [
        (Point(0.5, 0.5), Point(0.0, 0.0)),          # centre
        (Point(0.0, 0.5), Point(-HFW / 2, 0.0)),     # left edge
        (Point(1.0, 0.5), Point(HFW / 2, 0.0)),      # right edge
    ],
)
def test_point_to_metres_maps_normalised_to_centre_origin(normalised, expected):
    """(0-1, top-left origin) -> metres from the image centre."""
    got = TescanMicroscope._spot_burn_point_to_metres(normalised, hfw=HFW, resolution=RESOLUTION)
    assert got.x == pytest.approx(expected.x)
    assert got.y == pytest.approx(expected.y)


def test_point_to_metres_y_axis_points_up():
    """Image y grows downwards, DrawBeam y grows upwards, so the sign must flip."""
    top = TescanMicroscope._spot_burn_point_to_metres(Point(0.5, 0.0), hfw=HFW, resolution=RESOLUTION)
    bottom = TescanMicroscope._spot_burn_point_to_metres(Point(0.5, 1.0), hfw=HFW, resolution=RESOLUTION)
    assert top.y > 0
    assert bottom.y < 0
    assert top.y == pytest.approx(-bottom.y)


def test_point_to_metres_uses_pixel_aspect_not_hfw_for_y():
    """y extent is hfw scaled by the pixel aspect ratio, not hfw itself."""
    width, height = RESOLUTION
    bottom = TescanMicroscope._spot_burn_point_to_metres(Point(0.5, 1.0), hfw=HFW, resolution=RESOLUTION)
    assert bottom.y == pytest.approx(-(HFW / width) * (height / 2))


# --------------------------------------------------------------------------
# exposure
# --------------------------------------------------------------------------

def test_all_points_go_into_one_layer_as_timed_dots(monkeypatch):
    """A single layer holds one time-based dot per point."""
    m = make_microscope(monkeypatch)
    coords = [Point(0.25, 0.25), Point(0.5, 0.5), Point(0.75, 0.75)]

    m.run_spot_burn(coordinates=coords, exposure_time=3.0)

    db = m.connection.DrawBeam
    assert len(db.layers) == 1
    dots = db.layers[0].dots
    assert len(dots) == 3
    for dot in dots:
        assert dot.depth == 3.0
        assert dot.depth_unit == "Second"

    # the centre point must land on the origin, and the order must be preserved
    assert dots[1].x == pytest.approx(0.0)
    assert dots[1].y == pytest.approx(0.0)
    assert dots[0].x < dots[1].x < dots[2].x


def test_layer_is_loaded_started_and_unloaded_once(monkeypatch):
    m = make_microscope(monkeypatch)

    m.run_spot_burn(coordinates=[Point(0.5, 0.5), Point(0.4, 0.4)], exposure_time=1.0)

    # leading UnloadLayer clears any layer left loaded by a previous job
    assert m.connection.DrawBeam.calls == ["UnloadLayer", "LoadLayer", "Start", "UnloadLayer"]


def test_beam_is_prepared_once(monkeypatch):
    m = make_microscope(monkeypatch)
    m.run_spot_burn(coordinates=[Point(0.5, 0.5)], exposure_time=1.0)
    assert m._test_state["prepared"] == [BeamType.ION]


def test_out_of_bounds_coordinates_are_dropped(monkeypatch):
    m = make_microscope(monkeypatch)
    coords = [Point(0.5, 0.5), Point(1.5, 0.5), Point(-0.1, 0.2), Point(0.2, 0.2)]

    m.run_spot_burn(coordinates=coords, exposure_time=1.0)

    assert len(m.connection.DrawBeam.layers[0].dots) == 2


def test_nothing_runs_when_every_coordinate_is_out_of_bounds(monkeypatch):
    m = make_microscope(monkeypatch)

    m.run_spot_burn(coordinates=[Point(1.5, 0.5), Point(-0.1, 0.2)], exposure_time=1.0)

    assert m.connection.DrawBeam.calls == []


def test_layer_settings_use_the_spot_burn_preset_and_configured_defaults(monkeypatch):
    from fibsem.structures import FibsemMillingSettings

    m = make_microscope(monkeypatch)
    defaults = FibsemMillingSettings()

    m.run_spot_burn(coordinates=[Point(0.5, 0.5)], exposure_time=1.0)

    layer_settings = m.connection.DrawBeam.layers[0].settings
    assert layer_settings["preset"] == "30 keV; 100 pA"
    assert layer_settings["preset"] == tescan_module.SPOT_BURN_PRESET
    assert layer_settings["spotSize"] == defaults.spot_size
    assert layer_settings["spacing"] == defaults.spacing
    assert layer_settings["rate"] == defaults.rate
    assert layer_settings["writeFieldSize"] == HFW
    assert layer_settings["parallel"] is False


# --------------------------------------------------------------------------
# cancellation
# --------------------------------------------------------------------------

def test_stop_event_during_exposure_stops_the_exposition(monkeypatch):
    """Cancelling mid-exposure must stop DrawBeam and still unload the layer."""
    m = make_microscope(monkeypatch, busy_polls=10)
    stop_event = threading.Event()

    original = m.get_milling_state
    polls = {"n": 0}

    def counting_state():
        polls["n"] += 1
        if polls["n"] == 2:
            stop_event.set()
        return original()

    m.get_milling_state = counting_state

    m.run_spot_burn(
        coordinates=[Point(0.5, 0.5), Point(0.25, 0.25)],
        exposure_time=30.0,
        stop_event=stop_event,
    )

    assert "Stop" in m.connection.DrawBeam.calls
    assert m.connection.DrawBeam.calls[-1] == "UnloadLayer"  # still cleaned up


# --------------------------------------------------------------------------
# progress + validation
# --------------------------------------------------------------------------

def test_progress_uses_the_milling_progress_shape(monkeypatch):
    """Progress is reported exactly as run_milling reports it."""
    m = make_microscope(monkeypatch, busy_polls=2)
    events = collect_progress(m)

    m.run_spot_burn(coordinates=[Point(0.5, 0.5), Point(0.2, 0.2)], exposure_time=2.0)

    assert events, "expected at least one progress update"
    progress = events[0]["progress"]
    assert set(progress) == {
        "state", "milling_state", "start_time", "estimated_time", "remaining_time"
    }
    assert progress["state"] == "update"
    assert progress["estimated_time"] == 4.0  # 2 points x 2s
    assert 0.0 <= progress["remaining_time"] <= 4.0


def test_layer_is_unloaded_even_when_drawbeam_raises(monkeypatch):
    m = make_microscope(monkeypatch)

    def boom():
        raise RuntimeError("connection lost")

    m.get_milling_state = boom

    with pytest.raises(RuntimeError):
        m.run_spot_burn(coordinates=[Point(0.5, 0.5)], exposure_time=1.0)

    assert m.connection.DrawBeam.calls[-1] == "UnloadLayer"


def test_electron_beam_is_rejected(monkeypatch):
    m = make_microscope(monkeypatch)
    with pytest.raises(ValueError, match="ion beam"):
        m.run_spot_burn(
            coordinates=[Point(0.5, 0.5)], exposure_time=1.0, beam_type=BeamType.ELECTRON
        )


@pytest.mark.parametrize("exposure_time", [0.0, -1.0])
def test_non_positive_exposure_is_rejected(monkeypatch, exposure_time):
    m = make_microscope(monkeypatch)
    with pytest.raises(ValueError, match="exposure_time"):
        m.run_spot_burn(coordinates=[Point(0.5, 0.5)], exposure_time=exposure_time)
