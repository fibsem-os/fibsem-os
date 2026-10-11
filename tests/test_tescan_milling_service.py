"""Tescan milling through the milling service, against the calls it always made.

Each case runs over the fake SDK (tests/fixtures/tescan_sdk.py) through
``microscope.milling`` (TescanMilling). tests/fixtures/tescan_milling_calls.json holds
the DrawBeam calls, the presets activated and the results of the same milling by the
microscope's own DrawBeam code, before that code moved into the service; the service
must match them. What the service changes on purpose is asserted on its own: the field
of view is put back too.
"""

import json
import os
import types
from pathlib import Path

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.drivers.tescan import microscope as tescan_module
from fibsem.drivers.tescan.services import TescanMilling
from fibsem.structures import (
    BeamType,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemRectangleSettings,
    MillingState,
)
from tests.fixtures import tescan_sdk

MILL_PRESET = "30 keV; 1 nA"
IMAGING_PRESET = "30 keV; 100 pA"

DBStatus = types.SimpleNamespace(
    ProjectNotLoaded="ProjectNotLoaded",
    ProjectLoadedExpositionIdle="ProjectLoadedExpositionIdle",
    ProjectLoadedExpositionInProgress="ProjectLoadedExpositionInProgress",
    ProjectLoadedExpositionPaused="ProjectLoadedExpositionPaused",
    Unknown="Unknown",
)


class FakeDrawBeam(tescan_sdk._Recorded):
    """``connection.DrawBeam``: a layer that records its patterns, and a status."""

    def __init__(self, sdk):
        super().__init__(sdk, "DrawBeam")
        self.status = DBStatus.ProjectNotLoaded

    def Layer(self, name, settings):
        self._call("Layer", name, **settings)
        return tescan_sdk.Node(self._sdk, "Layer")

    def UnloadLayer(self):
        self._call("UnloadLayer")

    def Start(self):
        self._call("Start")
        self.status = DBStatus.ProjectLoadedExpositionInProgress

    def Pause(self):
        self._call("Pause")
        self.status = DBStatus.ProjectLoadedExpositionPaused

    def Resume(self):
        self._call("Resume")
        self.status = DBStatus.ProjectLoadedExpositionInProgress

    def Stop(self):
        self._call("Stop")
        self.status = DBStatus.ProjectLoadedExpositionIdle

    def GetStatus(self):
        self._call("GetStatus")
        return (self.status, 10.0, 2.0)

    def EstimateTime(self):
        self._call("EstimateTime")
        return 42.0


@pytest.fixture
def connect(monkeypatch, tescan_system):
    monkeypatch.setattr(tescan_module, "TESCAN_ELECTRON_TO_ION_SETTLE_TIME", 0)
    monkeypatch.setattr(
        tescan_module, "IEtching", lambda **kwargs: kwargs, raising=False
    )
    monkeypatch.setattr(tescan_module, "DBStatus", DBStatus, raising=False)
    monkeypatch.setattr(
        tescan_module,
        "DrawBeamStatusToPatterningState",
        {
            DBStatus.ProjectNotLoaded: MillingState.IDLE,
            DBStatus.ProjectLoadedExpositionIdle: MillingState.IDLE,
            DBStatus.ProjectLoadedExpositionInProgress: MillingState.RUNNING,
            DBStatus.ProjectLoadedExpositionPaused: MillingState.PAUSED,
            DBStatus.Unknown: MillingState.ERROR,
        },
        raising=False,
    )

    def make():
        fake = tescan_sdk.FakeTescan()
        fake.DrawBeam = FakeDrawBeam(fake)
        fake.FIB.Preset.names = [
            MILL_PRESET,
            IMAGING_PRESET,
            tescan_module.DEFAULT_IMAGING_PRESET,
        ]
        return tescan_sdk.connect(monkeypatch, tescan_system, fake)

    return make


@pytest.fixture
def tescan_system():
    return utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system


def mill(microscope, fake, imaging_preset=IMAGING_PRESET):
    if imaging_preset is not None:
        microscope.set_preset(imaging_preset, BeamType.ION)
    fake.log.clear()
    microscope.setup_milling(FibsemMillingSettings(preset=MILL_PRESET, hfw=80e-6))
    microscope.draw_rectangle(
        FibsemRectangleSettings(
            width=10e-6, height=5e-6, depth=1e-6, centre_x=0, centre_y=0
        )
    )
    microscope.draw_line(
        FibsemLineSettings(start_x=0, end_x=1e-6, start_y=0, end_y=0, depth=1e-6)
    )
    microscope.draw_circle(
        FibsemCircleSettings(radius=2e-6, depth=1e-6, centre_x=0, centre_y=0)
    )
    estimate = microscope.estimate_milling_time()
    microscope.start_milling()
    running = microscope.get_milling_state()
    microscope.pause_milling()
    paused = microscope.get_milling_state()
    microscope.resume_milling()
    microscope.finish_milling(imaging_current=20e-12, imaging_voltage=30e3)
    return estimate, running, paused


def drawbeam(fake):
    return [(p, a, k) for p, a, k in fake.log if p.startswith(("DrawBeam", "Layer"))]


def activated(fake):
    return [a[0] for p, a, _ in fake.log if p == "FIB.Preset.Activate"]


def test_tescan_builds_its_milling_service(connect):
    microscope, _ = connect()
    assert isinstance(microscope.milling, TescanMilling)
    assert microscope.milling.ion is microscope.beams[BeamType.ION]


with open(Path(__file__).parent / "fixtures" / "tescan_milling_calls.json") as f:
    OLD_CALLS = json.load(f)


def _plain(value):
    """As the old calls were stored: JSON, with an enum as its name."""
    return json.loads(json.dumps(value, default=lambda o: getattr(o, "name", repr(o))))


@pytest.mark.parametrize("imaging_preset", [IMAGING_PRESET, None])
def test_milling_makes_the_same_drawbeam_calls(connect, imaging_preset):
    microscope, fake = connect()
    old = OLD_CALLS[str(imaging_preset)]

    assert _plain(list(mill(microscope, fake, imaging_preset))) == old["result"]
    assert _plain(microscope.get_milling_state()) == old["state"]

    # the same layer, patterns, run and unload
    assert _plain(drawbeam(fake)) == old["drawbeam"]
    # the milling preset, then the one milling found (or the imaging default)
    restored = imaging_preset or tescan_module.DEFAULT_IMAGING_PRESET
    assert activated(fake) == old["activated"] == [MILL_PRESET, restored]


def test_the_service_puts_the_field_of_view_back(connect):
    microscope, fake = connect()
    fake.FIB.Optics.viewfield = 0.15  # mm
    microscope.setup_milling(FibsemMillingSettings(preset=MILL_PRESET, hfw=80e-6))
    fake.FIB.Optics.viewfield = 0.08  # the user zoomed in while it milled
    microscope.finish_milling()
    assert fake.FIB.Optics.viewfield == pytest.approx(0.15)


def test_the_ion_current_and_voltage_are_left_to_the_preset(connect):
    """Neither is settable on the ion column, so neither is saved, restored, or
    overridden: no "not supported" write after milling."""
    microscope, fake = connect()
    microscope.setup_milling(FibsemMillingSettings(preset=MILL_PRESET))
    assert set(microscope.milling._saved) == {"preset", "hfw"}
    fake.log.clear()
    microscope.finish_milling(imaging_current=20e-12, imaging_voltage=30e3)
    paths = [p for p, _, _ in fake.log]
    assert "FIB.Beam.SetCurrent" not in paths
    assert "FIB.Beam.SetVoltage" not in paths


def test_a_preset_that_fails_to_come_back_still_ends_milling(connect, monkeypatch):
    microscope, fake = connect()
    microscope.set_preset(IMAGING_PRESET, BeamType.ION)
    microscope.setup_milling(FibsemMillingSettings(preset=MILL_PRESET))

    def fail(name):
        raise RuntimeError("preset unavailable")

    monkeypatch.setattr(fake.FIB.Preset, "Activate", fail)
    fake.log.clear()
    microscope.finish_milling()  # must not raise

    assert "DrawBeam.UnloadLayer" in [p for p, _, _ in fake.log]
    assert microscope.milling._saved is None


def test_stop_uses_a_second_connection(connect):
    microscope, fake = connect()
    microscope.setup_milling(FibsemMillingSettings(preset=MILL_PRESET))
    microscope.start_milling()
    microscope.stop_milling()  # Automation is patched to return the same fake
    assert fake.DrawBeam.status == DBStatus.ProjectLoadedExpositionIdle


def test_stop_connects_on_the_port_the_microscope_connected_on(connect, monkeypatch):
    microscope, fake = connect()
    microscope.connect_to_microscope(ip_address="localhost", port=8301)
    opened = []

    def automation(*args, **kwargs):
        opened.append((args, kwargs))
        return fake

    monkeypatch.setattr(tescan_module, "Automation", automation, raising=False)
    microscope.stop_milling()
    assert opened == [(("localhost",), {"port": 8301})]


def test_tescan_mills_with_the_settings_it_says(connect):
    from tests.fixtures.milling_reads import fields_setup_reads

    microscope, _ = connect()
    supported = microscope.milling.supported_settings()
    # the preset sets the current and voltage, so the recipe's are not used
    assert "milling_current" not in supported
    assert "milling_voltage" not in supported
    assert MILL_PRESET in supported["preset"].choices
    assert supported["milling_channel"].choices == (BeamType.ION,)
    read = fields_setup_reads(
        microscope.milling, FibsemMillingSettings(preset=MILL_PRESET, hfw=80e-6)
    )
    assert set(supported) == read
    # an electron beam it can't mill with
    assert microscope.milling.supported_settings(BeamType.ELECTRON) == {}
    directions = microscope.milling.supported_pattern_settings()["scan_direction"]
    assert "Flyback" in directions.choices  # the fallback a pattern draws with


def _finishing(fake, running_looks=2):
    """GetStatus reports the exposition running for *running_looks* looks, then over."""
    looks = []

    def get_status():
        fake.DrawBeam._call("GetStatus")
        looks.append(1)
        if len(looks) > running_looks:
            fake.DrawBeam.status = DBStatus.ProjectLoadedExpositionIdle
        return (fake.DrawBeam.status, 10.0, 2.0 * len(looks))

    fake.DrawBeam.GetStatus = get_status


def _drawn(microscope):
    microscope.setup_milling(FibsemMillingSettings(preset=MILL_PRESET, hfw=80e-6))
    microscope.draw_rectangle(
        FibsemRectangleSettings(
            width=10e-6, height=5e-6, depth=1e-6, centre_x=0, centre_y=0
        )
    )


def test_run_milling_loads_runs_and_unloads_the_layer(connect):
    microscope, fake = connect()
    microscope.milling.poll_interval = 0
    _drawn(microscope)
    _finishing(fake)
    updates = []
    microscope.milling_progress_signal.connect(updates.append)
    fake.log.clear()

    microscope.run_milling()

    paths = [p for p, _, _ in fake.log]
    for step in ("DrawBeam.LoadLayer", "connection.Progress.Show", "DrawBeam.Start"):
        assert step in paths
    assert paths.index("DrawBeam.LoadLayer") < paths.index("DrawBeam.EstimateTime")
    assert paths.index("DrawBeam.EstimateTime") < paths.index("DrawBeam.Start")
    # the bar follows DrawBeam's own times, and goes away before the layer does
    assert ["connection.Progress.SetPercents", [20.0], {}] in fake.log
    assert paths[-2:] == ["connection.Progress.Hide", "DrawBeam.UnloadLayer"]
    # and so does the progress: DrawBeam's total and elapsed, not the estimate
    progress = microscope.milling.progress.cached
    assert progress.milling_state is MillingState.IDLE
    assert (progress.estimated_time, progress.remaining_time) == (10.0, 4.0)
    assert [u.remaining_time for u in updates] == [8.0, 6.0, 4.0]


def test_a_failure_while_milling_stops_and_unloads_then_raises(connect):
    """The old loop lost the error (an UnboundLocalError in its `finally`) and left
    the layer loaded and running."""
    microscope, fake = connect()
    microscope.milling.poll_interval = 0
    _drawn(microscope)

    status = fake.DrawBeam.GetStatus
    looks = []

    def broken():
        looks.append(1)
        if len(looks) == 2:  # the run's second look; the stop's own look works
            raise RuntimeError("connection lost")
        return status()

    fake.DrawBeam.GetStatus = broken
    fake.log.clear()
    with pytest.raises(RuntimeError, match="connection lost"):
        microscope.run_milling()

    paths = [p for p, _, _ in fake.log]
    assert "DrawBeam.Stop" in paths
    assert paths[-2:] == ["connection.Progress.Hide", "DrawBeam.UnloadLayer"]


def _no_layer(fake):
    """UnloadLayer raises, as DrawBeam does when no layer is loaded."""

    def unload():
        fake.DrawBeam._call("UnloadLayer")
        raise RuntimeError("no layer loaded")

    fake.DrawBeam.UnloadLayer = unload


def test_clearing_with_no_layer_loaded_is_quiet(connect):
    microscope, fake = connect()
    _no_layer(fake)
    fake.log.clear()
    for _ in range(3):
        microscope.clear_patterns()  # must not raise
    assert [p for p, _, _ in fake.log] == ["DrawBeam.UnloadLayer"] * 3


def test_a_second_setup_keeps_the_preset_milling_found(connect):
    microscope, fake = connect()
    microscope.set_preset(IMAGING_PRESET, BeamType.ION)
    microscope.setup_milling(FibsemMillingSettings(preset=MILL_PRESET))
    microscope.setup_milling(FibsemMillingSettings(preset=MILL_PRESET))
    microscope.finish_milling()
    assert activated(fake)[-1] == IMAGING_PRESET


def test_each_milling_puts_back_the_preset_it_found(connect):
    microscope, fake = connect()
    for found in (IMAGING_PRESET, tescan_module.DEFAULT_IMAGING_PRESET):
        microscope.set_preset(found, BeamType.ION)
        microscope.setup_milling(FibsemMillingSettings(preset=MILL_PRESET))
        microscope.finish_milling()
        assert activated(fake)[-1] == found


def test_finishing_survives_the_preset_and_the_unload_failing(connect, monkeypatch):
    microscope, fake = connect()
    microscope.set_preset(IMAGING_PRESET, BeamType.ION)
    microscope.setup_milling(FibsemMillingSettings(preset=MILL_PRESET))

    def fail(name):
        raise RuntimeError("preset unavailable")

    monkeypatch.setattr(fake.FIB.Preset, "Activate", fail)
    _no_layer(fake)
    microscope.finish_milling()  # must not raise
    assert microscope.milling._saved is None


def test_only_the_ion_beam_mills(connect):
    microscope, _ = connect()
    with pytest.raises(ValueError):
        microscope.setup_milling(
            FibsemMillingSettings(milling_channel=BeamType.ELECTRON)
        )
