"""The imaging service (`microscope.imaging`, FIB-1261) on the Demo: built over the
beams, ``auto_focus`` choosing the instrument's own autofocus or the sweep, and
``focus_sweep`` with its field of view, its stop and its result."""

import os
import threading
import time

import pytest

from fibsem import utils
from fibsem.autofunctions import autofocus
from fibsem.autofunctions.autofocus import (
    AutoFocusResult,
    AutoFocusSettings,
    FocusSweepPass,
)
from fibsem.services.imaging import Imaging
from fibsem.structures import BeamType, FibsemRectangle


@pytest.fixture
def microscope():
    previous = os.environ.get("FIBSEM_SIM_NO_DELAY")
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    yield microscope
    if previous is None:
        os.environ.pop("FIBSEM_SIM_NO_DELAY", None)
    else:
        os.environ["FIBSEM_SIM_NO_DELAY"] = previous


def _one_short_pass() -> AutoFocusSettings:
    return AutoFocusSettings(
        passes=[FocusSweepPass(search_range=20e-6, step_size=10e-6)],
        probe_resolution=(64, 64),
        use_autocontrast=False,
    )


def test_built_over_the_beams(microscope):
    imaging = microscope.imaging
    assert isinstance(imaging, Imaging)
    assert microscope.services["imaging"] is imaging
    assert imaging.electron is microscope.beams[BeamType.ELECTRON]
    assert imaging.ion is microscope.beams[BeamType.ION]


def test_auto_focus_uses_the_beams_own_autofocus(microscope, monkeypatch):
    beam = microscope.beams[BeamType.ELECTRON]
    assert beam.commands["auto_focus"].available
    calls = []
    monkeypatch.setattr(
        autofocus, "run_auto_focus", lambda *a, **k: pytest.fail("swept")
    )
    monkeypatch.setattr(
        type(beam), "_auto_focus", lambda self, area: calls.append(area)
    )
    area = FibsemRectangle(0.25, 0.25, 0.5, 0.5)

    microscope.auto_focus(BeamType.ELECTRON, reduced_area=area)

    assert calls == [area]


def test_auto_focus_sweeps_a_beam_without_its_own(microscope, monkeypatch):
    beam = microscope.beams[BeamType.ELECTRON]
    monkeypatch.setattr(type(beam), "_has_auto_focus", lambda self: False)
    assert not beam.commands["auto_focus"].available
    seen = {}

    def sweep(m, beam_type, hfw, settings, stop_event, path, name):
        seen.update(m=m, beam_type=beam_type, hfw=hfw, settings=settings)

    monkeypatch.setattr(autofocus, "run_auto_focus", sweep)
    area = FibsemRectangle(0.25, 0.25, 0.5, 0.5)

    microscope.auto_focus(BeamType.ELECTRON, reduced_area=area)

    # As the Odemis override did: the sweep at the beam's field of view.
    assert seen["m"] is microscope
    assert seen["beam_type"] is BeamType.ELECTRON
    assert seen["hfw"] == pytest.approx(beam.hfw.get_value())
    assert seen["settings"].reduced_area == area


def test_focus_sweep_runs_and_answers_the_result(microscope):
    beam = microscope.beams[BeamType.ELECTRON]
    result = microscope.imaging.focus_sweep(
        BeamType.ELECTRON, hfw=80e-6, settings=_one_short_pass()
    )
    assert isinstance(result, AutoFocusResult)
    assert result.n_iterations == 3
    assert beam.working_distance.get_value() == pytest.approx(result.working_distance)
    summary = result.to_dict()
    assert summary["n_iterations"] == 3
    assert "image" not in summary


def test_stop_ends_the_sweep_as_cancelled(microscope, monkeypatch):
    imaging = microscope.imaging
    stopped = []

    def sweep(m, beam_type, hfw, settings, stop_event, path, name):
        assert not stop_event.is_set()
        imaging.stop()  # as a second caller would, mid-run
        stopped.append(stop_event.is_set())

    monkeypatch.setattr(autofocus, "run_auto_focus", sweep)
    imaging.focus_sweep(BeamType.ELECTRON)
    assert stopped == [True]

    # A caller's own stop event still stops it, and a new sweep starts unstopped.
    caller_stop = threading.Event()

    def sweep_again(m, beam_type, hfw, settings, stop_event, path, name):
        assert not stop_event.is_set()
        caller_stop.set()
        stopped.append(stop_event.is_set())

    monkeypatch.setattr(autofocus, "run_auto_focus", sweep_again)
    imaging.focus_sweep(BeamType.ELECTRON, stop_event=caller_stop)
    assert stopped == [True, True]


def test_no_beam_no_service(microscope):
    from fibsem.services.imaging import bind_imaging

    class NoBeams:
        beams = {}

    assert bind_imaging(Imaging, NoBeams()) is None
    microscope.imaging = None
    with pytest.raises(ValueError, match="no imaging"):
        microscope.auto_focus(BeamType.ELECTRON)


def test_the_beams_own_autofocus_is_recorded(microscope):
    """It moves the working distance as the sweep does, so it is on the record too
    (FIB-1355)."""
    recorded = []
    microscope.session.record_signal.connect(
        lambda kind, payload: recorded.append((kind, payload))
    )
    beam = microscope.beams[BeamType.ELECTRON]
    beam.auto_focus(FibsemRectangle(0.25, 0.25, 0.5, 0.5))
    events = [payload for kind, payload in recorded if kind == "autofocus"]
    assert len(events) == 1
    event = events[0]
    assert event["status"] == "completed"
    assert event["method"] == "instrument"
    assert event["beam_type"] == "ELECTRON"
    assert event["reduced_area"] == FibsemRectangle(0.25, 0.25, 0.5, 0.5).to_dict()


def test_a_failed_beam_autofocus_is_recorded_and_raised(microscope, monkeypatch):
    recorded = []
    microscope.session.record_signal.connect(
        lambda kind, payload: recorded.append((kind, payload))
    )
    beam = microscope.beams[BeamType.ELECTRON]

    def fails(self, area):
        raise RuntimeError("no signal")

    monkeypatch.setattr(type(beam), "_auto_focus", fails)
    with pytest.raises(RuntimeError, match="no signal"):
        beam.auto_focus()
    (event,) = [payload for kind, payload in recorded if kind == "autofocus"]
    assert event["status"] == "failed"
    assert event["error"] == "RuntimeError: no signal"


def test_the_beam_commands_pass_through_by_beam_type(microscope, monkeypatch):
    """FIB-1261: acquire, autocontrast and live view are the beam's commands,
    reached through the service by beam type."""
    from fibsem.structures import FibsemImage, ImageSettings

    imaging = microscope.imaging
    image = imaging.acquire(BeamType.ION)
    assert isinstance(image, FibsemImage)
    assert image.metadata.image_settings.beam_type is BeamType.ION

    # The settings' own beam type wins over the argument.
    settings = ImageSettings(
        beam_type=BeamType.ELECTRON, resolution=(384, 256), hfw=80e-6
    )
    image = imaging.acquire(BeamType.ION, settings)
    assert image.metadata.image_settings.beam_type is BeamType.ELECTRON
    assert image.data.shape == (256, 384)

    beam = microscope.beams[BeamType.ION]
    areas = []
    monkeypatch.setattr(type(beam), "_autocontrast", lambda self, a: areas.append(a))
    area = FibsemRectangle(0.1, 0.1, 0.5, 0.5)
    imaging.autocontrast(BeamType.ION, area)
    assert areas == [area]

    electron = microscope.beams[BeamType.ELECTRON]
    imaging.start_live(BeamType.ELECTRON)
    try:
        assert electron.is_live
    finally:
        imaging.stop_live(BeamType.ELECTRON)
    # stop_live waits only briefly (two seconds) for the frame in flight. A frame
    # takes about one second here and longer on a loaded Windows runner, where the
    # beam still read live straight after stop_live; it must stop, not stop at once.
    deadline = time.monotonic() + 30
    while electron.is_live and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not electron.is_live


def test_the_microscope_images_through_the_service(microscope, monkeypatch):
    calls = []
    real = Imaging.acquire
    monkeypatch.setattr(
        Imaging,
        "acquire",
        lambda self, *a, **k: calls.append(a) or real(self, *a, **k),
    )
    microscope.acquire_image(beam_type=BeamType.ELECTRON)
    assert calls == [(BeamType.ELECTRON, None)]


def test_acquire_image_saves_and_records(microscope, tmp_path):
    """FIB-1261: acquire.py's image (new_image, now acquire_image) moved onto the service."""
    from fibsem.structures import ImageSettings

    recorded = []
    microscope.session.record_signal.connect(
        lambda kind, data: recorded.append((kind, data))
    )
    settings = ImageSettings(
        beam_type=BeamType.ION,
        resolution=(192, 128),
        hfw=100e-6,
        save=True,
        path=str(tmp_path),
        filename="trench",
    )
    image = microscope.imaging.acquire_image(settings)
    assert image.filepath == str(tmp_path / "trench_ib.tif")
    assert os.path.exists(image.filepath)
    kind, data = recorded[-1]
    assert kind == "image_acquired" and data["path"] == image.filepath


def test_reference_images_leave_the_settings_alone(microscope, tmp_path):
    from fibsem import acquire
    from fibsem.structures import ImageSettings

    settings = ImageSettings(
        beam_type=BeamType.ION,
        resolution=(96, 64),
        hfw=100e-6,
        save=False,
        path=str(tmp_path),
        filename="ref",
    )
    eb, ib = microscope.imaging.reference_images(settings)
    assert eb.metadata.image_settings.beam_type is BeamType.ELECTRON
    assert ib.metadata.image_settings.beam_type is BeamType.ION
    assert settings.beam_type is BeamType.ION

    sem, fib = acquire.acquire_channels(microscope, settings, acquire_sem=False)
    assert sem is None and fib is not None

    sets = microscope.imaging.reference_image_sets(
        settings, [200e-6, 80e-6], filename="ref", ion=False
    )
    assert [pair[1] for pair in sets] == [None, None]
    assert [os.path.basename(pair[0].filepath) for pair in sets] == [
        "ref_res_01_eb.tif",
        "ref_res_02_eb.tif",
    ]
    assert (settings.save, settings.hfw, settings.filename) == (False, 100e-6, "ref")
