"""The imaging service (`microscope.imaging`, FIB-1261) on the Demo: built over the
beams, ``auto_focus`` choosing the instrument's own autofocus or the sweep, and
``focus_sweep`` with its field of view, its stop and its result."""

import os
import threading

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
