"""Tests for TESCAN live image acquisition (_acquisition_worker).

TESCAN has no continuous/streaming API, so live imaging is a re-acquire loop that emits each
frame on the beam's acquisition signal until stop_acquisition() sets the stop event. The base
FibsemMicroscope.start_acquisition/stop_acquisition drive the thread; only the worker is
TESCAN-specific.

No hardware or Tescan SDK required: the microscope is created without __init__ and acquire_image
is stubbed.
"""

import threading
import time
from typing import List

import pytest

from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import BeamType, FibsemImage


def make_microscope(acquire_side_effect=None):
    """Create a TescanMicroscope whose acquire_image returns a sentinel per beam."""
    m = object.__new__(TescanMicroscope)
    # per-instance event/thread so tests don't share the class-level defaults
    m._stop_acquisition_event = threading.Event()
    m._acquisition_thread = None

    calls = {"n": 0, "beams": []}
    m._test_calls = calls

    def fake_acquire(image_settings=None, beam_type=None):
        calls["n"] += 1
        calls["beams"].append(beam_type)
        if acquire_side_effect is not None:
            acquire_side_effect(calls["n"])
        img = object.__new__(FibsemImage)
        img._beam_type = beam_type  # tag so emitted frames are identifiable
        return img

    m.acquire_image = fake_acquire
    return m


def collect(signal) -> List:
    received: List = []
    signal.connect(lambda img: received.append(img))
    return received


def _wrap_with_side_effect(fn, side_effect):
    """Wrap acquire_image so `side_effect(call_number)` runs after each call."""
    state = {"n": 0}

    def wrapped(image_settings=None, beam_type=None):
        state["n"] += 1
        img = fn(image_settings=image_settings, beam_type=beam_type)
        side_effect(state["n"])
        return img

    return wrapped


def run_worker_briefly(m, beam_type, frames=3):
    """Run the worker until at least `frames` frames are emitted, then stop and join.

    The worker checks the stop event between acquire and emit, so the acquisition that
    trips the stop is deliberately not emitted; stop one acquire later so `frames` emit.
    """
    def stop_after(n):
        if n > frames:
            m._stop_acquisition_event.set()

    m.acquire_image = _wrap_with_side_effect(m.acquire_image, stop_after)
    m._acquisition_thread = threading.Thread(
        target=m._acquisition_worker, args=(beam_type,), daemon=True
    )
    m._acquisition_thread.start()
    m._acquisition_thread.join(timeout=5)
    assert not m._acquisition_thread.is_alive(), "worker did not stop"


# --------------------------------------------------------------------------

def test_worker_stops_when_event_is_set_before_start():
    """A stop event set up-front means zero acquisitions."""
    m = make_microscope()
    m._stop_acquisition_event.set()

    m._acquisition_worker(BeamType.ELECTRON)

    assert m._test_calls["n"] == 0


def test_electron_frames_go_to_the_sem_signal():
    m = make_microscope()
    sem = collect(m.sem_acquisition_signal)
    fib = collect(m.fib_acquisition_signal)

    run_worker_briefly(m, BeamType.ELECTRON, frames=3)

    assert len(sem) >= 3
    assert len(fib) == 0
    assert all(img._beam_type is BeamType.ELECTRON for img in sem)


def test_ion_frames_go_to_the_fib_signal():
    m = make_microscope()
    sem = collect(m.sem_acquisition_signal)
    fib = collect(m.fib_acquisition_signal)

    run_worker_briefly(m, BeamType.ION, frames=3)

    assert len(fib) >= 3
    assert len(sem) == 0
    assert all(img._beam_type is BeamType.ION for img in fib)


def test_worker_acquires_repeatedly_until_stopped():
    m = make_microscope()

    run_worker_briefly(m, BeamType.ELECTRON, frames=5)

    assert m._test_calls["n"] >= 5


def test_no_frame_emitted_when_stopped_mid_iteration():
    """If the stop event is set between acquire and emit, that frame is not emitted."""
    m = make_microscope()
    sem = collect(m.sem_acquisition_signal)

    # stop as soon as the first acquire returns, before the emit check
    def stop_immediately(n):
        m._stop_acquisition_event.set()

    m.acquire_image = _wrap_with_side_effect(m.acquire_image, stop_immediately)
    m._acquisition_worker(BeamType.ELECTRON)

    assert m._test_calls["n"] == 1  # acquired once
    assert len(sem) == 0            # but the stop check dropped it before emit


def test_worker_survives_acquire_errors_without_raising():
    def boom(n):
        raise RuntimeError("socket died")

    m = make_microscope(acquire_side_effect=boom)

    # must not propagate out of the worker (it runs on a daemon thread)
    m._acquisition_worker(BeamType.ELECTRON)

    assert m._test_calls["n"] == 1


def test_start_and_stop_acquisition_drive_the_worker():
    """End-to-end through the base start/stop machinery."""
    m = make_microscope()
    sem = collect(m.sem_acquisition_signal)

    m.start_acquisition(BeamType.ELECTRON)
    assert m.is_acquiring

    # let a few frames accumulate
    deadline = time.monotonic() + 3
    while len(sem) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)

    m.stop_acquisition()
    assert not m.is_acquiring
    assert len(sem) >= 3
