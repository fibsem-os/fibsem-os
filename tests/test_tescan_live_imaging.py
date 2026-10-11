"""Tests for TESCAN live image acquisition (``TescanBeam._live``).

TESCAN has no continuous/streaming API, so live imaging is a re-acquire loop that emits
each frame on the beam's ``live_frame`` until it is stopped. The shared ``Beam`` runs the
loop on its thread and forwards frames to the microscope's old signals; that, and the
SEM/FIB signal each beam reaches, are covered over the fake SDK in
``tests/test_tescan_imaging.py``. These pin the loop itself.

No hardware or Tescan SDK required: the loop runs on a stand-in for the beam whose
acquire is stubbed.
"""

import threading
from types import SimpleNamespace

from psygnal import Signal

from fibsem.drivers.tescan.devices import TescanBeam


class _Frames:
    live_frame = Signal(object)


def make_beam(side_effect=None):
    """A stand-in with what ``_live`` uses: ``acquire`` and ``live_frame``."""
    beam = _Frames()
    beam.calls = 0
    beam.frames = []
    beam.live_frame.connect(beam.frames.append)

    def acquire():
        beam.calls += 1
        if side_effect is not None:
            side_effect(beam.calls)
        return SimpleNamespace(n=beam.calls)

    beam.acquire = acquire
    return beam


def test_a_stop_set_before_it_starts_acquires_nothing():
    beam, stop = make_beam(), threading.Event()
    stop.set()
    TescanBeam._live(beam, stop)
    assert beam.calls == 0


def test_it_acquires_and_emits_until_stopped():
    stop = threading.Event()

    def stop_after(n):
        if n > 5:  # the acquisition that trips the stop is not emitted
            stop.set()

    beam = make_beam(stop_after)
    TescanBeam._live(beam, stop)
    assert beam.calls == 6
    assert [frame.n for frame in beam.frames] == [1, 2, 3, 4, 5]


def test_no_frame_is_emitted_when_stopped_mid_iteration():
    stop = threading.Event()
    beam = make_beam(lambda n: stop.set())
    TescanBeam._live(beam, stop)
    assert beam.calls == 1  # acquired once
    assert beam.frames == []  # but the stop check dropped it before emit


def test_an_acquire_error_ends_the_loop_without_raising():
    def boom(n):
        raise RuntimeError("socket died")

    beam = make_beam(boom)
    # must not propagate out of the loop (it runs on a daemon thread)
    TescanBeam._live(beam, threading.Event())
    assert beam.calls == 1
