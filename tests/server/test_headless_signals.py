"""A headless server stops on a terminate signal as on Ctrl-C (headless round
12): a service stop or a Windows close used to leave the run going."""

import os
import signal
import threading

import pytest

from fibsem.server.headless import _stop_on_terminate


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread()
    or not hasattr(signal, "SIGTERM")
    or os.name == "nt",
    reason="signals reach the main thread, and SIGTERM ends a Windows process",
)
def test_a_terminate_signal_stops_serving():
    before = signal.getsignal(signal.SIGTERM)
    stop = threading.Event()
    try:
        _stop_on_terminate(stop)
        os.kill(os.getpid(), signal.SIGTERM)
        assert stop.wait(5)
    finally:
        signal.signal(signal.SIGTERM, before)


def test_off_the_main_thread_nothing_is_installed():
    before = signal.getsignal(signal.SIGTERM)
    worker = threading.Thread(target=_stop_on_terminate, args=(threading.Event(),))
    worker.start()
    worker.join()
    assert signal.getsignal(signal.SIGTERM) is before
