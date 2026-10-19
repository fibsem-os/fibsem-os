"""A wait on a run's stops ends the moment one is set, not at its next poll."""

import threading
import time

from fibsem.cancellation import AnyStopEvent, StopEvent


def test_setting_either_stop_wakes_a_wait_on_both():
    run, task = StopEvent(), StopEvent()
    either = AnyStopEvent(run, task)
    threading.Timer(0.05, task.set).start()

    started = time.monotonic()
    assert either.wait(10)
    assert time.monotonic() - started < 1.0


def test_a_wait_with_nothing_set_times_out_unset():
    assert AnyStopEvent(StopEvent()).wait(0.05) is False


def test_a_stop_set_before_the_wait_returns_at_once():
    stop = StopEvent()
    stop.set()
    assert AnyStopEvent(stop).wait(0) is True


def test_a_wait_reads_a_nested_token():
    run = StopEvent()
    token = AnyStopEvent(run)
    threading.Timer(0.05, run.set).start()
    assert AnyStopEvent(StopEvent(), token).wait(10)
