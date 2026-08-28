"""The request/answer primitive is headless: no Qt, no window, no microscope.

These tests pin the properties the design leans on — the latch, exception
propagation, the abort predicate, the timeout, and answer-beats-abort — so the
first safety net for this subsystem runs in plain ``tests/``, which CI has always
run, rather than only in ``tests/ui``.
"""

import dataclasses
import threading
import time

import pytest

from fibsem.applications.autolamella.workflows.interaction import (
    Confirm,
    SetImages,
    ask,
)


class InlineResponder:
    """Answers on the calling thread, inside ``submit`` — before the wait begins."""

    def __init__(self, answer=None, error=None):
        self.requests = []
        self._answer = answer
        self._error = error

    def submit(self, request, future):
        self.requests.append(request)
        if self._error is not None:
            future.set_exception(self._error)
        else:
            future.set_result(self._answer)


class ThreadResponder:
    """Answers from another thread after a delay — the wait must actually wait."""

    def __init__(self, answer, delay_s):
        self._answer = answer
        self._delay_s = delay_s

    def submit(self, request, future):
        def _answer_later():
            time.sleep(self._delay_s)
            future.set_result(self._answer)

        threading.Thread(target=_answer_later, daemon=True).start()


class SilentResponder:
    """Accepts the request and never completes the future."""

    def submit(self, request, future):
        pass


def test_an_inline_answer_comes_straight_back():
    responder = InlineResponder(answer=True)
    request = Confirm("Continue?")

    assert ask(responder, request) is True
    assert responder.requests == [request]


def test_a_deferred_answer_is_waited_for():
    # The delay spans more than one 0.1 s poll slice, so a pass proves the loop
    # re-polls rather than answering only when the result is already set.
    responder = ThreadResponder(answer="landed", delay_s=0.25)
    started = time.monotonic()

    assert ask(responder, Confirm("Continue?")) == "landed"
    assert time.monotonic() - started >= 0.2


def test_a_responder_failure_reraises_on_the_caller():
    boom = RuntimeError("widget fell over")
    responder = InlineResponder(error=boom)

    with pytest.raises(RuntimeError) as excinfo:
        ask(responder, SetImages())
    assert excinfo.value is boom


def test_silence_times_out_with_the_request_named():
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="SetImages"):
        ask(SilentResponder(), SetImages(), timeout=0.3)
    assert time.monotonic() - started < 3


def test_abort_interrupts_the_wait():
    started = time.monotonic()

    with pytest.raises(InterruptedError):
        ask(SilentResponder(), Confirm("Continue?"), abort=lambda: True)
    assert time.monotonic() - started < 3


def test_an_answer_already_set_beats_abort():
    # Both an answer and an abort are available on the first poll; the answer wins,
    # so a Stop pressed just as the operator answers cannot eat the answer.
    responder = InlineResponder(answer=False)

    assert ask(responder, Confirm("Continue?"), abort=lambda: True) is False


def test_requests_are_immutable():
    request = Confirm("Continue?")

    with pytest.raises(dataclasses.FrozenInstanceError):
        request.message = "changed"
