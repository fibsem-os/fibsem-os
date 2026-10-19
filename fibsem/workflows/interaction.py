"""Typed instructions from the workflow thread to the window supervising it.

Each instruction (show these images, load this milling config) is a request on its
own future; the workflow waits for the acknowledgement. Status rides
``workflow_status_signal`` as a ``WorkflowStatusEvent``, and what the run puts to
someone is a proposal on the record (``fibsem.workflows.proposals``), not a request.

* **:class:`Responder`** — the transport is a protocol with one method, not a Qt
  signal. The workflow layer needs no Qt and a test needs no window.
* **:func:`ask` / :func:`wait_for`** — a ``concurrent.futures.Future`` per call
  carries the acknowledgement, the error and the fact of completion. The wait polls
  in 0.1 s slices, which bounds how fast an abort is noticed.

A request carries its context: a responder must be able to act on it from the
request alone. Instructions are answered by a machine, so silence means something is
broken: pass a ``timeout``. Cancellation raises ``InterruptedError``, the same
exception the workflow abort path already raises.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as _FutureTimeoutError
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
)

if TYPE_CHECKING:
    from fibsem.detection.detection import DetectedFeatures
    from fibsem.fm.structures import ChannelSettings
    from fibsem.milling.tasks import FibsemMillingTaskConfig
    from fibsem.structures import FibsemImage

__all__ = [
    "Request",
    "ClearSpotBurn",
    "SetImages",
    "SetMillingConfig",
    "ClearMillingConfig",
    "SetFluorescenceChannels",
    "Responder",
    "ask",
    "wait_for",
]


R = TypeVar("R")

# How often the wait re-checks ``abort`` and the deadline. An answer that is already
# set returns immediately regardless; this only bounds how stale a cancel can be.
_POLL_INTERVAL_S = 0.1


class Request(Generic[R]):
    """A single instruction to the supervisor; ``R`` is the type of its answer.

    Subclasses are frozen dataclasses. An instruction subclasses ``Request[None]``
    and its "answer" is the acknowledgement that the responder has finished
    acting on it.
    """


# --- instructions: the workflow waits for the acknowledgement ---------------------


@dataclass(frozen=True)
class SetImages(Request[None]):
    """Display these acquisition images."""

    sem_image: Optional["FibsemImage"] = None
    fib_image: Optional["FibsemImage"] = None


@dataclass(frozen=True)
class SetMillingConfig(Request[None]):
    """Load this config into the milling editor."""

    config: "FibsemMillingTaskConfig"


@dataclass(frozen=True)
class ClearMillingConfig(Request[None]):
    """Clear the milling editor."""


@dataclass(frozen=True)
class ShowDetection(Request[None]):
    """Put what the model found on the detection widget, for the ``detection``
    proposal asked on the record next."""

    detection: "DetectedFeatures"


@dataclass(frozen=True)
class ClearSpotBurn(Request[None]):
    """Clear the spot burn widget's points and leave workflow mode."""


@dataclass(frozen=True)
class SetFluorescenceChannels(Request[None]):
    """Load these channel settings into the fluorescence widget."""

    channels: Sequence["ChannelSettings"]


# --- transport --------------------------------------------------------------------


# Instructions are answered by a machine, so silence past this is a wedged GUI
# thread, not thinking. Generous because the GUI thread may legitimately be busy
# painting; the real waits are milliseconds.
INSTRUCTION_TIMEOUT_S = 30


def _abort_requested(run: Optional[Any]) -> bool:
    """Whether the workflow has been asked to stop. Never raises.

    Asks the run's manager (or the session's runner, which asks it): the same
    predicate a task's ``_check_for_abort`` uses, so the two cannot disagree
    about what counts as cancelled. Also the ``abort`` predicate for
    :func:`ask`.
    """
    return bool(getattr(run, "should_abort", False))


class Responder(Protocol):
    """Whoever answers requests: the Qt window when supervised, a stub in tests.

    ``submit`` is called on the workflow thread and must not block — hand the request
    off (to the GUI thread, to a queue) and return. Every accepted future must
    eventually be completed: ``set_result`` with the answer, or ``set_exception`` if
    acting on the request failed. An exception set here re-raises on the workflow
    thread inside :func:`wait_for`, where a real handler exists — instead of escaping
    a Qt slot and taking the process down.
    """

    def submit(self, request: Request[R], future: "Future[R]") -> None:
        """Deliver ``request``, later completing ``future`` with its answer."""
        ...


def wait_for(
    future: "Future[R]",
    *,
    abort: Optional[Callable[[], bool]] = None,
    timeout: Optional[float] = None,
    description: str = "UI request",
) -> R:
    """Block until ``future`` completes, staying responsive to cancellation.

    Returns the answer, or re-raises whatever the responder set. Raises
    ``InterruptedError`` once ``abort()`` goes true, and ``TimeoutError`` once
    ``timeout`` seconds pass — an answer that is already set always wins over both.
    ``abort`` is a predicate rather than an event because the real source is
    ``task_manager.should_abort``, a property.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            return future.result(timeout=_POLL_INTERVAL_S)
        except _FutureTimeoutError:
            if abort is not None and abort():
                # Cancel before leaving: nobody will read the acknowledgement.
                # The race where the responder completes it first is benign.
                future.cancel()
                raise InterruptedError(f"{description} cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                future.cancel()
                raise TimeoutError(f"No response to {description} within {timeout} s")


def ask(
    responder: Responder,
    request: Request[R],
    *,
    abort: Optional[Callable[[], bool]] = None,
    timeout: Optional[float] = None,
) -> R:
    """Submit ``request`` to ``responder`` and wait for the answer.

    The future is created here and shared with nobody else, so only this
    request's responder can complete it.
    """
    future: "Future[R]" = Future()
    responder.submit(request, future)
    return wait_for(
        future, abort=abort, timeout=timeout, description=type(request).__name__
    )
