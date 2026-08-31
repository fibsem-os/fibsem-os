"""Typed requests from the workflow thread to whoever is supervising it.

``workflow_update_signal`` is one ``pyqtSignal(dict)`` carrying three different
mechanisms: fire-and-forget status, one-way instructions the workflow waits on, and
questions that need a value back. The waiting is a hand-rolled RPC — a shared mutable
flag plus a ``time.sleep`` poll — and the answers come back through unsynchronised
attributes on the main window, or by reaching directly into its widgets.

This module is the replacement's foundation, deliberately unwired: nothing imports it
yet, and each call site converts on its own.

* **Payload types** — one dataclass per interaction, split by mechanism. A question
  declares its answer type through ``Request[R]``; an instruction is ``Request[None]``,
  completed with a bare acknowledgement. One type per question rather than one type
  with optional fields: which-field-is-set dispatch is the shape being removed.
* **:class:`Responder`** — the transport is a protocol with one method, not a Qt
  signal. The workflow layer needs no Qt, a test needs no window, and headless mode
  stops being ``if parent_ui is None`` at the top of every helper and becomes just
  another implementation.
* **:func:`ask` / :func:`wait_for`** — a ``concurrent.futures.Future`` per call
  carries the answer, the error and the fact of completion. It belongs to one caller,
  so no other emitter can release it — unlike ``WAITING_FOR_UI_UPDATE``, which any
  emission clears for every waiter. The wait polls in 0.1 s slices; that bounds how
  fast an abort is noticed, not how fast an answer arrives.

Two rules keep the types honest, one in each direction:

* **A request carries its context.** A responder must be able to answer from the
  request alone. If a request only works because one particular responder can see
  what happens to be on screen, it is under-specified.
* **A response carries the value.** Never a pointer into the UI for the caller to
  read back afterwards.

Timeouts scale with who answers. Instructions are answered by a machine, so silence
means something is broken: pass a ``timeout``. Questions are answered by a human, so
silence means thinking: no timeout, but pass ``abort`` so Stop still works while the
prompt is up. Cancellation raises ``InterruptedError`` — the same exception the
workflow abort path already raises — so a cancelled request unwinds through the one
path call sites already have, rather than adding a second.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as _FutureTimeoutError
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
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
    from fibsem.imaging.spot import SpotBurnSettings
    from fibsem.milling.tasks import FibsemMillingTaskConfig
    from fibsem.structures import FibsemImage, FibsemRectangle, Point

__all__ = [
    "Request",
    "Confirm",
    "ConfirmDetection",
    "EditAlignmentArea",
    "PickPOI",
    "RunMillingTask",
    "SetImages",
    "SetMillingConfig",
    "ClearMillingConfig",
    "SetFluorescenceChannels",
    "SetSpotBurnSettings",
    "ClearSpotBurn",
    "Responder",
    "ask",
    "wait_for",
]

R = TypeVar("R")

# How often the wait re-checks ``abort`` and the deadline. An answer that is already
# set returns immediately regardless; this only bounds how stale a cancel can be.
_POLL_INTERVAL_S = 0.1


class Request(Generic[R]):
    """A single interaction with the supervisor; ``R`` is the type of the answer.

    Subclasses are frozen dataclasses. A question's answer is a value (never a
    widget to read); an instruction subclasses ``Request[None]`` and its "answer"
    is the acknowledgement that the responder has finished acting on it.
    """


# --- questions: the workflow needs a value back -----------------------------------


@dataclass(frozen=True)
class Confirm(Request[bool]):
    """A yes/no prompt. ``True`` means the positive choice was taken."""

    message: str
    positive: str = "Continue"
    negative: Optional[str] = None


@dataclass(frozen=True)
class ConfirmDetection(Request["DetectedFeatures"]):
    """Show detected features for correction; answer with the (possibly moved) set."""

    detection: "DetectedFeatures"


@dataclass(frozen=True)
class EditAlignmentArea(Request["FibsemRectangle"]):
    """Let the supervisor adjust an alignment area; answer with the final area."""

    initial: "FibsemRectangle"
    message: str = "Edit Alignment Area"


@dataclass(frozen=True)
class PickPOI(Request[Optional["Point"]]):
    """Ask for a point of interest on ``image``, in microscope image coordinates.

    The image travels in the request — in-process that is a reference, not a copy —
    so the responder is never answering about whatever happens to be on screen.
    """

    image: "FibsemImage"
    initial: Optional["Point"] = None
    message: str = "Select Point of Interest"


@dataclass(frozen=True)
class RunMillingTask(Request["FibsemMillingTaskConfig"]):
    """Hand over a milling config to edit and (if ``enabled``) run; answer with the
    config as actually used. Absorbs today's ``start_milling_signal`` sibling poll:
    running the mill is the responder's job, not a second channel's.

    ``confirm`` is the supervision mode: True shows the prompt (Run Milling reruns
    after edits, Continue ends the question); False runs once unprompted when
    ``enabled``, or just delivers the config back when not.
    """

    config: "FibsemMillingTaskConfig"
    enabled: bool = True
    confirm: bool = True
    message: str = "Run Milling"


# --- instructions: one-way, answered with a bare acknowledgement ------------------


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
class SetFluorescenceChannels(Request[None]):
    """Load these channel settings into the fluorescence widget."""

    channels: Sequence["ChannelSettings"]


@dataclass(frozen=True)
class SetSpotBurnSettings(Request[None]):
    """Load these spot-burn settings into the spot-burn widget."""

    settings: "SpotBurnSettings"


@dataclass(frozen=True)
class ClearSpotBurn(Request[None]):
    """Clear the spot-burn widget."""


# --- transport --------------------------------------------------------------------


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
                # Cancel before leaving: this tells the responder nobody will
                # read an answer, so it must not act for this question again —
                # concretely, a Stop mid-mill must not resurrect the prompt when
                # the cancelled mill finishes. cancel() always succeeds here
                # (the future has no runner), and the race where the responder
                # completes it first is benign: the answer is simply dropped.
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

    The one call sites use, for questions and instructions alike — an instruction is
    simply asked for its acknowledgement. The future is created here and shared with
    nobody else, so only this request's responder can complete it.
    """
    future: "Future[R]" = Future()
    responder.submit(request, future)
    return wait_for(
        future, abort=abort, timeout=timeout, description=type(request).__name__
    )
