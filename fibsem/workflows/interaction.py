"""Typed requests from the workflow thread to whoever is supervising it.

This replaced ``workflow_update_signal``: one ``pyqtSignal(dict)`` that carried
three different mechanisms — fire-and-forget status, one-way instructions the
workflow waited on, and questions that needed a value back. The waiting was a
hand-rolled RPC (a shared mutable flag plus a ``time.sleep`` poll), and answers
came back through unsynchronised attributes on the main window, or by reaching
directly into its widgets. Now every instruction and question is a request here,
each on its own future; status rides ``workflow_status_signal`` as a
``WorkflowStatusEvent``.

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
from contextlib import nullcontext
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
)

from fibsem.control import lending

if TYPE_CHECKING:
    from fibsem.detection.detection import DetectedFeatures
    from fibsem.fm.structures import ChannelSettings
    from fibsem.milling.tasks import FibsemMillingTaskConfig
    from fibsem.structures import FibsemImage, FibsemRectangle, Point

__all__ = [
    "Request",
    "Confirm",
    "ConfirmDetection",
    "EditAlignmentArea",
    "PickPOI",
    "ClearSpotBurn",
    "SetImages",
    "SetMillingConfig",
    "ClearMillingConfig",
    "SetFluorescenceChannels",
    "Responder",
    "InvalidAnswerValue",
    "StalePromptError",
    "ask",
    "wait_for",
]


class StalePromptError(Exception):
    """An answer named a question that is no longer the pending one.

    Raised back to a remote answerer (never to the workflow thread) when the
    question id it echoed does not match the standing question — the prompt it saw
    was answered, withdrawn, or replaced while its answer was in flight.
    """


R = TypeVar("R")

# How often the wait re-checks ``abort`` and the deadline. An answer that is already
# set returns immediately regardless; this only bounds how stale a cancel can be.
_POLL_INTERVAL_S = 0.1


class InvalidAnswerValue(ValueError):
    """A remote answer's value does not fit the question; nothing was answered."""


# Which displayed frame a remote answerer needs beside the question: the FIB view's
# frame, where the widget the question is asked on draws. "fallback" when the
# question carries no image of its own; "prefer" when the view's frame wins over the
# question's, because the view may have moved on since it was asked.
VIEW_FRAME_FALLBACK = "fallback"
VIEW_FRAME_PREFER = "prefer"


class Request(Generic[R]):
    """A single interaction with the supervisor; ``R`` is the type of the answer.

    Subclasses are frozen dataclasses. A question's answer is a value (never a
    widget to read); an instruction subclasses ``Request[None]`` and its "answer"
    is the acknowledgement that the responder has finished acting on it.

    ``lends_control``: while it stands, the run lends its control of the
    microscope (FIB-1270), so whoever answers may act on the instrument first,
    one action at a time. Only questions -- never an instruction, and not the
    spot burn, which the responder runs itself and which takes no control of
    its own yet.

    A question also says everything a remote answerer (an agent over the App
    Server) needs, so the server reads every question the same way:
    :meth:`describe` is what it is told, ``value_shape`` the edit an answer may
    carry (None: none), :meth:`parse_value` checks one, :meth:`value_to_wire`
    writes one back, and ``view_frame`` and :meth:`draw` say which picture goes
    with it. Qt-free: the widget side of each question is the responder's.
    """

    lends_control: ClassVar[bool] = False
    value_shape: ClassVar[Optional[str]] = None
    # A remote answer must carry a value: its action says what the answer means.
    requires_value: ClassVar[bool] = False
    view_frame: ClassVar[Optional[str]] = None

    def decided(self, answer: Any) -> Optional[str]:
        """What *answer* decided, in a few words for a line about it ("Point of
        interest at (+5, −2) µm"); None when the button that answered says it."""
        return None

    def describe(self) -> Dict[str, Any]:
        """What a remote answerer is told: the message, the options and the
        context, as plain values. An ``image`` is left an image (a FibsemImage
        or an array); the server makes a preview of it. A question that says
        nothing more is described by its fields, images left out."""
        import dataclasses

        described: Dict[str, Any] = {}
        if dataclasses.is_dataclass(self):
            for f in dataclasses.fields(self):
                value = getattr(self, f.name)
                if hasattr(value, "data") or type(value).__name__ == "ndarray":
                    continue  # images only travel as previews, never raw
                described[f.name] = value
        return described

    def parse_value(self, value: Dict[str, Any]) -> Any:
        """A remote answer's value as the edit it stands for, checked against
        this question. Raises :class:`InvalidAnswerValue` saying what is wrong."""
        raise InvalidAnswerValue(f"{type(self).__name__} answers carry no value.")

    def value_to_wire(self, value: Any) -> Dict[str, Any]:
        """A value (as :meth:`parse_value` makes one, or the widget holds one)
        in the shape an answer gives it."""
        return value.to_dict()

    def same_value(self, live: Any, value: Any, frame: Any = None) -> bool:
        """Whether ``value`` is what the widget already shows (``live``), so
        answering with it changes nothing. ``frame`` is the image a point is
        placed on, where that matters."""
        return False

    def response_for(self, value: Any) -> Optional[bool]:
        """The button a value's answer stands for, when the value says (an
        action); None when the answer's own yes/no does."""
        return None

    def draw(self, frame: "FibsemImage", live: Any) -> Any:
        """The question drawn over ``frame`` (the view's), as an RGB array, or
        None to show the frame as it is. ``live`` is the widget's value now."""
        return None


# --- questions: the workflow needs a value back -----------------------------------


@dataclass(frozen=True)
class Confirm(Request[bool]):
    """A yes/no prompt. ``True`` means the positive choice was taken."""

    lends_control = True

    message: str
    positive: str = "Continue"
    negative: Optional[str] = None

    def describe(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "positive": self.positive,
            "negative": self.negative,
        }


@dataclass(frozen=True)
class ConfirmDetection(Request["DetectedFeatures"]):
    """Show detected features for correction; answer with the (possibly moved) set."""

    lends_control = True
    value_shape = (
        '{"features": [{"name", "px": {"x", "y"}}]}: source-image pixels, origin '
        "top-left, +y down; only the named features move"
    )

    detection: "DetectedFeatures"

    def describe(self) -> Dict[str, Any]:
        detection = self.detection
        return {
            "message": "Confirm the detected features (answering yes accepts "
            "the positions as currently shown; a value moves the named ones "
            "first).",
            "features": [
                {"name": f.name, "px": f.px.to_dict()}
                for f in getattr(detection, "features", [])
            ],
            "pixelsize": getattr(detection, "pixelsize", None),
            "coordinates": "feature px are source-image pixels (origin "
            "top-left, +y down); pixelsize is metres per source pixel",
            "image": getattr(detection, "image", None),
        }

    def parse_value(self, value: Dict[str, Any]) -> Dict[str, "Point"]:
        """``{name: Point}`` in source-image pixels: only features the detection
        has, each inside the image."""
        from fibsem.structures import Point

        names = [f.name for f in getattr(self.detection, "features", [])]
        features = value.get("features")
        if not isinstance(features, list) or not features:
            raise InvalidAnswerValue(
                f"a {type(self).__name__} value is {self.value_shape}."
            )
        image = getattr(self.detection, "image", None)
        shape = getattr(getattr(image, "data", image), "shape", None)
        moved: Dict[str, Point] = {}
        for entry in features:
            name = entry.get("name") if isinstance(entry, dict) else None
            if name not in names:
                raise InvalidAnswerValue(
                    f"no detected feature named {name!r}; the features are {names}."
                )
            if name in moved:
                raise InvalidAnswerValue(f"feature {name!r} is given twice.")
            try:
                px = Point.from_dict(entry["px"])
            except Exception:
                raise InvalidAnswerValue(
                    f"feature {name!r} needs px: {{x, y}}, numbers in "
                    "source-image pixels."
                ) from None
            if shape is not None and len(shape) >= 2:
                height, width = shape[0], shape[1]
                if not (0 <= px.x < width and 0 <= px.y < height):
                    raise InvalidAnswerValue(
                        f"feature {name!r} at ({px.x:.1f}, {px.y:.1f}) px is "
                        f"outside the {width} x {height} px image."
                    )
            moved[name] = px
        return moved

    def value_to_wire(self, value: Dict[str, "Point"]) -> Dict[str, Any]:
        return {
            "features": [
                {"name": name, "px": px.to_dict()} for name, px in value.items()
            ]
        }

    def same_value(self, live: Any, value: Any, frame: Any = None) -> bool:
        # Features are in image pixels: half a pixel, as for a point.
        return live is not None and all(
            live.get(name) is not None
            and abs(live[name].x - px.x) <= 0.5
            and abs(live[name].y - px.y) <= 0.5
            for name, px in value.items()
        )


@dataclass(frozen=True)
class EditAlignmentArea(Request["FibsemRectangle"]):
    """Let the supervisor adjust an alignment area; answer with the final area.

    The area is fractions of the FIB frame; the frame itself is display state
    the workflow does not own, so it does not travel in the request — the
    App Server attaches the FIB display image when it serializes the
    question (the same seam that serves ``/app/experiment/images``).
    """

    lends_control = True
    value_shape = (
        "{left, top, width, height}: fractions of the frame, origin top-left, "
        "inside the frame"
    )
    view_frame = VIEW_FRAME_FALLBACK

    initial: "FibsemRectangle"
    message: str = "Edit Alignment Area"

    def decided(self, answer: Any) -> Optional[str]:
        try:
            width, height = float(answer.width), float(answer.height)
        except (AttributeError, TypeError, ValueError):
            return "Alignment area set"
        return f"Alignment area {width:.0%} × {height:.0%} of the frame"

    def describe(self) -> Dict[str, Any]:
        return {
            "message": "Confirm the alignment area (answering yes accepts "
            "the area as currently shown).",
            "initial": self.initial.to_dict() if self.initial is not None else None,
            "coordinates": "areas are fractions of the frame (origin "
            "top-left), so they need no rescaling against the preview",
            "image": None,
        }

    def parse_value(self, value: Dict[str, Any]) -> "FibsemRectangle":
        from fibsem.structures import FibsemRectangle

        try:
            rect = FibsemRectangle.from_dict(value)
        except Exception:
            raise InvalidAnswerValue(
                f"an EditAlignmentArea value is {self.value_shape}."
            ) from None
        if not rect.is_valid_reduced_area:
            raise InvalidAnswerValue(
                "alignment area out of bounds: left/top must be >= 0, "
                "width/height > 0, and the rectangle must fit inside "
                "the frame (left+width <= 1, top+height <= 1)."
            )
        return rect

    def same_value(self, live: Any, value: Any, frame: Any = None) -> bool:
        return live is not None and all(
            abs(getattr(live, f) - getattr(value, f)) <= 1e-6
            for f in ("left", "top", "width", "height")
        )


@dataclass(frozen=True)
class PickPOI(Request[Optional["Point"]]):
    """Ask for a point of interest, in microscope image coordinates.

    ``image`` is the FIB frame the task took for it. The marker sits on the FIB
    view, so once the view has moved on (an agent's approved moves and their
    images), the responder places, reads and serves it on the view's frame
    instead, with ``image`` the fallback (FIB-1399).
    """

    lends_control = True
    value_shape = "{x, y}: metres in microscope image coordinates, origin centre, +y up"
    view_frame = VIEW_FRAME_PREFER

    image: "FibsemImage"
    initial: Optional["Point"] = None
    message: str = "Select Point of Interest"

    def decided(self, answer: Any) -> Optional[str]:
        if answer is None:
            return "No point of interest"
        try:
            x, y = float(answer.x) * 1e6, float(answer.y) * 1e6
        except (AttributeError, TypeError, ValueError):
            return "Point of interest set"
        return f"Point of interest at ({x:+.1f}, {y:+.1f}) µm".replace("-", "−")

    def describe(self) -> Dict[str, Any]:
        return {
            "message": "Pick the point of interest on the image.",
            "initial": self.initial.to_dict() if self.initial is not None else None,
            "coordinates": "points are microscope image coordinates: "
            "metres, origin at the image centre, +x right, +y UP",
            "image": self.image,
        }

    def parse_value(self, value: Dict[str, Any]) -> "Point":
        from fibsem.structures import Point

        try:
            point = Point.from_dict(value)
        except Exception:
            raise InvalidAnswerValue(
                f"a PickPOI value is {self.value_shape}."
            ) from None
        try:
            pixel_size = self.image.metadata.pixel_size.x
            half_width = self.image.data.shape[1] / 2 * pixel_size
            half_height = self.image.data.shape[0] / 2 * pixel_size
        except Exception:
            raise InvalidAnswerValue(
                "the pending question's image is missing its scale; "
                "the point cannot be validated."
            ) from None
        if abs(point.x) > half_width or abs(point.y) > half_height:
            raise InvalidAnswerValue(
                f"point ({point.x:.3e}, {point.y:.3e}) m is outside the "
                f"image: |x| <= {half_width:.3e}, |y| <= {half_height:.3e}."
            )
        return point

    def same_value(self, live: Any, value: Any, frame: Any = None) -> bool:
        # A point within half a pixel of the frame the marker sits on is the
        # same point: the marker cannot sit any closer.
        if live is None:
            return False
        frame = frame if frame is not None else self.image
        tolerance = frame.metadata.pixel_size.x / 2
        return abs(live.x - value.x) <= tolerance and abs(live.y - value.y) <= tolerance


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
class ClearSpotBurn(Request[None]):
    """Clear the spot burn widget's points and leave workflow mode."""


@dataclass(frozen=True)
class SetFluorescenceChannels(Request[None]):
    """Load these channel settings into the fluorescence widget."""

    channels: Sequence["ChannelSettings"]


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
    # Lent before the question is put, so nobody who sees it is refused.
    with lending() if request.lends_control else nullcontext():
        responder.submit(request, future)
        answer = wait_for(
            future, abort=abort, timeout=timeout, description=type(request).__name__
        )
    return answer
