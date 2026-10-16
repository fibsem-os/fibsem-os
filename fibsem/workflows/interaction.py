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

import logging
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as _FutureTimeoutError
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

from fibsem.control import lending
from fibsem.workflows.proposals import DETECTION, Proposal

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
    "ReviewDetection",
    "EditAlignmentArea",
    "PickPOI",
    "RunMillingTask",
    "MillingDecision",
    "MillingEdit",
    "ACTION_RUN",
    "ACTION_EDIT",
    "ACTION_SKIP",
    "ACTION_ACCEPT",
    "RunSpotBurn",
    "SpotBurnDecision",
    "SpotBurnEdit",
    "RunAgainQuestion",
    "LOOP_ACTIONS",
    "ClearSpotBurn",
    "SetImages",
    "SetMillingConfig",
    "ClearMillingConfig",
    "SetFluorescenceChannels",
    "Responder",
    "InvalidAnswerValue",
    "StalePromptError",
    "apply_milling_edit",
    "apply_spot_burn_edit",
    "answered",
    "answered_by",
    "ask",
    "mark_answered_by",
    "proposal_for",
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

    # --- on the record (FIB-1025) ---------------------------------------------
    #
    # A question that carries a value -- where the model put the features, say --
    # can go on the item's record as a proposal, and be answered by a decision on
    # it. These say how it crosses into the record and back. A question with no
    # ``proposal_kind`` is not recorded and stays on the prompt path.

    # The proposal kind it is recorded as.
    proposal_kind: ClassVar[Optional[str]] = None
    # Whether the answer is given *on* the question's image -- a marker dragged,
    # an area drawn. Such a question with no saved image is not recorded at
    # all: a review nobody can see is worse than none, and the prompt path still
    # shows the picture it has in memory. A request carries its context, so a
    # missing image is the asker's defect to fix, not something to paper over.
    needs_image: ClassVar[bool] = False

    def proposal_values(self) -> Dict[str, Any]:
        """What was proposed, in the kind's value names."""
        raise NotImplementedError(f"{type(self).__name__} is not recorded.")

    def answer_from_values(self, values: Dict[str, Any]) -> Any:
        """The decided values as the answer the asker expects -- the same type
        it would have got from a prompt, so the task cannot tell which
        responder answered it."""
        raise NotImplementedError(
            f"{type(self).__name__} carries no value, so no decision answers it."
        )

    def proposal_provenance(self) -> Dict[str, Any]:
        """What proposed this, and anything else about the proposer worth
        keeping. The question's class is what asked, not what answered it -- a
        reader wants the model, and a delta is only comparable against the
        same one."""
        return {}

    def proposal_image(self, folder: str) -> str:
        """The file the values sit on, relative to the item's ``folder``, the
        way every proposal names its reference image. Empty when the question
        has no image on disk."""
        return ""

    def on_answered(self, answer: Any) -> None:
        """Whatever else answering entails -- what the prompt path did on the
        click besides answering, so that a question moved onto the record does
        not quietly stop doing it."""


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
class ReviewDetection(ConfirmDetection):
    """The same question, from an asker that says who it is -- so it can go on
    that item's record and be answered in the Review tab (FIB-1025).

    A type of its own rather than two optional fields on ``ConfirmDetection``:
    that request, the prompt it puts up and the helper that asks it are used as
    they are, in this repository and outside it, and stay exactly that. Asking
    this one instead is a choice a caller makes.

    ``item_id`` and ``task_name`` are the item the detection was made for and
    the task that made it. A responder has no other honest way to learn them:
    "whatever is running" is a guess about the screen, not something the
    request said. A responder that cannot record the question asks it as the
    ``ConfirmDetection`` it also is.
    """

    item_id: str = ""
    task_name: str = ""

    proposal_kind = DETECTION
    needs_image = True

    def proposal_values(self) -> Dict[str, Any]:
        from fibsem.detection.review import detection_values

        return detection_values(self.detection)

    def answer_from_values(self, values: Dict[str, Any]) -> "DetectedFeatures":
        from fibsem.detection.review import decided_features

        return decided_features(self.detection, values)

    def proposal_provenance(self) -> Dict[str, Any]:
        from fibsem.detection.review import detection_provenance

        return detection_provenance(self.detection)

    def proposal_image(self, folder: str) -> str:
        from fibsem.detection.review import detection_image_file

        return detection_image_file(self.detection, folder)

    def on_answered(self, answer: "DetectedFeatures") -> None:
        from fibsem.detection.review import record_training_data

        record_training_data(self.detection, answer)


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


# What can be decided at a milling question (FIB-1297). Each has one meaning,
# whenever it is asked: the yes/no it replaced meant "mill" at the first ask and
# "mill again" after a pass, and an answerer reading one for the other milled twice.
# The answers to a question whose asker runs something and asks again (the mill,
# the spot burn): see :class:`RunAgainQuestion`.
ACTION_RUN = "run"  # run it as it then stands, then ask again
ACTION_EDIT = "edit"  # take it as it then stands and ask again; nothing runs
ACTION_SKIP = "skip"  # go on without running it (before any pass)
ACTION_ACCEPT = "accept"  # go on with what was run (after a pass)
LOOP_ACTIONS = (ACTION_RUN, ACTION_EDIT, ACTION_SKIP, ACTION_ACCEPT)


class RunAgainQuestion:
    """What a run-and-ask-again question shares: the mill's and the spot burn's.

    The asker owns the loop (FIB-1236): it runs on its own thread when the
    answer is ``run``, then asks again with the pass counted and the run's
    error, if it failed. The answer is an action and what was shown as it then
    stood, so whoever answers (the operator in the widget, an agent over the
    App Server) changes what runs the same way: by answering with it. While
    the question stands it lends the run's control, as any question does; the
    run itself is never under the question.

    A subclass is a frozen dataclass with ``message``, ``passes`` (runs for
    this question, failed ones included: 0 at the first ask) and
    ``last_error`` (the last run's error, if it failed), and optionally
    ``enabled`` (False: confirm what is shown; nothing runs). It names its
    buttons and words below.
    """

    lends_control = True
    requires_value = True
    run_label: ClassVar[str] = "Run"
    again_label: ClassVar[str] = "Run Again"
    noun: ClassVar[str] = "run"  # "Press Run Milling to mill"
    without: ClassVar[str] = "without running it"  # "or Skip to go on ..."
    failed_label: ClassVar[str] = "Failed"
    off_hint: ClassVar[str] = "Press Continue when it is right."

    @property
    def _enabled(self) -> bool:
        return getattr(self, "enabled", True)

    @property
    def actions(self) -> Tuple[str, ...]:
        """What may be decided now."""
        if not self._enabled:
            return (ACTION_EDIT, ACTION_ACCEPT)
        if self.passes == 0:
            return (ACTION_RUN, ACTION_EDIT, ACTION_SKIP)
        return (ACTION_RUN, ACTION_EDIT, ACTION_ACCEPT)

    @property
    def positive(self) -> str:
        if not self._enabled:
            return "Continue"
        return self.run_label if self.passes == 0 else self.again_label

    @property
    def negative(self) -> Optional[str]:
        if not self._enabled:
            return None
        return "Skip" if self.passes == 0 else "Accept"

    def label_for(self, action: str) -> str:
        """The button, or what it does, for ``action``: a suggestion's label."""
        if action == ACTION_EDIT:
            return "Apply Edits"
        if action == ACTION_RUN:
            return self.positive
        return self.negative or self.positive

    def decided(self, answer: Any) -> Optional[str]:
        action = getattr(answer, "action", None)
        if action == ACTION_SKIP:
            return f"Skip, {self.without}"
        if action == ACTION_ACCEPT and self.passes:
            return f"Accept: don't {self.noun} again"
        return self.label_for(action) if action else None

    def action_for(self, clicked_yes: bool) -> str:
        """What a button means here."""
        if not self._enabled:
            return ACTION_ACCEPT
        if clicked_yes:
            return ACTION_RUN
        return ACTION_SKIP if self.passes == 0 else ACTION_ACCEPT

    @property
    def prompt(self) -> str:
        """The prompt's text: the asker's message, what the buttons do now, and
        the last run's error."""
        if not self._enabled:
            hint = self.off_hint
        elif self.passes == 0:
            hint = (
                f"Press {self.run_label} to {self.noun}, or Skip to go on "
                f"{self.without}."
            )
        else:
            hint = f"Press {self.again_label} to {self.noun} again, or Accept to go on."
        text = f"{self.message} {hint}"
        if self.last_error:
            return f"{self.failed_label}: {self.last_error}\n\n{text}"
        return text

    def _loop_state(self, off: str, ran: str) -> Dict[str, Any]:
        """The loop's half of ``describe``: the state in words, and the fields."""
        if not self._enabled:
            state = off
        elif self.passes == 0:
            state = "Nothing has run yet."
        else:
            state = f"{ran} {self.passes} time(s)."
            if self.last_error:
                state += f" The last run failed: {self.last_error}"
        return {
            "message": f"{self.message}\n\n{state}",
            "passes": self.passes,
            "last_error": self.last_error,
            "actions": list(self.actions),
            "positive": self.positive,
            "negative": self.negative,
        }

    def _parse_action(self, value: Dict[str, Any], keys: Sequence[str]) -> str:
        """The value's action, checked against what may be decided now, and its
        keys against ``keys`` (besides action)."""
        action = value.get("action")
        if action not in LOOP_ACTIONS:
            raise InvalidAnswerValue(
                f"action is required: one of {list(self.actions)} now."
            )
        if action not in self.actions:
            raise InvalidAnswerValue(
                f"{action!r} can't be decided now: one of {list(self.actions)}."
            )
        unknown = sorted(set(value) - {"action", *keys})
        if unknown:
            raise InvalidAnswerValue(
                f"unknown keys {unknown}: a value has action, {', '.join(keys)}."
            )
        return action

    def response_for(self, value: Any) -> bool:
        """The button the action stands for: yes for the positive one."""
        if not self._enabled:
            return True
        return value.action in (ACTION_RUN, ACTION_EDIT)


@dataclass(frozen=True)
class MillingDecision:
    """The answer to a milling question: what to do, and the config as it stood
    when it was decided. The asker owns the loop: it mills on ``run``, asks
    again on ``run`` and ``edit``, and ends on ``skip`` or ``accept``."""

    action: str
    config: "FibsemMillingTaskConfig"


@dataclass(frozen=True)
class MillingEdit:
    """A remote answer's value at a milling question: the action, and the edits
    to make first. ``config`` (a whole config document) replaces the shown one,
    for structure: stages added, removed, reordered, a pattern type changed.
    ``patch`` then sets fields by dotted path, as ``item_update_task_config``'s
    patch does, with a stage named by its name or its index
    (``stages.Polish 1.pattern.depth``)."""

    action: str
    config: Optional[Dict[str, Any]] = None
    patch: Optional[Dict[str, Any]] = None

    @property
    def changes(self) -> bool:
        return bool(self.config) or bool(self.patch)


def resolve_stage_names(config: "FibsemMillingTaskConfig", patch: Dict[str, Any]):
    """``patch`` with each ``stages.<name>.`` segment as the stage's index, which
    is what :func:`fibsem.server.config_patch.apply_patch` resolves.

    A stage is named by its name or its index. A name no stage has, or more
    than one has, is refused with the names there are."""
    names = [stage.name for stage in config.stages]
    resolved: Dict[str, Any] = {}
    for path, value in patch.items():
        head, sep, rest = str(path).partition(".")
        if head != "stages" or not sep:
            resolved[path] = value
            continue
        segment, sep, tail = rest.partition(".")
        if segment.isdigit():
            resolved[path] = value
            continue
        # A stage name may hold dots: match the longest name the path starts with.
        matches = [
            name for name in names if rest == name or rest.startswith(name + ".")
        ]
        if not matches:
            raise InvalidAnswerValue(
                f"{path!r}: no stage named {segment!r}; the stages are {names}."
            )
        name = max(matches, key=len)
        if names.count(name) > 1:
            raise InvalidAnswerValue(
                f"{path!r}: more than one stage is named {name!r}; name it by "
                "its index instead."
            )
        tail = rest[len(name) :]
        resolved[f"stages.{names.index(name)}{tail}"] = value
    return resolved


def apply_milling_edit(
    config: "FibsemMillingTaskConfig", edit: MillingEdit
) -> "FibsemMillingTaskConfig":
    """``config`` with ``edit``'s changes made: its whole config first, then its
    patch. A new config; ``config`` itself is not touched. Raises
    :class:`InvalidAnswerValue` naming what is wrong, with nothing changed."""
    from copy import deepcopy

    from fibsem.milling.tasks import FibsemMillingTaskConfig, task_overridden_path
    from fibsem.server.config_patch import PatchError, apply_patch

    if edit.config:
        try:
            edited = FibsemMillingTaskConfig.from_dict(deepcopy(edit.config))
        except Exception as exc:  # noqa: BLE001 - any bad document is refused
            raise InvalidAnswerValue(f"config could not be read: {exc}") from None
    else:
        edited = deepcopy(config)
    if edit.patch:
        patch = resolve_stage_names(edited, edit.patch)
        for path in patch:
            note = task_overridden_path(path)
            if note is not None:
                raise InvalidAnswerValue(f"{path!r}: {note}")
        try:
            apply_patch(edited, patch)
        except PatchError as exc:
            raise InvalidAnswerValue(f"{exc.path!r}: {exc}") from None
    return edited


@dataclass(frozen=True)
class RunMillingTask(RunAgainQuestion, Request[MillingDecision]):
    """Ask what to do with a milling config: run it, change it, or go on.

    A :class:`RunAgainQuestion`: the asker mills on the workflow thread
    (FIB-1236), and the answer is a :class:`MillingDecision`, the action and
    the config as the editor then held it. ``enabled`` False is a config to
    confirm without milling it. ``output_dir`` is the asking task's folder,
    which every run writes under.
    """

    value_shape = (
        '{"action": "run" | "edit" | "skip" | "accept", "config": {...}, '
        '"patch": {"<dotted path>": value}}: action is required. run mills the '
        "config as it then stands and asks again; edit makes the changes and asks "
        "again, nothing mills; skip goes on without milling (only before a "
        "pass); accept goes on with what was milled (only after a pass, or when "
        "milling is off). config (optional) is a whole config document, as "
        "served in config, replacing the shown one: for adding, removing or "
        "reordering stages, or changing a pattern type. patch (optional) then "
        "sets fields by dotted path, SI units, as item_update_task_config "
        "does; a stage is named by its name or index, e.g. "
        '"stages.Polish 1.pattern.depth". A field marked not_used in fields '
        "is refused."
    )
    view_frame = VIEW_FRAME_PREFER
    run_label = "Run Milling"
    again_label = "Mill Again"
    noun = "mill"
    without = "without milling"
    failed_label = "Milling failed"
    off_hint = "Press Continue when the config is right."

    config: "FibsemMillingTaskConfig"
    enabled: bool = True
    message: str = "Mill the task."
    output_dir: Optional[str] = None
    passes: int = 0
    last_error: Optional[str] = None

    def describe(self) -> Dict[str, Any]:
        from fibsem.milling.tasks import describe_config_fields

        return {
            **self._loop_state(
                off="Milling is off for this task: confirm the config; nothing mills.",
                ran="The mill has run",
            ),
            "enabled": self.enabled,
            "task_name": self.config.name,
            "config": self.config.to_dict(),
            "fields": describe_config_fields(self.config),
            "coordinates": "pattern points are metres from the image centre, "
            "+x right, +y up; the image shows the patterns as the config "
            "stands",
            "image": None,
        }

    def parse_value(self, value: Dict[str, Any]) -> MillingEdit:
        action = self._parse_action(value, ("config", "patch"))
        config, patch = value.get("config"), value.get("patch")
        if config is not None and not isinstance(config, dict):
            raise InvalidAnswerValue("config is a whole config document (an object).")
        if patch is not None and not isinstance(patch, dict):
            raise InvalidAnswerValue('patch is {"<dotted path>": value}.')
        edit = MillingEdit(action=action, config=config, patch=patch)
        # Checked against the config as asked; the responder makes the same
        # changes to the config as it stands when the answer lands.
        apply_milling_edit(self.config, edit)
        return edit

    def value_to_wire(self, value: Any) -> Dict[str, Any]:
        # The live half: the config as the editor holds it.
        return {"config": value.to_dict()}

    def same_value(self, live: Any, value: Any, frame: Any = None) -> bool:
        if not value.changes:
            return True
        if live is None:
            return False
        try:
            return apply_milling_edit(live, value).to_dict() == live.to_dict()
        except InvalidAnswerValue:
            return False

    def draw(self, frame: "FibsemImage", live: Any) -> Any:
        """The config's patterns over ``frame``, as the milling editor draws
        them: ``live`` is the config as it now stands, else the one asked."""
        from fibsem.milling.patterning.plotting import render_milling_patterns

        config = live if live is not None else self.config
        return render_milling_patterns(frame, config.enabled_stages)


@dataclass(frozen=True)
class SpotBurnDecision:
    """The answer to a spot burn question: what to do, and the settings (points,
    current, exposure) as they stood when it was decided."""

    action: str
    settings: "SpotBurnSettings"


@dataclass(frozen=True)
class SpotBurnEdit:
    """A remote answer's value at a spot burn question: the action, and the
    changes to make first. ``points`` replaces the points; ``milling_current``
    and ``exposure_time`` set the burn's. None leaves each as it stands."""

    action: str
    points: Optional[List["Point"]] = None
    milling_current: Optional[float] = None
    exposure_time: Optional[float] = None

    @property
    def changes(self) -> bool:
        return (
            self.points is not None
            or self.milling_current is not None
            or self.exposure_time is not None
        )


def apply_spot_burn_edit(
    settings: "SpotBurnSettings", edit: SpotBurnEdit
) -> "SpotBurnSettings":
    """``settings`` with ``edit``'s changes made; a new settings."""
    from dataclasses import replace

    changes: Dict[str, Any] = {"coordinates": list(settings.coordinates)}
    if edit.points is not None:
        changes["coordinates"] = list(edit.points)
    if edit.milling_current is not None:
        changes["milling_current"] = edit.milling_current
    if edit.exposure_time is not None:
        changes["exposure_time"] = edit.exposure_time
    return replace(settings, **changes)


@dataclass(frozen=True)
class RunSpotBurn(RunAgainQuestion, Request[Optional[SpotBurnDecision]]):
    """Ask what to do with the spot burn: burn it, change it, or go on.

    A :class:`RunAgainQuestion`: the asker burns on the workflow thread, and
    the answer is a :class:`SpotBurnDecision`, the action and the settings as
    the widget then held them. None without a spot-burn widget: optional
    hardware must not fail a workflow.
    """

    value_shape = (
        '{"action": "run" | "edit" | "skip" | "accept", "points": [{"x", "y"}], '
        '"milling_current": A, "exposure_time": s}: action is required. run '
        "burns the points as they then stand and asks again; edit makes the "
        "changes and asks again, nothing burns; skip goes on without burning "
        "(only before a pass); accept goes on with what was burned (only after "
        "a pass). points (optional) replaces the points: fractions of the frame, "
        "origin top-left, +y down. milling_current (amps) and exposure_time "
        "(seconds per point) are optional."
    )
    view_frame = VIEW_FRAME_FALLBACK
    run_label = "Run Spot Burn"
    again_label = "Burn Again"
    noun = "burn"
    without = "without burning"
    failed_label = "Spot burn failed"

    settings: "SpotBurnSettings"
    message: str = "Run Spot Burn"
    passes: int = 0
    last_error: Optional[str] = None

    def describe(self) -> Dict[str, Any]:
        from fibsem.imaging.spot import EXPOSURE_LIMITS

        settings = self.settings
        return {
            **self._loop_state(off="", ran="The spot burn has run"),
            "points": [pt.to_dict() for pt in getattr(settings, "coordinates", [])],
            "milling_current": getattr(settings, "milling_current", None),
            "exposure_time": getattr(settings, "exposure_time", None),
            "exposure_limits": list(EXPOSURE_LIMITS),
            "coordinates": "points are fractions of the frame (origin "
            "top-left, +y down), so they need no rescaling against the "
            "preview; milling_current is in amps, exposure_time in seconds "
            "per point",
            "image": None,
        }

    def parse_value(self, value: Dict[str, Any]) -> SpotBurnEdit:
        import math

        from fibsem.imaging.spot import EXPOSURE_LIMITS
        from fibsem.services.spot_burn import in_frame
        from fibsem.structures import Point

        action = self._parse_action(
            value, ("points", "milling_current", "exposure_time")
        )
        points = value.get("points")
        parsed: Optional[List[Point]] = None
        if points is not None:
            if not isinstance(points, list):
                raise InvalidAnswerValue(
                    'points is [{"x", "y"}], fractions of the frame.'
                )
            parsed = []
            for entry in points:
                try:
                    pt = Point.from_dict(entry)
                except Exception:
                    raise InvalidAnswerValue(
                        "each spot burn point needs numeric x, y: fractions of "
                        "the frame."
                    ) from None
                if not in_frame(pt):
                    raise InvalidAnswerValue(
                        f"point ({pt.x:.3f}, {pt.y:.3f}) is outside the frame: x "
                        "and y are fractions, 0 to 1."
                    )
                parsed.append(pt)
        numbers: Dict[str, Optional[float]] = {}
        for key in ("milling_current", "exposure_time"):
            number = value.get(key)
            if number is None:
                numbers[key] = None
                continue
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or number <= 0
            ):
                raise InvalidAnswerValue(f"{key} is a positive number.")
            numbers[key] = float(number)
        low, high = EXPOSURE_LIMITS
        exposure = numbers["exposure_time"]
        if exposure is not None and not low <= exposure <= high:
            raise InvalidAnswerValue(
                f"exposure_time {exposure:g} s is outside {low:g} to {high:g} s."
            )
        return SpotBurnEdit(action=action, points=parsed, **numbers)

    def value_to_wire(self, value: Any) -> Dict[str, Any]:
        # The live half: the settings as the widget holds them.
        return {
            "points": [point.to_dict() for point in value.coordinates],
            "milling_current": value.milling_current,
            "exposure_time": value.exposure_time,
        }

    def same_value(self, live: Any, value: Any, frame: Any = None) -> bool:
        import math

        if not value.changes:
            return True
        if live is None:
            return False
        edited = apply_spot_burn_edit(live, value)
        # Points are 0-1 fractions of the frame.
        return (
            len(live.coordinates) == len(edited.coordinates)
            and all(
                abs(a.x - b.x) <= 1e-6 and abs(a.y - b.y) <= 1e-6
                for a, b in zip(live.coordinates, edited.coordinates)
            )
            and math.isclose(live.milling_current, edited.milling_current)
            and math.isclose(live.exposure_time, edited.exposure_time)
        )


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


# --- the record ------------------------------------------------------------------
#
# How a question crosses into the record and back: what it looks like as a
# ``Proposal``, and what a decision on it looks like as the answer the asking
# task expects. Nothing here asks, waits or releases. Whoever owns the wait does
# that -- the responder, which records a question on the way to putting it up,
# not as a second way of asking. Nor does it make the question one that can be
# left for later: the run is still held with the instrument where the task left
# it, and the answer is still needed now; this is only what gets written down.


def proposal_for(request: Request, experiment: Any, item: Any) -> Optional[Proposal]:
    """``request`` as a proposal from the task now running on ``item``, or
    ``None`` when it is not to be recorded and stays on the prompt path: it
    carries no value, or it is answered on an image that was never saved.

    Builds the record, naming the picture it is about; it does not put the
    proposal on the item. That is ``Experiment.ask_proposal``, which is also
    what marks it as the one the run is parked on.
    """
    if request.proposal_kind is None:
        return None
    reference_image = request.proposal_image(str(experiment.item_path(item)))
    if request.needs_image and not reference_image:
        logging.warning(
            f"{type(request).__name__} on {item.name} names no saved image, so it "
            "is not recorded for review; it is asked as a prompt instead."
        )
        return None
    proposal = Proposal(
        kind=request.proposal_kind,
        values=request.proposal_values(),
        provenance={
            # The run the decision must name. Read off the item rather than
            # passed in, so it is the run that is actually in progress --
            # which is also what lets the decision land while it runs.
            "task_id": item.task_state.task_id,
            "proposer": type(request).__name__,
            **request.proposal_provenance(),
        },
    )
    proposal.provenance["reference_image"] = reference_image
    return proposal


def answered(request: Request, answer: Any) -> None:
    """Do whatever else answering ``request`` entails (``on_answered``).

    Never raises. By now the decision is on the record and the answer is on its
    way to the task; an export that fails is worth a log line, not the run.
    """
    try:
        request.on_answered(answer)
    except Exception:
        logging.exception(
            f"{type(request).__name__} was answered, but what follows an answer failed"
        )


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


# Who answered the last question this thread of control asked, as the responder
# marked it on the answer's future. A context variable, as fibsem.acting's mark
# is: the task that asked reads it on its own thread.
_answered_by: "ContextVar[Optional[str]]" = ContextVar(
    "fibsem_answered_by", default=None
)


def mark_answered_by(future: "Future", who: Optional[str]) -> None:
    """Say on ``future`` who is answering it, before completing it: ``"agent"``
    for an answer over the App Server, ``"operator"`` for a click. A responder
    that never says is taken to be the operator."""
    future.answered_by = who  # type: ignore[attr-defined]


def answered_by() -> Optional[str]:
    """Who answered the last question :func:`ask` returned from on this thread
    of control, as the responder marked it; None when it did not say (FIB-1328).

    The task that asked records the decision the answer makes, on its own
    thread, so it cannot read who clicked from the GUI thread that did."""
    return _answered_by.get()


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
    _answered_by.set(getattr(future, "answered_by", None))
    return answer
