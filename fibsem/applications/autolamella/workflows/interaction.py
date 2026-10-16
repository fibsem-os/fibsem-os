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
    "MILL_RUN",
    "MILL_EDIT",
    "MILL_SKIP",
    "MILL_ACCEPT",
    "RunSpotBurn",
    "SetImages",
    "SetMillingConfig",
    "ClearMillingConfig",
    "SetFluorescenceChannels",
    "Responder",
    "InvalidAnswerValue",
    "StalePromptError",
    "apply_milling_edit",
    "answered_by",
    "ask",
    "mark_answered_by",
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
MILL_RUN = "run"  # mill the config as it then stands, then ask again
MILL_EDIT = "edit"  # take the config as it then stands and ask again; nothing mills
MILL_SKIP = "skip"  # go on without milling (before any pass)
MILL_ACCEPT = "accept"  # go on with what was milled (after a pass)
MILL_ACTIONS = (MILL_RUN, MILL_EDIT, MILL_SKIP, MILL_ACCEPT)


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
class RunMillingTask(Request[MillingDecision]):
    """Ask what to do with a milling config: run it, change it, or go on.

    The asker owns the mill and the loop (FIB-1236): it mills on the workflow
    thread when the answer is ``run``, and asks again with the pass counted and
    the run's error, if it failed. The responder only shows the config, takes
    the edits, and answers with a :class:`MillingDecision` -- the action and
    the config as it then stood. So what is milled is always the asker's, and
    whoever answers (the operator in the editor, an agent over the App Server)
    changes it the same way: by answering with it.

    ``passes`` is how many times the mill has run for this question, failed runs
    included: 0 at the first ask. ``last_error`` is the last run's error, if it
    failed. ``enabled`` False is a config to confirm without milling it.

    While the question stands, the run lends its control of the microscope, as
    any other question does: whoever answers may image, move and re-align
    between passes. The mill itself is not under the question.

    ``output_dir`` is the asking task's folder, which every run writes under.
    """

    lends_control = True
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
    requires_value = True
    view_frame = VIEW_FRAME_PREFER

    config: "FibsemMillingTaskConfig"
    enabled: bool = True
    message: str = "Mill the task."
    output_dir: Optional[str] = None
    passes: int = 0
    last_error: Optional[str] = None

    @property
    def actions(self) -> Tuple[str, ...]:
        """What may be decided now."""
        if not self.enabled:
            return (MILL_EDIT, MILL_ACCEPT)
        if self.passes == 0:
            return (MILL_RUN, MILL_EDIT, MILL_SKIP)
        return (MILL_RUN, MILL_EDIT, MILL_ACCEPT)

    @property
    def positive(self) -> str:
        if not self.enabled:
            return "Continue"
        return "Run Milling" if self.passes == 0 else "Mill Again"

    @property
    def negative(self) -> Optional[str]:
        if not self.enabled:
            return None
        return "Skip" if self.passes == 0 else "Accept"

    def label_for(self, action: str) -> str:
        """The button, or what it does, for ``action``: a suggestion's label."""
        if action == MILL_EDIT:
            return "Apply Edits"
        if action == MILL_RUN:
            return self.positive
        return self.negative or self.positive

    def action_for(self, clicked_yes: bool) -> str:
        """What a button means here."""
        if not self.enabled:
            return MILL_ACCEPT
        if clicked_yes:
            return MILL_RUN
        return MILL_SKIP if self.passes == 0 else MILL_ACCEPT

    @property
    def prompt(self) -> str:
        """The prompt's text: the asker's message, what the buttons do now, and
        the last run's error."""
        if not self.enabled:
            hint = "Press Continue when the config is right."
        elif self.passes == 0:
            hint = "Press Run Milling to mill, or Skip to go on without milling."
        else:
            hint = "Press Mill Again to mill again, or Accept to go on."
        text = f"{self.message} {hint}"
        if self.last_error:
            return f"Milling failed: {self.last_error}\n\n{text}"
        return text

    def describe(self) -> Dict[str, Any]:
        from fibsem.milling.tasks import describe_config_fields

        if not self.enabled:
            state = "Milling is off for this task: confirm the config; nothing mills."
        elif self.passes == 0:
            state = "Nothing has been milled yet."
        else:
            state = f"The mill has run {self.passes} time(s)."
            if self.last_error:
                state += f" The last run failed: {self.last_error}"
        return {
            "message": f"{self.message}\n\n{state}",
            "passes": self.passes,
            "enabled": self.enabled,
            "last_error": self.last_error,
            "actions": list(self.actions),
            "positive": self.positive,
            "negative": self.negative,
            "task_name": self.config.name,
            "config": self.config.to_dict(),
            "fields": describe_config_fields(self.config),
            "coordinates": "pattern points are metres from the image centre, "
            "+x right, +y up; the image shows the patterns as the config "
            "stands",
            "image": None,
        }

    def parse_value(self, value: Dict[str, Any]) -> MillingEdit:
        action = value.get("action")
        if action not in MILL_ACTIONS:
            raise InvalidAnswerValue(
                f"action is required: one of {list(self.actions)} now."
            )
        if action not in self.actions:
            raise InvalidAnswerValue(
                f"{action!r} can't be decided now: one of {list(self.actions)}."
            )
        config, patch = value.get("config"), value.get("patch")
        if config is not None and not isinstance(config, dict):
            raise InvalidAnswerValue("config is a whole config document (an object).")
        if patch is not None and not isinstance(patch, dict):
            raise InvalidAnswerValue('patch is {"<dotted path>": value}.')
        unknown = sorted(set(value) - {"action", "config", "patch"})
        if unknown:
            raise InvalidAnswerValue(
                f"unknown keys {unknown}: a value has action, config and patch."
            )
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

    def response_for(self, value: MillingEdit) -> bool:
        """The button the action stands for: yes for the positive one."""
        if not self.enabled:
            return True
        return value.action in (MILL_RUN, MILL_EDIT)

    def draw(self, frame: "FibsemImage", live: Any) -> Any:
        """The config's patterns over ``frame``, as the milling editor draws
        them: ``live`` is the config as it now stands, else the one asked."""
        from fibsem.milling.patterning.plotting import render_milling_patterns

        config = live if live is not None else self.config
        return render_milling_patterns(frame, config.enabled_stages)


@dataclass(frozen=True)
class RunSpotBurn(Request[Optional["SpotBurnSettings"]]):
    """Hand over spot-burn settings to place points and run; answer with the
    settings as actually used — or None without a spot-burn widget (optional
    hardware must not fail a workflow).

    Run Spot Burn runs the widget's current points, the widget's finished
    signal re-raises the prompt, and Continue answers with ``get_settings()``
    and clears the widget.

    It does not lend control: the responder runs the burn itself, so a burn
    could start under a borrower. It moves to the asker's loop, as
    :class:`RunMillingTask` did (FIB-1236), in its own change.
    """

    value_shape = (
        '{"points": [{"x", "y"}]}: fractions of the frame, origin top-left, '
        "+y down; current and exposure stay the protocol's"
    )
    view_frame = VIEW_FRAME_FALLBACK

    settings: "SpotBurnSettings"
    message: str = "Run Spot Burn"

    def describe(self) -> Dict[str, Any]:
        settings = self.settings
        return {
            "message": f"{self.message}. Answering yes burns the points "
            "as currently shown and asks again; answering no continues. A "
            "value replaces the points first.",
            "positive": "Run Spot Burn",
            "negative": "Continue",
            "points": [pt.to_dict() for pt in getattr(settings, "coordinates", [])],
            "milling_current": getattr(settings, "milling_current", None),
            "exposure_time": getattr(settings, "exposure_time", None),
            "coordinates": "points are fractions of the frame (origin "
            "top-left, +y down), so they need no rescaling against the "
            "preview; current and exposure are the protocol's",
            "image": None,
        }

    def parse_value(self, value: Dict[str, Any]) -> List["Point"]:
        from fibsem.services.spot_burn import in_frame
        from fibsem.structures import Point

        points = value.get("points")
        if not isinstance(points, list):
            raise InvalidAnswerValue(f"a RunSpotBurn value is {self.value_shape}.")
        parsed: List[Point] = []
        for entry in points:
            try:
                pt = Point.from_dict(entry)
            except Exception:
                raise InvalidAnswerValue(
                    "each spot burn point needs numeric x, y: fractions of the frame."
                ) from None
            if not in_frame(pt):
                raise InvalidAnswerValue(
                    f"point ({pt.x:.3f}, {pt.y:.3f}) is outside the frame: x and y "
                    "are fractions, 0 to 1."
                )
            parsed.append(pt)
        return parsed

    def value_to_wire(self, value: List["Point"]) -> Dict[str, Any]:
        return {"points": [point.to_dict() for point in value]}

    def same_value(self, live: Any, value: Any, frame: Any = None) -> bool:
        # Points are 0-1 fractions of the frame.
        return (
            live is not None
            and len(live) == len(value)
            and all(
                abs(a.x - b.x) <= 1e-6 and abs(a.y - b.y) <= 1e-6
                for a, b in zip(live, value)
            )
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
