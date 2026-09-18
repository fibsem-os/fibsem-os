"""Answering a question through the record instead of through a prompt.

A supervised task asks by handing a ``Request`` to a ``Responder`` and blocking
on a future. ``QtResponder`` answers by putting a prompt up and completing the
future when it is clicked. This one answers by **recording the question as a
proposal** and completing the future when a decision lands on it (FIB-1025).

It is the same ``Responder`` protocol, so a caller swaps which object it asks
and changes nothing else. What changes is what an answer is:

* the question is on the record while it is open, so it survives being looked
  away from, and the Review tab can show it beside everything else waiting;
* who answered, what they changed and when are on the lamella rather than in a
  log keyed by image filename;
* **anything** that goes through ``Experiment.decide`` answers it -- a click in
  the Review tab, an agent over ``/app/decide`` -- because the decision landing
  is what releases the run, not the click.

It does **not** make the question deferrable. The task is still parked with the
instrument where it left it, and the answer is still needed now; this is only
where the answer comes from.

A parallel track for now: nothing in the workflow asks through this yet. The
live path is unchanged until the renderer for the kind exists and the swap is
made deliberately.
"""

from __future__ import annotations

import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type

from fibsem.applications.autolamella.proposals import (
    DETECTION,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.workflows.interaction import (
    ConfirmDetection,
    Request,
)

if TYPE_CHECKING:
    from concurrent.futures import Future

    from fibsem.applications.autolamella.structures import Experiment

__all__ = [
    "QUESTIONS_DIR",
    "QuestionAdapter",
    "ReviewResponder",
    "adapter_for",
    "register_adapter",
]

# Where a question's own files go, under the item's folder. Its own directory
# so the item's outputs stay the task's outputs: these are not results, they
# are what somebody was shown when they were asked.
QUESTIONS_DIR = "questions"


# ---------------------------------------------------------------------------
# What a request looks like as a proposal, and what a decision looks like as
# an answer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionAdapter:
    """How one kind of request crosses into the record and back.

    ``to_values`` is what the model proposed, in the kind's value names.
    ``to_answer`` turns the decided values back into the answer the asking
    task expects -- the same type it would have got from a prompt, so the
    task cannot tell which responder answered it.
    """

    kind: str
    to_values: Callable[[Any], Dict[str, Any]]
    to_answer: Callable[[Any, Dict[str, Any]], Any]
    # What proposed this, and anything else about the proposer worth keeping.
    # The request's class name is what asked, not what answered it -- a reader
    # wants the model, and a delta is only comparable against the same one.
    to_provenance: Callable[[Any], Dict[str, Any]] = lambda _request: {}
    # Puts the image the values sit on beside the item's other outputs and
    # names it the way every proposal names its reference image. Empty when
    # the question has no image at all.
    to_image: Callable[[Any, str, str], str] = lambda _request, _folder, _stem: ""


_ADAPTERS: Dict[Type[Request], QuestionAdapter] = {}


def register_adapter(
    request_type: Type[Request], adapter: QuestionAdapter
) -> QuestionAdapter:
    _ADAPTERS[request_type] = adapter
    return adapter


def adapter_for(request: Request) -> Optional[QuestionAdapter]:
    """The adapter for this request, or ``None`` when it has none -- a question
    that carries no value, which stays on the prompt path for now."""
    return _ADAPTERS.get(type(request))


# --- feature detection ------------------------------------------------------


def _detection_values(request: ConfirmDetection) -> Dict[str, Any]:
    return {
        "features": [{"name": f.name, "px": f.px} for f in request.detection.features]
    }


def _detection_answer(request: ConfirmDetection, values: Dict[str, Any]) -> Any:
    """The decided points put back on a copy of the detection.

    A copy because the request is frozen and the original is what the delta is
    measured against. Only the points move: the mask and the rgb are the
    model's output and a correction does not change what it saw.
    """
    decided = {
        f.get("name", ""): f.get("px")
        for f in values.get("features", [])
        if isinstance(f, dict)
    }
    answer = deepcopy(request.detection)
    for feature in answer.features:
        px = decided.get(feature.name)
        if px is not None:
            feature.px = px
    return answer


def _detection_image(request: ConfirmDetection, folder: str, stem: str) -> str:
    """Write the image the question is about, and name it relative to ``folder``.

    Saved here rather than looked for. ``DetectedFeatures`` carries its image
    in memory and whether the task ever wrote one depends on its image
    settings -- and a question answered by dragging a marker cannot be asked
    without a picture, so the image is part of the question rather than a
    nicety. Writing it also means the record says what was decided on, which
    is the whole complaint this path exists to fix.

    (Guessing at a file the task wrote is worse than it looks: ``acquire``
    saves as the settings filename plus a beam suffix and hands back metadata
    without it, so the obvious guess finds nothing.)
    """
    image = request.detection.fibsem_image
    if image is None or not folder:
        return ""
    try:
        os.makedirs(os.path.join(folder, QUESTIONS_DIR), exist_ok=True)
        image.save(os.path.join(folder, QUESTIONS_DIR, stem))
    except Exception:
        logging.exception("Could not save the image a question was asked on")
        return ""
    # Stored with a forward slash whatever the platform: a reference image is
    # named relative to the item's folder and readers join it back on, so a
    # record written on Windows has to be readable anywhere.
    return f"{QUESTIONS_DIR}/{stem}.tif"


def _detection_provenance(request: ConfirmDetection) -> Dict[str, Any]:
    """Which model said this. ``ConfirmDetection`` is the question's type, not
    its author, and a correction only means something measured against the
    checkpoint that produced it."""
    checkpoint = str(getattr(request.detection, "checkpoint", "") or "")
    name = os.path.splitext(os.path.basename(checkpoint))[0] if checkpoint else ""
    return {
        "proposer": name or "segmentation model",
        "checkpoint": checkpoint,
        "features": [f.name for f in request.detection.features],
    }


register_adapter(
    ConfirmDetection,
    QuestionAdapter(
        kind=DETECTION,
        to_values=_detection_values,
        to_answer=_detection_answer,
        to_image=_detection_image,
        to_provenance=_detection_provenance,
    ),
)


def _image_stem(task_name: str, task_id: str, asked_at: float) -> str:
    """A file name for one question's image: the task, the run, and when it was
    asked.

    All three, because the first two are not enough. A task may ask several
    times in one run -- two detections back to back is ordinary -- and those
    share a run id, so keying on the run alone lets the second write over the
    first's picture and leaves the superseded proposal pointing at the wrong
    image.

    The time comes off the proposal rather than a counter somewhere: a counter
    is only unique if every question goes through the same object, which is
    true of a task today and is not something a file name should depend on.
    """
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", f"{task_name}-{task_id}").strip("-")
    return f"question_{safe.lower()}_{int(asked_at * 1000)}"


# ---------------------------------------------------------------------------
# The responder
# ---------------------------------------------------------------------------


class ReviewResponder:
    """Records each question on ``item_id``'s ``task_name`` and waits for a
    decision on it.

    One per task that wants to ask this way, because a proposal is keyed by
    the item and the task it came from and a bare ``submit(request, future)``
    does not carry either.

    A task may ask more than once -- two detections back to back is ordinary --
    and they go through the same responder, one at a time: the workflow thread
    blocks on each answer before the next is asked. Each replaces the last on
    the record, keeping it underneath if it was answered, so the inbox shows
    the question now open and the history holds the rest.
    """

    def __init__(self, experiment: "Experiment", item_id: str, task_name: str) -> None:
        self._experiment = experiment
        self._item_id = item_id
        self._task_name = task_name
        # (future, request, adapter) for the question now open, if any. The
        # workflow thread blocks on each question in turn, so there is never
        # more than one.
        self._open: Optional[Tuple["Future", Request, QuestionAdapter]] = None

    # -- asking --------------------------------------------------------------

    def submit(self, request: Request, future: "Future") -> None:
        """Record the question and return. Called on the workflow thread, which
        then blocks on ``future`` -- so nothing here may wait for an answer."""
        adapter = adapter_for(request)
        if adapter is None:
            future.set_exception(
                NotImplementedError(
                    f"{type(request).__name__} carries no value to record; "
                    "ask a prompt responder instead."
                )
            )
            return
        item = self._experiment.get_item_by_id(self._item_id)
        if item is None:
            future.set_exception(
                LookupError(f"No item with id {self._item_id!r} to ask about.")
            )
            return
        proposal = Proposal(
            kind=adapter.kind,
            values=adapter.to_values(request),
            provenance={
                # The run the decision must name. Read off the item rather than
                # passed in, so it is the run that is actually in progress --
                # which is also what lets the decision land while it runs.
                "task_id": item.task_state.task_id,
                "proposer": type(request).__name__,
                **adapter.to_provenance(request),
                "in_run": True,
            },
        )
        # After the proposal exists, because the picture is named after when
        # the question was asked and that is the proposal's own timestamp.
        proposal.provenance["reference_image"] = adapter.to_image(
            request,
            str(self._experiment.item_path(item)),
            _image_stem(self._task_name, item.task_state.task_id, proposal.created_at),
        )
        # Through the experiment, not straight onto the item: this runs on the
        # workflow thread, and recording writes to the record and wakes the
        # Review tab, both of which belong on the main one.
        if not self._experiment.ask_proposal(self._item_id, self._task_name, proposal):
            future.set_exception(
                LookupError(f"Could not record a question on {self._item_id!r}.")
            )
            return
        self._open = (future, request, adapter)
        # One place to undo everything, whatever ends the wait: an answer, an
        # abort that cancels the future, or a timeout.
        future.add_done_callback(self._closed)
        self._experiment.decided.connect(self._on_decided)
        logging.info(
            {
                "msg": "question_recorded",
                "item": item.name,
                "task_name": self._task_name,
                "kind": adapter.kind,
            }
        )

    # -- answering -----------------------------------------------------------

    def _on_decided(self, item_id: str, task_name: str) -> None:
        """A decision landed somewhere. Release the run if it was on ours."""
        open_question = self._open
        if open_question is None:
            return
        if item_id != self._item_id or task_name != self._task_name:
            return
        future, request, adapter = open_question
        if future.done():
            return  # aborted or timed out; the answer is nobody's to read
        item = self._experiment.get_item_by_id(item_id)
        proposal = item.proposals.get(task_name) if item is not None else None
        decision = proposal.current if proposal is not None else None
        if decision is None:
            return
        if decision.outcome is DecisionOutcome.Confirmed:
            future.set_result(adapter.to_answer(request, decision.values))
        elif decision.outcome is DecisionOutcome.Rejected:
            # Raised on the workflow thread inside wait_for, where the task's
            # own failure path handles it -- the same as any other refusal to
            # go on.
            future.set_exception(
                RuntimeError(
                    f"{task_name} was rejected: {decision.reason or 'no reason given'}"
                )
            )
        else:
            # Withdrawn: the question was taken back, so nothing is coming.
            future.cancel()

    def _closed(self, future: "Future") -> None:
        """The wait ended, however it ended: stop listening and close the
        question, so a later decision on the same record does not try to
        release a run that has moved on."""
        self._open = None
        try:
            self._experiment.decided.disconnect(self._on_decided)
        except Exception:  # noqa: BLE001 - already disconnected is fine
            pass
        item = self._experiment.get_item_by_id(self._item_id)
        proposal = item.proposals.get(self._task_name) if item is not None else None
        if proposal is not None:
            proposal.asking = False
