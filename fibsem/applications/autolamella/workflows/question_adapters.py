"""How a question crosses into the record and back.

A supervised task asks by handing a ``Request`` to a ``Responder`` and blocking
on a future. Some of those questions carry a value -- where the model put the
features, say -- and that value, who corrected it and by how much belong on the
item rather than in a log keyed by image filename (FIB-1025).

This module is the translation and nothing else: what a request looks like as a
``Proposal``, and what a decision on it looks like as the answer the asking task
expects. It does **not** ask, wait or release. Whoever owns the wait does that,
and there is one of those: recording a question is something the responder does
on the way to putting it up, not a second way of asking.

It does **not** make the question one that can be left for later either. The
run is still held with the instrument where the task left it, and the answer is
still needed now; this is only what gets written down.

``QtResponder`` is that owner. It records a question when the request says who
is asking and the Review tab is there to answer it in, and asks the way it
always has when not.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type

from fibsem import conversions
from fibsem.applications.autolamella.proposals import DETECTION, Proposal
from fibsem.applications.autolamella.workflows.interaction import (
    ConfirmDetection,
    Request,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment

__all__ = [
    "QuestionAdapter",
    "adapter_for",
    "answer_from",
    "answered",
    "proposal_for",
    "register_adapter",
]


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
    # The file the values sit on, named the way every proposal names its
    # reference image: relative to the item's folder. Empty when the question
    # has no image on disk.
    to_image: Callable[[Any, str], str] = lambda _request, _folder: ""
    # Whether the answer is given *on* that image -- a marker dragged, an area
    # drawn. Such a question with no saved image is not recorded at all: a
    # review nobody can see is worse than none, and the prompt path still shows
    # the picture it has in memory. A request carries its context, so a missing
    # image is the asker's defect to fix, not something to paper over here.
    needs_image: bool = False
    # What else has to happen once the question is answered, given the request
    # and the answer -- whatever the prompt path did on the click besides
    # answering, so that a question moved onto the record does not quietly stop
    # doing it.
    on_answered: Callable[[Any, Any], None] = lambda _request, _answer: None


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

    ``feature_m`` moves with ``px``. It is what every consumer actually reads --
    the stage moves in ``core.py`` and ``undercut.py`` take ``feature_m``, never
    ``px`` -- so a corrected pixel with the model's metres beside it would be
    recorded as a correction and then milled where the model said. The
    detection widget recomputes it on read-back for the same reason.
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
        feature.feature_m = conversions.image_to_microscope_image_coordinates(
            feature.px, answer.image.data, answer.pixelsize
        )
    return answer


def _detection_image(request: ConfirmDetection, folder: str) -> str:
    """The file the task saved the image to, named relative to ``folder``.

    Nothing is written here. ``take_image_and_detect_features`` always saves
    what it detects on, and a saved ``FibsemImage`` knows its own file:
    ``filepath`` is set by the write, so it carries the beam suffix and the
    extension that ``acquire`` adds and the image settings do not. That is the
    path to record -- the one actually written, not one rebuilt from a stem --
    and it is the same file the rest of the record names, so a correction here
    and anything else said about that acquisition are about one image.

    Empty when the image was never saved, which leaves the question
    unrecorded: see ``QuestionAdapter.needs_image``.
    """
    image = request.detection.fibsem_image
    path = str(getattr(image, "filepath", "") or "")
    if not path:
        return ""
    # Relative to the item's folder when it is under it, so the record
    # survives a moved experiment; in full when it is not, which a reader's
    # join onto the folder leaves alone. Forward slashes whatever the
    # platform: a record written on Windows has to be readable anywhere.
    try:
        relative = os.path.relpath(path, folder) if folder else path
    except ValueError:  # a different drive, on Windows
        relative = path
    if relative.startswith(".."):
        relative = path
    return relative.replace(os.sep, "/")


def _detection_answered(request: ConfirmDetection, answer: Any) -> None:
    """Write the training data the Detection tab writes on its Continue click:
    the image, the mask and a row per feature with how far it was moved, and
    the ``feature_detection`` log records the reports are built from.

    The delta is on the item's record now, but those readers have not moved,
    and the mask is in neither the record nor anywhere else.
    """
    import numpy as np

    from fibsem.detection import utils as det_utils

    if answer.mask is not None:
        # PIL needs uint8 to write it; the widget normalises it the same way.
        answer.mask = np.asarray(answer.mask).astype(np.uint8)
    det_utils.save_ml_feature_data(
        det=answer, initial_features=request.detection.features
    )


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
        needs_image=True,
        on_answered=_detection_answered,
        to_provenance=_detection_provenance,
    ),
)


# ---------------------------------------------------------------------------
# The two crossings
# ---------------------------------------------------------------------------


def proposal_for(
    request: Request, experiment: "Experiment", item: Any
) -> Optional[Proposal]:
    """``request`` as a proposal from the task now running on ``item``, or
    ``None`` when it is not to be recorded and stays on the prompt path: it
    carries no value, or it is answered on an image that was never saved.

    Builds the record, naming the picture it is about; it does not put the
    proposal on the item. That is ``Experiment.ask_proposal``, which is also
    what marks it as the one the run is parked on.
    """
    adapter = adapter_for(request)
    if adapter is None:
        return None
    reference_image = adapter.to_image(request, str(experiment.item_path(item)))
    if adapter.needs_image and not reference_image:
        logging.warning(
            f"{type(request).__name__} on {item.name} names no saved image, so it "
            "is not recorded for review; it is asked as a prompt instead."
        )
        return None
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
    proposal.provenance["reference_image"] = reference_image
    return proposal


def answer_from(request: Request, values: Dict[str, Any]) -> Any:
    """The decided ``values`` as the answer ``request``'s asker expects -- the
    same type it would have got from a prompt, so the task cannot tell that the
    answer came off the record."""
    adapter = adapter_for(request)
    if adapter is None:
        raise NotImplementedError(
            f"{type(request).__name__} carries no value, so no decision answers it."
        )
    return adapter.to_answer(request, values)


def answered(request: Request, answer: Any) -> None:
    """Do whatever else answering ``request`` entails (``on_answered``).

    Never raises. By now the decision is on the record and the answer is on its
    way to the task; an export that fails is worth a log line, not the run.
    """
    adapter = adapter_for(request)
    if adapter is None:
        return
    try:
        adapter.on_answered(request, answer)
    except Exception:
        logging.exception(
            f"{type(request).__name__} was answered, but what follows an answer failed"
        )
