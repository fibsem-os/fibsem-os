from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Sequence

from fibsem import milling
from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.workflows.interaction import (
    Confirm,
    ConfirmDetection,
    EditAlignmentArea,
    PickPOI,
    SetImages,
    ask,
)
from fibsem.detection import detection
from fibsem.detection import utils as det_utils
from fibsem.detection.detection import DetectedFeatures, Feature
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    Point,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

# Instructions are answered by a machine, so silence past this is a wedged GUI
# thread, not thinking. Generous because the GUI thread may legitimately be busy
# painting; the real waits are milliseconds.
INSTRUCTION_TIMEOUT_S = 30


# CORE UI FUNCTIONS -> PROBS SEPARATE FILE
def _abort_requested(parent_ui: Optional["AutoLamellaUI"]) -> bool:
    """Whether the workflow has been asked to stop. Never raises.

    Asks the manager rather than reading its stop event: the same predicate
    AutoLamellaTask._check_for_abort uses, so the two cannot disagree about what
    counts as cancelled. Falls back to the legacy UI event for the window before
    the manager exists. Also the ``abort`` predicate for :func:`interaction.ask`.
    """
    if parent_ui is None:
        return False
    task_manager = getattr(parent_ui, "_task_manager", None)
    if task_manager is not None:
        return task_manager.should_abort
    return parent_ui._workflow_stop_event.is_set()


def _check_for_abort(
    parent_ui: Optional["AutoLamellaUI"], msg: str = "Workflow aborted by user."
) -> bool:
    if _abort_requested(parent_ui):
        raise InterruptedError(msg)
    return False


def update_detection_ui(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,  # TODO: deprecate
    checkpoint: str,
    features: Sequence[Feature],
    parent_ui: Optional["AutoLamellaUI"] = None,
    validate: bool = True,
    msg: str = "Lamella",
    position: Optional[FibsemStagePosition] = None,
) -> DetectedFeatures:
    feat_str = ", ".join([f.name for f in features])
    if len(feat_str) > 15:
        feat_str = feat_str[:15] + "..."
    update_status_ui(parent_ui, f"{msg}: Detecting Features ({feat_str})...")

    det = detection.take_image_and_detect_features(
        microscope=microscope,
        image_settings=image_settings,
        features=features,
        point=position,
        checkpoint=checkpoint,
    )

    if validate and parent_ui is not None:
        # The answer IS the (possibly corrected) feature set, read and saved on
        # the GUI thread that owns the widget — no _get_detected_features
        # read-back across the seam, no detection_confirmed_signal round trip.
        # No timeout: a human answers, and silence means thinking.
        det = ask(
            parent_ui.ui_responder,
            ConfirmDetection(detection=det),
            abort=lambda: _abort_requested(parent_ui),
        )
    else:
        det_utils.save_ml_feature_data(det)

    # TODO: set images in ui here
    return det


def set_images_ui(
    parent_ui: Optional["AutoLamellaUI"],
    eb_image: Optional[FibsemImage] = None,
    ib_image: Optional[FibsemImage] = None,
):
    # headless mode
    if parent_ui is None:
        return

    _check_for_abort(parent_ui)

    # TMP: prevent milling images overwriting existing
    if eb_image is not None:
        eb_image.metadata.image_settings.save = False
    if ib_image is not None:
        ib_image.metadata.image_settings.save = False

    # One future per call, owned by this caller — unlike WAITING_FOR_UI_UPDATE,
    # no other emitter can release it. A GUI-side failure re-raises here, on the
    # workflow thread, instead of aborting the process out of a queued slot.
    ask(
        parent_ui.ui_responder,
        SetImages(sem_image=eb_image, fib_image=ib_image),
        abort=lambda: _abort_requested(parent_ui),
        timeout=INSTRUCTION_TIMEOUT_S,
    )


def update_status_ui(
    parent_ui: Optional["AutoLamellaUI"],
    msg: str,
    workflow_info: Optional[str] = None,
    status_bar: Optional[str] = None,
    check_abort: bool = True,
) -> None:
    """Put a line on the workflow-information label or the status bar.

    A status point doubles as a cancellation point: a task that reports progress
    is interrupted here once Stop has been pressed. ``check_abort=False`` is for
    the manager's closing line after a Stop -- "Workflow cancelled by user." -- which is
    the one status that has to get out precisely because the run was aborted.
    """
    if parent_ui is None:
        logging.info(msg or status_bar or "")
        return

    if check_abort:
        _check_for_abort(parent_ui)

    # Local import: the tasks package pulls this module in through its manager,
    # so a top-level import of tasks.status is a cycle.
    from fibsem.applications.autolamella.workflows.tasks.status import (
        WorkflowStatusEvent,
    )

    parent_ui.workflow_status_signal.emit(
        WorkflowStatusEvent(
            message=msg, workflow_info=workflow_info, status_bar=status_bar
        )
    )


def ask_user(
    parent_ui: Optional["AutoLamellaUI"],
    msg: str,
    pos: str,
    neg: Optional[str] = None,
) -> bool:
    """A yes/no prompt. The last variant (spot burn) converted with RunSpotBurn,
    so every ask is the typed path now: the answer arrives on this call's own
    future when a button is clicked — USER_RESPONSE and the polled flag are not
    involved. No timeout: a human answers, and silence means thinking; ``abort``
    keeps Stop working while the prompt is up.
    """
    if parent_ui is None:
        logging.warning(
            f"User input requested in headless mode: {msg}, always returning True."
        )
        return True

    return ask(
        parent_ui.ui_responder,
        Confirm(message=msg, positive=pos, negative=neg),
        abort=lambda: _abort_requested(parent_ui),
    )


def ask_user_continue_workflow(
    parent_ui, msg: str = "Continue with the next stage?", validate: bool = True
):

    ret = True
    if validate:
        ret = ask_user(parent_ui=parent_ui, msg=msg, pos="Continue", neg="Exit")
    return ret


def update_alignment_area_ui(
    alignment_area: FibsemRectangle,
    parent_ui: Optional["AutoLamellaUI"],
    msg: str = "Edit Alignment Area",
    validate: bool = True,
) -> FibsemRectangle:
    """Show the editable alignment area and return the (possibly edited) area.

    The answer IS the area: the Continue click reads it and hides the overlay on
    the GUI thread, so the old second handshake — emit "clear", poll
    WAITING_FOR_UI_UPDATE, then read across the seam — is gone with it.
    No timeout: a human answers, and silence means thinking.
    """

    _check_for_abort(parent_ui)

    # headless mode, return the alignment area
    if parent_ui is None or not validate:
        return alignment_area

    return ask(
        parent_ui.ui_responder,
        EditAlignmentArea(initial=alignment_area, message=msg),
        abort=lambda: _abort_requested(parent_ui),
    )


def select_poi_ui(
    parent_ui: Optional["AutoLamellaUI"],
    image: Optional[FibsemImage],
    msg: str = "Select Point of Interest",
    validate: bool = True,
    initial_poi: Optional[Point] = None,
) -> Optional[Point]:
    """Show a draggable POI marker on ``image`` and return the picked point.

    The image travels in the request (the marker's coordinates only mean
    something against it), and the answer IS the point: the Continue click reads
    the marker, converts, and takes the overlay down on the GUI thread — no
    SELECTED_POI read-back, no second WAITING_FOR_UI_UPDATE handshake.
    No timeout: a human answers, and silence means thinking.
    """
    _check_for_abort(parent_ui)

    if parent_ui is None or not validate or image is None:
        return None

    return ask(
        parent_ui.ui_responder,
        PickPOI(image=image, initial=initial_poi, message=msg),
        abort=lambda: _abort_requested(parent_ui),
    )


def update_experiment_ui(
    parent_ui: Optional["AutoLamellaUI"], experiment: Experiment
) -> None:

    # headless mode
    if parent_ui is None:
        return

    parent_ui.update_experiment_signal.emit(deepcopy(experiment))
