from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from fibsem import milling
from fibsem.detection import detection
from fibsem.detection import utils as det_utils
from fibsem.detection.detection import DetectedFeatures, Feature
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
)
from fibsem.workflows.interaction import (
    ConfirmDetection,
    SetImages,
    ask,
)
from fibsem.workflows.status import post_status

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

# Instructions are answered by a machine, so silence past this is a wedged GUI
# thread, not thinking. Generous because the GUI thread may legitimately be busy
# painting; the real waits are milliseconds.
INSTRUCTION_TIMEOUT_S = 30


# CORE UI FUNCTIONS -> PROBS SEPARATE FILE
def _abort_requested(run: Optional[Any]) -> bool:
    """Whether the workflow has been asked to stop. Never raises.

    Asks the run's manager (or the session's runner, which asks it): the same
    predicate AutoLamellaTask._check_for_abort uses, so the two cannot disagree
    about what counts as cancelled. Also the ``abort`` predicate for
    :func:`interaction.ask`.
    """
    return bool(getattr(run, "should_abort", False))


def _check_for_abort(
    run: Optional[Any], msg: str = "Workflow aborted by user."
) -> bool:
    if _abort_requested(run):
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
    runner = parent_ui.session.runner if parent_ui is not None else None
    update_status_ui(
        runner.manager if runner is not None else None,
        f"{msg}: Detecting Features ({feat_str})...",
    )

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
            abort=lambda: _abort_requested(runner),
        )
    else:
        det_utils.save_ml_feature_data(det)

    # TODO: set images in ui here
    return det


def set_images_ui(
    run: Optional[Any],
    eb_image: Optional[FibsemImage] = None,
    ib_image: Optional[FibsemImage] = None,
):
    """Keep a task's images as the latest of each beam (the run's ``images``),
    then have the window, when one answers, show them."""
    if run is None:
        return

    _check_for_abort(run)

    images = getattr(run, "images", None)
    if images is not None:
        if eb_image is not None:
            images.set(eb_image, BeamType.ELECTRON)
        if ib_image is not None:
            images.set(ib_image, BeamType.ION)
    responder = getattr(run, "responder", None)
    if responder is None:
        return

    # TMP: prevent milling images overwriting existing
    if eb_image is not None:
        eb_image.metadata.image_settings.save = False
    if ib_image is not None:
        ib_image.metadata.image_settings.save = False

    # One future per call, owned by this caller — unlike WAITING_FOR_UI_UPDATE,
    # no other emitter can release it. A GUI-side failure re-raises here, on the
    # workflow thread, instead of aborting the process out of a queued slot.
    ask(
        responder,
        SetImages(sem_image=eb_image, fib_image=ib_image),
        abort=lambda: _abort_requested(run),
        timeout=INSTRUCTION_TIMEOUT_S,
    )


def update_status_ui(
    run: Optional[Any],
    msg: Optional[str],
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
    if check_abort:
        _check_for_abort(run)
    post_status(run, msg, workflow_info=workflow_info, status_bar=status_bar)
