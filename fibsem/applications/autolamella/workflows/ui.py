from __future__ import annotations

import logging
from typing import Any, List, Optional

from fibsem import milling
from fibsem.milling import FibsemMillingStage
from fibsem.structures import BeamType, FibsemImage
from fibsem.workflows.interaction import (
    SetImages,
    ask,
)
from fibsem.workflows.status import post_status

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
