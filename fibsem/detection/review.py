"""How a detection is recorded for review, and how a correction to it is
applied (FIB-1025).

A detection a task records goes on the item's record as its feature points;
the decided points come back onto a copy of the detection, and the training
data the Detection tab writes on Continue is written for it too.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict

from fibsem import conversions

__all__ = [
    "apply_feature_px",
    "decided_features",
    "detection_image_file",
    "detection_provenance",
    "detection_values",
    "record_training_data",
]


def detection_values(detection: Any) -> Dict[str, Any]:
    """What the model proposed, in the kind's value names."""
    return {"features": [{"name": f.name, "px": f.px} for f in detection.features]}


def decided_features(detection: Any, values: Dict[str, Any]) -> Any:
    """The decided points put back on a copy of the detection.

    A copy because the original is what the delta is measured against. Only
    the points move: the mask and the rgb are the model's output and a
    correction does not change what it saw.

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
    answer = deepcopy(detection)
    apply_feature_px(answer, decided)
    return answer


def apply_feature_px(detection: Any, px_by_name: Dict[str, Any]) -> None:
    """Move the named features of ``detection`` to new pixels, in place, and
    recompute every ``feature_m`` from ``px`` (see :func:`decided_features`).
    Features not named stay where they are."""
    for feature in detection.features:
        px = px_by_name.get(feature.name)
        if px is not None:
            feature.px = px
        feature.feature_m = conversions.image_to_microscope_image_coordinates(
            feature.px, detection.image.data, detection.pixelsize
        )


def detection_image_file(detection: Any, folder: str) -> str:
    """The file the task saved the image to, named relative to ``folder``.

    Nothing is written here. ``take_image_and_detect_features`` always saves
    what it detects on, and a saved ``FibsemImage`` knows its own file:
    ``filepath`` is set by the write, so it carries the beam suffix and the
    extension that ``acquire`` adds and the image settings do not. That is the
    path to record -- the one actually written, not one rebuilt from a stem --
    and it is the same file the rest of the record names, so a correction here
    and anything else said about that acquisition are about one image.

    Empty when the image was never saved.
    """
    image = detection.fibsem_image
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


def record_training_data(detection: Any, answer: Any) -> None:
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
    det_utils.save_ml_feature_data(det=answer, initial_features=detection.features)


def detection_provenance(detection: Any) -> Dict[str, Any]:
    """Which model said this. The question's type is not its author, and a
    correction only means something measured against the checkpoint that
    produced it. A registered model is named ``name@version``, whose dots are
    the version's, not a file's extension."""
    checkpoint = str(getattr(detection, "checkpoint", "") or "")
    name = os.path.basename(checkpoint)
    if "@" not in name:
        name = os.path.splitext(name)[0]
    return {
        "proposer": name or "segmentation model",
        "checkpoint": checkpoint,
        "features": [f.name for f in detection.features],
    }
