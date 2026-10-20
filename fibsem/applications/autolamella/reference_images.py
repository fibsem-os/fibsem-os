"""The image a proposal's values sit on, for whoever looks at it: an agent's
proposal list, the agent card's preview."""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional, Union

from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import FibsemImage
from fibsem.workflows.proposals import Proposal

__all__ = [
    "load_reference_image",
    "preview_of",
    "reference_image_label",
    "fm_stack_label",
]


def load_reference_image(
    experiment: Any, item: Any, proposal: Proposal, key: str = "reference_image"
) -> Optional[Union[FibsemImage, FluorescenceImage]]:
    """The image the proposal's values sit on, from its provenance: a file
    name relative to the item's folder (``Experiment.item_path``). A
    fluorescence result (an OME-TIFF, channels and planes) loads as a
    ``FluorescenceImage``; a beam image as a ``FibsemImage``."""
    path = proposal.provenance.get(key)
    if not path:
        return None
    path = os.path.join(str(experiment.item_path(item)), path)
    if not os.path.exists(path):
        logging.warning(f"Reference image not found: {path}")
        return None
    fluorescence = path.lower().endswith((".ome.tif", ".ome.tiff"))
    try:
        if fluorescence:
            return FluorescenceImage.load(path)
        return FibsemImage.load(path)
    except Exception:
        logging.exception(f"Could not load the reference image: {path}")
        return None


def preview_of(image: Any) -> Optional[Any]:
    """What an agent is shown of a proposal's image: the beam image itself,
    or a fluorescence result's channel composite (max over z, tinted per
    channel), the same one the FM canvas and the thumbnail show."""
    if isinstance(image, FluorescenceImage):
        from fibsem.fm.preview import composite_projection

        try:
            return composite_projection(image)
        except Exception:
            logging.exception("Could not project the fluorescence image")
            return None
    return image


_REFERENCE_SUFFIXES = ("_ib.tif", "_eb.tif")


def reference_image_label(filename: str, task_names: Iterable[str]) -> str:
    """What to call a reference image in a picker: ``Task · stage``, not its filename.

    The workflow names them ``ref_<task>_<stage>_<beam>.tif``, and the task can
    carry spaces or underscores depending on the version that wrote it. The stem is
    matched against the lamella's own task names, longest first, so a task whose
    name is a prefix of another's cannot claim the other's images. Anything that
    matches no task (``ref_alignment``, a timestamped acquisition) keeps its stem,
    with underscores read as spaces.
    """
    stem = filename
    if stem.startswith("ref_"):
        stem = stem[len("ref_") :]
    for suffix in _REFERENCE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for task in sorted(task_names, key=len, reverse=True):
        for spelled in (task, task.replace(" ", "_")):
            if stem == spelled:
                return task
            if stem.startswith(spelled + "_"):
                stage = stem[len(spelled) + 1 :].replace("_", " ")
                return f"{task} · {stage}"
    return stem.replace("_", " ")


def fm_stack_label(filename: str) -> str:
    """``<lamella>-zstack-HH-MM-SS.ome.tiff`` reads as ``Z-stack HH:MM:SS``."""
    stem = filename
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    marker = "-zstack-"
    if marker in stem:
        clock = stem.split(marker, 1)[1]
        parts = clock.split("-")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return "Z-stack " + ":".join(parts)
    return stem
