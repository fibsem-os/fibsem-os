"""The labels the lamella editor gives a lamella's reference images and FM
stacks."""

from __future__ import annotations

from typing import Iterable

__all__ = ["reference_image_label", "fm_stack_label"]


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
