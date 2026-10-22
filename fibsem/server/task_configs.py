"""A task config's version and the version-guarded patch every app's agent
edits go through (FIB-864): reads serve the version, writes echo it, and a
mismatch is refused as stale. The editors' write of a task config, and the
save after it, are here too."""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

from fibsem.server.app_context import content_hash, find_item
from fibsem.server.config_patch import PatchError, apply_patch
from fibsem.workflows.edit_recording import PendingEdits, serialise
from fibsem.workflows.events import to_plain
from fibsem.workflows.task_config import WORKFLOW_FIELDS

__all__ = [
    "WORKFLOW_FIELD_VERBS",
    "config_version",
    "patch_config",
    "plain_changes",
    "save_edits",
    "save_experiment",
    "set_task_config_values",
    "task_config_at",
    "workflow_field_refusal",
]


def config_version(config) -> str:
    """A content hash naming exactly this state of a task config's settings.

    Computed from the serialized document, less the task's name, so the same
    content hashes identically wherever it is held.
    """
    data = config.to_dict()
    for name in WORKFLOW_FIELDS:
        data.pop(name, None)
    return content_hash(to_plain(data))


WORKFLOW_FIELD_VERBS = {
    "attention": "protocol_set_supervision",
    "requires": "protocol_set_task_requirements",
}


def workflow_field_refusal(patch: dict) -> Optional[dict]:
    """Why a task-config *patch* is refused for naming attention or requires,
    as an ``invalid_patch`` answer; None when it names neither. Both are the
    plan entry's, with their own verbs."""
    for path in patch:
        field = str(path).split(".", 1)[0]
        if field in WORKFLOW_FIELD_VERBS:
            return {
                "invalid_patch": f"{field} is not set by a config patch; use "
                f"{WORKFLOW_FIELD_VERBS[field]}.",
                "path": path,
            }
    return None


def plain_changes(changes: List[Tuple[str, Any, Any]]) -> List[Dict[str, Any]]:
    return [
        {"path": p, "old": to_plain(old), "new": to_plain(new)}
        for p, old, new in changes
    ]


def patch_config(config, patch: dict, version: str) -> Dict[str, Any]:
    """Patch *config* when *version* names its state: what changed and its new
    version, or why not, with nothing changed. Saving is the caller's."""
    refused = workflow_field_refusal(patch)
    if refused:
        return {"applied": False, **refused}
    if config_version(config) != version:
        return {"applied": False, "stale": True}
    try:
        changes = apply_patch(config, patch)
    except PatchError as exc:
        return {"applied": False, "invalid_patch": str(exc), "path": exc.path}
    return {
        "applied": True,
        "changes": plain_changes(changes),
        "version": config_version(config),
    }


def save_experiment(experiment, what: str, save_protocol: bool = False) -> bool:
    """Persist now: an agent's edit has no editor's debounced save to ride, so
    without this write a restart silently reverts it."""
    try:
        experiment.save(save_protocol=save_protocol)
        return True
    except Exception:
        logging.exception(
            f"save after {what} failed; the change is applied in memory but "
            "not yet on disk"
        )
        return False


def task_config_at(experiment, level: str, item_name: str, task_name: str):
    """The task config an edit at *level* names: an item's (``item``) or the
    protocol's, with the item holding it (None for the protocol's), and
    None; or None, None and the refusal, as an answer's fields."""
    if level == "protocol":
        protocol = getattr(experiment, "task_protocol", None)
        config_map = getattr(protocol, "task_config", None)
        if config_map is None:
            return None, None, {"error": "no protocol is loaded"}
        config = dict(config_map).get(task_name)
        if config is None:
            return (
                None,
                None,
                {
                    "error": f"No task named {task_name!r} in the protocol.",
                    "task_names": list(config_map.keys()),
                },
            )
        return None, config, None
    lamella, refusal = find_item(experiment, item_name)
    if refusal:
        return None, None, refusal
    config = dict(lamella.task_config).get(task_name)
    if config is None:
        return (
            None,
            None,
            {
                "error": f"No task config named {task_name!r} on {item_name!r}.",
                "task_names": list(lamella.task_config.keys()),
            },
        )
    return lamella, config, None


def save_edits(experiment, level: str, what: str = "config edit") -> bool:
    """Write the experiment after an edit at *level*, with the protocol file
    for the protocol's: an agent's edit at once, an editor's once its edits
    settle (``fibsem/ui/edit_recording.py``)."""
    return save_experiment(experiment, what, level == "protocol")


def set_task_config_values(
    experiment,
    level: str,
    item_name: str,
    task_name: str,
    values: Dict[str, Any],
    edits: PendingEdits,
) -> dict:
    """Set *values* on one task config, as the editors do: each by its path (a
    field, or ``milling.<key>`` for one milling config), whole and copied, so
    the config never shares an object with the form that sent it.

    Each value that changes is touched on *edits* first, under its path
    (``protocol.``-prefixed for the protocol's). No version check: an editor
    writes the config it shows, and is redrawn when anything else writes it.
    Not saved here: the editor saves (:func:`save_edits`) once its edits
    settle."""
    lamella, config, refusal = task_config_at(experiment, level, item_name, task_name)
    if refusal:
        return {"applied": False, **refusal}
    changes = []
    for path, value in values.items():
        *parents, leaf = path.split(".")
        owner = config
        for part in parents:
            owner = _value_at(owner, part)
        old = _value_at(owner, leaf)
        if serialise(old) == serialise(value):
            continue
        target = path if level == "item" else f"protocol.{path}"
        edits.touch(
            lamella,
            task_name,
            target,
            lambda owner=owner, leaf=leaf: _value_at(owner, leaf),
        )
        if isinstance(owner, dict):
            owner[leaf] = copy.deepcopy(value)
        else:
            setattr(owner, leaf, copy.deepcopy(value))
        changes.append((path, old, value))
    if changes:
        where = f"{item_name} / {task_name}" if level == "item" else task_name
        logging.info(
            f"config edit ({level}): {where}: " + ", ".join(p for p, _o, _n in changes)
        )
    return {"applied": True, "task_name": task_name, "changes": plain_changes(changes)}


def _value_at(owner: Any, name: str) -> Any:
    return owner[name] if isinstance(owner, dict) else getattr(owner, name)
