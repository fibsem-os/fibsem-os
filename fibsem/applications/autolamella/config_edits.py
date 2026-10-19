"""An agent's edits to an experiment's configs and items, with no window.

The App Server's config edits: patch an item's task config, the protocol's,
or the item's own fields; reorder a milling config's stages; re-copy the
protocol onto an item. Each runs where the experiment is edited
(``session.run_edit``), checks the version it was written against beside the
apply, saves, and is recorded as the agent's edit (FIB-1034, FIB-1062). A
window, when one is attached, only flushes its editors first and redraws
after (``AppWindow`` in ``server/context.py``).

Each returns a plain dict: the applied changes, or a structured refusal.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Optional

from fibsem.acting import AGENT, acting
from fibsem.applications.autolamella.edit_recording import (
    PendingEdits,
    touch_agent_patch,
)
from fibsem.applications.autolamella.server.context import (
    ITEM_PATCH_FIELDS,
    config_version,
    find_item,
    item_fields_version,
)
from fibsem.server.config_patch import PatchError, apply_patch
from fibsem.workflows.events import to_plain


def apply_agent_edit(
    experiment: Any,
    microscope: Callable[[], Any],
    level: str,
    item_name: str,
    task_name: str,
    patch: Any,
    version: str,
) -> Dict[str, Any]:
    """Apply the agent's edit at *level*, recorded as the agent's: what it
    changed, before and after, on *microscope*'s record. *level* is ``item``
    or ``protocol`` (a task-config patch), ``item_fields``, ``reorder_stages``
    or ``apply_protocol``."""
    edits = PendingEdits(microscope, via="agent patch")
    with acting(AGENT):
        touch_agent_patch(edits, experiment, level, item_name, task_name, patch)
        try:
            return _apply(experiment, level, item_name, task_name, patch, version)
        finally:
            edits.flush()


def _apply(experiment, level, item_name, task_name, patch, version) -> dict:
    if experiment is None:
        return {"applied": False, "error": "no experiment is loaded"}
    if level == "item_fields":
        return patch_item_fields(experiment, item_name, patch, version)
    if level == "reorder_stages":
        return reorder_milling_stages(experiment, item_name, task_name, patch, version)
    if level == "apply_protocol":
        return apply_protocol_to_item(experiment, item_name, patch.get("task_names"))
    return patch_task_config(experiment, level, item_name, task_name, patch, version)


def _save(experiment, what: str, save_protocol: bool = False) -> bool:
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


def _changes(changes) -> list:
    return [
        {"path": p, "old": to_plain(old), "new": to_plain(new)}
        for p, old, new in changes
    ]


def patch_task_config(
    experiment, level: str, item_name: str, task_name: str, patch: dict, version: str
) -> dict:
    """Patch one task config: an item's (*level* ``item``) or the protocol's.
    The version check sits beside the apply, so an edit landing in between is
    refused as stale, never half-merged."""
    if level == "protocol":
        protocol = getattr(experiment, "task_protocol", None)
        config_map = getattr(protocol, "task_config", None)
        if config_map is None:
            return {"applied": False, "error": "no protocol is loaded"}
        config = dict(config_map).get(task_name)
        if config is None:
            return {
                "applied": False,
                "error": f"No task named {task_name!r} in the protocol.",
                "task_names": list(config_map.keys()),
            }
    else:
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"applied": False, **refusal}
        config = dict(lamella.task_config).get(task_name)
        if config is None:
            return {
                "applied": False,
                "error": f"No task config named {task_name!r} on {item_name!r}.",
                "task_names": list(lamella.task_config.keys()),
            }
    if config_version(config) != version:
        return {"applied": False, "stale": True}
    try:
        changes = apply_patch(config, patch)
    except PatchError as exc:
        return {"applied": False, "invalid_patch": str(exc), "path": exc.path}
    where = f"{item_name} / {task_name}" if level == "item" else task_name
    logging.info(
        f"agent config patch applied ({level}): {where}: "
        + ", ".join(f"{p}: {old!r} -> {new!r}" for p, old, new in changes)
    )
    saved = _save(experiment, "agent config patch", level == "protocol")
    result = {
        "applied": True,
        "saved": saved,
        "task_name": task_name,
        "changes": _changes(changes),
        "version": config_version(config),
    }
    if level == "item":
        result["item_name"] = item_name
    return result


def patch_item_fields(experiment, item_name: str, patch: dict, version: str) -> dict:
    """Patch the lamella's own editable fields, ITEM_PATCH_FIELDS only.

    Alignment validity is checked after the apply and reverted on failure (a
    rectangle is only judgeable whole). A defect edit stamps ``updated_at``.
    Moving the POI moves the patterns that follow it, and moving the area
    moves the fiducial not milled yet, as the window's own moves do.
    """
    lamella, refusal = find_item(experiment, item_name)
    if refusal:
        return {"applied": False, **refusal}
    for path in patch:
        if path.split(".", 1)[0] not in ITEM_PATCH_FIELDS:
            return {
                "applied": False,
                "invalid_patch": f"{path!r} is not an editable item field; "
                f"editable: {sorted(ITEM_PATCH_FIELDS)}",
                "path": path,
            }
    if item_fields_version(lamella) != version:
        return {"applied": False, "stale": True}
    try:
        changes = apply_patch(lamella, patch, none_types={"description": str})
    except PatchError as exc:
        return {"applied": False, "invalid_patch": str(exc), "path": exc.path}
    area = lamella.alignment_area
    if area is not None and not area.is_valid_reduced_area:
        for path, old, _new in reversed(changes):
            apply_patch(lamella, {path: old if not hasattr(old, "name") else old.name})
        return {
            "applied": False,
            "invalid_patch": "the patched alignment area is out of bounds "
            "(left/top >= 0, width/height > 0, inside the frame); "
            "nothing was applied.",
            "path": None,
        }
    fields = {p.split(".", 1)[0] for p, _o, _n in changes}
    if "defect" in fields:
        from fibsem.util.timestamps import now

        lamella.defect.updated_at = now()
    synced_tasks = []
    if "poi" in fields:
        try:
            synced_tasks = list(lamella.sync_tasks_to_poi())
        except Exception:
            logging.exception("pattern sync after POI patch failed")
    if "alignment_area" in fields:
        from fibsem.applications.autolamella.workflows.tasks.fiducial import (
            follow_alignment_area,
        )

        try:
            synced_tasks += follow_alignment_area(lamella)
        except Exception:
            logging.exception("fiducial sync after alignment area patch failed")
    logging.info(
        f"agent item patch applied: {item_name}: "
        + ", ".join(f"{p}: {old!r} -> {new!r}" for p, old, new in changes)
        + (f" (patterns synced: {', '.join(synced_tasks)})" if synced_tasks else "")
    )
    saved = _save(experiment, "agent item patch")
    return {
        "applied": True,
        "saved": saved,
        "item_name": item_name,
        "synced_tasks": synced_tasks,
        "changes": _changes(changes),
        "version": item_fields_version(lamella),
    }


def reorder_milling_stages(
    experiment, item_name: str, task_name: str, payload: dict, version: str
) -> dict:
    """Reorder the stages inside one milling config, by name: never adds,
    drops or duplicates a stage, and never reorders a config the caller
    hasn't seen. *payload* has ``level``, ``milling_key`` and ``order``."""
    level = payload.get("level", "item")
    milling_key = payload.get("milling_key")
    order = payload.get("order") or []
    if level == "protocol":
        protocol = getattr(experiment, "task_protocol", None)
        config_map = getattr(protocol, "task_config", None)
        if config_map is None:
            return {"applied": False, "error": "no protocol is loaded"}
        config = dict(config_map).get(task_name)
    else:
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"applied": False, **refusal}
        config = dict(lamella.task_config).get(task_name)
    if config is None:
        return {"applied": False, "error": f"No task config named {task_name!r}."}
    milling = getattr(config, "milling", None) or {}
    if milling_key not in milling:
        return {
            "applied": False,
            "invalid_patch": f"{milling_key!r} is not a milling config of "
            f"{task_name!r}; known: {sorted(milling.keys())}",
            "path": milling_key,
        }
    if config_version(config) != version:
        return {"applied": False, "stale": True}
    stages = milling[milling_key].stages
    names = [s.name for s in stages]
    if len(set(names)) != len(names):
        return {
            "applied": False,
            "invalid_patch": "stage names are not unique; reorder by "
            "name is ambiguous — rename the stages first.",
            "path": milling_key,
        }
    if sorted(order) != sorted(names):
        return {
            "applied": False,
            "invalid_patch": f"order must be exactly the current stages "
            f"in a new sequence; current: {names}",
            "path": milling_key,
        }
    by_name = {s.name: s for s in stages}
    milling[milling_key].stages = [by_name[n] for n in order]
    logging.info(
        f"agent reordered stages ({level}): "
        f"{item_name or 'protocol'} / {task_name} / {milling_key}: "
        f"{names} -> {order}"
    )
    saved = _save(experiment, "stage reorder", level == "protocol")
    return {
        "applied": True,
        "saved": saved,
        "task_name": task_name,
        "milling_key": milling_key,
        "order": list(order),
        "changes": _changes([(f"milling.{milling_key}.stages", names, list(order))]),
        "version": config_version(config),
    }


def apply_protocol_to_item(
    experiment, item_name: str, task_names: Optional[list]
) -> dict:
    """Deep-copy the protocol's task configs onto one item, as creation does,
    with the copied acquisitions pointed at the item's own directory.
    Wholesale by design, so no version check. ``task_names`` of None means
    every task the protocol defines."""
    lamella, refusal = find_item(experiment, item_name)
    if refusal:
        return {"applied": False, **refusal}
    protocol = getattr(experiment, "task_protocol", None)
    config_map = getattr(protocol, "task_config", None)
    if config_map is None:
        return {"applied": False, "error": "no protocol is loaded"}
    names = list(task_names) if task_names else list(config_map.keys())
    unknown = [n for n in names if n not in config_map]
    if unknown:
        return {
            "applied": False,
            "error": f"Not in the protocol: {unknown!r}.",
            "task_names": list(config_map.keys()),
        }
    for name in names:
        lamella.task_config[name] = deepcopy(config_map[name])
    lamella._sync_imaging_paths()
    logging.info(f"agent applied protocol to {item_name}: {', '.join(names)}")
    saved = _save(experiment, "protocol apply")
    return {
        "applied": True,
        "saved": saved,
        "item_name": item_name,
        "task_names": names,
    }
