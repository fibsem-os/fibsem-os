"""Edits to an experiment's configs and items, with no window.

The App Server's config edits: patch an item's task config, the protocol's,
or the item's own fields; reorder a milling config's stages; re-copy the
protocol onto an item. The editors' task config edits, copy onto other
lamellae and global edit of the reference imaging are here too, beside the
agent's, and the save a config edit makes (:func:`save_edits`). Each agent
edit runs where the experiment is edited (``session.run_edit``), checks the
version it was written against beside the apply, saves, and is recorded as the agent's edit (FIB-1034, FIB-1062). A
window writes its editors' pending edits before any edit runs (its executor)
and redraws when ``session.changed`` says the edit applied. The versions an
edit is checked against are here too, for the server's reads to serve.

Each returns a plain dict: the applied changes, or a structured refusal.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from fibsem.acting import AGENT, acting
from fibsem.applications.autolamella.edit_recording import (
    PendingEdits,
    serialise,
    touch_agent_patch,
)
from fibsem.server.app_context import content_hash, find_item, item_pose_positions
from fibsem.server.config_patch import PatchError, apply_patch, check_patch
from fibsem.server.task_configs import (
    config_version,
    patch_config,
    plain_changes,
    workflow_field_refusal,
)
from fibsem.workflows.status import TaskStatus

# The item-document fields an agent may patch: what a lamella IS — geometry,
# verdict, notes. Everything else on the object (poses, ids, paths, history)
# is either derived, hardware-adjacent, an *outcome* rather than an input
# (milling_angle is recorded from where Setup put the stage — writing it
# would move nothing), or the record itself.
ITEM_PATCH_FIELDS = ("poi", "alignment_area", "description", "defect")


def item_fields_version(lamella) -> str:
    """A content hash naming the state of an item's patchable fields and poses.

    The item document's version, hashing the ITEM_PATCH_FIELDS and where the
    poses put the stage — so a task re-recording geometry or moving the
    milling pose rotates it, and a pending patch goes stale (FIB-1420): a POI
    is in the milling frame, so one chosen before the pose moved names a
    different place after. History growing does not rotate it.
    """
    from fibsem.workflows.events import to_plain

    fields = {
        name: to_plain(getattr(lamella, name, None)) for name in ITEM_PATCH_FIELDS
    }
    fields["poses"] = item_pose_positions(lamella)
    return content_hash(fields)


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


def task_config_at(experiment, level: str, item_name: str, task_name: str):
    """The task config an edit at *level* names: an item's (``item``) or the
    protocol's, with the lamella holding it (None for the protocol's), and
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
    settle (``ui/protocol/edit_recording.py``)."""
    return _save(experiment, what, level == "protocol")


def patch_task_config(
    experiment, level: str, item_name: str, task_name: str, patch: dict, version: str
) -> dict:
    """Patch one task config: an item's (*level* ``item``) or the protocol's.
    The version check sits beside the apply, so an edit landing in between is
    refused as stale, never half-merged."""
    _lamella, config, refusal = task_config_at(experiment, level, item_name, task_name)
    if refusal:
        return {"applied": False, **refusal}
    patched = patch_config(config, patch, version)
    if not patched["applied"]:
        return patched
    where = f"{item_name} / {task_name}" if level == "item" else task_name
    logging.info(
        f"agent config patch applied ({level}): {where}: "
        + ", ".join(
            f"{c['path']}: {c['old']!r} -> {c['new']!r}" for c in patched["changes"]
        )
    )
    saved = save_edits(experiment, level, "agent config patch")
    result = {
        "applied": True,
        "saved": saved,
        "task_name": task_name,
        "changes": patched["changes"],
        "version": patched["version"],
    }
    if level == "item":
        result["item_name"] = item_name
    return result


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


# The workflow fields a task-config patch may not set, and the verb that sets each.
_ITEM_NONE_TYPES = {"description": str}


def _not_editable(path: str) -> Optional[str]:
    if path.split(".", 1)[0] in ITEM_PATCH_FIELDS:
        return None
    return (
        f"{path!r} is not an editable item field; editable: {sorted(ITEM_PATCH_FIELDS)}"
    )


def item_patch_refusal(lamella, patch: dict) -> Optional[str]:
    """Why :func:`patch_item_fields` would refuse *patch*, setting nothing; None
    when it would apply. An alignment area is judged whole, once applied."""
    for path in patch:
        why = _not_editable(path)
        if why is not None:
            return why
    try:
        check_patch(lamella, patch, none_types=_ITEM_NONE_TYPES)
    except PatchError as exc:
        return str(exc)
    return None


def patch_item_fields(experiment, item_name: str, patch: dict, version: str) -> dict:
    """The App Server's item patch: :func:`edit_item_fields` on the item named
    *item_name*, saved."""
    lamella, refusal = find_item(experiment, item_name)
    if refusal:
        return {"applied": False, **refusal}
    result = edit_item_fields(lamella, patch, version)
    if result["applied"]:
        result["saved"] = _save(experiment, "agent item patch")
    return result


def edit_item_fields(lamella, patch: dict, version: str) -> dict:
    """Patch the lamella's own editable fields, ITEM_PATCH_FIELDS only: the one
    edit of them, the App Server's and the lamella editors'. Not saved.

    Alignment validity is checked after the apply and reverted on failure (a
    rectangle is only judgeable whole). A defect edit is a new record (see
    :func:`_edited_defect`). Moving the POI moves the patterns that follow
    it, and moving the area moves the fiducial not milled yet.
    """
    for path in patch:
        why = _not_editable(path)
        if why is not None:
            return {"applied": False, "invalid_patch": why, "path": path}
    if item_fields_version(lamella) != version:
        return {"applied": False, "stale": True}
    try:
        changes = apply_patch(lamella, patch, none_types=_ITEM_NONE_TYPES)
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
        lamella.defect = _edited_defect(lamella.defect, changes)
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
        f"item patch applied: {lamella.name}: "
        + ", ".join(f"{p}: {old!r} -> {new!r}" for p, old, new in changes)
        + (f" (patterns synced: {', '.join(synced_tasks)})" if synced_tasks else "")
    )
    return {
        "applied": True,
        "item_name": lamella.name,
        "synced_tasks": synced_tasks,
        "changes": plain_changes(changes),
        "version": item_fields_version(lamella),
    }


def _edited_defect(record, changes) -> Any:
    """The defect record after a patch, as a new record, so every display of
    it redraws (they listen for its replacement). A new verdict starts afresh,
    keeping only the fields the patch names; it is stamped when judged and
    unstamped when taken back."""
    from fibsem.util.timestamps import now
    from fibsem.workflows.quality import Verdict

    named = {
        p.split(".", 1)[1]: new for p, _o, new in changes if p.startswith("defect.")
    }
    old_verdict = next(
        (old for p, old, _n in changes if p == "defect.verdict"), record.verdict
    )
    if record.verdict is not old_verdict:
        record = type(record)(**named)
    else:
        record = copy.copy(record)
    record.updated_at = None if record.verdict is Verdict.UNASSESSED else now()
    return record


def poi_patch(point) -> dict:
    """The item patch that moves an item's point of interest to *point*."""
    return {"poi.x": float(point.x), "poi.y": float(point.y)}


def apply_correlation(
    result,
    lamella,
    run_folder: Optional[str],
    root: Any,
    move_poi: Callable[[Any], dict],
    record: Callable[[str, Dict[str, Any]], None],
) -> dict:
    """Record correlation *result* with *record*, then move *lamella*'s point of
    interest to its target with *move_poi* (an item patch of ``poi``): the
    correlation dialog's Continue and the App Server's apply_item_correlation.
    Recorded first, so the record reads the correlation before the point it
    moved. *result* has a fitted point."""
    from fibsem.correlation.history import correlation_record

    try:
        record("correlation", correlation_record(result, lamella, run_folder, root))
    except Exception:  # noqa: BLE001 - recording must not matter
        logging.debug("could not record the correlation", exc_info=True)
    return move_poi(result.poi[0].px_m)


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
    saved = save_edits(experiment, level, "stage reorder")
    return {
        "applied": True,
        "saved": saved,
        "task_name": task_name,
        "milling_key": milling_key,
        "order": list(order),
        "changes": plain_changes(
            [(f"milling.{milling_key}.stages", names, list(order))]
        ),
        "version": config_version(config),
    }


def apply_protocol_to_item(
    experiment, item_name: str, task_names: Optional[list]
) -> dict:
    """Copy the protocol's task configs onto one item, as the editors' apply
    does (``Experiment.apply_lamella_config``): the item keeps its milling
    pattern positions. No version check. ``task_names`` of None means every
    task the protocol defines."""
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
    _, saved = copy_task_configs(experiment, [lamella.name], names)
    return {
        "applied": True,
        "saved": saved,
        "item_name": item_name,
        "task_names": names,
    }


def copy_task_configs(
    experiment,
    lamella_names: List[str],
    task_names: List[str],
    source_name: Optional[str] = None,
    update_base_protocol: bool = False,
) -> Tuple[int, bool]:
    """Copy *task_names*' configs onto *lamella_names*, from the protocol or
    from lamella *source_name*, each keeping its milling pattern positions
    (``Experiment.apply_lamella_config``), and onto the protocol too when
    *update_base_protocol*; saved. How many lamellae were updated, and whether
    it saved. The lamella editor's Apply to other lamellae and the App
    Server's apply_protocol."""
    count = experiment.apply_lamella_config(
        lamella_names=lamella_names,
        task_names=task_names,
        source_lamella_name=source_name,
        update_base_protocol=update_base_protocol,
    )
    return count, _save(experiment, "config apply", update_base_protocol)


def set_reference_imaging(
    experiment,
    task_names: Sequence[str],
    reference_imaging: Any,
    milling_fov: float,
    lamellae: Sequence[Any] = (),
) -> Tuple[int, List[Tuple[str, str]]]:
    """Global edit: set the reference imaging and each milling config's field
    of view on *task_names*, the protocol's and those of each of *lamellae*
    that has the task; saved with the protocol. Only these two settings: a
    lamella keeps everything else it was tuned to (FIB-1071).

    A lamella's task that is running is left alone: the task holds that config
    object, so an edit would change its settings mid-run. Returns how many of
    the protocol's tasks have milling, and the (lamella, task) pairs skipped
    as running."""
    milling = 0
    skipped: List[Tuple[str, str]] = []
    for task_name in task_names:
        task_config = experiment.task_protocol.task_config.get(task_name)
        if not task_config:
            continue
        if _set_imaging(task_config, reference_imaging, milling_fov):
            milling += 1
        for lamella in lamellae:
            lamella_config = lamella.task_config.get(task_name)
            if lamella_config is None:
                continue
            if _is_running(lamella, task_name):
                skipped.append((lamella.name, task_name))
                continue
            _set_imaging(lamella_config, reference_imaging, milling_fov)
    _save(experiment, "global edit", save_protocol=True)
    return milling, skipped


def _set_imaging(task_config: Any, reference_imaging: Any, milling_fov: float) -> bool:
    """Set a task's reference imaging, and each milling config's field of view;
    whether it has milling. The save path is kept: the dialog's settings carry
    none, and a None path fails the next image saved with them."""
    path = task_config.reference_imaging.imaging.path
    task_config.reference_imaging = copy.deepcopy(reference_imaging)
    task_config.reference_imaging.imaging.path = path
    for milling_config in (task_config.milling or {}).values():
        milling_config.field_of_view = milling_fov
    return bool(task_config.milling)


def _is_running(lamella: Any, task_name: str) -> bool:
    """Whether *task_name* is the task running on *lamella* now."""
    state = getattr(lamella, "task_state", None)
    return (
        state is not None
        and state.name == task_name
        and state.status == TaskStatus.InProgress
    )
