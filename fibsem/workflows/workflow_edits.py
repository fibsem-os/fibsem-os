"""Edits to an experiment's workflow that the window and the agent both make.

Each applies the change through the protocol's own rule, saves it with
the protocol and puts it on the session's record as ``workflow_changed``,
through the ``record(kind, payload)`` it is given. Each answers a plain dict:
what was applied, or why not.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from fibsem.workflows.attention import Attention

WORKFLOW_CHANGED = "workflow_changed"

Record = Callable[[str, Dict[str, Any]], None]


def save_workflow_change(experiment, change: Dict[str, Any], record: Record) -> bool:
    """Save the protocol after a workflow edit and record *change*; whether it
    saved. A failed save leaves the change in memory, and on the record."""
    saved = True
    try:
        experiment.save(save_protocol=True)
    except Exception:
        saved = False
        logging.exception(
            "save after a workflow change failed; the change is applied in "
            "memory but not yet on disk"
        )
    record(WORKFLOW_CHANGED, change)
    return saved


def set_supervision(
    experiment, task_name: str, supervise: bool, record: Record
) -> Dict[str, Any]:
    """Set whether *task_name* asks for supervision, in the live protocol.

    The workflow reads it at every decision point, so a change takes effect
    at the next one, mid-run included. A supervised proposal goes to the
    operator and the agent alike, and the first decision wins (FIB-1035).
    """
    protocol = experiment.task_protocol
    supervise = bool(supervise)
    why = protocol.set_attention(
        task_name, Attention.supervised if supervise else Attention.automated
    )
    if why:
        return {
            "applied": False,
            "error": why,
            "task_names": protocol.workflow_config.workflow,
        }
    saved = save_workflow_change(
        experiment,
        {"field": "attention", "task_name": task_name, "supervise": supervise},
        record,
    )
    return {
        "applied": True,
        "saved": saved,
        "task_name": task_name,
        "supervise": supervise,
    }


def add_task(
    experiment, task_type: str, task_name: str, record: Record
) -> Dict[str, Any]:
    """Add a task of a type the protocol registers (``experiment.add_task``):
    its default settings, at the end of the workflow."""
    try:
        experiment.add_task(task_type, task_name)
    except ValueError as exc:
        return {"applied": False, "error": str(exc)}
    name = task_name.strip()
    saved = save_workflow_change(experiment, {"field": "tasks", "added": name}, record)
    return {"applied": True, "saved": saved, "task_name": name, "task_type": task_type}


def remove_task(experiment, task_name: str, record: Record) -> Dict[str, Any]:
    """Remove a task from the protocol and the workflow
    (``experiment.remove_task``). Task history stays: it is the record of
    what ran."""
    experiment.remove_task(task_name)
    saved = save_workflow_change(
        experiment, {"field": "tasks", "removed": task_name}, record
    )
    return {"applied": True, "saved": saved}


def reorder_tasks(experiment, order: List[str], record: Record) -> Dict[str, Any]:
    """Put the workflow's tasks in *order*: every task, each once. Refused when
    a task would come before a task it requires."""
    why = experiment.task_protocol.reorder(list(order))
    if why:
        return {"applied": False, "error": why}
    saved = save_workflow_change(
        experiment, {"field": "order", "order": list(order)}, record
    )
    return {"applied": True, "saved": saved}


def set_requirements(
    experiment,
    task_name: str,
    requires: Optional[List[str]],
    required: Optional[bool],
    record: Record,
) -> Dict[str, Any]:
    """Set the tasks an item must have completed before *task_name*, and
    whether it is required. Refused when a requirement is not an earlier
    task."""
    protocol = experiment.task_protocol
    why = protocol.set_requirements(task_name, requires, required)
    if why:
        return {"applied": False, "error": why}
    change = {
        "task_name": task_name,
        "requires": protocol.requirements(task_name),
        "required": protocol.workflow_config.get(task_name).required,
    }
    saved = save_workflow_change(
        experiment, {"field": "requirements", **change}, record
    )
    return {"applied": True, "saved": saved, **change}


def set_schedule(
    experiment, task_name: str, scheduled_at: Optional[datetime], record: Record
) -> Dict[str, Any]:
    """Set (or clear, None) when *task_name* may start: up to two days ahead
    (``RunPlan.set_schedule``)."""
    config = experiment.task_protocol.workflow_config
    why = config.set_schedule(task_name, scheduled_at)
    if why:
        return {
            "applied": False,
            "error": why,
            "error_type": "not_found"
            if config.get(task_name) is None
            else "invalid_value",
            "task_names": config.workflow,
        }
    iso = scheduled_at.isoformat() if scheduled_at else None
    saved = save_workflow_change(
        experiment,
        {"field": "scheduled_at", "task_name": task_name, "scheduled_at": iso},
        record,
    )
    return {
        "applied": True,
        "saved": saved,
        "task_name": task_name,
        "scheduled_at": iso,
    }
