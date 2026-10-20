"""Edits to an experiment's workflow that the window and the agent both make.

Each applies the change through the workflow config's own rule, saves it with
the protocol and puts it on the session's record as ``workflow_changed``,
through the ``record(kind, payload)`` it is given. Each answers a plain dict:
what was applied, or why not.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

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
    config = experiment.task_protocol.workflow_config
    supervise = bool(supervise)
    why = config.set_attention(
        task_name, Attention.supervised if supervise else Attention.automated
    )
    if why:
        return {"applied": False, "error": why, "task_names": config.workflow}
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
