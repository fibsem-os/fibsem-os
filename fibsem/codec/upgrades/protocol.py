"""Task protocols, their configs and experiments as older versions wrote them, in
the current shape.

- A task config's parameters sat under ``parameters``; they are at its top level
  now, beside ``milling`` and ``reference_imaging``.
- A protocol said ``_id``, ``tasks`` and ``order``; it says ``id``, ``task_config``
  and ``workflow_config`` (``{tasks: [{name, required, scheduled_at}]}``) now.
- A task's ``attention`` and ``requires`` are on its plan entry; format 3 put
  them on its config. v0.5.2 and earlier wrote a ``supervise`` flag in place of
  ``attention``.
- An AutoLamella protocol kept its plan under ``workflow``. One from before
  task protocols becomes one (`fibsem.codec.upgrades.pre_task_protocol`).
- A plan entry that is null was read as an optional task with no name. A
  protocol with no plan at all plans every task it has.

Presence, not a version, decides each step, and a key already in its new place
wins. Each step copies what it is given.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from fibsem.codec.upgrades import pre_task_protocol

SUPERVISED = "supervised"
AUTOMATED = "automated"


# On a task's plan entry, not its config.
PLAN_FIELDS = ("attention", "requires")


def task_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """A task config with its ``parameters`` at the top level, and nothing of
    its plan entry."""
    data = dict(data or {})
    for key in PLAN_FIELDS:
        data.pop(key, None)
    parameters = data.pop("parameters", None)
    if isinstance(parameters, dict):
        data = {**parameters, **data}
    return data


def run_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    """A run plan whose every entry is a dict, with ``attention`` for
    ``supervise``; a task named on its own is required."""
    data = dict(data or {})
    tasks = []
    for entry in data.get("tasks") or []:
        if entry is None:
            entry = {"name": "", "required": False}
        elif isinstance(entry, str):
            entry = {"name": entry}
        entry = dict(entry)
        supervise = entry.pop("supervise", None)
        if supervise is not None:
            entry.setdefault("attention", SUPERVISED if supervise else AUTOMATED)
        tasks.append(entry)
    data["tasks"] = tasks
    return data


def task_protocol(data: Dict[str, Any]) -> Dict[str, Any]:
    """An engine protocol (grid, volume) with the current keys."""
    data = copy.deepcopy(data or {})
    if "_id" in data:
        data.setdefault("id", data.pop("_id"))
    if "tasks" in data:
        data.setdefault("task_config", data.pop("tasks"))
    configs = data.get("task_config") or {}
    if "order" in data:
        order = data.pop("order") or []
        data.setdefault("workflow_config", {"tasks": list(order)})
    # with no plan at all, every task is in it
    data.setdefault("workflow_config", {"tasks": list(configs)})
    data["workflow_config"] = run_plan(data.get("workflow_config") or {})
    for entry in data["workflow_config"]["tasks"]:
        config = configs.get(entry.get("name")) or {}
        for key in PLAN_FIELDS:
            if key in config:
                entry.setdefault(key, config[key])
    data["task_config"] = {
        name: task_config(config) for name, config in configs.items()
    }
    return data


def autolamella_protocol(data: Dict[str, Any]) -> Dict[str, Any]:
    """An AutoLamella protocol with the current keys, and its grid tasks
    upgraded too."""
    if pre_task_protocol.is_pre_task(data):
        data = pre_task_protocol.upgrade(data)
    data = copy.deepcopy(data or {})
    if "workflow" in data:
        data.setdefault("workflow_config", data.pop("workflow"))
    data = task_protocol(data)
    if data.get("grid_tasks") is not None:
        data["grid_tasks"] = task_protocol(data["grid_tasks"])
    return data


def experiment(data: Dict[str, Any]) -> Dict[str, Any]:
    """An experiment.yaml whose lamellae's configs and protocol (``protocol``,
    when it carries one) are in the current shape."""
    data = dict(data or {})
    data["positions"] = [
        {
            **position,
            "task_config": {
                name: task_config(config)
                for name, config in (position.get("task_config") or {}).items()
            },
        }
        for position in data.get("positions") or []
    ]
    if data.get("protocol") is not None:
        data["protocol"] = autolamella_protocol(data["protocol"])
    return data
