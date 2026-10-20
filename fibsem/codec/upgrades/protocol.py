"""Task protocols, their configs and experiments as older versions wrote them, in
the current shape.

- A task config's parameters sat under ``parameters``; they are at its top level
  now, beside ``milling`` and ``reference_imaging``.
- A protocol said ``_id``, ``tasks`` and ``order``; it says ``id``, ``task_config``
  and ``workflow_config`` (``{tasks: [{name, required, scheduled_at}]}``) now.
- An AutoLamella protocol kept its plan under ``workflow``, with each task's
  ``attention`` and ``requires`` there. Both are on the task's config now. v0.5.2
  and earlier wrote a ``supervise`` flag in place of ``attention``.
- A plan entry that is null was read as an optional task with no name.

Presence, not a version, decides each step, and a key already in its new place
wins. Each step copies what it is given.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

SUPERVISED = "supervised"
AUTOMATED = "automated"


def task_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """A task config with its ``parameters`` at the top level."""
    data = dict(data or {})
    parameters = data.pop("parameters", None)
    if isinstance(parameters, dict):
        data = {**parameters, **data}
    return data


def run_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    """A run plan whose every entry is a dict; a task named on its own is
    required."""
    data = dict(data or {})
    tasks = []
    for entry in data.get("tasks") or []:
        if entry is None:
            entry = {"name": "", "required": False}
        elif isinstance(entry, str):
            entry = {"name": entry}
        tasks.append(dict(entry))
    data["tasks"] = tasks
    return data


def task_protocol(data: Dict[str, Any]) -> Dict[str, Any]:
    """An engine protocol (grid, volume) with the current keys."""
    data = copy.deepcopy(data or {})
    if "_id" in data:
        data.setdefault("id", data.pop("_id"))
    if "tasks" in data:
        data.setdefault("task_config", data.pop("tasks"))
    data["task_config"] = {
        name: task_config(config)
        for name, config in (data.get("task_config") or {}).items()
    }
    if "order" in data:
        order = data.pop("order") or []
        data.setdefault("workflow_config", {"tasks": list(order)})
    data["workflow_config"] = run_plan(data.get("workflow_config") or {})
    return data


def autolamella_protocol(data: Dict[str, Any]) -> Dict[str, Any]:
    """An AutoLamella protocol with the current keys, each task's attention and
    requires on its config, and its grid tasks upgraded too."""
    data = copy.deepcopy(data or {})
    if "workflow" in data:
        data.setdefault("workflow_config", data.pop("workflow"))
    data = task_protocol(data)
    configs = data["task_config"]
    for entry in data["workflow_config"]["tasks"]:
        supervise = entry.pop("supervise", None)
        attention = entry.pop("attention", None)
        requires = entry.pop("requires", None)
        config = configs.get(entry.get("name"))
        if config is None:
            continue
        if attention is None and supervise is not None:
            attention = SUPERVISED if supervise else AUTOMATED
        if attention is not None:
            config.setdefault("attention", attention)
        if requires is not None:
            config.setdefault("requires", requires)
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
