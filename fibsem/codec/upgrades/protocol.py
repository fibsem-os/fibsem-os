"""Task protocols, their configs and experiments as older versions wrote them, in
the current shape.

- A task config's parameters sat under ``parameters``; they are at its top level
  now, beside ``milling`` and ``reference_imaging``. Its milling recipes are
  upgraded too (`fibsem.codec.upgrades.milling`).
- A protocol said ``_id``, ``tasks`` and ``order``; it says ``id``, ``task_config``
  and ``workflow_config`` (``{tasks: [{name, required, scheduled_at}]}``) now.
- A task's ``attention`` and ``requires`` are on its plan entry; format 3 put
  them on its config. v0.5.2 and earlier wrote a ``supervise`` flag in place of
  ``attention``.
- An AutoLamella protocol kept its plan under ``workflow``. One from before
  task protocols becomes one (`fibsem.codec.upgrades.pre_task_protocol`).
- A v0.5.0 lamella kept its objective position beside its poses.
- A plan entry's attention or requires that doesn't read is automated, or
  none, with a warning: one unreadable field must not drop the protocol.
- A config's channels, autofocus and overview settings are read as their own
  upgrades read them (`typed_fields`).
- A plan entry that is null was read as an optional task with no name. A
  protocol with no plan at all plans every task it has.

Presence, not a version, decides each step, and a key already in its new place
wins. Each step copies what it is given.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import typing
from typing import Any, Callable, Dict

from fibsem.codec.upgrades import milling, pre_task_protocol, records

SUPERVISED = "supervised"
AUTOMATED = "automated"


# On a task's plan entry, not its config.
PLAN_FIELDS = ("attention", "requires")


def task_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """A task config with its ``parameters`` at the top level, its milling in
    the current shape, and nothing of its plan entry."""
    data = dict(data or {})
    for key in PLAN_FIELDS:
        data.pop(key, None)
    parameters = data.pop("parameters", None)
    if isinstance(parameters, dict):
        data = {**parameters, **data}
    if "milling" in data:
        data["milling"] = milling.task_milling(data["milling"])
    return data


def _names(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def run_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    """A run plan whose every entry is a dict, with ``attention`` for
    ``supervise``; a task named on its own is required."""
    from fibsem.workflows.attention import attention_from

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
        where = f"task '{entry.get('name', '')}'"
        if "attention" in entry:
            entry["attention"] = attention_from(entry["attention"], where).value
        requires = entry.get("requires")
        if requires is not None and not _names(requires):
            logging.warning(
                f"{where} has requires {requires!r}, not a list of task names; "
                "read as none."
            )
            entry["requires"] = []
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


def lamella(data: Dict[str, Any]) -> Dict[str, Any]:
    """A lamella whose configs and records are in the current shape, with the
    objective position v0.5.0 kept beside its poses on its fluorescence pose."""
    data = records.item(data, verdict="defect")
    data["task_config"] = {
        name: task_config(config)
        for name, config in (data.get("task_config") or {}).items()
    }
    objective = data.pop("objective_position", None)
    fluorescence = (data.get("poses") or {}).get("FLUORESCENCE")
    if objective is not None and fluorescence is not None:
        if fluorescence.get("objective_position") is None:
            data["poses"] = {
                **data["poses"],
                "FLUORESCENCE": {**fluorescence, "objective_position": objective},
            }
    return data


def experiment(data: Dict[str, Any]) -> Dict[str, Any]:
    """An experiment.yaml whose lamellae and grids are in the current shape. A
    ``protocol`` it carries was never read, and is dropped."""
    data = dict(data or {})
    data.pop("protocol", None)
    data["positions"] = [lamella(p) for p in data.get("positions") or []]
    data["grids"] = [records.item(g) for g in data.get("grids") or []]
    return data


def _upgrades() -> Dict[type, Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """The upgrade for each type a task config holds that has one."""
    from fibsem.autofunctions.autofocus import AutoFocusSettings
    from fibsem.codec.upgrades.autofocus import settings as autofocus
    from fibsem.codec.upgrades.fluorescence import channel
    from fibsem.codec.upgrades.overview import acquisition_settings
    from fibsem.fm.structures import ChannelSettings
    from fibsem.structures import OverviewAcquisitionSettings

    return {
        ChannelSettings: channel,
        AutoFocusSettings: autofocus,
        OverviewAcquisitionSettings: acquisition_settings,
    }


def typed_fields(cls: type, data: Dict[str, Any]) -> Dict[str, Any]:
    """``data`` for the task config class ``cls``, with each field whose type
    has an upgrade (a channel, a list of them) upgraded."""
    upgrades = _upgrades()
    hints = typing.get_type_hints(cls, localns={t.__name__: t for t in upgrades})
    data = dict(data)
    for f in dataclasses.fields(cls):
        value = data.get(f.name)
        if value is None:
            continue
        hint = hints.get(f.name)
        args = typing.get_args(hint)
        if hint is int and isinstance(value, (float, str)):
            # older files hold some as text, or with a fraction: whole units
            try:
                data[f.name] = int(float(value))
            except ValueError:
                pass
        elif hint in upgrades and isinstance(value, dict):
            data[f.name] = upgrades[hint](value)
        elif typing.get_origin(hint) is list and args and args[0] in upgrades:
            data[f.name] = [upgrades[args[0]](v) for v in value]
        elif len(args) == 2 and type(None) in args:  # Optional
            (inner,) = [a for a in args if a is not type(None)]
            if inner in upgrades and isinstance(value, dict):
                data[f.name] = upgrades[inner](value)
    return data
