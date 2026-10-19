"""A task's settings, and the tasks a protocol offers: shared by every app.

Flat on disk: every field at the top level, nested dataclasses through their own
``to_dict`` / ``from_dict``, enums by name. An app's config classes extend
:class:`TaskConfig`; its protocol extends :class:`TaskProtocol` and says how a
saved config becomes a typed one (its task registry).
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Union,
    get_type_hints,
)

from fibsem.structures import Configurable, Display, param
from fibsem.workflows.attention import Attention, attention_from

if TYPE_CHECKING:
    from fibsem.plugins.loader import PluginRegistry


@dataclass
class TaskConfig(Configurable, ABC):
    """Configuration for one task: the fields that say how it takes part in the
    workflow, and the task's own parameters after them."""

    task_type: ClassVar[str]
    display_name: ClassVar[str]
    task_name: str = param(
        "",
        doc="The task's name, unique within the protocol: the key the workflow knows it by.",
    )  # unique within a protocol; the key the workflow uses
    # Whether its questions are asked (supervised) or recorded for someone to
    # check (automated). One per task name, shared by every item like the rest
    # of the config.
    attention: Attention = param(
        Attention.automated,
        doc="Whether the task's questions are asked while it runs (supervised) or recorded Unreviewed without asking (automated).",
        display=Display("Attention"),
    )
    # The tasks, by name, whose result this one uses: it waits while one of
    # them is still queued for the item, and is skipped when one of them did
    # not complete. Empty: it runs whatever happened before it on the item.
    requires: List[str] = param(
        doc="The tasks, by name, whose result this one uses. It waits while one of them is still queued for the item, and is skipped when one did not complete. Empty runs whatever happened before it.",
        display=Display("Requires"),
        default_factory=list,
    )

    @property
    def estimated_duration(self) -> float:
        """Conservative forward estimate of one run of this task on one item, in
        seconds: what the run confirmation, the timeline and Add to queue quote.
        Zero here; each task type adds what its run does (see
        :mod:`fibsem.timing`).
        """
        return 0.0

    @property
    def parameters(self) -> tuple:
        """The task-specific fields, in declaration order: what a form shows."""
        return tuple(f.name for f in fields(self) if f.name not in WORKFLOW_FIELDS)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "task_type": self.task_type,
            "task_name": self.task_name,
            "attention": self.attention.value,
            "requires": list(self.requires),
        }
        for name in self.parameters:
            data[name] = _serialise(getattr(self, name))
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        hints = get_type_hints(cls)
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            if f.name == "attention":
                kwargs[f.name] = attention_from(
                    data[f.name], f"task '{data.get('task_name', '')}'"
                )
                continue
            if f.name == "requires":
                kwargs[f.name] = _requires(data[f.name], data.get("task_name", ""))
                continue
            kwargs[f.name] = _deserialise(hints.get(f.name), data[f.name])
        unknown = set(data) - {f.name for f in fields(cls)} - {"task_type"}
        for key in sorted(unknown):
            logging.warning(f"Unknown field '{key}' in {cls.__name__}; ignored.")
        return cls(**kwargs)


# Fields that say how a task takes part in the workflow, not how it runs: not
# form parameters, and serialised by the base.
WORKFLOW_FIELDS = ("task_name", "attention", "requires")


def _requires(value: Any, task_name: str) -> List[str]:
    """Stored requirements as task names, or none with a warning: a malformed
    value must not drop the task from the protocol."""
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    logging.warning(
        f"Task '{task_name}' has requires {value!r}, not a list of task "
        "names; read as none."
    )
    return []


def _serialise(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialise(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialise(v) for k, v in value.items()}
    return value


def _deserialise(hint: Any, value: Any) -> Any:
    """Rebuild a field from its serialised form, guided by the annotation."""
    if value is None or hint is None:
        return value
    origin = getattr(hint, "__origin__", None)
    if origin is Union:  # Optional[X]
        inner = [a for a in hint.__args__ if a is not type(None)]
        return _deserialise(inner[0], value) if len(inner) == 1 else value
    if origin in (list, List) and isinstance(value, list):  # List[X]
        (item_hint,) = getattr(hint, "__args__", (None,))
        return [_deserialise(item_hint, v) for v in value]
    if isinstance(hint, type):
        if issubclass(hint, Enum) and isinstance(value, str):
            return hint[value]
        if hasattr(hint, "from_dict") and isinstance(value, dict):
            return hint.from_dict(value)
    return value


def load_task_config(
    registry: "PluginRegistry", data: Dict[str, Any], kind: str = "task"
) -> TaskConfig:
    """A typed config from its saved form, via ``registry`` (task_type -> task)."""
    task_type = data.get("task_type")
    task_cls = registry.get(task_type)
    if task_cls is None:
        raise KeyError(f"{kind.capitalize()} type '{task_type}' is not registered.")
    return task_cls.config_cls.from_dict(data)


def load_task_configs(
    registry: "PluginRegistry",
    ddict: Optional[Dict[str, Dict[str, Any]]],
    kind: str = "task",
) -> Dict[str, TaskConfig]:
    """name -> config, skipping (with a warning) any type this build does not know."""
    configs: Dict[str, TaskConfig] = {}
    for name, data in (ddict or {}).items():
        try:
            config = load_task_config(registry, data, kind)
        except KeyError as e:
            logging.warning(f"{kind.capitalize()} '{name}' skipped: {e}")
            continue
        config.task_name = name
        configs[name] = config
    return configs


@dataclass
class TaskProtocol:
    """The tasks a protocol offers, and their settings. Which tasks run in a
    session, and in what order, is chosen at run time; ``order`` is the default a
    task list shows. An app's protocol says how its saved configs load
    (:meth:`load_task_configs`) and what its tasks are called (``NOUN``)."""

    NOUN: ClassVar[str] = "task"

    name: str = "Task Protocol"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_config: Dict[str, TaskConfig] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)

    @classmethod
    def load_task_configs(
        cls, ddict: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, TaskConfig]:
        """name -> typed config, from the saved ``tasks`` section."""
        raise NotImplementedError

    def add(self, config: TaskConfig) -> TaskConfig:
        if not config.task_name:
            raise ValueError(f"A {self.NOUN} config needs a task_name.")
        self.task_config[config.task_name] = config
        if config.task_name not in self.order:
            self.order.append(config.task_name)
        return config

    def remove(self, task_name: str) -> None:
        self.task_config.pop(task_name, None)
        if task_name in self.order:
            self.order.remove(task_name)

    def reorder(self, order: List[str]) -> str:
        """Put the tasks in *order*: every task, each once. Why not, with
        nothing changed, or ""."""
        if sorted(order) != sorted(self.ordered_task_names):
            return f"Give every {self.NOUN}, each once."
        self.order = list(order)
        return ""

    def set_requires(self, task_name: str, requires: List[str]) -> str:
        """Set the tasks *task_name* requires on the same item, kept in run
        order. Why not, with nothing changed, or ""."""
        config = self.task_config.get(task_name)
        if config is None:
            return f"No {self.NOUN} named {task_name!r}."
        others = [n for n in self.ordered_task_names if n != task_name]
        wrong = [name for name in requires if name not in others]
        if wrong:
            return (
                f"{task_name} can require {', '.join(others) or 'no task'}; "
                f"not {', '.join(wrong)}."
            )
        config.requires = [n for n in others if n in requires]
        return ""

    def requirements(self, task_name: str) -> List[str]:
        """The tasks ``task_name`` requires on the same item; none for a task
        the protocol does not have."""
        config = self.task_config.get(task_name)
        return list(config.requires) if config is not None else []

    @property
    def ordered_task_names(self) -> List[str]:
        """`order` first, then anything in `task_config` it forgot to mention."""
        names = [n for n in self.order if n in self.task_config]
        names += [n for n in self.task_config if n not in names]
        return names

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "name": self.name,
            "tasks": {k: v.to_dict() for k, v in self.task_config.items()},
            "order": list(self.ordered_task_names),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        data = data or {}
        protocol = cls(
            name=data.get("name", cls.name),
            task_config=cls.load_task_configs(data.get("tasks", {})),
            order=list(data.get("order", [])),
        )
        if data.get("_id"):
            protocol.id = data["_id"]
        return protocol
