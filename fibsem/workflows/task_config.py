"""A task's settings, the tasks a protocol offers, and the plan a run follows:
shared by every app.

Flat on disk: every field at the top level, nested dataclasses through their own
``to_dict`` / ``from_dict``, enums by name. An app's config classes extend
:class:`TaskConfig`; its protocol extends :class:`TaskProtocol` and says how a
saved config becomes a typed one (its task registry). Older shapes are read
through `fibsem.codec.upgrades.protocol`.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Union,
    get_type_hints,
)

import yaml
from psygnal import evented

from fibsem.codec.upgrades import protocol as upgrade
from fibsem.structures import Configurable, param
from fibsem.util.timestamps import now
from fibsem.workflows.attention import Attention, attention_from

if TYPE_CHECKING:
    from fibsem.plugins.loader import PluginRegistry


@dataclass
class TaskConfig(Configurable, ABC):
    """Configuration for one task: its name, and the task's own parameters
    after it. How it takes part in a run is its plan entry's (:class:`TaskPlan`)."""

    task_type: ClassVar[str]
    display_name: ClassVar[str]
    # Parameters this task type used to have. Saved protocols and experiments
    # still carry them; they are dropped on load without the "Unknown field"
    # warning, which is for keys nobody recognises.
    retired_parameters: ClassVar[FrozenSet[str]] = frozenset()
    task_name: str = param(
        "",
        doc="The task's name, unique within the protocol: the key the workflow knows it by.",
    )  # unique within a protocol; the key the workflow uses

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
        }
        # the form's parameters first, then what is saved beside them
        names = list(self.parameters)
        names += [
            f.name
            for f in fields(self)
            if f.name not in WORKFLOW_FIELDS and f.name not in names
        ]
        for name in names:
            data[name] = _serialise(getattr(self, name))
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        hints = get_type_hints(cls)
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            kwargs[f.name] = _deserialise(hints.get(f.name), data[f.name])
        unknown = (
            set(data)
            - {f.name for f in fields(cls)}
            - {"task_type"}
            - cls.retired_parameters
        )
        for key in sorted(unknown):
            logging.warning(f"Unknown field '{key}' in {cls.__name__}; ignored.")
        return cls(**kwargs)


# The field that names a task, not how it runs: not a form parameter, and
# serialised by the base.
WORKFLOW_FIELDS = ("task_name",)


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
    if origin in (dict, Dict) and isinstance(value, dict):  # Dict[str, X]
        _, item_hint = getattr(hint, "__args__", (None, None))
        return {k: _deserialise(item_hint, v) for k, v in value.items()}
    if isinstance(hint, type):
        if issubclass(hint, Enum) and isinstance(value, str):
            return hint[value]
        if hasattr(hint, "from_dict") and isinstance(value, dict):
            return hint.from_dict(value)
        if hint in (int, float) and not isinstance(value, bool):
            return _number(hint, value)
    return value


def _number(hint: type, value: Any) -> Any:
    """A number field's value as its type: older files hold some as strings,
    and an int field written as 10.0. Left as it is, with a warning, when it
    is not a number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        logging.warning(f"{value!r} is not a number; kept as it is.")
        return value
    return int(number) if hint is int else number


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
            config = load_task_config(registry, {**data, "task_name": name}, kind)
        except KeyError as e:
            logging.warning(f"{kind.capitalize()} '{name}' skipped: {e}")
            continue
        configs[name] = config
    return configs


# A task may be scheduled to start up to this many days ahead.
SCHEDULE_DAYS_AHEAD = 2


@evented
@dataclass
class WorkflowOptions:
    """How a run ends: whether the beams go off and the summary dialog opens."""

    turn_beams_off: bool = False
    # Open the run summary dialog when a run ends. The run-end toast, with its
    # Show summary link, comes either way; off is for runs nobody is watching
    # the end of -- an agent starting many short runs, or overnight.
    show_run_summary: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowOptions":
        data = dict(data)
        # How long a run waited on reviews, before nothing waited on one.
        data.pop("review_wait", None)
        return cls(**data)


@dataclass
class TaskPlan:
    """One task in the plan a run follows: whether its questions are asked,
    the tasks it needs first, whether an item is done without it, and when it
    may start."""

    name: str
    # Whether its questions are asked (supervised) or recorded for someone to
    # check (automated).
    attention: Attention = Attention.automated
    # The tasks, by name, whose result this one uses: it waits while one of
    # them is still queued for the item, and is skipped when one of them did
    # not complete. Empty: it runs whatever happened before it on the item.
    requires: List[str] = field(default_factory=list)
    required: bool = True
    scheduled_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "attention": self.attention.value,
            "requires": list(self.requires),
            "required": self.required,
            "scheduled_at": self.scheduled_at.isoformat()
            if self.scheduled_at is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPlan":
        scheduled_at = data.get("scheduled_at")
        name = data.get("name", "")
        return cls(
            name=name,
            attention=attention_from(
                data.get("attention", Attention.automated.value), f"task '{name}'"
            ),
            requires=_requires(data.get("requires") or [], name),
            required=bool(data.get("required", True)),
            scheduled_at=datetime.fromisoformat(scheduled_at)
            if isinstance(scheduled_at, str)
            else scheduled_at,
        )


@dataclass
class RunPlan:
    """The tasks a run offers, in order: the default a task list shows, and
    which of them an item needs to be done."""

    tasks: List[TaskPlan] = field(default_factory=list)

    @property
    def workflow(self) -> List[str]:
        return [task.name for task in self.tasks]

    @property
    def required_tasks(self) -> List[str]:
        return [task.name for task in self.tasks if task.required]

    def get(self, task_name: str) -> Optional[TaskPlan]:
        return next((task for task in self.tasks if task.name == task_name), None)

    def add(self, task_name: str, required: bool = True) -> TaskPlan:
        task = self.get(task_name)
        if task is None:
            task = TaskPlan(name=task_name, required=required)
            self.tasks.append(task)
        return task

    def remove(self, task_name: str) -> None:
        self.tasks[:] = [task for task in self.tasks if task.name != task_name]

    def is_completed(self, item: Any) -> bool:
        """Whether *item* has completed every required task."""
        completed = item.completed_tasks
        return all(name in completed for name in self.required_tasks)

    def get_scheduled_at(self, task_name: str) -> Optional[datetime]:
        task = self.get(task_name)
        return task.scheduled_at if task is not None else None

    def reorder(self, order: List[str]) -> str:
        """Put the tasks in *order*: every task, each once. Why not, with
        nothing changed, or ""."""
        if sorted(order) != sorted(self.workflow):
            return "Give every task in the workflow, each once."
        by_name = {task.name: task for task in self.tasks}
        self.tasks[:] = [by_name[name] for name in order]
        return ""

    def set_schedule(self, task_name: str, scheduled_at: Optional[datetime]) -> str:
        """Set when *task_name* may start, no more than SCHEDULE_DAYS_AHEAD
        days from now; None clears it, and a time already past starts it at
        once. A naive time is local. Why not, with nothing changed, or ""."""
        task = self.get(task_name)
        if task is None:
            return f"No task named {task_name!r} in the workflow."
        if scheduled_at is not None and scheduled_at.astimezone() - now() > timedelta(
            days=SCHEDULE_DAYS_AHEAD
        ):
            return (
                f"{task_name} can be scheduled up to {SCHEDULE_DAYS_AHEAD} days "
                "from now."
            )
        task.scheduled_at = scheduled_at
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {"tasks": [task.to_dict() for task in self.tasks]}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        data = upgrade.run_plan(data or {})
        return cls(tasks=[TaskPlan.from_dict(task) for task in data["tasks"]])


@dataclass
class TaskProtocol:
    """The tasks a protocol offers, their settings, and the plan a run follows
    (``workflow_config``): which tasks run in a session, and in what order, is
    chosen at run time, and the plan is the default a task list shows. An
    app's protocol says how its saved configs load (:meth:`load_task_configs`)
    and what its tasks are called (``NOUN``)."""

    NOUN: ClassVar[str] = "task"
    # A task runs after the ones it requires only when they come first in the
    # plan: the item's runs follow its order.
    REQUIRES_IN_ORDER: ClassVar[bool] = True

    name: str = "Task Protocol"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_config: Dict[str, TaskConfig] = field(default_factory=dict)
    workflow_config: RunPlan = field(default_factory=RunPlan)

    @classmethod
    def load_task_configs(
        cls, ddict: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, TaskConfig]:
        """name -> typed config, from the saved ``task_config`` section."""
        raise NotImplementedError

    @classmethod
    def task_classes(cls) -> Dict[str, type]:
        """task_type -> task class, for every task this protocol can add."""
        raise NotImplementedError

    def add_new(self, task_type: str, task_name: str) -> TaskConfig:
        """Add a supervised task of a registered *task_type*, named
        *task_name*, with its default settings, at the end of the plan.
        Raises ValueError for an unknown type, no name, or a name in use."""
        task_name = task_name.strip()
        classes = self.task_classes()
        if task_type not in classes:
            raise ValueError(f"No {self.NOUN} type {task_type!r}.")
        if not task_name:
            raise ValueError(f"A {self.NOUN} needs a name.")
        if task_name in self.task_config:
            raise ValueError(f"There is already a {self.NOUN} named {task_name!r}.")
        config = classes[task_type].config_cls()
        config.task_name = task_name
        self.add(config)
        self.workflow_config.get(task_name).attention = Attention.supervised
        return config

    def add(self, config: TaskConfig) -> TaskConfig:
        if not config.task_name:
            raise ValueError(f"A {self.NOUN} config needs a task_name.")
        self.task_config[config.task_name] = config
        self.workflow_config.add(config.task_name)
        return config

    def remove(self, task_name: str) -> None:
        """Remove *task_name*'s config and plan entry. A task that required
        it no longer does."""
        self.task_config.pop(task_name, None)
        self.workflow_config.remove(task_name)
        for entry in self.workflow_config.tasks:
            if task_name in entry.requires:
                entry.requires = [r for r in entry.requires if r != task_name]

    def reorder(self, order: List[str]) -> str:
        """Put the tasks in *order*: every task, each once, none before a task
        it requires. Why not, with nothing changed, or ""."""
        if sorted(order) != sorted(self.ordered_task_names):
            return f"Give every {self.NOUN}, each once."
        issues = self.validate(order)
        if issues:
            return " ".join(issues)
        for name in order:
            self.workflow_config.add(name)
        return self.workflow_config.reorder(list(order))

    def set_requires(self, task_name: str, requires: List[str]) -> str:
        """Set the tasks *task_name* requires on the same item, which must come
        before it. Why not, with nothing changed, or ""."""
        entry = self.workflow_config.get(task_name)
        if entry is None or task_name not in self.task_config:
            return f"No {self.NOUN} named {task_name!r} in the workflow."
        names = self.ordered_task_names
        if self.REQUIRES_IN_ORDER:
            allowed = names[: names.index(task_name)]
            can = f"can only require {self.NOUN}s before it"
        else:
            allowed = [n for n in names if n != task_name]
            can = "can require"
        wrong = [name for name in requires if name not in allowed]
        if wrong:
            return (
                f"{task_name} {can}: "
                f"{', '.join(allowed) or 'none'}; not {', '.join(wrong)}."
            )
        entry.requires = [n for n in allowed if n in requires]
        return ""

    def validate(self, order: Optional[List[str]] = None) -> List[str]:
        """What is wrong with the plan in *order* (its own by default): a
        task requiring one the plan does not have, or one after it."""
        names = list(order) if order is not None else self.workflow_config.workflow
        issues = []
        for i, name in enumerate(names):
            for req in self.requirements(name):
                if req not in names:
                    issues.append(f"Task '{name}' requires unknown task '{req}'.")
                elif self.REQUIRES_IN_ORDER and req not in names[:i]:
                    issues.append(
                        f"Task '{name}' requires '{req}' which comes after it in the workflow."
                    )
        return issues

    def set_requirements(
        self,
        task_name: str,
        requires: Optional[List[str]] = None,
        required: Optional[bool] = None,
    ) -> str:
        """Set the tasks *task_name* requires and whether the plan requires it;
        None leaves either as it is. Why not, with nothing changed, or ""."""
        plan = self.workflow_config.get(task_name)
        if plan is None or task_name not in self.task_config:
            return f"No {self.NOUN} named {task_name!r} in the workflow."
        if requires is not None:
            why = self.set_requires(task_name, list(requires))
            if why:
                return why
        if required is not None:
            plan.required = bool(required)
        return ""

    def requirements(self, task_name: str) -> List[str]:
        """The tasks ``task_name`` requires on the same item; none for a task
        the plan does not list."""
        entry = self.workflow_config.get(task_name)
        return list(entry.requires) if entry is not None else []

    def get_attention(self, task_name: str) -> Attention:
        """Who decides the task's record, and when; automated for a task the
        plan does not list."""
        entry = self.workflow_config.get(task_name)
        if entry is None or task_name not in self.task_config:
            return Attention.automated
        return entry.attention

    def set_attention(self, task_name: str, attention: Attention) -> str:
        """Set who decides *task_name*'s record. Why not, with nothing
        changed, or "". The status bar's and the task row's chips, and the App
        Server's supervision."""
        entry = self.workflow_config.get(task_name)
        if entry is None or task_name not in self.task_config:
            return f"No {self.NOUN} named {task_name!r} in the workflow."
        entry.attention = Attention(attention)
        return ""

    @property
    def ordered_task_names(self) -> List[str]:
        """The plan's tasks first, then anything in `task_config` it does not
        list."""
        names = [n for n in self.workflow_config.workflow if n in self.task_config]
        names += [n for n in self.task_config if n not in names]
        return names

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task_config": {k: v.to_dict() for k, v in self.task_config.items()},
            "workflow_config": self.workflow_config.to_dict(),
        }

    @classmethod
    def load(cls, filename: str):
        with open(filename) as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data)

    def save(self, filename: str) -> None:
        """Save the task protocol to a YAML file."""
        with open(filename, "w") as file:
            yaml.safe_dump(
                self.to_dict(),
                file,
                indent=4,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        data = upgrade.task_protocol(data or {})
        protocol = cls(
            name=data.get("name", cls.name),
            task_config=cls.load_task_configs(data.get("task_config", {})),
            workflow_config=RunPlan.from_dict(data.get("workflow_config")),
        )
        if data.get("id"):
            protocol.id = data["id"]
        return protocol
