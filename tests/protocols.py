"""AutoLamella task protocols for tests: a plan of tasks, each with its config,
where the protocol keeps a task's attention and requirements."""

import weakref
from typing import Optional, Sequence, Tuple

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskConfig,
    AutoLamellaTaskProtocol,
)
from fibsem.applications.autolamella.workflows.tasks.reference_image import (
    AcquireReferenceImageConfig,
)
from fibsem.workflows.attention import Attention
from fibsem.workflows.task_config import TaskPlan

Planned = Tuple[TaskPlan, AutoLamellaTaskConfig]

# Configs planned() made up; add_planned() puts their attention and requires on
# a config the protocol already has, rather than replacing it.
_STAND_INS: "weakref.WeakValueDictionary[int, AutoLamellaTaskConfig]" = (
    weakref.WeakValueDictionary()
)


def planned(
    name: str,
    required: bool = True,
    attention: Attention = Attention.automated,
    requires: Sequence[str] = (),
    scheduled_at=None,
    config: Optional[AutoLamellaTaskConfig] = None,
) -> Planned:
    """One task of a plan, and its config in the protocol: *config*, or a
    reference image task's, with *attention* and *requires*."""
    if config is None:
        config = AcquireReferenceImageConfig()
        _STAND_INS[id(config)] = config
    config.task_name = name
    config.attention = attention
    config.requires = list(requires)
    return TaskPlan(name=name, required=required, scheduled_at=scheduled_at), config


def task_protocol(*tasks: Planned, **fields) -> AutoLamellaTaskProtocol:
    """A protocol whose plan lists *tasks*, in order, with their configs."""
    protocol = AutoLamellaTaskProtocol(**fields)
    add_planned(protocol, *tasks)
    return protocol


def add_planned(protocol: AutoLamellaTaskProtocol, *tasks: Planned) -> None:
    """Put *tasks* at the end of *protocol*'s plan, their configs in it."""
    for plan, config in tasks:
        existing = protocol.task_config.get(plan.name)
        if existing is not None and _STAND_INS.get(id(config)) is config:
            existing.attention = config.attention
            existing.requires = list(config.requires)
        else:
            protocol.task_config[plan.name] = config
        protocol.workflow_config.tasks.append(plan)
