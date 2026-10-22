"""AutoLamella task protocols for tests: a plan of tasks, each with its config."""

import weakref
from typing import Optional, Sequence, Tuple

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskConfig,
    AutoLamellaTaskProtocol,
)
from fibsem.workflows.attention import Attention
from fibsem.workflows.site_tasks.reference_image import (
    AcquireReferenceImageConfig,
)
from fibsem.workflows.task_config import TaskPlan

Planned = Tuple[TaskPlan, AutoLamellaTaskConfig]

# Configs planned() made up; add_planned() keeps a config the protocol already
# has rather than replacing it with one of these.
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
    """One task of a plan, with *attention* and *requires*, and its config in
    the protocol: *config*, or a reference image task's."""
    if config is None:
        config = AcquireReferenceImageConfig()
        _STAND_INS[id(config)] = config
    config.task_name = name
    plan = TaskPlan(
        name=name,
        attention=attention,
        requires=list(requires),
        required=required,
        scheduled_at=scheduled_at,
    )
    return plan, config


def task_protocol(*tasks: Planned, **fields) -> AutoLamellaTaskProtocol:
    """A protocol whose plan lists *tasks*, in order, with their configs."""
    protocol = AutoLamellaTaskProtocol(**fields)
    add_planned(protocol, *tasks)
    return protocol


def add_planned(protocol: AutoLamellaTaskProtocol, *tasks: Planned) -> None:
    """Put *tasks* at the end of *protocol*'s plan, their configs in it."""
    for plan, config in tasks:
        existing = protocol.task_config.get(plan.name)
        if existing is None or _STAND_INS.get(id(config)) is not config:
            protocol.task_config[plan.name] = config
        protocol.workflow_config.tasks.append(plan)
