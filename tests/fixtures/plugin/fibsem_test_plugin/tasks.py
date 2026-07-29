"""Fixture AutoLamella task."""

from dataclasses import dataclass, field
from typing import ClassVar, Type

from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask

from fibsem_test_plugin import TASK_TYPE


@dataclass
class FixtureTaskConfig(AutoLamellaTaskConfig):
    task_type: ClassVar[str] = TASK_TYPE
    display_name: ClassVar[str] = "Fixture Task"

    number: int = field(default=7, metadata={"help": "A round-trippable parameter"})


class FixtureTask(AutoLamellaTask):
    config: FixtureTaskConfig
    config_cls: ClassVar[Type[FixtureTaskConfig]] = FixtureTaskConfig

    def _run(self) -> None:
        raise NotImplementedError("fixture task is never executed")
