"""Fixture grid task."""

from dataclasses import dataclass
from typing import ClassVar, Type

from fibsem.applications.autolamella.workflows.tasks.grid import (
    GridTask,
    GridTaskConfig,
)
from fibsem_test_plugin import GRID_TASK_TYPE


@dataclass
class FixtureGridTaskConfig(GridTaskConfig):
    task_type: ClassVar[str] = GRID_TASK_TYPE
    display_name: ClassVar[str] = "Fixture Grid Task"


class FixtureGridTask(GridTask):
    config: FixtureGridTaskConfig
    config_cls: ClassVar[Type[FixtureGridTaskConfig]] = FixtureGridTaskConfig

    def _run(self) -> None:
        raise NotImplementedError("fixture grid task is never executed")
