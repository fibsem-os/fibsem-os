"""Fixture volume task."""

from dataclasses import dataclass
from typing import ClassVar, Type

from fibsem.applications.volume.tasks import VolumeTask
from fibsem.workflows.task_config import TaskConfig
from fibsem_test_plugin import VOLUME_TASK_TYPE


@dataclass
class FixtureVolumeTaskConfig(TaskConfig):
    task_type: ClassVar[str] = VOLUME_TASK_TYPE
    display_name: ClassVar[str] = "Fixture Volume Task"


class FixtureVolumeTask(VolumeTask):
    config: FixtureVolumeTaskConfig
    config_cls: ClassVar[Type[FixtureVolumeTaskConfig]] = FixtureVolumeTaskConfig

    def _run(self) -> None:
        raise NotImplementedError("fixture volume task is never executed")
