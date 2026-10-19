"""The volume tasks a protocol offers, and their settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

from fibsem.applications.volume.tasks import (
    VolumeFIBMillingTaskConfig,
    VolumeSEMAcquisitionTaskConfig,
    load_volume_task_configs,
)
from fibsem.workflows.task_config import TaskConfig, TaskProtocol


@dataclass
class VolumeProtocol(TaskProtocol):
    """The tasks each slice runs, in ``order``, with their settings: shared by
    every position, which keeps its own poses and point of interest."""

    NOUN: ClassVar[str] = "volume task"

    name: str = "Volume Protocol"

    @classmethod
    def load_task_configs(
        cls, ddict: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, TaskConfig]:
        return load_volume_task_configs(ddict)

    @classmethod
    def default(cls) -> "VolumeProtocol":
        """Mill a slice, then image it."""
        protocol = cls()
        protocol.add(VolumeFIBMillingTaskConfig())
        protocol.add(VolumeSEMAcquisitionTaskConfig())
        return protocol
