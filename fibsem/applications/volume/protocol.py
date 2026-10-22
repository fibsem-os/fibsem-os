"""The volume tasks a protocol offers, and their settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional

from fibsem.applications.volume.tasks import (
    VolumeFIBMillingTaskConfig,
    VolumeSEMAcquisitionTaskConfig,
    VolumeSetupImagingTaskConfig,
    VolumeSetupMillingTaskConfig,
    get_volume_tasks,
    load_volume_task_configs,
)
from fibsem.workflows.attention import Attention
from fibsem.workflows.task_config import TaskConfig, TaskProtocol, WorkflowOptions

# Where volume protocol files are saved and listed, by name.
PROTOCOL_FOLDER = os.path.join(os.path.dirname(__file__), "protocol")


@dataclass
class VolumeProtocol(TaskProtocol):
    """The tasks a position is set up with and each slice runs, in ``order``,
    with their settings: shared by every position, which keeps its own poses
    and point of interest."""

    NOUN: ClassVar[str] = "volume task"

    name: str = "Volume Protocol"
    options: WorkflowOptions = field(default_factory=WorkflowOptions)

    @classmethod
    def load_task_configs(
        cls, ddict: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, TaskConfig]:
        return load_volume_task_configs(ddict)

    @classmethod
    def task_classes(cls) -> Dict[str, type]:
        return get_volume_tasks()

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), "options": self.options.to_dict()}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "VolumeProtocol":
        protocol = super().from_dict(data)
        protocol.options = WorkflowOptions.from_dict((data or {}).get("options", {}))
        return protocol

    @classmethod
    def default(cls) -> "VolumeProtocol":
        """Set each position up, asking, then per slice mill it and image it:
        the image needs the cut."""
        protocol = cls()
        for setup in (VolumeSetupMillingTaskConfig(), VolumeSetupImagingTaskConfig()):
            protocol.add(setup)
            protocol.set_attention(setup.task_name, Attention.supervised)
        mill, image = VolumeFIBMillingTaskConfig(), VolumeSEMAcquisitionTaskConfig()
        protocol.add(mill)
        protocol.add(image)
        protocol.set_requires(image.task_name, [mill.task_name])
        return protocol
