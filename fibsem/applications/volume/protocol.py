"""The volume tasks a protocol offers, and their settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

from fibsem.applications.volume.tasks import (
    VolumeFIBMillingTaskConfig,
    VolumeSEMAcquisitionTaskConfig,
    VolumeSetupImagingTaskConfig,
    VolumeSetupMillingTaskConfig,
    load_volume_task_configs,
)
from fibsem.workflows.attention import Attention
from fibsem.workflows.task_config import TaskConfig, TaskProtocol


@dataclass
class VolumeProtocol(TaskProtocol):
    """The tasks a position is set up with and each slice runs, in ``order``,
    with their settings: shared by every position, which keeps its own poses
    and point of interest."""

    NOUN: ClassVar[str] = "volume task"

    name: str = "Volume Protocol"

    @classmethod
    def load_task_configs(
        cls, ddict: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, TaskConfig]:
        return load_volume_task_configs(ddict)

    @classmethod
    def default(cls) -> "VolumeProtocol":
        """Set each position up, asking, then per slice mill it and image it:
        the image needs the cut."""
        protocol = cls()
        for setup in (VolumeSetupMillingTaskConfig(), VolumeSetupImagingTaskConfig()):
            setup.attention = Attention.supervised
            protocol.add(setup)
        mill, image = VolumeFIBMillingTaskConfig(), VolumeSEMAcquisitionTaskConfig()
        protocol.add(mill)
        protocol.add(image)
        protocol.set_requires(image.task_name, [mill.task_name])
        return protocol
