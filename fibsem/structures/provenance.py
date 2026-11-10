"""What produced an image, the same shape whatever kind of image it is.

A beam image (`FibsemImageMetadata`) and an FM image (`FluorescenceImageMetadata`)
record the same provenance under different names and types: which instrument,
its geometry, which experiment, item and task, who, when, and where the stage
was. Each has a ``provenance`` property that reads its own fields into an
`ImageProvenance`, so a reader handles one shape. Nothing here is saved: the
files keep their own fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional

from fibsem.structures.settings import FibsemStagePosition
from fibsem.structures.system import (
    FibsemExperimentRef,
    FibsemHardwareGeometry,
    FibsemUser,
    SystemInfo,
)

__all__ = ["ImageProvenance", "InstrumentRef"]


@dataclass(frozen=True)
class InstrumentRef:
    """Which instrument took an image, as the image records it. Each field is
    None when the image doesn't say."""

    name: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    software_version: Optional[str] = None
    fibsem_version: Optional[str] = None

    @classmethod
    def of(cls, info: Any) -> Optional["InstrumentRef"]:
        """From a beam image's `SystemInfo` or an FM image's dict; None from None."""
        if isinstance(info, SystemInfo):
            read = lambda key: getattr(info, key, None)  # noqa: E731
        elif isinstance(info, Mapping):
            read = info.get
        else:
            return None
        return cls(**{key: read(key) for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ImageProvenance:
    """What produced an image: the instrument and its geometry, the experiment
    (with its item and task), who, when, and where the stage was. A field is
    None when the image doesn't record it, as an older file may not."""

    instrument: Optional[InstrumentRef] = None
    geometry: Optional[FibsemHardwareGeometry] = None
    experiment: Optional[FibsemExperimentRef] = None
    user: Optional[FibsemUser] = None
    acquired_at: Optional[datetime] = None
    stage_position: Optional[FibsemStagePosition] = None

    @property
    def item(self) -> Optional[str]:
        """The item (a lamella, or a grid) the image was taken for."""
        return getattr(self.experiment, "item_name", None)

    @property
    def task(self) -> Optional[str]:
        """The task the image was taken in."""
        return getattr(self.experiment, "task_name", None)
