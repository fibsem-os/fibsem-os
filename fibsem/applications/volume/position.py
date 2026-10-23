"""A volume position: one site milled and imaged slice by slice."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

from psygnal import evented

from fibsem.structures import (
    DEFAULT_ALIGNMENT_AREA,
    FibsemRectangle,
    MicroscopeState,
    Point,
)
from fibsem.util.timestamps import now, to_datetime, to_iso
from fibsem.workflows.item import MILLING_POSE, SiteItem
from fibsem.workflows.proposals import Proposal, proposals_from_dict, proposals_to_dict
from fibsem.workflows.status import TaskState, TaskStatus

# The pose a position's face is imaged at: the milling pose when it has none
# of its own.
SEM_POSE = "SEM_IMAGING"

# The alignment references a position's tasks align to, in its folder.
FIB_REFERENCE = "ref_alignment_ib.tif"
SEM_REFERENCE = "ref_alignment_eb.tif"


@evented
@dataclass
class VolumePosition(SiteItem):
    """A site cut into ``iterations`` slices, each one iteration of its tasks.

    What a slice needs is worked out from the record, not counted on the
    position: where a slice is milled from ``poi`` and its number (see
    ``tasks.milling``), and which slices are done from ``task_history``. A resume after a stop reads
    the same answer the run would have.
    """

    item_type: ClassVar[str] = "position"
    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    poses: Dict[str, MicroscopeState] = field(default_factory=dict)
    # The point of interest: the start slice's centre, which the slices step on from.
    poi: Point = field(default_factory=Point)
    # Where the alignment references are taken, in fractions of the frame.
    alignment_area: FibsemRectangle = field(
        default_factory=lambda: FibsemRectangle.from_dict(DEFAULT_ALIGNMENT_AREA)
    )
    iterations: int = 10
    task_state: TaskState = field(default_factory=TaskState)
    task_history: List[TaskState] = field(default_factory=list)
    created_at: Optional[datetime] = field(default_factory=now)
    proposals: List[Proposal] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        self.task_state.item_id = self.id

    def pose(self, name: str) -> MicroscopeState:
        """The pose ``name``; the imaging pose falls back to the milling pose."""
        if name == SEM_POSE and SEM_POSE not in self.poses:
            name = MILLING_POSE
        return super().pose(name)

    def poi_sync_plan(self, point: Point) -> Tuple[List[str], List[Tuple[Any, Point]]]:
        """Nothing on the position follows the point: each slice is placed
        from ``poi`` as it is milled."""
        return [], []

    def iteration_completed(self, iteration: int, task_names: Sequence[str]) -> bool:
        """Whether every one of ``task_names`` completed for slice ``iteration``."""
        done = {
            state.name
            for state in self.task_history
            if state.iteration == iteration and state.status is TaskStatus.Completed
        }
        return all(name in done for name in task_names)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "_id": self.id,
            "description": self.description,
            "poses": {k: v.to_dict() for k, v in self.poses.items()},
            "poi": self.poi.to_dict(),
            "alignment_area": self.alignment_area.to_dict(),
            "iterations": self.iterations,
            "task_state": self.task_state.to_dict(),
            "task_history": [t.to_dict() for t in self.task_history],
            "created_at": to_iso(self.created_at),
            "proposals": proposals_to_dict(self.proposals),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VolumePosition":
        return cls(
            name=data["name"],
            id=data.get("_id") or str(uuid.uuid4()),
            description=data.get("description", ""),
            poses={
                k: MicroscopeState.from_dict(v)
                for k, v in (data.get("poses") or {}).items()
            },
            poi=Point.from_dict(data.get("poi") or {}),
            alignment_area=FibsemRectangle.from_dict(
                data.get("alignment_area") or DEFAULT_ALIGNMENT_AREA
            ),
            iterations=int(data.get("iterations", 10)),
            task_state=TaskState.from_dict(data.get("task_state", {})),
            task_history=[TaskState.from_dict(t) for t in data.get("task_history", [])],
            created_at=to_datetime(data.get("created_at", now())),
            proposals=proposals_from_dict(data.get("proposals")),
        )

    def __repr__(self) -> str:
        return f"VolumePosition(name={self.name}, iterations={self.iterations})"
