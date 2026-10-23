"""Grids as a workflow knows them, and an experiment that tracks them.

A :class:`GridRecord` is a grid as an item: tasks run on it, it keeps their
record and a verdict, and what was placed over its overviews. Distinct from the
hardware's ``SampleGrid``, to which it is linked by name. :class:`ExperimentGrids`
is what an experiment does with them; every application on grids mixes it in.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from psygnal import evented

from fibsem.util.timestamps import now, to_datetime, to_iso
from fibsem.utils import _plain
from fibsem.workflows.actions import note_record
from fibsem.workflows.item import ExperimentItem
from fibsem.workflows.proposals import Proposal, proposals_from_dict, proposals_to_dict
from fibsem.workflows.quality import QualityRecord
from fibsem.workflows.status import TaskState, TaskStatus


@evented
@dataclass
class OverlayRecord:
    """Something placed by hand over a grid's overview, and where it was put.

    The grid bars today; a fluorescence overview next. What is stored is a
    placement on the *sample*: an offset in metres along its surface and a turn in
    degrees, relative to `reference` -- the grid centre when there is none -- so the
    same record draws in every view, squashed by whatever that view's foreshortening
    is. The view does not belong in here.

    One record per placed thing, on the grid rather than the experiment: a placement is
    about one grid, and switching grids should swap it with them.
    """

    kind: str  # "gridbar" | "image"
    dx: float = 0.0  # metres along the sample surface from the reference
    dy: float = 0.0
    rotation: float = 0.0  # degrees, clockwise on screen
    scale: float = 1.0
    # An image mirrored left to right by the user, about its own axis.
    mirrored: bool = False
    # Grid bars: the lattice's pitch and bar width, in metres.
    pitch: Optional[float] = None
    bar_width: Optional[float] = None
    # An image: its file, relative to the grid's folder.
    source: Optional[str] = None
    # What it was aligned against: an overview's file, relative to the grid's folder,
    # and the view it was shown in. None means the grid centre.
    reference: Optional[str] = None
    view: Optional[str] = None
    # How it was placed by a fit, when it was: the point pairs and the residual.
    fit: Dict[str, Any] = field(default_factory=dict)
    # An image: how it is shown -- opacity, signal only, and per channel its colour,
    # visibility, opacity, gamma and contrast. Empty means as loaded.
    display: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Aware, written as ISO 8601 with its offset; older files hold a
    # POSIX float, which reads as this machine's zone.
    created_at: Optional[datetime] = field(default_factory=now)

    def to_dict(self) -> dict:
        # Plain floats throughout: a numpy scalar that slipped in from a canvas
        # cannot be represented by the YAML writer, and a save that raises half-way
        # is worse than a wrong number.
        return {
            "kind": self.kind,
            "dx": float(self.dx),
            "dy": float(self.dy),
            "rotation": float(self.rotation),
            "scale": float(self.scale),
            "mirrored": bool(self.mirrored),
            "pitch": None if self.pitch is None else float(self.pitch),
            "bar_width": None if self.bar_width is None else float(self.bar_width),
            "source": self.source,
            "reference": self.reference,
            "view": self.view,
            "fit": _plain(self.fit),
            "display": _plain(self.display),
            "id": self.id,
            "created_at": to_iso(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OverlayRecord":
        return cls(
            kind=str(data.get("kind", "")),
            dx=float(data.get("dx", 0.0) or 0.0),
            dy=float(data.get("dy", 0.0) or 0.0),
            rotation=float(data.get("rotation", 0.0) or 0.0),
            scale=float(data.get("scale", 1.0) or 1.0),
            mirrored=bool(data.get("mirrored", False)),
            pitch=data.get("pitch"),
            bar_width=data.get("bar_width"),
            source=data.get("source"),
            reference=data.get("reference"),
            view=data.get("view"),
            fit=dict(data.get("fit") or {}),
            display=dict(data.get("display") or {}),
            id=data.get("id") or str(uuid.uuid4()),
            created_at=to_datetime(data.get("created_at", now())),
        )


@dataclass
class GridRecord(ExperimentItem):
    """A grid as the workflow knows it, distinct from the hardware's `SampleGrid`.

    Linked to the hardware by name and to lamellae by `Lamella.grid_id -> id`.
    Deliberately holds no slot, stage position, loaded state or copy of the
    hardware grid: all of that is resolved live against the stage, so a record
    stays valid when its grid changes slot or leaves the magazine. A record exists
    from the moment a grid is *available* (present in the inventory), so twelve
    grids can be queued before any has been loaded.

    `task_state` and `task_history` are the same types the lamella side uses,
    and the same rule applies: never replace `task_state` wholesale, mutate it.
    """

    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    quality: QualityRecord = field(default_factory=QualityRecord)
    task_state: TaskState = field(default_factory=TaskState)
    task_history: List[TaskState] = field(default_factory=list)
    # Aware, written as ISO 8601 with its offset; older files hold a
    # POSIX float, which reads as this machine's zone.
    created_at: Optional[datetime] = field(default_factory=now)
    proposals: List[Proposal] = field(default_factory=list)  # as on Lamella
    # What has been placed by hand over this grid's overviews; see `OverlayRecord`.
    overlays: List[OverlayRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        self.task_state.item_id = self.id

    def overlay_of(self, kind: str) -> Optional[OverlayRecord]:
        """The one record of *kind* that has no source -- the grid bars, say."""
        return next(
            (o for o in self.overlays if o.kind == kind and o.source is None), None
        )

    def set_overlay(self, record: OverlayRecord) -> None:
        """Keep *record*, replacing the one it stands in for.

        By id when the record has been seen before, otherwise by kind for a
        source-less record: there is one lattice of grid bars per grid.
        """
        for index, existing in enumerate(self.overlays):
            same = existing.id == record.id or (
                record.source is None
                and existing.source is None
                and existing.kind == record.kind
            )
            if same:
                self.overlays[index] = record
                return
        self.overlays.append(record)

    def set_note(self, note: str, **fields: Any) -> Optional[Dict[str, Any]]:
        """Keep *note* as the grid's note (an empty one clears it) and answer
        its note record (with *fields*, as ``note_record``) for whoever wrote
        it, or None when it was cleared."""
        self.description = note
        if not note:
            return None
        return note_record(note, item_id=self.id, item_name=self.name, **fields)

    @property
    def is_failure(self) -> bool:
        """Whether the latest run on this grid ended in failure. Not a verdict on
        the grid -- see `quality` for that, which a person sets and a task never
        does, same as a lamella."""
        return self.task_state.status is TaskStatus.Failed

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "_id": self.id,
            "description": self.description,
            "quality": self.quality.to_dict(),
            "task_state": self.task_state.to_dict(),
            "task_history": [t.to_dict() for t in self.task_history],
            "created_at": to_iso(self.created_at),
            "proposals": proposals_to_dict(self.proposals),
            "overlays": [o.to_dict() for o in self.overlays],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridRecord":
        return cls(
            name=data["name"],
            id=data.get("_id") or str(uuid.uuid4()),
            description=data.get("description", ""),
            quality=QualityRecord.from_dict(data.get("quality")),
            task_state=TaskState.from_dict(data.get("task_state", {})),
            task_history=[TaskState.from_dict(t) for t in data.get("task_history", [])],
            created_at=to_datetime(data.get("created_at", now())),
            proposals=proposals_from_dict(data.get("proposals")),
            overlays=[OverlayRecord.from_dict(o) for o in data.get("overlays", [])],
        )

    def __repr__(self) -> str:
        return f"GridRecord(name={self.name!r}, quality={self.quality.verdict.name}, tasks={len(self.task_history)})"


class ExperimentGrids:
    """The grids an experiment tracks. A mixin for an experiment with a
    ``grids`` list; put it before the experiment base so ``item_path`` reaches
    the base for other items."""

    grids: List[GridRecord]

    def get_grid_by_name(self, name: str) -> Optional[GridRecord]:
        return next((g for g in self.grids if g.name == name), None)

    def get_grid_by_id(self, grid_id: Optional[str]) -> Optional[GridRecord]:
        if grid_id is None:
            return None
        return next((g for g in self.grids if g.id == grid_id), None)

    def add_grid(self, grid: GridRecord) -> GridRecord:
        """Track a grid. Names are the link to the hardware, so they are unique."""
        if self.get_grid_by_name(grid.name) is not None:
            raise ValueError(f"Grid '{grid.name}' already exists in the experiment.")
        self.grids.append(grid)
        return grid

    def remove_grid(self, name: str) -> Optional[GridRecord]:
        """Stop tracking a grid. What sits on it is released, not deleted."""
        grid = self.get_grid_by_name(name)
        if grid is None:
            return None
        self._release_grid(grid)
        self.grids.remove(grid)
        return grid

    def sync_grids_from_inventory(self, stage) -> List[GridRecord]:
        """Create a record for every grid the inventory reports present.

        Idempotent, matched by name: a grid already tracked is left alone, with its
        history; a grid whose hardware has gone keeps its record too, and the UI
        reads "not present" for it from the inventory. Returns the records added.
        """
        added: List[GridRecord] = []
        for entry in stage.grid_inventory():
            if not entry.present or not entry.name:
                continue
            if self.get_grid_by_name(entry.name) is None:
                added.append(self.add_grid(GridRecord(name=entry.name)))
        return added

    def grid_path(self, grid: GridRecord) -> Path:
        """Where a grid's task output goes: `grids/<name>/` under the experiment.

        Its own directory so it can never collide with a lamella's, which sits
        directly under the experiment path. Not created here; a task creates its
        own output directory when it first writes.
        """
        return Path(self.path) / "grids" / grid.name

    def item_path(self, item: Any) -> Path:
        """The directory an item's recorded outputs are relative to: a grid's
        ``grid_path``, derived rather than stored on its record, which is
        renamed in place; any other item's own."""
        if isinstance(item, GridRecord):
            return self.grid_path(item)
        return super().item_path(item)

    def _release_grid(self, grid: GridRecord) -> None:
        """What sits on a grid that is no longer tracked; nothing by default."""
