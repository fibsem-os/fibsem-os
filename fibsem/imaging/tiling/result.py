"""What a tiled overview run did: one record for the beam and fluorescence runners.

`OverviewResult` is filled in as a run goes and handed back when it ends, to the
Overview tabs, the grid overview tasks and the imaging service alike. Its `to_dict`
is the full record, written beside the tiles as `overview-metadata.json`; `summary` is the
short form the App Server answers an agent with.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fibsem.imaging.tiling.layout_check import TileLayoutCheck
from fibsem.imaging.tiling.registration import Registration
from fibsem.structures.settings import FibsemStagePosition

# Written into the run's tile folder, beside the tiles it describes.
METADATA_FILE = "overview-metadata.json"


@dataclass
class TileRecord:
    """One tile of the grid: where it was planned, where the stage read back after
    the move, and where its picture says it is (``shift``, metres from the plan, y
    down, from the registration). ``acquired`` is False for a tile turned off in the
    mask (out of reach, say) or not reached before a stop."""

    row: int
    col: int
    planned: Optional[FibsemStagePosition] = None
    position: Optional[FibsemStagePosition] = None
    acquired: bool = False
    path: Optional[str] = None
    shift: Optional[List[float]] = None
    objective_position: Optional[float] = None  # fluorescence only

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "row": self.row,
            "col": self.col,
            "acquired": self.acquired,
            "planned": _position(self.planned),
            "position": _position(self.position),
            "path": self.path,
            "shift": self.shift,
        }
        if self.objective_position is not None:
            out["objective_position"] = self.objective_position
        return out


@dataclass
class OverviewResult:
    """A tiled overview run: what it was asked to do, what it did, and where it put
    the files.

    ``settings`` are the settings as used (the resolution the instrument scans, the
    tiles turned off), ``requested_resolution`` the size asked for when the
    instrument scans another. ``estimate`` is the run's own estimate in seconds:
    the beam runner's counts the scanning only, the fluorescence runner's the whole
    run. ``image`` is the mosaic, not part of the record.
    """

    modality: str  # `MODALITY_BEAM` or `MODALITY_FLUORESCENCE` (tiling.progress)
    status: str = "running"
    error: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    requested_resolution: Optional[Tuple[int, int]] = None
    centre: Optional[FibsemStagePosition] = None
    beam_type: Optional[str] = None
    scan_rotation: Optional[float] = None
    pixel_size: Optional[float] = None
    fov: Optional[Tuple[float, float]] = None
    tiles: List[TileRecord] = field(default_factory=list)
    layout: Optional[Dict[str, Any]] = None
    registration: Optional[Registration] = None
    warning: Optional[str] = None
    autofocus: List[Dict[str, Any]] = field(default_factory=list)
    started: Optional[datetime] = None
    ended: Optional[datetime] = None
    estimate: Optional[float] = None
    mosaic_path: Optional[str] = None
    tiles_directory: Optional[str] = None
    record_path: Optional[str] = None
    fluorescence: Optional[Dict[str, Any]] = None
    image: Any = field(default=None, repr=False, compare=False)

    # -- filled in by the runners ----------------------------------------------

    def start(self) -> None:
        self.started = datetime.now()

    def end(self, status: str, error: Optional[str] = None) -> None:
        self.status, self.error = status, error
        self.ended = datetime.now()

    def tile(self, row: int, col: int) -> TileRecord:
        for record in self.tiles:
            if (record.row, record.col) == (row, col):
                return record
        record = TileRecord(row=row, col=col)
        self.tiles.append(record)
        return record

    def apply_registration(self, registration: Registration) -> None:
        self.registration = registration
        for key, shift in registration.shifts.items():
            row, col = (int(v) for v in key.split(","))
            self.tile(row, col).shift = shift

    def report_layout(self, check: Optional[TileLayoutCheck]) -> Optional[str]:
        """Log the layout check and set the run's warning: the check's, and the
        registration's note when it was asked to align the tiles and did not, or the
        operator would take the mosaic for an aligned one. Returns the warning."""
        warning = None
        if check is not None:
            try:
                summary = check.summary()
                logging.info(summary.describe())
                warning = summary.warning
                if warning:
                    logging.warning(warning)
            except Exception as e:  # noqa: BLE001 - a report, not the run
                logging.warning(f"Could not report the tile layout check: {e}")
        registration = self.registration
        if (
            registration is not None
            and registration.requested
            and not registration.applied
        ):
            warning = f"{warning} {registration.note}" if warning else registration.note
        self.warning = warning
        return warning

    # -- read ------------------------------------------------------------------

    @property
    def tiles_acquired(self) -> int:
        return sum(t.acquired for t in self.tiles)

    @property
    def tiles_total(self) -> int:
        return len(self.tiles)

    @property
    def tile_paths(self) -> List[str]:
        """Each saved tile's path, in the order acquired (the runners add the tiles
        in the order they visit them)."""
        return [t.path for t in self.tiles if t.path]

    @property
    def duration(self) -> Optional[float]:
        if self.started is None or self.ended is None:
            return None
        return (self.ended - self.started).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """The full record, as written to ``overview-metadata.json``."""
        return {
            "modality": self.modality,
            "status": self.status,
            "error": self.error,
            "settings": self.settings,
            "requested_resolution": _pair(self.requested_resolution),
            "centre": _position(self.centre),
            "beam_type": self.beam_type,
            "scan_rotation": self.scan_rotation,
            "pixel_size": self.pixel_size,
            "fov": _pair(self.fov),
            "tiles_acquired": self.tiles_acquired,
            "tiles_total": self.tiles_total,
            "warning": self.warning,
            "layout": self.layout,
            "registration": (
                self.registration.to_dict() if self.registration else None
            ),
            "started": _time(self.started),
            "ended": _time(self.ended),
            "duration": self.duration,
            "estimate": self.estimate,
            "autofocus": self.autofocus,
            "fluorescence": self.fluorescence,
            "paths": self._paths(),
            "tiles": [t.to_dict() for t in self.tiles],
        }

    def summary(self) -> Dict[str, Any]:
        """The short form: what an agent reads to decide what to do next. The
        per-tile detail and the layout check's pairs are in the record file."""
        layout = self.layout or {}
        steps = {
            axis: {
                "step_fraction": s.get("step_fraction"),
                "angle_deg": s.get("angle_deg"),
            }
            for axis, s in (layout.get("summary") or {}).items()
            if axis in ("x", "y") and s
        }
        out = {
            "status": self.status,
            "tiles_acquired": self.tiles_acquired,
            "tiles_total": self.tiles_total,
            "tiles_skipped": [[t.row, t.col] for t in self.tiles if not t.acquired],
            "resolution": self.settings.get("resolution"),
            "requested_resolution": _pair(self.requested_resolution),
            "pixel_size": self.pixel_size,
            "fov": _pair(self.fov),
            "centre": _position(self.centre),
            "warning": self.warning,
            "layout": steps or None,
            "registration": (
                self.registration.summary() if self.registration else None
            ),
            "duration": self.duration,
            "estimate": self.estimate,
            "paths": self._paths(),
        }
        if self.error:
            out["error"] = self.error
        return out

    def _paths(self) -> Dict[str, Optional[str]]:
        return {
            "mosaic": self.mosaic_path,
            "tiles": self.tiles_directory,
            "record": self.record_path,
        }

    # -- write -----------------------------------------------------------------

    def save(self, directory: Optional[str], check: Optional[TileLayoutCheck]) -> None:
        """Write the record into the run's tile folder as `overview-metadata.json`, with the
        layout check's pairs. Best effort: a run is not failed by its record."""
        if directory is None:
            return
        path = os.path.join(directory, METADATA_FILE)
        try:
            if check is not None:
                self.layout = check.record()
            self.record_path = path
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:  # noqa: BLE001 - the tiles are already saved
            self.record_path = None
            logging.warning(f"Could not write the overview record: {e}")


def _position(position: Optional[FibsemStagePosition]) -> Optional[Dict[str, Any]]:
    return position.to_dict() if position is not None else None


def _pair(value) -> Optional[List[Any]]:
    return list(value) if value is not None else None


def _time(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None
