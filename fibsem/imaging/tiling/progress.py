"""Typed events emitted while a tiled acquisition runs.

``FibsemMicroscope.tiled_acquisition_signal`` is shared by the beam and fluorescence
tilers. The event class identifies what happened; ``modality`` identifies which
imaging path produced it. Missing modality retains the signal's historical meaning:
beam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import numpy as np

from fibsem.structures import FibsemImage

MODALITY_BEAM = "beam"
MODALITY_FLUORESCENCE = "fluorescence"


class TiledEventType(str, Enum):
    """Closed set of event shapes carried by the tiled signal."""

    PHASE = "phase"
    COUNTED_PHASE = "counted-phase"
    TILE_STARTED = "tile-started"
    FLUORESCENCE_TILE_COUNT = "fluorescence-tile-count"
    BEAM_TILE_COMPLETED = "beam-tile-completed"
    FLUORESCENCE_TILE_COMPLETED = "fluorescence-tile-completed"
    TERMINAL = "terminal"
    COUNTED_TERMINAL = "counted-terminal"


class TiledPhase(str, Enum):
    """Non-terminal work surrounding the completed-tile tally."""

    COMPUTING_POSITIONS = "computing-positions"
    MOVING = "moving"
    ACQUIRING = "acquiring"
    TILES_ACQUIRED = "tiles-acquired"
    STITCHING = "stitching"
    SAVING = "saving"


class TiledOutcome(str, Enum):
    """How an originating tiled acquisition ended."""

    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, kw_only=True)
class TiledAcquisitionEvent:
    """Base type for every event on ``tiled_acquisition_signal``.

    ``modality`` intentionally remains a string rather than an enum: consumers can
    safely ignore or generically present a future modality, and omitting it continues
    to mean beam for callers constructing an event with the default.
    """

    modality: str = MODALITY_BEAM


@dataclass(frozen=True, kw_only=True)
class TiledPhaseEvent(TiledAcquisitionEvent):
    event_type: TiledEventType = field(init=False, default=TiledEventType.PHASE)
    phase: TiledPhase


@dataclass(frozen=True, kw_only=True)
class CountedTiledPhaseEvent(TiledAcquisitionEvent):
    event_type: TiledEventType = field(
        init=False, default=TiledEventType.COUNTED_PHASE
    )
    phase: TiledPhase
    completed: int
    total: int
    message: str


@dataclass(frozen=True, kw_only=True)
class TileStartedEvent(TiledAcquisitionEvent):
    event_type: TiledEventType = field(init=False, default=TiledEventType.TILE_STARTED)
    row_index: int
    column_index: int
    rows: int
    columns: int


@dataclass(frozen=True, kw_only=True)
class FluorescenceTileCountEvent(TiledAcquisitionEvent):
    event_type: TiledEventType = field(
        init=False, default=TiledEventType.FLUORESCENCE_TILE_COUNT
    )
    completed: int
    total: int
    estimated_total_seconds: float
    estimated_remaining_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True, kw_only=True)
class BeamTileCompletedEvent(TiledAcquisitionEvent):
    event_type: TiledEventType = field(
        init=False, default=TiledEventType.BEAM_TILE_COMPLETED
    )
    completed: int
    total: int
    row_index: int
    column_index: int
    rows: int
    columns: int
    image: np.ndarray
    preview: FibsemImage
    message: str


@dataclass(frozen=True, kw_only=True)
class FluorescenceTileCompletedEvent(TiledAcquisitionEvent):
    event_type: TiledEventType = field(
        init=False, default=TiledEventType.FLUORESCENCE_TILE_COMPLETED
    )
    completed: int
    total: int
    row_index: int
    column_index: int
    rows: int
    columns: int
    image: np.ndarray
    preview_stride: int
    estimated_total_seconds: float
    estimated_remaining_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True, kw_only=True)
class TiledTerminalEvent(TiledAcquisitionEvent):
    event_type: TiledEventType = field(init=False, default=TiledEventType.TERMINAL)
    outcome: TiledOutcome
    message: str
    error: str | None = None


@dataclass(frozen=True, kw_only=True)
class CountedTiledTerminalEvent(TiledAcquisitionEvent):
    event_type: TiledEventType = field(
        init=False, default=TiledEventType.COUNTED_TERMINAL
    )
    outcome: TiledOutcome
    message: str
    completed: int
    total: int
    error: str | None = None


TiledEvent = Union[
    TiledPhaseEvent,
    CountedTiledPhaseEvent,
    TileStartedEvent,
    FluorescenceTileCountEvent,
    BeamTileCompletedEvent,
    FluorescenceTileCompletedEvent,
    TiledTerminalEvent,
    CountedTiledTerminalEvent,
]


def modality_of(event: TiledAcquisitionEvent) -> str:
    """Return the producer modality, with the legacy beam default."""
    return event.modality or MODALITY_BEAM


def is_modality(event: TiledAcquisitionEvent, modality: str) -> bool:
    """Whether *event* came from *modality*."""
    return modality_of(event) == modality


__all__ = [
    "MODALITY_BEAM",
    "MODALITY_FLUORESCENCE",
    "TiledEventType",
    "TiledPhase",
    "TiledOutcome",
    "TiledAcquisitionEvent",
    "TiledPhaseEvent",
    "CountedTiledPhaseEvent",
    "TileStartedEvent",
    "FluorescenceTileCountEvent",
    "BeamTileCompletedEvent",
    "FluorescenceTileCompletedEvent",
    "TiledTerminalEvent",
    "CountedTiledTerminalEvent",
    "TiledEvent",
    "modality_of",
    "is_modality",
]
