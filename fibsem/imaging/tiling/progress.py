"""What a tiled acquisition says about itself while it runs.

The imaging service's ``tiled_progress`` signal is shared by the beam tiler and the
fluorescence tileset runner, which report on it through `report_tiled_progress`.
`TiledProgress` is the whole contract, and `TiledStatus` is the whole vocabulary.

One flat record rather than a class per payload shape. Every consumer asks the same
three questions -- is this mine, does it carry counts, is there something to draw --
and none dispatches over a set of types; an `isinstance` check that misses a sibling
class is a run that never finishes, with nothing raised.

`modality` is a field and only a field, so there are not two discriminators that can
disagree.

Presentation is not on the wire. A producer reports *what happened*; each consumer
words it. `error` is the one free string, and it carries a *reason* -- an exception, a
stage-limits rejection naming every offending tile -- never a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Optional, Tuple

from fibsem.codec import NotSaved
from fibsem.progress import ProgressStatus, ends

# The beam tiler: SEM or FIB, since no consumer distinguishes them -- both draw into the
# same overview.
MODALITY_BEAM = "beam"
# The fluorescence tileset runner.
MODALITY_FLUORESCENCE = "fluorescence"

__all__ = [
    "MODALITY_BEAM",
    "MODALITY_FLUORESCENCE",
    "TiledStatus",
    "TiledProgress",
    "report_tiled_progress",
]


class TiledStatus(ProgressStatus):
    """Where a tiled run has got to.

    One vocabulary covering the whole run, so one field says whether a run is over.
    `TILES_ACQUIRED` is the runner saying the last tile is in and the stage is parked;
    `FINISHED` is the whole run being over, mosaic written.

    A `str` mixin so a member compares equal to its own value: these are persisted
    nowhere, but they do end up in log lines, and `"finished"` reads better than
    `TiledStatus.FINISHED`.
    """

    # The run's opening report: nothing acquired, how many there are to do, and an ETA
    # if the producer has one. One member for both producers even though they reach it
    # differently -- the beam tiler before it computes the grid, the fluorescence runner
    # after -- because what a consumer does with it is identical.
    STARTING = "starting"
    MOVING = "moving"
    # A tile is beginning. Not a progress report: emitted *before* the tile, so a count
    # taken from it is always one short.
    TILE_STARTED = "tile-started"
    # A tile is in and painted into the mosaic. The one report per tile that carries
    # counts, and the only one that carries a preview.
    TILE_COLLECTED = "tile-collected"
    # The runner is done and the stage is back where it started. Not the end of the run:
    # the mosaic still has to be stitched and written, and on the fluorescence side those
    # happen in the widget rather than the runner.
    TILES_ACQUIRED = "tiles-acquired"
    STITCHING = "stitching"
    SAVING = "saving"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


ends(TiledStatus.FINISHED, TiledStatus.CANCELLED, TiledStatus.FAILED)


# `eq=False`, so equality and hashing stay identity-based: `preview` holds an image
# whose `__eq__` compares a numpy array elementwise, so a generated `__eq__` would raise
# on two distinct events, and `hash()` would raise outright.
@dataclass(frozen=True, eq=False)
class TiledProgress:
    """One report from a tiled acquisition.

    Most fields are absent on most reports, and that is the point: a stage move carries
    no counts, a tile carries no error, and a consumer reads whichever it understands.
    `status` is the only thing every report has, and it is the only required field.

    Tile indices are 0-based on the wire, matching every other index in the codebase.
    `display_tile` is the only thing that adds one.
    """

    status: TiledStatus
    modality: str = MODALITY_BEAM
    # Tiles **completed**, not the tile in flight.
    completed: Optional[int] = None
    # Tiles that will actually be acquired, not grid cells: a progress bar that stops at
    # 9/25 on a successful sparse run reads as a failure.
    total: Optional[int] = None
    # The mosaic so far, as a placeable image -- whole canvas rather than the single
    # tile, so the receiver needs no state of its own and simply redisplays what it is
    # given, which is also what makes a late subscriber correct.
    #
    # The type varies with the modality, and that is not overloading: both sides mean
    # "the mosaic so far", and each carries the pixel size and stage position that place
    # it. A `FibsemImage` cannot hold the fluorescence one -- `check_data_format` accepts
    # `ndim == 2` only, and a fluorescence preview is (channels, y, x).
    #
    # Decimated, and the decimation is already in the metadata's pixel size. Coarser
    # pixels over a smaller count cover the same ground, so a real-space display needs
    # nothing else to put it in the right place at the right size.
    #
    # A `FibsemImage` or a `FluorescenceImage`. Pixels, so never in an event.
    preview: Annotated[Optional[Any], NotSaved] = None
    row_index: Optional[int] = None
    column_index: Optional[int] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    estimated_total_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    # Why a run ended, when the status alone does not say enough -- an exception's text,
    # or a stage-limits rejection naming every offending tile. The *reason*, never the
    # label: a consumer words `FAILED` itself, and puts this behind it.
    error: Optional[str] = None
    # Something about a run that finished which the operator should know -- the tiles
    # not lining up with the stage steps (`layout_check`). A finished run, not a
    # failed one: the mosaic is saved either way.
    warning: Optional[str] = None

    @property
    def display_tile(self) -> Optional[Tuple[int, int]]:
        """`(row, column)` of the tile this report is about, 1-based, for people.

        The one place the offset is applied: tile 0 is "1" to a reader and 0 to a
        `_ordered[...]` lookup.
        """
        if self.row_index is None or self.column_index is None:
            return None
        return self.row_index + 1, self.column_index + 1


def report_tiled_progress(microscope: Any, progress: TiledProgress) -> None:
    """Report tile progress on the microscope's imaging service
    (``microscope.imaging.tiled_progress``). A microscope with no beam has no imaging
    service, so its fluorescence overview reports to no one."""
    imaging = microscope.imaging
    if imaging is not None:
        imaging.tiled_progress.emit(progress)
