"""What a tiled acquisition says about itself while it runs.

`FibsemMicroscope.tiled_acquisition_signal` is shared by the beam tiler and the
fluorescence tileset runner. It used to carry a bare dict with no declared shape, so
every consumer worked out what it had received from which keys happened to be present.
That held while `imaging.tiled` was the only emitter; with a second producer the same
key came to mean different things depending on who sent it, which is the root of
FIB-736 and FIB-739.

`TiledProgress` is the whole contract, and `TiledStatus` is the whole vocabulary.

One flat record rather than a class per payload shape. Every consumer asks the same
three questions -- is this mine, does it carry counts, is there something to draw --
and none dispatches over an exhaustive set of types, so types buy nothing that optional
fields do not. A hierarchy also fails in a way a flat record cannot: an `isinstance`
check that quietly misses a sibling class is a run that never finishes, with nothing
raised anywhere.

`modality` is a field and only a field. Encoding it in the type as well -- a
`BeamTileCompleted` beside a `FluorescenceTileCompleted` -- gives two discriminators
that can disagree, and the one that disagrees silently paints a fluorescence mosaic
into the beam overview (FIB-725).

Presentation is not on the wire. A producer reports *what happened*; each consumer
words it. `imaging.tiled` is a headless module and was building English for three
widgets that each wanted different wording, so the two producers had drifted to
"Stitching Tiles" and "Stitching tiles..." for the same state. `error` is the one free
string, and it carries a *reason* -- an exception, a stage-limits rejection naming
every offending tile -- never a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    # Annotation-only, so this module stays free of a runtime dependency on either image
    # type. `fibsem.microscope` imports it to type the signal, and it is imported in turn
    # by both producers -- a runtime import here would put `fibsem.fm.structures` in the
    # path of every one of them for a name that is never called.
    from fibsem.fm.structures import FluorescenceImage
    from fibsem.structures import FibsemImage

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
]


class TiledStatus(str, Enum):
    """Where a tiled run has got to.

    One vocabulary covering the whole run rather than a phase enum plus an outcome enum.
    Splitting them means a consumer needs both fields to know whether a run is over, and
    the two overlap on the word "finished" -- which is what the split was protecting
    against in the first place. Both members survive here under names that cannot be
    confused: `TILES_ACQUIRED` is the runner saying the last tile is in and the stage is
    parked, `FINISHED` is the whole run being over, mosaic written.

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
    # A tile is beginning. Deliberately not a progress report: emitted *before* the tile,
    # a count taken from it is always one short, and a bar driven off it stopped at 3/4
    # for a whole four-tile run (FIB-736).
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

    @property
    def is_terminal(self) -> bool:
        """Whether this ends the run.

        On the status rather than restated at each consumer. Three of them need the
        question answered, and a membership tuple copied into three files is a tuple
        that drifts -- the next status added to the enum is one somebody forgets to add
        to one of them, and the symptom is a progress bar that never clears.
        """
        return self in _TERMINAL_STATUSES


_TERMINAL_STATUSES = frozenset(
    {TiledStatus.FINISHED, TiledStatus.CANCELLED, TiledStatus.FAILED}
)


# `eq=False`, so equality and hashing stay identity-based.
#
# The generated `__eq__` compares every field, and `preview` holds an image whose own
# `__eq__` compares a numpy array elementwise -- `FluorescenceImage` is a dataclass, so
# it has one. Comparing two events then raises `ValueError: truth value of an array with
# more than one element is ambiguous`, and only sometimes: comparing an event to *itself*
# short-circuits on identity and returns True, so the naive test passes and the failure
# waits for two distinct events. `hash()` raises `TypeError` outright. Identity semantics
# are both honest and what a one-shot progress report actually wants.
@dataclass(frozen=True, eq=False)
class TiledProgress:
    """One report from a tiled acquisition.

    Most fields are absent on most reports, and that is the point: a stage move carries
    no counts, a tile carries no error, and a consumer reads whichever it understands.
    `status` is the only thing every report has, and it is the only required field --
    which is also what keeps this constructible on Python 3.8, where `kw_only` does not
    exist and required fields have to come first.

    Tile indices are 0-based on the wire, matching every other index in the codebase.
    `display_tile` is the only thing that adds one.
    """

    status: TiledStatus
    modality: str = MODALITY_BEAM
    # Tiles **completed**, not the tile in flight. The beam tiler increments and then
    # emits, so its count has always meant this; the fluorescence side used to say it
    # twice per tile with two meanings, and a consumer choosing between them got a bar
    # that changed scale at every boundary (FIB-736, FIB-739).
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
    preview: Optional[Union["FibsemImage", "FluorescenceImage"]] = None
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

    @property
    def display_tile(self) -> Optional[Tuple[int, int]]:
        """`(row, column)` of the tile this report is about, 1-based, for people.

        The one place the offset is applied. Tile 0 is "1" to a reader and 0 to a
        `_ordered[...]` lookup, and the two spellings sat one function apart on the
        fluorescence runner -- the log said `[1/3][1/3]` while the payload said row 0 --
        so the conversion belongs somewhere single and testable rather than inline at
        whichever call site happens to need it.
        """
        if self.row_index is None or self.column_index is None:
            return None
        return self.row_index + 1, self.column_index + 1
