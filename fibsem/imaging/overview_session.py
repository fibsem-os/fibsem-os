"""What an overview tab shows and does, without Qt.

One per overview of the connected microscope, the beam one and the fluorescence
one, held on the app's session (`AppSession.overview`). The widget draws it and
binds to it; a host writes the positions and the lock here, not on the widget.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Iterable, List, Optional, Set

from psygnal import Signal

from fibsem.structures import FibsemStagePosition

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.microscope import FibsemMicroscope

logger = logging.getLogger(__name__)

BEAM = "beam"
FLUORESCENCE = "fm"
OVERVIEW_KINDS = (BEAM, FLUORESCENCE)

# What `changed` names.
POSITIONS = "positions"
SELECTION = "selection"
LOCK = "lock"
RUNNING = "running"


class OverviewSession:
    """One overview's marked positions, selection, lock, run state, latest
    mosaic and cached stage position, and its stage moves.

    ``changed`` fires with the part that changed (``POSITIONS``, ``SELECTION``,
    ``LOCK``, ``RUNNING``), on the thread that changed it.
    """

    changed = Signal(str)

    def __init__(self, microscope: "FibsemMicroscope", kind: str = BEAM) -> None:
        if kind not in OVERVIEW_KINDS:
            raise ValueError(f"No overview kind {kind!r}; one of {OVERVIEW_KINDS}.")
        if kind == FLUORESCENCE and microscope.fm is None:
            raise ValueError("This microscope has no fluorescence detector.")
        self.microscope = microscope
        self.kind = kind
        self.positions: List[FibsemStagePosition] = []
        # Names a host flagged (for an experiment, a defective lamella).
        self.flagged: Set[str] = set()
        self.selected: Optional[str] = None
        # Whether a run is under way, and whether a host allows one to start: kept
        # apart, so a workflow ending can't re-enable a tab whose run is still going.
        self.running = False
        self.interactive = True
        self.lock_reason = "a workflow is running"
        # The last overview acquired: a FibsemImage, or a FluorescenceImage.
        self.mosaic = None
        # Where the stage is, as last told; everything drawn from it agrees.
        self.stage_position: Optional[FibsemStagePosition] = None

    def set_positions(
        self,
        positions: Iterable[FibsemStagePosition],
        flagged: Optional[Iterable[str]] = None,
    ) -> None:
        """Mark these positions, replacing whatever was marked; *flagged* names
        the ones to draw in the warning colour."""
        self.positions = list(positions)
        self.flagged = set(flagged or ())
        self.changed.emit(POSITIONS)

    def select(self, name: Optional[str]) -> None:
        """Highlight one marked position by name, or none."""
        self.selected = name
        self.changed.emit(SELECTION)

    def set_interactive(self, enabled: bool, reason: str = "") -> None:
        """Allow or forbid starting work, for a host that has taken the
        instrument. *reason* completes "Cannot move the stage while ___"."""
        self.interactive = bool(enabled)
        self.lock_reason = reason or "a workflow is running"
        self.changed.emit(LOCK)

    def set_running(self, running: bool) -> None:
        self.running = bool(running)
        self.changed.emit(RUNNING)

    def read_stage_position(self) -> Optional[FibsemStagePosition]:
        """Read where the stage is and keep it. A failed read keeps the last."""
        try:
            self.stage_position = deepcopy(self.microscope.get_stage_position())
        except Exception as e:
            logger.debug(f"Could not read the stage position: {e}")
        return self.stage_position

    def move_refusal(self) -> Optional[str]:
        """Why the stage may not be moved from this overview now, or None."""
        if self.running:
            return "Cannot move the stage during an acquisition."
        if not self.interactive:
            return f"Cannot move the stage while {self.lock_reason}."
        return None

    def move_to(self, target: FibsemStagePosition) -> None:
        """Drive the stage to *target*; blocks. Raises when the move fails.

        The position is read back either way, which publishes where the stage
        stopped through ``stage.position.changed``.
        """
        try:
            self.microscope.safe_absolute_stage_movement(target)
        finally:
            try:
                self.microscope.get_stage_position()
            except Exception as e:
                logger.debug(f"Could not confirm the stage position after moving: {e}")

    def stage_limits(self):
        """The stage's travel limits, or None when it has none to give."""
        stage = self.microscope.stage
        return stage.position.limits if stage is not None else None
