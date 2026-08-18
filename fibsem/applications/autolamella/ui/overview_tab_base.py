"""What both overview tabs do with an experiment, regardless of what is imaging it.

`AutoLamellaOverviewTab` and `AutoLamellaFluorescenceOverviewTab` drive different widgets
over different poses, and they were doing it through two copies of the same plumbing:
both hold a lamella list beside a canvas, both answer the window's selection sync, both
turn a clicked crosshair into a lamella. A fix to one left the other behaving differently
-- and both are wired into the same handler in `AutoLamellaMainUI`, so "differently" here
means one of four lists disagreeing with the other three.

**Only the parts whose code is identical live here.** Measured method by method with
docstrings stripped (FIB-697): seven were byte-identical and `_on_marker_clicked` differed
only in `logging` against `logger`. Those eight are below.

Deliberately *not* here, for now:

* `refresh_microscope` and `_drop_overview` -- 96% and 98% identical, which is close
  enough to look safe and not close enough to move without reading both carefully.
* `refresh_positions`, `_on_add_requested`, `_on_move_requested`, `_on_move_to_requested`,
  `_on_remove_requested`, `__init__` -- these differ because of **which pose is read**,
  which is the real difference between the two tabs and wants a `_pose_of` hook rather
  than a copy-paste into a base class.

Finishing that is the rest of FIB-697. Starting with the identical eight means this
change cannot alter behaviour, which is worth more than doing it in one go.

# The contract

This is a mixin over attributes its subclasses own, not a constructor. Each subclass sets
`autolamella_ui`, `overview`, `lamella_list` and `_syncing_selection` in its own
`__init__`, and supplies `refresh_positions`. The base deliberately does not build any of
them: the two tabs construct their widgets differently and at different times, and a base
`__init__` that half-built them would be a third thing to keep in step.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

logger = logging.getLogger(__name__)


class AutoLamellaOverviewTabBase(QWidget):
    """The experiment-facing half of an overview tab, shared by both modalities."""

    # Whether there is a live widget to drive. The window listens so it can enable or
    # disable the tab; neither tab touches the tab bar itself.
    availability_changed = pyqtSignal(bool)
    # A lamella was picked in this tab's own list. The window forwards it to the other
    # lists; nothing here selects them directly, which is what keeps the sync in one
    # place instead of four.
    lamella_selected = pyqtSignal(object)

    # ── what the window asks ─────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Whether there is a widget to drive, i.e. whether the tab does anything."""
        return self.overview is not None

    @property
    def microscope(self):
        return self.autolamella_ui.microscope if self.autolamella_ui is not None else None

    @property
    def experiment(self):
        return self.autolamella_ui.experiment if self.autolamella_ui is not None else None

    def refresh_positions(self) -> None:
        """Mark the experiment's lamellae on the canvas, at this tab's own pose.

        Left to the subclass: the pose is the difference between the two tabs, and on a
        compustage the milling and fluorescence poses share x, y, z **and** r, differing
        only in tilt. Marking the wrong one puts every lamella on the far side of the
        grid while looking entirely plausible.
        """
        raise NotImplementedError

    def refresh_experiment(self) -> None:
        """Tell the overview widget where to save, and what to mark."""
        if self.overview is None:
            return
        experiment = self.experiment
        self.overview.set_save_directory(
            str(experiment.path) if experiment is not None else None
        )
        self.refresh_positions()

    def set_selected(self, lamella) -> None:
        """Highlight the selected lamella.

        Called from each of the window's selection handlers *before* their
        `_syncing_selection` guard, and deliberately: that guard exists to stop several
        lists selecting each other in circles, and this is not another list. It emits
        nothing and only repaints.
        """
        if self.overview is None:
            return
        self.overview.set_selected_position(
            lamella.name if lamella is not None else None
        )
        if not self._syncing_selection and lamella is not None:
            self.lamella_list.select(lamella.name)

    def set_interactive(self, enabled: bool) -> None:
        """Allow or forbid starting work, for a host that has taken the instrument."""
        if self.overview is None:
            return
        self.overview.set_interactive(enabled)

    # ── the list ─────────────────────────────────────────────────────────

    def _on_list_selection(self, lamella) -> None:
        """A row was clicked: highlight it on the canvas, and tell the window.

        The flag is what stops the round trip. The window answers by syncing every list
        it knows about, this tab included, and re-selecting the row under a click that is
        still happening moves the selection out from under the user.

        Checked as well as set: `_on_marker_clicked` selects the row itself, and the row
        answering by coming back through here announced the same lamella twice.
        """
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            if self.overview is not None:
                self.overview.set_selected_position(
                    lamella.name if lamella is not None else None
                )
            self.lamella_selected.emit(lamella)
        finally:
            self._syncing_selection = False

    def _on_marker_clicked(self, name: str) -> None:
        """A crosshair on the canvas was clicked: select that lamella everywhere.

        The canvas has already highlighted it -- it knows the name it drew. What it
        cannot do is turn a name into a lamella, so the list and the window are told from
        here, through the same path a row click takes.
        """
        experiment = self.experiment
        if experiment is None:
            return
        lamella = next((p for p in experiment.positions if p.name == name), None)
        if lamella is None:
            logger.debug(f"Clicked {name!r}, which is not in the experiment.")
            return
        self._syncing_selection = True
        try:
            self.lamella_list.select(name)
            self.lamella_selected.emit(lamella)
        finally:
            self._syncing_selection = False

    # Set by the subclass, in its own `__init__`. Annotations only -- they document
    # the contract and enforce nothing, so a subclass that forgets one fails on the
    # first click rather than at construction.
    autolamella_ui: object
    overview: Optional[QWidget]
    lamella_list: QWidget
    _syncing_selection: bool
