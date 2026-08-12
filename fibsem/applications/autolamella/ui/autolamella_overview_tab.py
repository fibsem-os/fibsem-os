"""The Overview tab: the FIB/SEM overview, plus what an experiment adds to it.

`FibsemOverviewWidget` knows nothing about lamellae. That is deliberate and worth
keeping -- it is what lets the same widget open standalone against a simulator, and be
built in a test from a microscope alone. It says *where* a user pointed and leaves the
meaning to whoever is listening.

This is whoever is listening, and it is the beam-side twin of
`AutoLamellaFluorescenceOverviewTab`. It lives in the autolamella package because
everything it adds is autolamella's: experiments, lamellae, defects and poses.

**Milling poses, not fluorescence ones** -- the opposite of its twin. This canvas looks
at the sample from the beam side, so a lamella is marked where the beam sees it. On a
compustage the two are 180 degrees apart, so passing the wrong one would put every
marker on the far side of the grid.

Why this exists at all
----------------------
`FibsemMinimapWidget` is the last module under `fibsem/ui/` that imports AutoLamella
(FIB-560): it reaches through `parent_widget.experiment` in nineteen places, types its
parent as `AutoLamellaUI`, and knows about `DefectType`, `sync_fluorescence_pose` and
milling protocol keys. It cannot simply be moved into the application package, because
it has a live generic host in `FibsemUI`. Splitting it in two along the line the parent
contract already suggests is the answer: the widget for anyone with a microscope, this
for anyone with an experiment.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from fibsem.applications.autolamella.poses import sync_fluorescence_pose
from fibsem.applications.autolamella.structures import DefectType
from fibsem.applications.autolamella.ui.lamella_name_list_widget import (
    LamellaNameListWidget,
)
from fibsem.ui import notification_service
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.applications.autolamella.structures import Lamella

logger = logging.getLogger(__name__)

# Defect states that make a lamella worth flagging on the canvas. Both mean "do not
# just re-target this": one has failed, the other needs work. The tab this replaces
# drew them red and orange; there is one warning colour here, because the list beside
# the canvas already distinguishes them and a second colour on a crosshair does not.
FLAGGED_DEFECT_STATES = (DefectType.FAILURE, DefectType.REWORK)


class AutoLamellaOverviewTab(QWidget):
    """Drives `FibsemOverviewWidget` on behalf of an experiment.

    Built empty and filled in on connection, like its fluorescence twin: the overview
    widget requires a microscope at construction, and at the point the tab is reserved
    there may be no microscope at all.
    """

    # Whether there is a live widget to drive. The window listens so it can enable or
    # disable the tab; this object deliberately does not touch the tab bar.
    availability_changed = pyqtSignal(bool)
    # A lamella was picked in this tab's own list. The window forwards it to the other
    # lists; nothing here selects them directly, which is what keeps the sync in one
    # place instead of four.
    lamella_selected = pyqtSignal(object)

    def __init__(self, autolamella_ui, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.autolamella_ui = autolamella_ui
        self.overview: Optional[FibsemOverviewWidget] = None
        # The microscope the current widget was built for. A reconnection hands out a
        # new object, and the old widget would go on reading geometry from an
        # instrument nobody is driving, so the identity is worth keeping.
        self._microscope = None
        # Set while this tab is the one driving a selection, so the highlight it gets
        # back does not re-enter the list and fight whatever the user just clicked.
        self._syncing_selection = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lamella_list = LamellaNameListWidget()
        self.lamella_list.enable_defect_button(True)
        self.lamella_list.enable_actions_button(True)
        self.lamella_list.enable_move_to_action(True)
        self.lamella_list.enable_remove_button(True)
        self.lamella_list.lamella_selected.connect(self._on_list_selection)
        self.lamella_list.move_to_requested.connect(self._on_move_to_requested)
        self.lamella_list.remove_requested.connect(self._on_remove_requested)
        self.lamella_list.defect_changed.connect(self._on_defect_changed)

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

    def refresh_microscope(self) -> None:
        """Build, rebuild or drop the overview widget to match the instrument.

        Called on every connection. Rebuilds when the microscope object has changed,
        because the widget holds its microscope for life and would otherwise be talking
        to the previous one.
        """
        microscope = self.microscope
        if microscope is None:
            self._drop_overview()
            self.availability_changed.emit(False)
            return

        if self.overview is not None:
            if self._microscope is microscope:
                return
            self._drop_overview()

        try:
            self.overview = FibsemOverviewWidget(microscope)
        except Exception as e:
            logger.error(f"Could not create the Overview tab: {e}")
            self.availability_changed.emit(False)
            return

        self.overview.position_add_requested.connect(self._on_add_requested)
        self.overview.position_move_requested.connect(self._on_move_requested)
        self.overview.position_selected.connect(self._on_marker_clicked)
        self.overview.add_settings_section("Lamella Positions", self.lamella_list)

        self._microscope = microscope
        self.layout().addWidget(self.overview)
        self.availability_changed.emit(True)
        self.refresh_experiment()

    def _drop_overview(self) -> None:
        """Retire the current overview widget, if there is one.

        `close()` rather than `deleteLater()` alone: the widget's `closeEvent` is what
        releases its psygnal subscriptions on the microscope, and those outlive the
        widget -- a stale one left connected emits into a torn-down Qt object.
        """
        if self.overview is None:
            return
        # Taken back before the widget goes. `add_settings_section` reparents the list
        # into the overview's column, so Qt destroying the overview would destroy the
        # list with it, and the next `refresh_microscope` would hand a dead C++ object
        # to `add_settings_section`.
        self.lamella_list.setParent(self)
        self.lamella_list.hide()
        try:
            self.overview.close()
        except Exception as e:
            logger.debug(f"Error closing the overview widget: {e}")
        self.layout().removeWidget(self.overview)
        self.overview.deleteLater()
        self.overview = None
        self._microscope = None

    def refresh_experiment(self) -> None:
        """Tell the overview widget where to save, and what to mark."""
        if self.overview is None:
            return
        experiment = self.experiment
        self.overview.set_save_directory(
            str(experiment.path) if experiment is not None else None
        )
        self.refresh_positions()

    def refresh_positions(self) -> None:
        """Mark the experiment's lamellae on the canvas.

        **Milling poses**, which is what `lamella.stage_position` is. This canvas looks
        at the sample from the beam side; marking the fluorescence poses here would put
        every lamella where the FM sees it, which on a compustage is the other side of
        the grid.
        """
        if self.overview is None:
            return
        experiment = self.experiment
        if experiment is None:
            self.overview.set_positions([])
            self.lamella_list.set_lamella([])
            return

        lamellae = list(experiment.positions)
        self.lamella_list.set_lamella(lamellae)

        positions, flagged = [], []
        for lamella in lamellae:
            if lamella.stage_position is None:
                continue
            # Named here rather than relying on whatever the stored position carries:
            # the marker's label is the lamella's name, and a position saved before
            # names were stamped on them would otherwise draw an unlabelled crosshair.
            position = deepcopy(lamella.stage_position)
            position.name = lamella.name
            positions.append(position)
            if lamella.defect.state in FLAGGED_DEFECT_STATES:
                flagged.append(lamella.name)

        self.overview.set_positions(positions, flagged=flagged)

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

    @property
    def is_acquiring(self) -> bool:
        """Whether an overview acquisition is running. The window asks before moving."""
        return self.overview is not None and self.overview.is_acquiring

    # ── the list ─────────────────────────────────────────────────────────

    def _on_list_selection(self, lamella) -> None:
        """A row was clicked: highlight it on the canvas, and tell the window."""
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
        cannot do is turn a name into a lamella.
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

    def _on_move_to_requested(self, lamella) -> None:
        """Drive the stage to a lamella's milling pose."""
        if lamella is None or self.overview is None:
            return
        if lamella.stage_position is None:
            notification_service.show_toast(
                f"{lamella.name} has no position to move to.", "warning"
            )
            return
        self.overview.move_to(lamella.stage_position)

    def _on_defect_changed(self, lamella) -> None:
        """Persist a defect set from the list, and re-flag the canvas.

        The row writes `lamella.defect` and emits; nothing was listening, so the change
        lived in memory and was gone on reload (FIB-564). Redrawing as well as saving,
        because the marker's colour is one of the things the defect decides.
        """
        experiment = self.experiment
        if experiment is None:
            return
        experiment.save()
        self.refresh_positions()

    def _on_remove_requested(self, lamella) -> None:
        """Remove a lamella from the experiment. The row already confirmed."""
        experiment = self.experiment
        if experiment is None or lamella is None:
            return
        try:
            experiment.positions.remove(lamella)
        except ValueError:
            logger.debug(f"Cannot remove {lamella.name!r}: not in the experiment.")
            return
        experiment.save()
        self.refresh_positions()
        notification_service.show_toast(f"Removed {lamella.name}.", "info")

    # ── turning a request into a lamella ─────────────────────────────────

    def _on_add_requested(self, position) -> None:
        """A user asked for a new lamella at a point on the overview."""
        if self.autolamella_ui is None or self.experiment is None:
            notification_service.show_toast(
                "Load an experiment before marking positions.", "warning"
            )
            return
        try:
            lamella = self.autolamella_ui.add_new_lamella(stage_position=position)
        except Exception as e:
            logger.error(f"Could not add a lamella from the overview: {e}")
            notification_service.show_toast(str(e), "error")
            return

        self.refresh_positions()
        self.set_selected(lamella)
        notification_service.show_toast(f"Added {lamella.name}.", "info")

    def _on_move_requested(self, name: str, position) -> None:
        """A user asked to move a marked lamella to a point on the overview.

        Moves the milling pose, and takes the fluorescence one with it. They describe
        one piece of sample from two sides, so leaving the other behind would have it go
        on naming where this lamella *used to be* -- and nothing about a stale pose
        looks wrong.
        """
        experiment = self.experiment
        if experiment is None:
            return
        lamella = next((p for p in experiment.positions if p.name == name), None)
        if lamella is None:
            logger.debug(f"Cannot move {name!r}: no such lamella in the experiment.")
            return

        lamella.stage_position = position
        lamella.update_milling_angle(self.microscope)
        sync_fluorescence_pose(self.microscope, lamella)

        experiment.save()
        self.refresh_positions()
        notification_service.show_toast(f"Moved {name}.", "info")
