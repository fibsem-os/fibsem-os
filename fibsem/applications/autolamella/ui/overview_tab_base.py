"""What both overview tabs do with an experiment, regardless of what is imaging it.

`AutoLamellaOverviewTab` and `AutoLamellaFluorescenceOverviewTab` drive different widgets
over different poses, and they were doing it through two copies of the same plumbing:
both hold a lamella list beside a canvas, both answer the window's selection sync, both
turn a clicked crosshair into a lamella. A fix to one left the other behaving differently
-- and both are wired into the same handler in `AutoLamellaMainUI`, so "differently" here
means one of four lists disagreeing with the other three.

# The pose is the difference

Almost everything that looked different between the two turned out to be one question
asked twice: *which pose does this tab mean?* The beam tab marks, moves to and drags the
**milling** pose; the fluorescence tab the **fluorescence** pose. That is `_pose_of`, and
with it in place `refresh_positions`, `_on_move_to_requested` and `_on_add_requested`
have one implementation each.

**This hook is where a mistake hides.** On a compustage the two poses share x, y, z *and*
r, and differ only in tilt -- -23 degrees against -180. A test asserting where a marker
landed cannot tell them apart, and a canvas marking the wrong one looks entirely
plausible while putting every lamella on the far side of the grid. Anything checking this
has to assert the tilt (FIB-709).

# What is deliberately still written twice

`_on_move_requested` -- dragging a marked lamella to a new point. The beam tab writes the
milling pose and syncs the fluorescence one after it; the fluorescence tab derives both
through `build_lamella_poses`, and asks first, because from a canvas showing only the
fluorescence side it is not visible that the milling pose moves too. Those are different
operations that happen to share a name, and folding them together behind a hook would
hide the one part a reader most needs to see.

# The contract

A base class *and* a constructor: it builds the layout, the lamella list and the common
connections, because that part was identical. Subclasses supply the pieces below and are
free to add their own -- the beam tab's defect handling, the fluorescence tab's objective
position -- which is the point of two widgets sharing parts rather than one widget with a
mode flag.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from fibsem.applications.autolamella.ui.lamella_name_list_widget import (
    LamellaNameListWidget,
)
from fibsem.ui import notification_service

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.structures import FibsemStagePosition

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
    # Whether a run is in progress in this tab, forwarded from the widget. The window
    # listens so it can stop the *other* overview taking the stage (FIB-706); it
    # connects to the tab rather than the widget because the widget is rebuilt on every
    # reconnection and a subscription to the old one would be stale and undisconnectable.
    acquiring_changed = pyqtSignal(bool)

    # What this canvas marks, in the words a message needs: "has no fluorescence pose to
    # move to" against "has no position to move to". A user reading the second one on the
    # fluorescence tab would go looking for a position the lamella has.
    POSE_NOUN = "position"
    # Which overview this is, for the lines logged about it.
    OVERVIEW_NOUN = "overview"

    def __init__(self, autolamella_ui, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.autolamella_ui = autolamella_ui
        # Built empty and filled in on connection: both overview widgets require a
        # microscope at construction -- every scale on their canvases comes from the
        # instrument -- and at the point a tab is reserved there may be no microscope.
        self.overview: Optional[QWidget] = None
        # The microscope the current widget was built for. A reconnection hands out a
        # new object, and the old widget would go on reading geometry from an instrument
        # nobody is driving (FIB-433), so the identity is worth keeping.
        self._microscope = None
        # Set while this tab is the one driving a selection, so the highlight it gets
        # back does not re-enter the list and fight whatever the user just clicked.
        self._syncing_selection = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lamella_list = LamellaNameListWidget()
        self.lamella_list.enable_actions_button(True)
        self.lamella_list.enable_move_to_action(True)
        self.lamella_list.enable_remove_button(True)
        self.lamella_list.lamella_selected.connect(self._on_list_selection)
        self.lamella_list.move_to_requested.connect(self._on_move_to_requested)
        self.lamella_list.remove_requested.connect(self._on_remove_requested)
        self._configure_list()

    # ── what a subclass supplies ─────────────────────────────────────────

    def _pose_of(self, lamella) -> Optional["FibsemStagePosition"]:
        """Where this tab's canvas puts *lamella*, or None if it cannot place it.

        The one hook that explains most of the difference between the two tabs. Returning
        None is not an error: a lamella with no fluorescence pose is the normal state on
        a system without an FM, and one with no milling pose can be loaded from an older
        experiment. Both are reported rather than dropped in silence.
        """
        raise NotImplementedError

    def _build_overview(self, microscope) -> QWidget:
        """Construct this tab's overview widget for *microscope*."""
        raise NotImplementedError

    def _can_build(self, microscope) -> bool:
        """Whether to hold a live widget for *microscope* at all.

        Two different questions with one shape: the beam tab answers a feature flag, the
        fluorescence tab a capability (is there a detector). Either way a False means the
        widget is dropped rather than merely hidden -- it subscribes to the microscope
        for its lifetime, so one left behind goes on working for a tab nobody can open.
        """
        return True

    def _configure_list(self) -> None:
        """Whatever else this tab's lamella list needs, beyond the three common actions."""

    def _show_positions(self, positions, lamellae) -> None:
        """Hand the placeable positions to the widget.

        Overridable because the beam tab has one more thing to say -- which lamellae are
        flagged as defective. That concept belongs to the beam side and stays there
        rather than becoming a base-class notion the fluorescence tab passes empty.
        """
        self.overview.set_positions(positions)

    def _add_lamella_kwargs(self) -> Dict[str, Any]:
        """Anything beyond the position that this tab knows about a new lamella."""
        return {}

    def _on_move_requested(self, name: str, position) -> None:
        """A user dragged a marked lamella to a new point. See the module docstring for
        why this one is not shared."""
        raise NotImplementedError

    # ── what the window asks ─────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Whether there is a widget to drive, i.e. whether the tab does anything."""
        return self.overview is not None

    @property
    def is_acquiring(self) -> bool:
        """Whether an overview acquisition is running here.

        Asked *of the other tab* before anything drives the stage: a click-to-move on one
        overview while the other is mid-tileset does not fail loudly -- the run goes on
        stamping tiles with the poses it planned, so the mosaic comes out plausible and
        wrong (FIB-706). Both widgets answer this; only one tab used to pass it on.
        """
        return self.overview is not None and self.overview.is_acquiring

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
        if microscope is None or not self._can_build(microscope):
            self._drop_overview()
            self.availability_changed.emit(False)
            return

        if self.overview is not None:
            if self._microscope is microscope:
                return
            self._drop_overview()

        try:
            self.overview = self._build_overview(microscope)
        except Exception as e:
            logger.error(f"Could not create the {self.OVERVIEW_NOUN} tab: {e}")
            self.availability_changed.emit(False)
            return

        self.overview.position_add_requested.connect(self._on_add_requested)
        self.overview.position_move_requested.connect(self._on_move_requested)
        self.overview.position_selected.connect(self._on_marker_clicked)
        # Signal to signal: the widget's answer is this tab's answer, and forwarding it
        # by hand would be a second place for the two to disagree.
        self.overview.acquiring_changed.connect(self.acquiring_changed)
        # In the settings column rather than beside it: the positions are the subject of
        # this tab, and a column of their own would read as a third pane.
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
        # to `add_settings_section`. The list belongs to this tab and outlives any one
        # overview, so it is moved out rather than left to be collected.
        #
        # Kept as a precondition rather than because a test proves it: under the
        # offscreen platform the deferred delete does not actually run, so the failure
        # it prevents cannot be reproduced there.
        self.lamella_list.setParent(self)
        self.lamella_list.hide()
        try:
            self.overview.close()
        except Exception as e:
            logger.debug(f"Error closing the {self.OVERVIEW_NOUN} widget: {e}")
        self.layout().removeWidget(self.overview)
        self.overview.deleteLater()
        self.overview = None
        self._microscope = None
        # A tab with no widget is certainly not acquiring, and nothing else will say so:
        # the widget that would have emitted it is the one just dropped. Without this a
        # host that locked the other tab during a run would hold that lock forever if
        # the running tab were dropped -- a reconnection mid-run does exactly that.
        self.acquiring_changed.emit(False)

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
        """Mark the experiment's lamellae on the canvas, at this tab's own pose.

        Every lamella reaches the list, including the ones the canvas cannot place: the
        list is how you find out that a lamella exists but has no pose for this view.
        Dropping those from both would leave the log as the only place that says so.
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

        positions, unplaceable = [], []
        for lamella in lamellae:
            place = self._pose_of(lamella)
            if place is None:
                unplaceable.append(lamella.name)
                continue
            # Named here rather than relying on whatever the stored position carries:
            # the marker's label is the lamella's name, and a pose saved before names
            # were stamped on positions would otherwise draw an unlabelled marker.
            position = deepcopy(place)
            position.name = lamella.name
            positions.append(position)

        self._show_positions(positions, lamellae)
        if unplaceable:
            logger.info(
                f"{len(unplaceable)} lamella(e) have no {self.POSE_NOUN} and are not "
                f"shown on the {self.OVERVIEW_NOUN}: {', '.join(unplaceable)}"
            )

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

    def set_interactive(self, enabled: bool, reason: str = "") -> None:
        """Allow or forbid starting work, for a host that has taken the instrument.

        *reason* is carried through to the widget, which quotes it when it refuses a
        move: the widget sees one `False` whether a workflow owns the instrument or the
        other overview is mid-run, and only the host knows which.
        """
        if self.overview is None:
            return
        self.overview.set_interactive(enabled, reason)

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

    def _on_move_to_requested(self, lamella) -> None:
        """Drive the stage to a lamella's pose, as this tab means it.

        Not "the lamella's position": moving to the milling pose from the fluorescence
        canvas would swing the stage 180 degrees away from the view you asked to centre,
        and vice versa.
        """
        if lamella is None or self.overview is None:
            return
        place = self._pose_of(lamella)
        if place is None:
            notification_service.show_toast(
                f"{lamella.name} has no {self.POSE_NOUN} to move to.", "warning"
            )
            return
        self.overview.move_to(place)

    def _on_remove_requested(self, lamella) -> None:
        """Remove a lamella from the experiment.

        The row's own button already asked -- `_LamellaRow._on_remove_clicked` puts up
        the confirmation before it emits -- so this does not ask twice.

        Re-marks the canvas afterwards. The window rebuilds displays off the experiment's
        `removed` event too, so for one of the two tabs this is a second redraw of the
        same set; that is the safe direction to be wrong in, and the asymmetry where only
        one tab was covered is what made this worth sharing.
        """
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
            lamella = self.autolamella_ui.add_new_lamella(
                stage_position=position, **self._add_lamella_kwargs()
            )
        except Exception as e:
            logger.error(f"Could not add a lamella from the {self.OVERVIEW_NOUN}: {e}")
            notification_service.show_toast(str(e), "error")
            return

        self.refresh_positions()
        self.set_selected(lamella)
        notification_service.show_toast(f"Added {lamella.name}.", "info")
