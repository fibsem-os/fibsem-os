"""The FM Overview tab: the fluorescence overview, plus what an experiment adds to it.

`FMOverviewWidget` knows nothing about lamellae. That is deliberate and worth keeping --
it is what lets the same widget open standalone against a simulator, and be built in a
test from a microscope alone. It says *where* a user pointed and leaves the meaning to
whoever is listening.

This is whoever is listening. It lives in the autolamella package because everything it
adds is autolamella's: experiments, lamellae, poses, and the milling side of a sample the
fluorescence canvas cannot see.

A widget rather than a plain controller, because part of what it adds is *visible* --
`LamellaNameListWidget` needs real `Lamella` objects (it subscribes to
`lamella.events.description` and reads the defect and task state), so it cannot live
inside the fluorescence widget. Something has to lay the two out together, and a QWidget
that owns both is simpler than a controller reaching into another widget's layout.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

import fibsem.config as fibsem_cfg
from fibsem.applications.autolamella.poses import (
    FLUORESCENCE_ORIENTATION,
    build_lamella_poses,
)
from fibsem.ui import notification_service
from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget
from fibsem.ui.utils import message_box_ui
from fibsem.applications.autolamella.ui.lamella_name_list_widget import LamellaNameListWidget

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.applications.autolamella.structures import Lamella


class AutoLamellaFluorescenceOverviewTab(QWidget):
    """Drives `FMOverviewWidget` on behalf of an experiment.

    Built empty and filled in on connection: `FMOverviewWidget` requires a microscope
    with a fluorescence detector at construction -- it has no meaningful empty state,
    since every scale on its canvas comes from the camera -- and at the point the tab is
    reserved there may be no microscope at all.
    """

    # Whether there is a live widget to drive. The window listens so it can enable or
    # disable the tab; this object deliberately does not touch the tab bar, which is the
    # one thing about the tab that is not its business.
    availability_changed = pyqtSignal(bool)
    # A lamella was picked in this tab's own list. The window forwards it to the other
    # lists; nothing here selects them directly, which is what keeps the sync in one
    # place instead of four.
    lamella_selected = pyqtSignal(object)

    def __init__(self, autolamella_ui, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.autolamella_ui = autolamella_ui
        self.overview: Optional[FMOverviewWidget] = None
        # The microscope the current widget was built for. A reconnection hands out a new
        # object, and the old widget would go on reading geometry from an instrument
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
        """Build, rebuild or drop the fluorescence widget to match the instrument.

        Called on every connection. Rebuilds when the microscope object has changed,
        because the widget holds its microscope for life and would otherwise be talking
        to the previous one -- destructive of anything on the canvas, and the reason
        FIB-433 exists to do this without throwing the view away.
        """
        microscope = self.microscope

        if (
            not fibsem_cfg.FEATURE_FM_OVERVIEW_TAB_ENABLED
            or microscope is None
            or microscope.fm is None
        ):
            # Switched off, or no fluorescence detector. The tab stays in place but dead
            # rather than being removed, so the tab bar does not change shape between
            # systems; hiding it is the window's job.
            self._drop_overview()
            self.availability_changed.emit(False)
            return

        if self.overview is not None:
            if self._microscope is microscope:
                return
            self._drop_overview()

        try:
            self.overview = FMOverviewWidget(microscope)
        except Exception as e:
            logging.error(f"Could not create the FM overview tab: {e}")
            self.availability_changed.emit(False)
            return

        self.overview.position_add_requested.connect(self._on_add_requested)
        self.overview.position_move_requested.connect(self._on_move_requested)
        self.overview.position_selected.connect(self._on_marker_clicked)
        # In the settings column rather than beside it: the positions are the subject of
        # this tab, and a column of their own would read as a third pane.
        self.overview.add_settings_section("Lamella Positions", self.lamella_list)

        self._microscope = microscope
        self.layout().addWidget(self.overview)
        self.availability_changed.emit(True)
        self.refresh_experiment()

    def _drop_overview(self) -> None:
        """Retire the current fluorescence widget, if there is one.

        `close()` rather than `deleteLater()` alone: the widget's `closeEvent` is what
        releases its psygnal subscriptions on the microscope, and those outlive the
        widget -- a stale one left connected emits into a torn-down Qt object.
        """
        if self.overview is None:
            return
        # Taken back before the widget goes. `add_settings_section` reparents the list
        # into the overview's column, so Qt destroying the overview would destroy the
        # list with it, and the next `refresh_microscope` would hand a dead C++ object to
        # `add_settings_section`. The list belongs to this tab and outlives any one
        # overview, so it is moved out rather than left to be collected.
        #
        # Kept as a precondition rather than because a test proves it: under the
        # offscreen platform the deferred delete does not actually run, so the failure it
        # prevents cannot be reproduced here. `test_the_list_survives_the_overview_being_
        # rebuilt` covers the reuse, not this mechanism.
        self.lamella_list.setParent(self)
        self.lamella_list.hide()
        try:
            self.overview.close()
        except Exception as e:
            logging.debug(f"Error closing the FM overview widget: {e}")
        self.layout().removeWidget(self.overview)
        self.overview.deleteLater()
        self.overview = None
        self._microscope = None

    def refresh_experiment(self) -> None:
        """Tell the fluorescence widget where to save, and what to mark.

        Told rather than handed the experiment: the widget knows nothing about
        experiments, and keeping it that way is what lets it open standalone against a
        simulator and be constructed in a test from a microscope alone.
        """
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
        `_syncing_selection` guard, and deliberately: that guard exists to stop three
        lists selecting each other in circles, and this is not a fourth list. It emits
        nothing and only repaints, so it wants to run whichever list the selection came
        from -- which is exactly what the guard suppresses.
        """
        if self.overview is None:
            return
        self.overview.set_selected_position(
            lamella.name if lamella is not None else None
        )
        # Not while this tab is the one that raised the selection: the list already shows
        # what the user clicked, and re-selecting it would fight them mid-click.
        if not self._syncing_selection and lamella is not None:
            self.lamella_list.select(lamella.name)

    def refresh_positions(self) -> None:
        """Mark the experiment's lamellae on the fluorescence canvas.

        **Fluorescence poses, not the primary stage position.** A lamella carries one of
        each and they are different places on this canvas -- on a compustage the stage
        flips 180 degrees between them -- so passing the milling pose would mark every
        lamella somewhere the sample is not.

        A lamella with no fluorescence pose cannot be placed here at all, which is the
        normal state on a system with no FM and possible on one loaded from an older
        experiment. Those are counted rather than dropped in silence: a canvas showing
        three of five positions and saying nothing is worse than one saying which two it
        cannot show.
        """
        if self.overview is None:
            return
        experiment = self.experiment
        if experiment is None:
            self.overview.set_positions([])
            self.lamella_list.set_lamella([])
            return

        # Every lamella, including the ones the canvas cannot place. The list is how you
        # find out a lamella exists but has no fluorescence pose -- dropping those here
        # too would leave the log as the only place that says so.
        self.lamella_list.set_lamella(list(experiment.positions))

        positions, unplaceable = [], []
        for lamella in experiment.positions:
            pose = lamella.fluorescence_pose
            if pose is None or pose.stage_position is None:
                unplaceable.append(lamella.name)
                continue
            # Named here rather than relying on whatever the stored position carries:
            # the marker's label is the lamella's name, and a pose saved before names
            # were stamped on positions would otherwise draw an unlabelled dot.
            position = deepcopy(pose.stage_position)
            position.name = lamella.name
            positions.append(position)

        self.overview.set_positions(positions)
        if unplaceable:
            logging.info(
                f"{len(unplaceable)} lamella(e) have no fluorescence pose and are not "
                f"shown on the FM overview: {', '.join(unplaceable)}"
            )

    def set_interactive(self, enabled: bool) -> None:
        """Allow or forbid starting work, for a host that has taken the instrument."""
        if self.overview is None:
            return
        self.overview.set_interactive(enabled)

    # ── the list ─────────────────────────────────────────────────────────

    def _on_list_selection(self, lamella) -> None:
        """A row was clicked: highlight it on the canvas, and tell the window.

        The flag is what stops the round trip. The window answers by syncing every list
        it knows about, this tab included, and re-selecting the row under a click that
        is still happening moves the selection out from under the user.

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
            logging.debug(f"Clicked {name!r}, which is not in the experiment.")
            return
        self._syncing_selection = True
        try:
            self.lamella_list.select(name)
            self.lamella_selected.emit(lamella)
        finally:
            self._syncing_selection = False

    def _on_move_to_requested(self, lamella) -> None:
        """Drive the stage to a lamella's *fluorescence* pose.

        Not `stage_position`, which is the milling pose: this tab looks at the sample
        from the fluorescence side, and moving to the milling pose from here would swing
        the stage 180 degrees away from the view you asked to centre.
        """
        if lamella is None:
            return
        pose = lamella.fluorescence_pose
        if pose is None or pose.stage_position is None:
            notification_service.show_toast(
                f"{lamella.name} has no fluorescence pose to move to.", "warning"
            )
            return
        if self.overview is None:
            return
        self.overview.move_to(pose.stage_position)

    def _on_remove_requested(self, lamella) -> None:
        """Remove a lamella from the experiment.

        The row's own button already asked -- `_LamellaRow._on_remove_clicked` puts up
        the confirmation before it emits -- so this does not ask twice.
        """
        experiment = self.experiment
        if experiment is None or lamella is None:
            return
        try:
            experiment.positions.remove(lamella)
        except ValueError:
            logging.debug(f"Cannot remove {lamella.name!r}: not in the experiment.")
            return
        experiment.save()
        # `positions` is an EventedList, so the window's own rebuild has already run off
        # `removed` and re-marked this canvas. Saying it again would be a second full
        # redraw for nothing.
        notification_service.show_toast(f"Removed {lamella.name}.", "info")

    # ── turning a request into a lamella ─────────────────────────────────

    def _objective_position(self) -> Optional[float]:
        """Where the objective is right now, for a pose marked on the overview.

        The focus a user is actually looking through beats the objective's *configured*
        focus position, which `build_lamella_poses` falls back to: that one is a property
        of the instrument and says nothing about this sample. Anyone marking a position
        here has the feature in focus, and that focus is part of what they marked.

        Unless the objective is retracted, in which case its position is a parking spot
        ~10 mm from anything and nobody focused on anything. None hands the decision back
        to the fallback -- which matters because `fluorescence_selected` only asks
        whether an objective position exists, so a parked one would read as a focused
        lamella.
        """
        microscope = self.microscope
        if microscope is None or microscope.fm is None:
            return None
        try:
            objective = microscope.fm.objective
            if objective.state != "Inserted":
                logging.debug(
                    f"Objective is {objective.state}; falling back to its focus position."
                )
                return None
            return objective.position
        except Exception as e:
            logging.debug(f"Could not read the objective position: {e}")
            return None

    def _on_add_requested(self, position) -> None:
        """A user asked for a new lamella at a point on the overview.

        The orientation is *declared* rather than left to be derived. On a compustage
        deriving it would give the same answer; on an offset mount it would not, and the
        wrong answer there is a lamella with a milling pose 48 mm off the beam axis that
        nothing rejects until something tries to mill it (FIB-93). Declaring it turns
        that into a refusal here, where there is a user to tell.
        """
        if self.autolamella_ui is None or self.experiment is None:
            notification_service.show_toast(
                "Load an experiment before marking positions.", "warning"
            )
            return
        try:
            lamella = self.autolamella_ui.add_new_lamella(
                stage_position=position,
                objective_position=self._objective_position(),
                marked_at=FLUORESCENCE_ORIENTATION,
            )
        except Exception as e:
            logging.error(f"Could not add a lamella from the FM overview: {e}")
            notification_service.show_toast(str(e), "error")
            return

        self.refresh_positions()
        self.set_selected(lamella)
        notification_service.show_toast(f"Added {lamella.name}.", "info")

    def _on_move_requested(self, name: str, position) -> None:
        """A user asked to move a marked lamella to a point on the overview.

        Confirmed first, because moving one pose moves both: the milling pose is derived
        from this one, so a lamella dragged across the fluorescence view also stops being
        where the beam was going to mill it. That is usually the point -- the two describe
        one piece of sample -- but it is not visible from this canvas, which shows only
        the fluorescence side, so it gets said rather than assumed.
        """
        experiment = self.experiment
        if experiment is None:
            return
        lamella = next((p for p in experiment.positions if p.name == name), None)
        if lamella is None:
            logging.debug(f"Cannot move {name!r}: no such lamella in the experiment.")
            return
        if lamella.milling_pose is None:
            notification_service.show_toast(
                f"{name} has no milling pose to move.", "warning"
            )
            return

        try:
            poses = build_lamella_poses(
                microscope=self.microscope,
                position=position,
                objective_position=self._objective_position(),
                # Only reaches the result in one case, and it is worth it for that one:
                # a lamella with no fluorescence pose yet gets a whole new one below, and
                # it should be built on what this lamella recorded rather than on
                # whatever the microscope happens to be set to now. Everywhere else only
                # the stage positions are read, and those come from `position`.
                state=lamella.milling_pose,
                marked_at=FLUORESCENCE_ORIENTATION,
            )
        except Exception as e:
            logging.error(f"Could not move {name} from the FM overview: {e}")
            notification_service.show_toast(str(e), "error")
            return

        history = (
            f"\n\n{name} has already completed "
            f"{', '.join(lamella.completed_tasks)}."
            if lamella.completed_tasks
            else ""
        )
        if not message_box_ui(
            title=f"Move {name}?",
            text=(
                f"Move {name} to {poses.fluorescence.stage_position.pretty_string}?"
                f"\n\nThis moves the milling pose with it, to "
                f"{poses.milling.stage_position.pretty_string}."
                f"{history}"
            ),
            parent=self,
        ):
            return

        # Only the stage positions are replaced, so anything else the poses carry --
        # notably the objective position on a lamella that was focused by hand -- is
        # kept. `stage_position` is the milling pose's, via the property.
        lamella.stage_position = poses.milling.stage_position
        lamella.update_milling_angle(self.microscope)
        if lamella.fluorescence_pose is None:
            lamella.fluorescence_pose = poses.fluorescence
        else:
            lamella.fluorescence_pose.stage_position = poses.fluorescence.stage_position

        experiment.save()
        self.refresh_positions()
        self.autolamella_ui.update_ui()
        notification_service.show_toast(f"Moved {name}.", "info")
