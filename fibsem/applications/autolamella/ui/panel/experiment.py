"""The Experiment tab: the lamella list, and the selected lamella's poses and
objective position. Adding, moving and removing lamellae from here."""

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Optional

from PyQt5.QtWidgets import (
    QGridLayout,
    QMessageBox,
    QSizePolicy,
    QSpacerItem,
    QWidget,
)

from fibsem.applications.autolamella.poses import (
    add_lamella,
    remove_lamella,
    set_lamella_pose,
)
from fibsem.applications.autolamella.run_edits import (
    busy_lamellae,
    lamella_edit_refusal,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.ui.lamella.rows import LamellaNameListWidget
from fibsem.applications.autolamella.ui.lamella.selected import (
    SelectedLamellaController,
    SelectedLamellaWidget,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition
from fibsem.ui import notification_service
from fibsem.ui.run_refusal import refused_during_run

if TYPE_CHECKING:
    from fibsem.workflows.session import AppSession

__all__ = ["ExperimentTab"]


class ExperimentTab(QWidget):
    title = "Experiment"

    def __init__(
        self,
        session: "AppSession",
        *,
        movement_widget: Callable[[], Any],
        parent: Optional[QWidget] = None,
    ) -> None:
        """*movement_widget* is the panel's Movement tab, or None without a
        microscope: moves to a lamella go through it."""
        super().__init__(parent)
        self.session = session
        self._movement_widget = movement_widget

        self.grid_layout_experiment = QGridLayout(self)

        self.lamella_list = LamellaNameListWidget()
        self.lamella_list.enable_add_button(True)
        self.lamella_list.enable_defect_button(True)
        self.lamella_list.enable_move_to_action(True)
        self.lamella_list.enable_update_action(True)
        self.lamella_list.enable_remove_button(True)

        self.selected_lamella_widget = SelectedLamellaWidget()

        self.grid_layout_experiment.addWidget(self.lamella_list, 0, 0, 1, 2)
        self.grid_layout_experiment.addWidget(self.selected_lamella_widget, 1, 0, 1, 2)

        self.grid_layout_experiment.addItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), 2, 0, 1, 2
        )

        self.lamella_list.add_requested.connect(
            lambda: self.add_new_lamella(stage_position=None)  # type: ignore
        )
        self.lamella_list.remove_requested.connect(self._on_lamella_remove_requested)
        self.lamella_list.move_to_requested.connect(self._on_lamella_move_to_requested)
        self.lamella_list.update_requested.connect(self._on_lamella_update_requested)
        # `defect_changed` is deliberately not wired here. The window this widget sits
        # in connects the same signal to its own handler, which saves *and* redraws the
        # rows, the cards and the name list -- so a second handler here only bought a
        # second full `experiment.save()` for one toggled icon, which is 0.8 s of frozen
        # GUI at 100 lamella (FIB-682). Every other host of this list keeps its own
        # handler, because nothing else is listening for them (FIB-564).
        self.lamella_list.lamella_selected.connect(self.update_selected)

        self.selected_lamella = SelectedLamellaController(
            self.selected_lamella_widget,
            experiment=lambda: self.experiment,
            microscope=lambda: self.microscope,
            lamella=self.get_selected_lamella,
            move_stage=lambda position: self._movement_widget().move_to_position(
                position
            ),
            busy=lambda: busy_lamellae(self.session.runner, self.experiment),
        )

    @property
    def microscope(self) -> Optional[FibsemMicroscope]:
        return self.session.microscope

    @property
    def experiment(self) -> Optional[Experiment]:
        return self.session.experiment

    @property
    def protocol(self) -> Optional[AutoLamellaTaskProtocol]:
        experiment = self.session.experiment
        return experiment.task_protocol if experiment is not None else None

    def refresh(self) -> None:
        """Enable the list and the selected lamella for the session's state."""
        if self.session.runner.is_running:
            self.refresh_run_edits()
            return

        is_experiment_loaded = bool(self.experiment is not None)
        is_experiment_ready = self.microscope is not None and self.protocol is not None
        has_lamella = bool(self.experiment.positions) if is_experiment_loaded else False

        self.lamella_list.setEnabled(is_experiment_ready)

        # clear the panel when no lamella is selected; populated by update_selected otherwise
        if not has_lamella:
            self.selected_lamella_widget.set_lamella(None)

        self.refresh_run_edits()

        # Current Lamella Status
        if has_lamella and self.experiment is not None:
            self.update_selected()

    def update_items(self, latest: bool = False):
        if self.experiment is None:
            return
        if self.session.runner.is_running:
            return

        # detail lamella list
        preferred = (
            self.experiment.positions[-1].name
            if latest and self.experiment.positions
            else ""
        )
        self.lamella_list.set_lamella(
            self.experiment.positions, preferred_name=preferred
        )

    def update_selected(self, _lamella=None):
        # set the info for the current selected lamella
        if self.experiment is None or self.experiment.positions == []:
            return

        if self.protocol is None:
            return

        idx = self.lamella_list.selected_index
        if idx == -1:
            return

        lamella: Lamella = self.experiment.positions[idx]
        logging.debug(f"Updating Lamella UI for {lamella.status_info}")

        # refresh objective position + pose display for the selected lamella
        self.selected_lamella_widget.set_lamella(lamella)
        self.refresh_run_edits()

    def refresh_run_edits(self) -> None:
        """While a run is on, the selected lamella stays editable, the one the
        run is on included (only its running task's config waits, FIB-1505);
        the stage and objective are the run's."""
        self.selected_lamella_widget.setEnabled(True)
        self.selected_lamella_widget.set_instrument_actions_enabled(
            not self.session.runner.is_running
        )

    def move_to_lamella_position(self):
        """Move the stage to the position of the selected lamella."""
        if self.experiment is None or self.experiment.positions == []:
            return
        if self._movement_widget() is None:
            return

        idx = self.lamella_list.selected_index
        if idx == -1:
            return
        lamella: Lamella = self.experiment.positions[idx]
        stage_position = lamella.milling_pose.stage_position

        # confirmation dialog
        ret = QMessageBox.question(
            self,
            "Move to Lamella Position",
            f"Move to position of Lamella {lamella.name}?\n{stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        logging.info(f"Moving to position of {lamella.name}.")
        self._movement_widget().move_to_position(stage_position)

    def add_new_lamella(
        self,
        stage_position: Optional[FibsemStagePosition] = None,
        name: Optional[str] = None,
        objective_position: Optional[float] = None,
        marked_at: Optional[str] = None,
        grid_id: Optional[str] = None,
    ) -> Optional[Lamella]:
        """Add a lamella to the experiment.

        Args:
            stage_position: Where the lamella is, in any orientation -- which one is
                read off the position itself, so a position picked on the fluorescence
                side is taken as the fluorescence pose rather than as somewhere to mill.
                If None, the current stage position is used.
            name: The name of the lamella. If None, a default name will be generated.
            objective_position: The objective position of the lamella. If None, the 'focused' objective position is used.
            marked_at: The orientation *stage_position* is in, for a caller that knows.
                Left alone it is read off the position, which is right on a compustage
                and cannot be on an offset mount -- see `build_lamella_poses`.
            grid_id: The grid this lamella is on, for a caller that knows -- one
                marked on a grid's overview belongs to that grid whether or not it
                is on the stage. Left alone it is resolved from the stage.
        Returns:
            lamella: The created lamella; None when a run refuses reading the stage.
        """
        if stage_position is None and refused_during_run(
            self.session.runner, "add an item where the stage is"
        ):
            return None
        if self.experiment is None:
            raise ValueError("No experiment loaded. Please load an experiment first.")
        if self.protocol is None:
            raise ValueError("No protocol loaded. Please load a protocol first.")
        if self.microscope is None:
            raise ValueError(
                "No microscope connected. Please connect a microscope first."
            )

        lamella = add_lamella(
            self.experiment,
            self.microscope,
            stage_position=stage_position,
            name=name,
            objective_position=objective_position,
            marked_at=marked_at,
            grid_id=grid_id,
        )
        self.update_items(latest=True)
        self.refresh()

        return lamella

    def _on_lamella_move_to_requested(self, lamella):
        """Handle move-to request from the list row's actions menu."""
        self.lamella_list.select(lamella.name)
        self.move_to_lamella_position()

    def _on_lamella_update_requested(self, lamella):
        """Handle update-position request from the list row's actions menu."""
        self.lamella_list.select(lamella.name)
        self.update_lamella_position_ui()

    def _on_lamella_remove_requested(self, lamella):
        """Handle removal of a lamella via the list row's remove button.

        Confirmation is already handled by the row widget.
        """
        if self.experiment is None:
            return
        why = lamella_edit_refusal(
            self.session.runner, self.experiment, lamella, "remove it"
        )
        if why:
            # The row has gone already: put it back.
            self.lamella_list.set_lamella(
                self.experiment.positions, preferred_name=lamella.name
            )
            notification_service.show_toast(why, "warning")
            return
        if not remove_lamella(self.experiment, lamella):
            return
        logging.debug("Lamella removed from experiment")
        self.update_items(latest=True)
        self.refresh()

    def update_lamella_position_ui(self):
        """Update the stage position of the selected lamella to the current stage position."""

        if self.microscope is None:
            return
        if self.protocol is None:
            return
        if self.experiment is None or self.experiment.positions == []:
            return
        if refused_during_run(self.session.runner, "set a pose from the stage"):
            return

        # toggle between saving position and marking as ready
        idx = self.lamella_list.selected_index
        if idx == -1:
            logging.warning("No lamella is selected, cannot save.")
            return

        lamella: Lamella = self.experiment.positions[idx]
        current_position = self.microscope.get_stage_position()

        # message box to confirm
        ret = QMessageBox.question(
            self,
            "Save Position Confirmation",
            f"Save new position for Lamella {lamella.name} position?\n\n"
            f"New Stage Position: {current_position.pretty}\n"
            f"Existing Stage Position: {lamella.stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        # As Set pose does: the stored objective position is kept, and the
        # milling angle and fluorescence pose follow the new milling pose.
        set_lamella_pose(
            self.experiment,
            self.microscope,
            lamella,
            "MILLING",
            deepcopy(self.microscope.get_microscope_state()),
        )

        self.update_items()
        self.refresh()

    def get_selected_lamella(self) -> Optional[Lamella]:
        """Get the currently selected lamella from the combobox.

        Returns:
            The selected lamella, or None if no experiment, no positions, or invalid selection.
        """
        if self.experiment is None:
            return None

        if not self.experiment.positions:
            return None

        idx = self.lamella_list.selected_index
        if idx == -1 or idx >= len(self.experiment.positions):
            return None

        return self.experiment.positions[idx]
