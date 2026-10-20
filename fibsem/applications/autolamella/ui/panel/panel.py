"""The right-hand panel of the AutoLamella window: the Experiment tab, the
microscope's tabs, the workflow line and the prompt bar.

The microscope's tabs (Image, Movement, Milling, Sample, Fluorescence,
Detection, Spot Burn) are built when a microscope connects and dropped when it
goes, by `show_microscope`.
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from PyQt5.QtWidgets import QGridLayout, QLabel, QTabWidget, QWidget
from superqt import ensure_main_thread

from fibsem.applications.autolamella.ui.panel.experiment import ExperimentTab
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition, MicroscopeSettings
from fibsem.ui import (
    DETECTION_AVAILABLE,
    FibsemImageSettingsWidget,
    FibsemMovementWidget,
    FibsemSpotBurnWidget,
    MillingTaskViewerWidget,
    notification_service,
)
from fibsem.ui.FibsemSampleWidget import FibsemSampleWidget
from fibsem.ui.widgets.custom_widgets import scrollable
from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget
from fibsem.ui.widgets.workflow.prompt_bar import PromptBar

if DETECTION_AVAILABLE:  # ml dependencies are optional
    from fibsem.ui.FibsemEmbeddedDetectionWidget import (
        FibsemEmbeddedDetectionUI as FibsemEmbeddedDetectionWidget,
    )

if TYPE_CHECKING:
    from fibsem.workflows.session import AppSession
    from fibsem.workflows.status import WorkflowStatusEvent

__all__ = ["MicroscopePanel"]


class MicroscopePanel(QWidget):
    def __init__(
        self,
        session: "AppSession",
        *,
        view_controller: Callable[[], Any],
        overview_acquiring: Callable[[], bool],
        parent: Optional[QWidget] = None,
    ) -> None:
        """*view_controller* is the window's quad view, once it has one;
        *overview_acquiring* says whether the Overview tab is mid-tileset."""
        super().__init__(parent)
        self.session = session
        self._view_controller = view_controller
        self._overview_acquiring = overview_acquiring
        # the microscope whose moves the stage readout follows
        self._stage_followed: Optional[FibsemMicroscope] = None

        self.image_widget: Optional[FibsemImageSettingsWidget] = None
        self.movement_widget: Optional[FibsemMovementWidget] = None
        self.spot_burn_widget: Optional[FibsemSpotBurnWidget] = None
        self.fm_control_widget: Optional[FMControlWidget] = None
        self.sample_widget: Optional[FibsemSampleWidget] = None
        self.milling_task_config_widget: Optional[MillingTaskViewerWidget] = None
        self.milling_tab: Optional[QWidget] = None  # the scroll area around it
        self.det_widget: Optional["FibsemEmbeddedDetectionWidget"] = None

        self.gridLayout = QGridLayout(self)
        self.tabWidget = QTabWidget(self)
        self.experiment_tab = ExperimentTab(
            session, movement_widget=lambda: self.movement_widget
        )
        self.tabWidget.addTab(self.experiment_tab, "Experiment")
        self.label_workflow_information = QLabel("Workflow Information")
        self.prompt_bar = PromptBar()

        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 2)
        self.gridLayout.addWidget(self.label_workflow_information, 2, 0, 1, 2)
        self.gridLayout.addWidget(self.prompt_bar, 3, 0, 1, 2)
        self.tabWidget.setCurrentIndex(0)

        self.set_current_workflow_message(msg=None, show=False)
        self.label_workflow_information.setWordWrap(True)

        session.runner.starting.connect(self._on_run_starting)

    def show_experiment_tab(self) -> None:
        self.tabWidget.setCurrentWidget(self.experiment_tab)

    def refresh_tabs(self) -> None:
        """The Experiment tab shows once a microscope is connected; Detection
        stays hidden until a proposal asks for it."""
        self.tabWidget.setTabVisible(
            self.tabWidget.indexOf(self.experiment_tab),
            self.session.microscope is not None,
        )
        if self.det_widget is not None:
            idx = self.tabWidget.indexOf(self.det_widget)
            self.tabWidget.setTabVisible(idx, False)  # hide detection tab for now

    def reset_after_run(self, image_path: str) -> None:
        """Put the tabs back as they were before the run: the Experiment tab
        in front, the editors cleared and their own buttons back."""
        self.show_experiment_tab()

        # clear milling task config
        if self.milling_task_config_widget is not None:
            self.milling_task_config_widget.clear()
            self.milling_task_config_widget.milling_widget.pushButton_run_milling.setVisible(
                True
            )

        # restore the spot burn widget: an aborted workflow may skip the ClearSpotBurn
        # that normally resets it, which would leave the Burn button hidden
        # (no-op after a normal completion, where ClearSpotBurn already ran)
        if self.spot_burn_widget is not None:
            self.spot_burn_widget.set_workflow_mode(False)
            self.spot_burn_widget.clear_points_layer()

        # clear detection layers
        if self.det_widget is not None:
            self.det_widget.clear_layers()

        # clear the image settings save settings etc
        if self.image_widget is not None:
            self.image_widget.checkBox_image_save_image.setChecked(False)
            self.image_widget.lineEdit_image_path.setText(image_path)
            self.image_widget.lineEdit_image_label.setText("default-image")

    @property
    def _overview_is_acquiring(self) -> bool:
        """Whether either overview modality is mid-tileset.

        Asked of the Overview tab rather than of a widget held here: the tab owns both
        modalities and rebuilds its widgets on every reconnection, so anything holding
        one of them directly would be answering for a widget that had been replaced.

        This used to ask the napari minimap, which means it stopped being true the
        moment the Overview tab took over acquisition -- a run from there did not
        suppress anything.
        """
        return self._overview_acquiring()

    @ensure_main_thread
    def _on_stage_position_updated(self, stage_position: FibsemStagePosition) -> None:
        """Callback for when the stage position is updated."""
        if self._overview_is_acquiring:
            # Mid-tileset the stage moves once per tile, and refreshing the readout on
            # each is churn nobody reads.
            return
        if self.movement_widget is not None:
            # pass the position from the signal; re-querying the microscope here races
            # with the worker thread driving the move (see TescanMicroscope socket lock)
            self.movement_widget.update_ui(stage_position=stage_position)

    def _follow_stage(self, microscope: Optional[FibsemMicroscope]) -> None:
        """Keep the stage readout on ``microscope``'s moves, and off any other's.

        The readout follows the microscope, not the experiment: a move with no
        experiment open still moves the stage (FIB-1421)."""
        followed = self._stage_followed
        if followed is microscope:
            return
        if followed is not None and followed.stage is not None:
            try:
                followed.stage.position.changed.disconnect(
                    self._on_stage_position_updated
                )
            except Exception as e:
                logging.debug(f"Could not disconnect microscope stage events: {e}")
        if microscope is not None and microscope.stage is not None:
            microscope.stage.position.changed.connect(self._on_stage_position_updated)
        self._stage_followed = microscope

    def front_tab(self, widget: QWidget) -> None:
        """Bring forward the tab that holds *widget*.

        The widget may be the tab itself or sit inside a wrapper (the Milling tab is
        a scroll area around its editor). ``setCurrentWidget`` on a widget that is
        not a direct tab is a silent no-op, which is how the responder stopped
        fronting the Milling tab once the wrapper arrived.
        """
        candidate = widget
        while candidate is not None and self.tabWidget.indexOf(candidate) == -1:
            candidate = candidate.parentWidget()
        if candidate is not None:
            # A hidden tab cannot be fronted: Detection is hidden until it is
            # asked for, so bringing it forward has to un-hide it first or the
            # call is a second silent no-op.
            self.tabWidget.setTabVisible(self.tabWidget.indexOf(candidate), True)
            self.tabWidget.setCurrentWidget(candidate)

    def show_microscope(
        self,
        microscope: Optional[FibsemMicroscope],
        settings: Optional[MicroscopeSettings],
    ) -> None:
        """Build the microscope's tabs, or drop them for None."""
        self._follow_stage(microscope)

        if microscope is not None:
            # reusable components
            self.image_widget = FibsemImageSettingsWidget(
                microscope=microscope,
                image_settings=settings.image,  # type: ignore
                parent=self,
                images=self.session.images,
                view_controller=self._view_controller,
                toggle_movement=lambda enable: (
                    self.movement_widget._toggle_interactions(enable, caller="ui")
                ),
                lamella_list=self.experiment_tab.lamella_list,
                experiment=lambda: self.session.experiment,
                revise_prompt=lambda msg, ok: self.prompt_bar.revise(
                    msg, can_continue=ok
                ),
            )
            self.movement_widget = FibsemMovementWidget(
                microscope=microscope,
                image_widget=self.image_widget,
                view_controller=self._view_controller,
                milling_widget=lambda: self.milling_task_config_widget,
                parent=self,
            )

            # add widgets to tabs
            self.tabWidget.addTab(self.image_widget, "Image")
            self.tabWidget.addTab(self.movement_widget, "Movement")
            self.milling_task_config_widget = MillingTaskViewerWidget(
                microscope=microscope,
                image_widget=self.image_widget,
                parent=self,
            )
            # The milling widget no longer scrolls itself; its tab does. The tab is
            # kept by name because indexOf / setCurrentWidget want the tab, not the
            # widget inside it -- see front_tab.
            self.milling_tab = scrollable(self.milling_task_config_widget)
            self.tabWidget.addTab(self.milling_tab, "Milling")

            # The hardware view of the grids: the holder, and the magazine when
            # there is one. Slot moves go through the Movement widget, the same
            # route as a saved position, so the readout and post-move images follow.
            self.sample_widget = FibsemSampleWidget(microscope=microscope, parent=self)
            self.sample_widget.move_to_requested.connect(
                self.movement_widget.move_to_position
            )
            self.tabWidget.addTab(self.sample_widget, "Sample")

            if microscope.fm is not None:
                self.fm_control_widget = FMControlWidget(
                    microscope=microscope,
                    parent=self,
                    view_controller=self._view_controller,
                    lamella_list=self.experiment_tab.lamella_list,
                    experiment=lambda: self.session.experiment,
                    set_host_enabled=self.setEnabled,
                )
                if settings is not None and settings.fm is not None:
                    self.fm_control_widget._apply_fluorescence_configuration(
                        settings.fm
                    )
                self.tabWidget.addTab(self.fm_control_widget, "Fluorescence")

            # add the detection widget if ml dependencies are available
            if DETECTION_AVAILABLE:
                self.det_widget = FibsemEmbeddedDetectionWidget(
                    parent=self, view_controller=self._view_controller
                )
                self.tabWidget.addTab(self.det_widget, "Detection")
                self.tabWidget.setTabVisible(
                    self.tabWidget.indexOf(self.det_widget), False
                )

            # spot burn widget (optional)
            self.spot_burn_widget = FibsemSpotBurnWidget(
                microscope=microscope,
                image_widget=self.image_widget,
                view_controller=self._view_controller,
                parent=self,
            )
            self.tabWidget.addTab(self.spot_burn_widget, "Spot Burn")
            self.tabWidget.setTabVisible(
                self.tabWidget.indexOf(self.spot_burn_widget), False
            )

            self.image_widget.acquisition_progress_signal.connect(
                self.handle_acquisition_update
            )
        else:
            if self.image_widget is None:
                return

            # remove tabs
            if self.sample_widget is not None:
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.sample_widget))
                self.sample_widget.deleteLater()
                self.sample_widget = None
            if self.fm_control_widget is not None:
                # deleteLater fires neither closeEvent nor close_widget, so tear
                # down the FM widget's external signal connections explicitly
                self.fm_control_widget._teardown_connections()
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.fm_control_widget))
                self.fm_control_widget.deleteLater()
                self.fm_control_widget = None
            if self.det_widget is not None:
                self.det_widget._teardown_connections()
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.det_widget))
                self.det_widget.deleteLater()
                self.det_widget = None
            if self.spot_burn_widget is not None:
                self.spot_burn_widget.disconnect_signals()
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.spot_burn_widget))
                self.spot_burn_widget.deleteLater()
                self.spot_burn_widget = None
            if self.milling_task_config_widget is not None:
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.milling_tab))
                self.milling_tab.deleteLater()  # owns the widget
                self.milling_tab = None
                self.milling_task_config_widget = None
            if self.movement_widget is not None:
                self.movement_widget._teardown_connections()
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.movement_widget))
                self.movement_widget.deleteLater()
                self.movement_widget = None
            if self.image_widget is not None:
                self.image_widget._teardown_connections()
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.image_widget))
                self.image_widget.acquisition_progress_signal.disconnect(
                    self.handle_acquisition_update
                )
                self.image_widget.deleteLater()
                self.image_widget = None

    def import_fm_configuration(self) -> None:
        """Load a fluorescence microscope configuration via the control widget."""
        if self.fm_control_widget is None:
            msg = "Fluorescence control not available. Connect to an FM-enabled microscope first."
            logging.warning(msg)
            notification_service.show_toast(msg, "warning")
            return

        try:
            self.fm_control_widget.import_fm_configuration()
        except Exception:
            logging.exception("Failed to load FM configuration from AutoLamella UI.")
            notification_service.show_toast(
                "Failed to load FM configuration. Check logs for details.", "error"
            )

    def export_fm_configuration(self) -> None:
        """Save the current fluorescence microscope configuration via the control widget."""
        if self.fm_control_widget is None:
            msg = "Fluorescence control not available. Connect to an FM-enabled microscope first."
            logging.warning(msg)
            notification_service.show_toast(msg, "warning")
            return

        try:
            self.fm_control_widget.export_fm_configuration()
        except Exception:
            logging.exception("Failed to save FM configuration from AutoLamella UI.")
            notification_service.show_toast(
                "Failed to save FM configuration. Check logs for details.", "error"
            )

    def set_spot_burn_widget_active(self, active: bool = True) -> None:
        """Set the spot burn widget active (sets the tab visible, activate point layer)."""
        if self.spot_burn_widget is None:
            return

        idx = self.tabWidget.indexOf(self.spot_burn_widget)
        self.tabWidget.setTabVisible(idx, active)
        if active:
            self.tabWidget.setCurrentIndex(idx)
            self.spot_burn_widget.set_active()
        else:
            self.spot_burn_widget.set_inactive()

    def set_current_workflow_message(
        self, msg: Optional[str] = None, show: bool = True
    ):
        """Set the current workflow information message"""
        if msg is not None:
            self.label_workflow_information.setText(msg)
        self.label_workflow_information.setVisible(show)

    def handle_acquisition_update(self, ddict: dict) -> None:
        if ddict.get("finished", False):
            self.experiment_tab.update_lamella_ui()

    def handle_workflow_status(self, event: "WorkflowStatusEvent") -> None:
        """Show a fire-and-forget status update. GUI thread, via the run relay.

        Two deliberate absences: no widget-existence guards (these two labels
        exist from construction, so there is nothing to raise about in a queued
        slot), and no touching of the waiting display state — a status update on
        its own channel can never release a blocked waiter, which is the point
        of the channel. A ``message`` of None says nothing about the prompt and
        leaves it standing — the responder pings this signal for chrome
        refreshes while its question is up.
        """
        if event.message is not None:
            self.prompt_bar.ask(event.message)
        self.set_current_workflow_message(event.workflow_info)

    def _on_run_starting(self, request) -> None:
        """Where the run was started, before its thread starts."""
        if self.milling_task_config_widget is not None:
            self.milling_task_config_widget.clear()
