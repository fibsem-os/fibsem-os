"""The microscope's control tabs, and the right-hand panel of an app's window
built on them.

`MicroscopeControls` holds the microscope's tabs (Image, Movement, Milling,
Sample, Manipulator, Fluorescence, Detection, Spot Burn), built when a
microscope connects and dropped when it goes, by `show_microscope`. Every
window uses it: fibsemOS directly, the apps through `MicroscopePanel`, which
adds the app's experiment tab, the workflow line and the prompt bar, and
redraws itself when the session changes and when the open experiment's
positions do.
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Tuple

from psygnal import EmissionInfo
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGridLayout,
    QLabel,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem.microscope import FibsemMicroscope
from fibsem.structures.settings import FibsemStagePosition
from fibsem.structures.system import MicroscopeSettings
from fibsem.theme.tokens import GRAY_ICON_COLOR
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemSampleWidget import FibsemSampleWidget
from fibsem.ui.FibsemSpotBurnWidget import FibsemSpotBurnWidget
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.item_selection import ItemSelection
from fibsem.ui.qt.connections import connect_on_gui_thread
from fibsem.ui.widgets.custom_widgets import scrollable
from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget
from fibsem.ui.widgets.workflow.prompt_bar import PromptBar
from fibsem.workflows.session import EXPERIMENT, MICROSCOPE

try:  # ml dependencies are optional
    from fibsem.ui.FibsemEmbeddedDetectionWidget import (
        FibsemEmbeddedDetectionUI as FibsemEmbeddedDetectionWidget,
    )

    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False

if TYPE_CHECKING:
    from fibsem.workflows.session import AppSession
    from fibsem.workflows.status import WorkflowStatusEvent

__all__ = ["ItemTab", "MicroscopeControls", "MicroscopePanel", "add_microscope_tab"]


class ItemTab(Protocol):
    """The panel's first tab, the app's: its experiment's items."""

    # the tab's name
    title: str
    # the item selection the Image and Fluorescence tabs follow; None for none
    item_selection: Optional[ItemSelection]

    def refresh(self) -> None: ...

    def update_items(self, latest: bool = False) -> None:
        """The items changed; *latest*: select the one just added."""

    def update_selected(self) -> None:
        """The selected item changed."""


class MicroscopeControls(QWidget):
    """The microscope's control tabs, in `tabWidget`. A host puts its own tabs
    beside them (the Connection tab, an app's experiment tab)."""

    def __init__(
        self,
        *,
        view_controller: Callable[[], Any],
        overview_acquiring: Callable[[], bool] = lambda: False,
        parent: Optional[QWidget] = None,
    ) -> None:
        """*view_controller* is the window's quad view, once it has one;
        *overview_acquiring* says whether an Overview tab is mid-tileset."""
        super().__init__(parent)
        self._view_controller = view_controller
        self._overview_acquiring = overview_acquiring
        # the microscope whose moves the stage readout follows
        self._stage_followed: Optional[FibsemMicroscope] = None

        self.image_widget: Optional[FibsemImageSettingsWidget] = None
        self.movement_widget: Optional[FibsemMovementWidget] = None
        self.spot_burn_widget: Optional[FibsemSpotBurnWidget] = None
        self.fm_control_widget: Optional[FMControlWidget] = None
        self.sample_widget: Optional[FibsemSampleWidget] = None
        self.manipulator_widget: Optional[FibsemManipulatorWidget] = None
        self.milling_task_config_widget: Optional[MillingTaskViewerWidget] = None
        self.milling_tab: Optional[QWidget] = None  # the scroll area around it
        self.det_widget: Optional["FibsemEmbeddedDetectionWidget"] = None

        self.gridLayout = QGridLayout(self)
        self.tabWidget = QTabWidget(self)
        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 2)

    # What an app's widgets add to the tabs; none without an app.

    def _images(self) -> Optional[Any]:
        """Where the latest image of each beam is kept."""
        return None

    def _item_selection(self) -> Optional[ItemSelection]:
        """The item selection the Image and Fluorescence tabs follow."""
        return None

    def _experiment(self) -> Optional[Any]:
        """The open experiment, where images can be saved."""
        return None

    def _revise_prompt(self, msg: str, can_continue: bool) -> None:
        """Say *msg* on the question a run is asking."""

    def handle_acquisition_update(self, ddict: dict) -> None:
        """An acquisition reported progress."""

    @property
    def _overview_is_acquiring(self) -> bool:
        """Whether either overview modality is mid-tileset.

        Asked of the Overview tab rather than of a widget held here: the tab owns both
        modalities and rebuilds its widgets on every reconnection, so anything holding
        one of them directly would be answering for a widget that had been replaced.
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
        experiment open still moves the stage."""
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
                images=self._images(),
                view_controller=self._view_controller,
                toggle_movement=lambda enable: (
                    self.movement_widget._toggle_interactions(enable, caller="ui")
                ),
                item_selection=self._item_selection(),
                experiment=self._experiment,
                revise_prompt=self._revise_prompt,
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
            # The tab scrolls the milling widget. The tab is
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

            if microscope.is_available("manipulator"):
                self.manipulator_widget = FibsemManipulatorWidget(
                    microscope=microscope,
                    settings=settings,
                    image_widget=self.image_widget,
                    parent=self,
                )
                self.tabWidget.addTab(self.manipulator_widget, "Manipulator")

            if microscope.fm is not None:
                self.fm_control_widget = FMControlWidget(
                    microscope=microscope,
                    parent=self,
                    view_controller=self._view_controller,
                    item_selection=self._item_selection(),
                    experiment=self._experiment,
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
                index = self.tabWidget.indexOf(self.sample_widget)
                self.tabWidget.setTabVisible(
                    self.tabWidget.insertTab(index, self.det_widget, "Detection"), False
                )

            # spot burn widget (optional)
            self.spot_burn_widget = FibsemSpotBurnWidget(
                microscope=microscope,
                image_widget=self.image_widget,
                view_controller=self._view_controller,
                parent=self,
            )
            index = self.tabWidget.indexOf(self.sample_widget)
            self.tabWidget.setTabVisible(
                self.tabWidget.insertTab(index, self.spot_burn_widget, "Spot Burn"), False
            )

            self.image_widget.acquisition_progress_signal.connect(
                self.handle_acquisition_update
            )
        else:
            if self.image_widget is None:
                return

            # remove tabs
            if self.manipulator_widget is not None:
                self.tabWidget.removeTab(
                    self.tabWidget.indexOf(self.manipulator_widget)
                )
                self.manipulator_widget.deleteLater()
                self.manipulator_widget = None
            if self.sample_widget is not None:
                self.sample_widget._teardown_connections()
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


class MicroscopePanel(MicroscopeControls):
    def __init__(
        self,
        session: "AppSession",
        *,
        experiment_tab: Callable[["MicroscopePanel"], ItemTab],
        view_controller: Callable[[], Any],
        overview_acquiring: Callable[[], bool],
        parent: Optional[QWidget] = None,
    ) -> None:
        """*experiment_tab* builds the app's tab for this panel;
        *view_controller* is the window's quad view, once it has one;
        *overview_acquiring* says whether the Overview tab is mid-tileset."""
        super().__init__(
            view_controller=view_controller,
            overview_acquiring=overview_acquiring,
            parent=parent,
        )
        self.session = session
        # the experiment whose position events the panel follows
        self._experiment_followed: Optional[Any] = None

        self.experiment_tab = experiment_tab(self)
        self.tabWidget.addTab(self.experiment_tab, self.experiment_tab.title)
        self.label_workflow_information = QLabel("Workflow Information")
        self.prompt_bar = PromptBar()

        self.gridLayout.addWidget(self.label_workflow_information, 2, 0, 1, 2)
        self.gridLayout.addWidget(self.prompt_bar, 3, 0, 1, 2)
        self.tabWidget.setCurrentIndex(0)

        self.set_current_workflow_message(msg=None, show=False)
        self.label_workflow_information.setWordWrap(True)

        session.runner.starting.connect(self._on_run_starting)
        connect_on_gui_thread(self, session.changed, self._on_session_changed)

    def _images(self) -> Optional[Any]:
        return self.session.images

    def _item_selection(self) -> Optional[ItemSelection]:
        return self.experiment_tab.item_selection

    def _experiment(self) -> Optional[Any]:
        return self.session.experiment

    def _revise_prompt(self, msg: str, can_continue: bool) -> None:
        self.prompt_bar.revise(msg, can_continue=can_continue)

    def refresh(self) -> None:
        """Redraw for the current state of the session."""
        self.experiment_tab.refresh()
        if self.session.runner.is_running:
            return

        # force order: connect -> experiment -> protocol
        self.refresh_tabs()

        # Nothing to say while idle. The same guidance is in the window's status
        # bar, which is where it belongs -- shown in both places it read as two
        # different messages that happened to agree. This label is for the question
        # a running workflow is asking, which the Yes/No buttons below it answer.
        self.prompt_bar.clear()

    def _on_session_changed(self, what: str) -> None:
        """Another microscope or experiment: follow it, redraw and show it. An
        edit to the open one from elsewhere: redraw."""
        experiment = self.session.experiment
        self._follow_experiment(experiment)
        if experiment is not None:
            self.experiment_tab.update_items()
        self.refresh()
        if experiment is not None and what in (MICROSCOPE, EXPERIMENT):
            self.show_experiment_tab()

    def _follow_experiment(self, experiment: Optional[Any]) -> None:
        """Redraw on ``experiment``'s position events, and stop for the last's.
        An experiment without them redraws on the session's changes only."""
        followed = self._experiment_followed
        if followed is experiment:
            return
        if followed is not None:
            try:
                followed.events.disconnect(self._on_experiment_event)  # type: ignore
            except Exception as e:
                logging.debug(f"Could not disconnect experiment events: {e}")
        if getattr(experiment, "events", None) is None:
            experiment = None
        if experiment is not None:
            experiment.events.connect(self._on_experiment_event)  # type: ignore
        self._experiment_followed = experiment

    @ensure_main_thread
    def _on_experiment_event(self, evt: EmissionInfo) -> None:
        """A position added (and selected), removed or changed (the minimap, a
        task, an edit)."""
        if evt.signal.name not in ["inserted", "removed", "changed"]:
            return
        self.experiment_tab.update_items(latest=evt.signal.name == "inserted")
        self.refresh()

    def show_experiment_tab(self) -> None:
        self.tabWidget.setCurrentWidget(self.experiment_tab)

    def refresh_tabs(self) -> None:
        """The experiment tab shows once a microscope is connected; Detection
        stays hidden until a proposal asks for it."""
        self.tabWidget.setTabVisible(
            self.tabWidget.indexOf(self.experiment_tab),
            self.session.microscope is not None,
        )
        if self.det_widget is not None:
            idx = self.tabWidget.indexOf(self.det_widget)
            self.tabWidget.setTabVisible(idx, False)  # hide detection tab for now

    def reset_after_run(self, image_path: str) -> None:
        """Put the tabs back as they were before the run: the experiment tab
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

    def set_current_workflow_message(
        self, msg: Optional[str] = None, show: bool = True
    ):
        """Set the current workflow information message"""
        if msg is not None:
            self.label_workflow_information.setText(msg)
        self.label_workflow_information.setVisible(show)

    def handle_acquisition_update(self, ddict: dict) -> None:
        if ddict.get("finished", False):
            self.experiment_tab.update_selected()

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


def add_microscope_tab(
    tab_widget: QTabWidget,
    view_controller: Any,
    controls: QWidget,
    sizes: Tuple[int, int] = (700, 550),
    controls_min_width: int = 500,
) -> QWidget:
    """Add the window's Microscope tab, the quad view beside *controls*, to
    *tab_widget*, and return it."""
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    splitter = QSplitter(Qt.Horizontal)
    splitter.setChildrenCollapsible(False)
    splitter.addWidget(view_controller.widget)
    splitter.addWidget(controls)
    splitter.setSizes(list(sizes))
    controls.setMinimumWidth(controls_min_width)
    layout.addWidget(splitter)
    tab_widget.addTab(
        container,
        fibsem_icon("mdi:microscope", color=GRAY_ICON_COLOR),
        "Microscope",
    )
    return container
