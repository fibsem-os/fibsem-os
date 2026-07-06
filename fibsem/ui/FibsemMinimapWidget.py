import logging
import os
import sys
import threading
import time
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from psygnal import EmissionInfo
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem import constants, conversions
from fibsem.ui import notification_service
from fibsem.applications.autolamella.protocol.constants import (
    FIDUCIAL_KEY,
    MICROEXPANSION_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    TRENCH_KEY,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    DefectType,
    Lamella,
)
from fibsem.imaging import tiled
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
    OverviewAcquisitionSettings,
    Point,
)
from fibsem.ui import FibsemMovementWidget, stylesheets
from fibsem.ui import utils as ui_utils
from fibsem.ui.widgets.custom_widgets import ContextMenu, ContextMenuConfig, LamellaNameListWidget, TitledPanel
from fibsem.ui.widgets.canvas.fm_composite import FMLayer, composite_fm_layers
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import MinimapShapesOverlay, ShapeSpec
from fibsem.ui.widgets.overview_acquisition_settings_widget import (
    OverviewAcquisitionSettingsWidget,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI


OVERVIEW_IMAGE_PARAMETERS = {
    "nrows": 3,
    "ncols": 3,
    "fov": 500, # um
    "dwell_time": 1.0, # us
    "autocontrast": True,
    "autogamma": False,
}

# Crosshair layer configuration constants
CROSSHAIR_CONFIG = {
    "layer_name": "stage-position",
    "crosshair_size": 50,
    "text_properties": {
        "color": "white",
        "font_size": 50,
        "anchor": "lower_left",
        "translation": (5, 55),
    },
    "line_style": {
        "edge_width": 5,
        "face_color": "transparent",
    },
    "colors": {
        "origin": "red",
        "current": "yellow",
        "saved_selected": "lime",
        "saved_unselected": "cyan",
        "grid": "red",
    },
}

OVERLAY_CONFIG = {
    "layer_name": "overlay-shapes",
    "text_properties": {
        "color": "white",
        "font_size": 50,
        "anchor": "upper_left",
        "translation": (-5, 5),
    },
    "rectangle_style": {
        "edge_width": 5,
        "face_color": "transparent",
        "opacity": 0.7,
    },
    "circle_style": {
        "edge_width": 40,
        "face_color": "transparent", 
        "opacity": 0.7,
    },
}

LABEL_INSTRUCTIONS = {
    "image-available": "Instructions: \nRight Click to Add/Move a Lamella Position or Double Click to Move the Stage...",
    "no-image": "Please take or load an overview image..."
}

# Floating "Display" popover on the canvas toolbar (napari-dark, like the layer controls).
_DISPLAY_PANEL_QSS = """
QFrame#minimapDisplayPanel {
    background: #262930;
    border: 1px solid #3a3f47;
    border-radius: 8px;
}
QFrame#minimapDisplayPanel QLabel#panelTitle {
    color: #9aa0a6;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
}
QFrame#minimapDisplayPanel QCheckBox {
    color: #d1d2d4;
    font-size: 12px;
    padding: 2px 0;
}
"""
DEFAULT_OVERVIEW_ACQUISITION_SETTINGS = OverviewAcquisitionSettings(
    image_settings=ImageSettings(
        hfw=OVERVIEW_IMAGE_PARAMETERS["fov"] * constants.MICRO_TO_SI,
        dwell_time=OVERVIEW_IMAGE_PARAMETERS["dwell_time"] * constants.MICRO_TO_SI,
        autocontrast=OVERVIEW_IMAGE_PARAMETERS["autocontrast"],
        autogamma=OVERVIEW_IMAGE_PARAMETERS["autogamma"],
        beam_type=BeamType.ELECTRON,
        save=True,
        path=None,  # will be set to experiment path when overview acquisition widget is initialized
        filename="overview-image",
    ),
    nrows=OVERVIEW_IMAGE_PARAMETERS["nrows"],
    ncols=OVERVIEW_IMAGE_PARAMETERS["ncols"],
)


def generate_gridbar_image(shape: Tuple[int, int], pixelsize: float, spacing: float, width: float) -> FibsemImage:
    """Generate an synthetic image of cryo gridbars."""
    arr = np.zeros(shape=shape, dtype=np.uint8)

    # create grid, grid bars thickness = 10px
    thickness_px = int(width / pixelsize)
    spacing_px = int(spacing / pixelsize)
    for i in range(0, arr.shape[0], spacing_px):
        arr[i:i+thickness_px, :] = 255
        arr[:, i:i+thickness_px] = 255

    # TODO: add metadata
    return FibsemImage(data=arr)

# TODO: migrate to properly scaled infinite canvas
# TODO: allow acquiring multiple overview images
# TODO: deprecate the need for the movement_widget widgets...
# TODO: update layer name for correlation layers, set from file?
# TODO: set combobox to all images in viewer 
class FibsemMinimapWidget(QWidget):
    _acquisition_finished = pyqtSignal(dict)

    def __init__(
        self,
        parent: 'AutoLamellaUI',
    ):
        super().__init__(parent=parent) # type: ignore
        self._setup_ui()

        self.parent_widget = parent

        # Image display model — the overview + correlation images are composited
        # into one RGB frame (FMLayer + composite_fm_layers) and shown on the
        # matplotlib FibsemImageCanvas built in _setup_ui. Replaces the napari
        # viewer + its image/shape/text layers.
        self.image: Optional[FibsemImage] = None
        self._overview_layer: Optional[FMLayer] = None
        self._correlation_layers: List[FMLayer] = []  # M5: correlation + gridbar
        self._canvas_shape: Optional[Tuple[int, int]] = None
        self.correlation_image_layers: List[str] = []

        self.correlation_mode_enabled: bool = False

        self._thread_stop_event = threading.Event()
        self._acquisition_worker: Optional[threading.Thread] = None

        # display options
        self.show_current_fov: bool = True
        self.show_overview_fov: bool = True
        self.show_saved_positions_fov: bool = True
        self.show_stage_limits: bool = True
        self.show_circle_overlays: bool = True
        self.show_tem_stage_limits: bool = False

        self.parent_widget.system_widget.connected_signal.connect(self._on_microscope_connected)

        self.setup_connections()
        self.draw_blank_image()

    def _setup_ui(self):
        # matplotlib canvas (left pane) — replaces the napari viewer. The overview +
        # correlation images composite onto it; overlays + clicks attach in later stages.
        self.canvas = FibsemImageCanvas(self)
        # A black background + ~2x view margin: the overview "floats" on black, and
        # overlays that extend beyond the image (stage limits, grid boundary) stay visible.
        self.canvas.set_background_color("black")
        self.canvas.set_view_margin(0.5)

        # The overview is shown as an RGB composite (overview base + correlation layers),
        # so the canvas's built-in grayscale contrast/gamma is a no-op on it. Re-point the
        # contrast popover at the overview *layer* (clim/gamma) and recomposite instead.
        try:
            self.canvas._contrast.changed.disconnect(self.canvas._apply_contrast)
        except (TypeError, RuntimeError):
            pass
        self.canvas._contrast.changed.connect(self._on_overview_contrast_changed)

        # entity-grouped overlays (see docs/design/overview-minimap-cutover.md): each
        # redraws on its own trigger. ReferenceFrame (static geometry) behind
        # CurrentPosition behind LamellaMarkers (the interactive one, on top).
        self._reference_overlay = MinimapShapesOverlay(zorder=4)
        self._current_overlay = MinimapShapesOverlay(zorder=5)
        self._lamella_overlay = MinimapShapesOverlay(zorder=6)
        for _ov in (self._reference_overlay, self._current_overlay, self._lamella_overlay):
            self.canvas.add_overlay(_ov)

        # Controls (right pane) build onto their own container so the canvas + controls
        # sit in one splitter owned by this widget. Both call sites now just place the
        # widget; neither manages a separate viewer window.
        controls_widget = QWidget()
        self.gridLayout = QGridLayout(controls_widget)

        # Scroll area — row 0
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.gridLayout_5 = QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 0, 0)

        # Bottom elements (outside scroll area)
        self.label_instructions = QLabel(LABEL_INSTRUCTIONS["no-image"])
        self.gridLayout.addWidget(self.label_instructions, 1, 0)

        self.pushButton_run_tile_collection = QPushButton("Run Tiled Acquisition")
        self.gridLayout.addWidget(self.pushButton_run_tile_collection, 2, 0)

        self.pushButton_cancel_acquisition = QPushButton("Cancel Acquisition")
        self.gridLayout.addWidget(self.pushButton_cancel_acquisition, 3, 0)

        self.progressBar_acquisition = QProgressBar()
        self.progressBar_acquisition.setValue(24)
        self.progressBar_acquisition.setAlignment(Qt.AlignCenter)
        self.progressBar_acquisition.setStyleSheet(stylesheets.MILLING_PROGRESS_BAR_STYLESHEET)
        self.gridLayout.addWidget(self.progressBar_acquisition, 4, 0)

        # --- Acquisition settings widget ---
        self.overview_acquisition_widget = OverviewAcquisitionSettingsWidget(self)
        self.gridLayout_5.addWidget(self.overview_acquisition_widget, 0, 0)

        # ── Positions panel ────────────────────────────────────────
        positions_content = QWidget()
        self.gridLayout_2 = QGridLayout(positions_content)
        self.gridLayout_2.setContentsMargins(4, 4, 4, 4)

        bold_font = QFont()
        bold_font.setBold(True)

        self.lamella_list = LamellaNameListWidget()
        self.lamella_list.enable_defect_button(True)
        self.lamella_list.enable_actions_button(True)
        self.lamella_list.enable_move_to_action(True)
        self.lamella_list.enable_remove_button(True)
        self.gridLayout_2.addWidget(self.lamella_list, 0, 0, 1, 2)

        self.label_position_info = QLabel("No Positions saved.")
        self.gridLayout_2.addWidget(self.label_position_info, 2, 0, 1, 2)

        # row 3 skipped (matches original .ui)
        self.label_pattern_overlay = QLabel("Pattern Overlay")
        self.label_pattern_overlay.setFont(bold_font)
        self.gridLayout_2.addWidget(self.label_pattern_overlay, 4, 0)

        self.checkBox_pattern_overlay = QCheckBox("Display Pattern")
        self.comboBox_pattern_overlay = QComboBox()
        self.gridLayout_2.addWidget(self.checkBox_pattern_overlay, 5, 0)
        self.gridLayout_2.addWidget(self.comboBox_pattern_overlay, 5, 1)

        self.positions_panel = TitledPanel("Positions", content=positions_content)
        self.positions_panel._btn_collapse.setChecked(True)
        self.gridLayout_5.addWidget(self.positions_panel, 1, 0)

        # ── Correlation panel ─────────────────────────────────────
        correlation_content = QWidget()
        self.gridLayout_4 = QGridLayout(correlation_content)
        self.gridLayout_4.setContentsMargins(4, 4, 4, 4)

        self.label_correlation_selected_layer = QLabel("Selected Layer")
        self.comboBox_correlation_selected_layer = QComboBox()
        self.gridLayout_4.addWidget(self.label_correlation_selected_layer, 0, 0)
        self.gridLayout_4.addWidget(self.comboBox_correlation_selected_layer, 0, 1, 1, 2)

        # row 1 skipped (matches original .ui)
        self.checkBox_gridbar = QCheckBox("Show Grid Overlay")
        self.label_gb_width = QLabel("Gridbar Width (um)")
        self.label_gb_spacing = QLabel("Gridbar Spacing (um)")
        self.gridLayout_4.addWidget(self.checkBox_gridbar, 2, 0)
        self.gridLayout_4.addWidget(self.label_gb_width, 2, 1)
        self.gridLayout_4.addWidget(self.label_gb_spacing, 2, 2)

        self.doubleSpinBox_gb_width = QDoubleSpinBox()
        self.doubleSpinBox_gb_width.setMaximum(10000.0)
        self.doubleSpinBox_gb_spacing = QDoubleSpinBox()
        self.doubleSpinBox_gb_spacing.setMaximum(10000.0)
        self.gridLayout_4.addWidget(self.doubleSpinBox_gb_width, 3, 1)
        self.gridLayout_4.addWidget(self.doubleSpinBox_gb_spacing, 3, 2)

        self.pushButton_enable_correlation = QPushButton("Enable Correlation Mode")
        self.gridLayout_4.addWidget(self.pushButton_enable_correlation, 4, 0, 1, 3)

        self.correlation_panel = TitledPanel("Correlation", content=correlation_content)
        self.correlation_panel._btn_collapse.setChecked(False)
        self.gridLayout_5.addWidget(self.correlation_panel, 2, 0)

        # ── Display Options — a floating popover on the canvas toolbar (like napari's
        # layer controls), toggled by the "eye" button, instead of a side panel.
        self._display_panel = QFrame(self)
        self._display_panel.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self._display_panel.setObjectName("minimapDisplayPanel")
        self._display_panel.setStyleSheet(_DISPLAY_PANEL_QSS)
        _dlo = QVBoxLayout(self._display_panel)
        _dlo.setContentsMargins(14, 12, 14, 14)
        _dlo.setSpacing(6)
        _display_title = QLabel("DISPLAY")
        _display_title.setObjectName("panelTitle")
        _dlo.addWidget(_display_title)
        self.checkBox_show_overview_fov = QCheckBox("Overview FOV")
        self.checkBox_show_overview_fov.setChecked(True)
        self.checkBox_show_saved_positions_fov = QCheckBox("Saved Positions FOV")
        self.checkBox_show_saved_positions_fov.setChecked(True)
        self.checkBox_show_stage_limits = QCheckBox("Stage Limits")
        self.checkBox_show_stage_limits.setChecked(True)
        self.checkBox_show_circle_overlays = QCheckBox("Circle Overlays")
        self.checkBox_show_circle_overlays.setChecked(True)
        self.checkBox_show_tem_stage_limits = QCheckBox("TEM Stage Limits")
        self.checkBox_show_tem_stage_limits.setChecked(False)
        for _cb in (self.checkBox_show_overview_fov, self.checkBox_show_saved_positions_fov,
                    self.checkBox_show_stage_limits, self.checkBox_show_circle_overlays,
                    self.checkBox_show_tem_stage_limits):
            _dlo.addWidget(_cb)
        self._display_panel.hide()

        # canvas toolbar button that toggles the display popover
        self._btn_display = self.canvas.add_toolbar_button(
            "mdi:eye-outline", "Display options", self._toggle_display_panel, checkable=True
        )

        self.pushButton_load_image = QPushButton("Load Image")
        self.gridLayout_5.addWidget(self.pushButton_load_image, 4, 0)

        self.pushButton_load_correlation_image = QPushButton("Load Correlation Image")
        self.gridLayout_5.addWidget(self.pushButton_load_correlation_image, 5, 0)

        # add strech to end of scroll content
        self.gridLayout_5.setRowStretch(6, 1)

        # assemble: canvas (left) | controls (right)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.canvas)
        splitter.addWidget(controls_widget)
        splitter.setSizes([700, 500])
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(splitter)

    def _toggle_display_panel(self) -> None:
        """Show/hide the Display popover from its canvas toolbar button."""
        if self._btn_display.isChecked():
            self._position_display_panel()
            self._display_panel.show()
            self._display_panel.raise_()
        else:
            self._display_panel.hide()

    def _position_display_panel(self) -> None:
        """Anchor the popover near the canvas top-right, just below the toolbar."""
        self._display_panel.adjustSize()
        anchor = self.canvas.mapToGlobal(QPoint(self.canvas.width() - 8, 44))
        self._display_panel.move(anchor.x() - self._display_panel.width(), anchor.y())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_display_panel", None) is not None and self._display_panel.isVisible():
            self._position_display_panel()

    def draw_blank_image(self):
        if self.microscope is None:
            return
        image: Optional[FibsemImage] = None
        orientation = self.microscope.get_stage_orientation()
        if orientation == "SEM": 
            beam_type = BeamType.ELECTRON 
        else: 
            beam_type = BeamType.ION
        ms = self.microscope.get_microscope_state(beam_type=beam_type)
        image = FibsemImage.generate_blank_image(resolution=(4096, 4096), hfw=4000e-6)
        image.metadata.image_settings.beam_type = beam_type  # type: ignore
        image.metadata.microscope_state = ms                # type: ignore
        image.metadata.system = self.microscope.system      # type: ignore
        self.update_viewer(image=image)

    def set_experiment(self):
        if self.parent_widget is None:
            raise ValueError("Parent widget is None, cannot proceed.")
        if self.parent_widget.experiment is None:
            raise ValueError("Experiment in parent widget is None, cannot proceed.")

        self._on_experiment_changed()
        self._update_position_display()
        self.draw_blank_image()
        self._update_position_display()

    @property
    def microscope(self) -> FibsemMicroscope:
        return self.parent_widget.microscope

    @property
    def movement_widget(self) -> FibsemMovementWidget:
        return self.parent_widget.movement_widget

    @property
    def protocol(self) -> Optional[AutoLamellaTaskProtocol]:
        if self.parent_widget is None:
            return None
        if self.parent_widget.experiment is None:
            return None
        return self.parent_widget.experiment.task_protocol

    def setup_connections(self):

        # acquisition buttons
        self.pushButton_run_tile_collection.clicked.connect(self.run_tile_collection)
        self.pushButton_cancel_acquisition.clicked.connect(self.cancel_acquisition)
        self.pushButton_load_image.clicked.connect(self.load_image)

        # initialise overview acquisition widget with defaults
        self.overview_acquisition_widget.settings_changed.connect(self.update_imaging_display)

        # position list signals
        self.lamella_list.lamella_selected.connect(self.update_current_selected_position)
        self.lamella_list.move_to_requested.connect(self._on_move_to_requested)
        self.lamella_list.remove_requested.connect(self._on_remove_requested)

        # signals
        self._acquisition_finished.connect(self.tile_collection_finished)

        # Correlation, gridbar, and milling-pattern overlays are disabled pending a
        # rework (see docs/design/minimap-correlation-gridbar-rework.md): the
        # composite / stretch-to-fit approach can't do properly aligned correlation.
        # Hide their controls; the underlying methods are no-op stubs.
        self.correlation_panel.setVisible(False)
        self.pushButton_load_correlation_image.setVisible(False)
        self.label_pattern_overlay.setVisible(False)
        self.checkBox_pattern_overlay.setVisible(False)
        self.comboBox_pattern_overlay.setVisible(False)

        # set styles
        self.pushButton_run_tile_collection.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_cancel_acquisition.setStyleSheet(stylesheets.STOP_WORKFLOW_BUTTON_STYLESHEET)
        self.progressBar_acquisition.setStyleSheet(stylesheets.PROGRESS_BAR_GREEN_STYLE)
        self.pushButton_enable_correlation.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_load_image.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_load_correlation_image.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)

        # display option checkboxes
        self.checkBox_show_overview_fov.toggled.connect(self._on_display_option_toggled)
        self.checkBox_show_saved_positions_fov.toggled.connect(self._on_display_option_toggled)
        self.checkBox_show_stage_limits.toggled.connect(self._on_display_option_toggled)
        self.checkBox_show_circle_overlays.toggled.connect(self._on_display_option_toggled)
        self.checkBox_show_tem_stage_limits.toggled.connect(self._on_display_option_toggled)

        # set italics for instructions
        self.label_instructions.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore

        # canvas click callbacks: single → select nearest position,
        # double → move stage, right → add/move-position context menu.
        self.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self.canvas.canvas_double_clicked.connect(self._on_canvas_double_clicked)
        self.canvas.canvas_right_clicked.connect(self._on_canvas_right_clicked)

        self._update_position_display()
        self.toggle_interaction(enable=True)

    def _on_microscope_connected(self):
        """Connect microscope signals to the widget when microscope is (re)connected."""
        if self.microscope is None:
            return
        try:
            self.microscope.tiled_acquisition_signal.disconnect(self.handle_tile_acquisition_progress)
        except Exception:
            pass
        try:    
            self.microscope.stage_position_changed.disconnect(self._on_stage_position_changed)
        except Exception:
            pass
        self.microscope.tiled_acquisition_signal.connect(self.handle_tile_acquisition_progress)
        self.microscope.stage_position_changed.connect(self._on_stage_position_changed)

    def _on_experiment_changed(self):
        """Handle when the experiment in the parent widget changes (e.g. new experiment loaded)."""
        if self.parent_widget.experiment is None:
            raise ValueError("Experiment in parent widget is None, cannot proceed.")

        path = str(self.parent_widget.experiment.path)
        DEFAULT_OVERVIEW_ACQUISITION_SETTINGS.image_settings.path = path
        self.overview_acquisition_widget.update_from_settings(DEFAULT_OVERVIEW_ACQUISITION_SETTINGS)
        try:
            self.parent_widget.experiment.events.disconnect(self._on_experiment_position_changed) # type: ignore
        except Exception:
            pass
        self.parent_widget.experiment.events.connect(self._on_experiment_position_changed) # type: ignore

        available_task_names = []
        if self.protocol is not None:
            available_task_names = [name for name in self.protocol.task_config.keys() if self.protocol.task_config[name].milling]
            self.comboBox_pattern_overlay.blockSignals(True)
            self.comboBox_pattern_overlay.clear()
            self.comboBox_pattern_overlay.addItems(available_task_names)
            if "Trench Milling" in available_task_names:
                self.comboBox_pattern_overlay.setCurrentText("Trench Milling")
            elif "Rough Milling" in available_task_names:
                self.comboBox_pattern_overlay.setCurrentText("Rough Milling")
            self.comboBox_pattern_overlay.blockSignals(False)
        if available_task_names:
            self.checkBox_pattern_overlay.setEnabled(True)
            self.comboBox_pattern_overlay.setToolTip("")
        else:
            self.checkBox_pattern_overlay.setEnabled(False)
            self.comboBox_pattern_overlay.setToolTip("No milling patterns available.")

    def _on_display_option_toggled(self):
        self.show_overview_fov = self.checkBox_show_overview_fov.isChecked()
        self.show_saved_positions_fov = self.checkBox_show_saved_positions_fov.isChecked()
        self.show_stage_limits = self.checkBox_show_stage_limits.isChecked()
        self.show_circle_overlays = self.checkBox_show_circle_overlays.isChecked()
        self.show_tem_stage_limits = self.checkBox_show_tem_stage_limits.isChecked()
        self.draw_current_stage_position()

    @property
    def lamellas(self) -> List[Lamella]:
        if self.parent_widget is None or self.parent_widget.experiment is None:
            return []
        return list(self.parent_widget.experiment.positions)

    @property
    def positions(self) -> List[FibsemStagePosition]:
        stage_positions = []
        for lam in self.lamellas:
            sp = deepcopy(lam.stage_position)
            sp.name = lam.name
            stage_positions.append(sp)
        return stage_positions

    def _lamella_color(self, lamella: Lamella, selected: bool) -> str:
        """Return a display color for a lamella based on its defect state and selection."""
        if selected:
            return CROSSHAIR_CONFIG["colors"]["saved_selected"]  # lime
        if lamella.defect.state == DefectType.FAILURE:
            return "red"
        if lamella.defect.state == DefectType.REWORK:
            return "orange"
        return CROSSHAIR_CONFIG["colors"]["saved_unselected"]  # cyan

    @property
    def selected_lamella(self) -> Optional[Lamella]:
        """Return the currently selected lamella, or None if nothing is selected."""
        return self.lamella_list.selected_lamella

    @ensure_main_thread
    def _on_experiment_position_changed(self, event: EmissionInfo):
        if self.parent_widget is None:
            return
        try:
            # logging.info(f"Experiment position emitted: {event.signal.name}, {event.path}")
            if event.signal.name not in ["inserted", "removed", "changed"]:
                return
            self._update_position_display()
        except Exception as e:
            logging.error(f"Error logging experiment position change: {e}")
            self.parent_widget.experiment.events.disconnect(self._on_experiment_position_changed) # type: ignore

    def get_overview_settings(self) -> OverviewAcquisitionSettings:
        """Get the current overview acquisition settings from the UI."""
        return self.overview_acquisition_widget.get_settings()

    def update_imaging_display(self):
        """Refresh overlays whenever acquisition settings change."""
        self.draw_current_stage_position()

    def run_tile_collection(self):
        """Run the tiled acquisition."""
        logging.info("running tile collection")

        overview_settings = self.get_overview_settings()
        image_settings = overview_settings.image_settings
        image_settings.save = True

        if not image_settings.filename:
            notification_service.show_toast("Please enter a filename for the image", "error")
            return

        # ui feedback
        self.toggle_interaction(enable=False)
        self._hide_overlay_layers()

        self._thread_stop_event.clear()
        self._acquisition_worker = threading.Thread(
            target=self._run_tile_collection,
            args=(self.microscope, overview_settings),
            daemon=True,
        )
        self._acquisition_worker.start()

    def tile_collection_finished(self, result: dict):
        self._acquisition_worker = None
        self._thread_stop_event.clear()
        if result.get("error"):
            notification_service.show_toast(str(result["error"]), "error")
        elif result.get("cancelled"):
            notification_service.show_toast("Tile collection cancelled.", "warning")
        else:
            notification_service.show_toast("Tile collection finished.")
        self.update_viewer(self.image)
        self.toggle_interaction(enable=True)

    def _run_tile_collection(
        self,
        microscope: FibsemMicroscope,
        overview_settings: OverviewAcquisitionSettings,
    ):
        """Threaded worker for tiled acquisition and stitching."""
        self._tiles_acquired: int = 0
        self._tile_total_count: int = 0
        _start_time = time.time()
        _error: Optional[Exception] = None
        try:
            self.image = tiled.tiled_image_acquisition_and_stitch(
                microscope=microscope,
                settings=overview_settings,
                stop_event=self._thread_stop_event,
            )
        except Exception as e:
            logging.error(f"Error in tile collection: {e}", exc_info=True)
            _error = e
        finally:
            elapsed = time.time() - _start_time
            cancelled = self._thread_stop_event.is_set()
            result = {
                "tiles": self._tiles_acquired,
                "total": self._tile_total_count,
                "elapsed": elapsed,
                "cancelled": cancelled,
                "error": _error,
            }
            self._acquisition_finished.emit(result)

    @ensure_main_thread
    def handle_tile_acquisition_progress(self, ddict: dict) -> None:
        """Callback for handling the tile acquisition progress."""

        # track counts for result dict
        count, total = ddict["counter"], ddict["total"]
        self._tiles_acquired = count
        self._tile_total_count = total

        # progress bar
        self.progressBar_acquisition.setMaximum(100)
        self.progressBar_acquisition.setValue(int(count/total*100))
        self.progressBar_acquisition.setFormat(f"{ddict['msg']} — {count}/{total} tiles (%p%)")

        image = ddict.get("image", None)
        if image is not None:
            self.update_viewer(image, tmp=True)

    def cancel_acquisition(self):
        """Cancel the tiled acquisition."""
        logging.info("Cancelling acquisition...")
        self._thread_stop_event.set()

    @property
    def is_acquiring(self) -> bool:
        """Check if the acquisition thread is running."""
        return self._acquisition_worker is not None and self._acquisition_worker.is_alive()

    def toggle_gridbar_display(self):
        """Disabled pending the correlation/gridbar rework
        (see docs/design/minimap-correlation-gridbar-rework.md)."""
        return

    def update_gridbar_layer(self):
        """Disabled pending the correlation/gridbar rework."""
        return

    def toggle_interaction(self, enable: bool = True):
        """Toggle the interactivity of the UI elements."""
        self.pushButton_run_tile_collection.setEnabled(enable)
        self.pushButton_cancel_acquisition.setVisible(not enable)
        self.progressBar_acquisition.setVisible(not enable)
        # reset progress bar
        self.progressBar_acquisition.setValue(0)

        if enable:
            self.pushButton_run_tile_collection.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
            self.pushButton_run_tile_collection.setText("Run Tile Collection")
        else:
            # self.pushButton_run_tile_collection.setStyleSheet(stylesheets.DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_run_tile_collection.setText("Running Tile Collection...")

        if self.image is None:
            self.label_instructions.setText(LABEL_INSTRUCTIONS["no-image"])

    def load_image(self):
        """Ask the user to select a file to load an image as overview or correlation image."""
        is_correlation = self.sender() == self.pushButton_load_correlation_image

        filename = ui_utils.open_existing_file_dialog(
            msg="Select image to load",
            path=str(self.overview_acquisition_widget.get_settings().image_settings.path or os.getcwd()),
            _filter="Image Files (*.tif *.tiff)",
            parent=self)

        if filename == "":
            notification_service.show_toast("No file selected..", "error")
            return

        # load the image
        image = FibsemImage.load(filename)
        
        if is_correlation:
            self.add_correlation_image(image)
        else:
            self.update_viewer(image)

    def update_viewer(self, image: Optional[FibsemImage] = None, tmp: bool = False):
        """Update the canvas with the image and overlays."""
        if image is not None:
            if not tmp:
                self.image = image
                arr = image.filtered_data
            else:
                arr = image  # raw array from a live tile update

            self._set_overview_array(arr, reset_view=not tmp)

            if tmp:
                return  # live tile update — image pixels only, skip the overlays

        if self.image:
            self.draw_current_stage_position()  # draw the current stage position on the image
            self._draw_milling_pattern_overlay()  # draw the reprojected positions on the image
            self.label_instructions.setText(LABEL_INSTRUCTIONS["image-available"])
        self._update_info_bar()

    def _on_overview_contrast_changed(self) -> None:
        """Drive the overview layer's contrast/gamma from the canvas popover, then
        recomposite. The composite is RGB, so the canvas's grayscale contrast can't
        touch it — we adjust the overview FMLayer's clim/gamma instead (mirrors the FM
        canvas). The popover's min/max are normalized [0, 1] over the data range."""
        layer = self._overview_layer
        if layer is None or layer.data is None:
            return
        ctrl = self.canvas._contrast
        if ctrl.is_default():
            layer.autocontrast = True
            layer.clim = None
            layer.gamma = 1.0
        else:
            d = layer.data
            dmin, dmax = float(d.min()), float(d.max())
            span = (dmax - dmin) or 1.0
            layer.autocontrast = False
            layer.clim = (dmin + ctrl.contrast_min * span, dmin + ctrl.contrast_max * span)
            layer.gamma = ctrl.gamma
        self._recomposite()

    def _set_overview_array(self, arr: np.ndarray, reset_view: bool = True) -> None:
        """Show *arr* as the grayscale base layer of the composite (overview image)."""
        if self._overview_layer is None:
            self._overview_layer = FMLayer(name="overview", data=arr, color="gray")
        else:
            self._overview_layer.data = arr
        self._recomposite(reset_view=reset_view)

    def _recomposite(self, reset_view: bool = False) -> None:
        """Blend the overview + correlation layers into one RGB frame and show it
        (mirrors ``FMCanvasWidget._recomposite``: rebuild the axes on a shape change,
        otherwise swap pixels only)."""
        if self._overview_layer is None or self._overview_layer.data is None:
            return
        layers = [l for l in [self._overview_layer, *self._correlation_layers] if l is not None]
        shape = self._overview_layer.data.shape[:2]
        rgb = composite_fm_layers(layers, shape)
        if rgb is None:
            return
        h, w = rgb.shape[:2]
        px = None
        if self.image is not None and self.image.metadata and self.image.metadata.pixel_size:
            px = self.image.metadata.pixel_size.x
        if self._canvas_shape != (h, w):
            self._canvas_shape = (h, w)
            self.canvas.set_array(rgb, pixel_size=px)
            if reset_view:
                self.canvas.reset_view()
        else:
            # Pass pixel_size so the scalebar tracks a same-shape scale change — e.g. the
            # real overview replacing the blank placeholder after a tiled acquisition,
            # where the progressive tmp updates already set the canvas to this shape.
            self.canvas.update_display(rgb, pixel_size=px)

    def _update_info_bar(self, stage_position: Optional[FibsemStagePosition] = None) -> None:
        """Refresh the canvas info bar (stage / milling angle / grid / objective),
        mirroring the napari ``update_text_overlay``."""
        try:
            if type(self.microscope).__name__ == "TescanMicroscope":
                return  # no stage-position display yet
            if stage_position is None:
                stage_position = self.microscope._stage_position
            orientation = self.microscope.get_stage_orientation(stage_position=stage_position)
            grid = self.microscope.current_grid
            milling_angle = self.microscope.get_current_milling_angle(stage_position=stage_position)
            obj_txt = ""
            if self.microscope.fm is not None:
                obj_pos = self.microscope.fm.objective.position
                obj_txt = f"OBJECTIVE: {obj_pos * constants.METRE_TO_MICRON:.1f} µm"
            text = (
                f"STAGE: {stage_position.pretty_string} [{orientation}] [{grid}]\n"
                f"MILLING ANGLE: {milling_angle:.1f}°\n{obj_txt}"
            )
            self.canvas.set_info_text(text)
        except Exception as e:
            logging.warning(f"Error updating minimap info bar: {e}")
            self.canvas.set_info_text(None)

    def _point_to_microscope_coordinates(self, x: float, y: float) -> Optional[Point]:
        """Convert a canvas click at image-pixel ``(x, y)`` to microscope-image
        coordinates, or ``None`` if outside the image. The canvas emits full-resolution
        image-pixel coords, so there is no napari ``world_to_data`` step."""
        if self.image is None or self.image.metadata is None:
            return None
        h, w = self.image.data.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            notification_service.show_toast(
                "Clicked outside image dimensions. Please click inside the image.", "warning"
            )
            return None
        return conversions.image_to_microscope_image_coordinates(
            coord=Point(x=x, y=y),
            image=self.image.data,
            pixelsize=self.image.metadata.pixel_size.x,
        )

    def _on_canvas_clicked(self, x: float, y: float, modifiers) -> None:
        """Single left-click → select the nearest saved position (≤50 µm)."""
        point = self._point_to_microscope_coordinates(x, y)
        if point is None:
            return
        stage_position = self.microscope.project_stable_move(
            dx=point.x, dy=point.y,
            beam_type=self.image.metadata.image_settings.beam_type,
            base_position=self.image.metadata.stage_position)
        self.check_closest_experiment_position(stage_position)

    def check_closest_experiment_position(self, clicked_position: FibsemStagePosition) -> None:
        """Check and print distances to all experiment positions, highlighting the closest one.

        Args:
            clicked_position: The stage position that was clicked on the minimap.
        """

        if not self.positions:
            logging.info("No experiment positions to compare.")
            return

        # Calculate distances to all positions
        distances = []
        for pos in self.positions:
            distance = clicked_position.euclidean_distance(pos)
            distances.append((pos.name, distance))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        # Highlight the closest
        closest_name, closest_dist = distances[0]

        # If closest position is within 50um, select it
        SELECTED_POSITION_THRESHOLD_MICRONS = 50.0
        if closest_dist < SELECTED_POSITION_THRESHOLD_MICRONS * constants.MICRO_TO_SI:
            self.lamella_list.select(closest_name)
            return

    def _on_canvas_double_clicked(self, x: float, y: float, modifiers) -> None:
        """Double left-click → move the stage to the clicked point (limit-checked)."""
        if self.parent_widget.is_workflow_running:
            notification_service.show_toast("Cannot move stage while workflow is running.", "warning")
            return
        point = self._point_to_microscope_coordinates(x, y)
        if point is None:
            return
        beam_type = self.image.metadata.image_settings.beam_type
        stage_position = self.microscope.project_stable_move(
            dx=point.x, dy=point.y,
            beam_type=beam_type,
            base_position=self.image.metadata.stage_position)
        if not stage_position.is_within_limits(self.microscope._stage.limits, axes=["x", "y"]):
            notification_service.show_toast("Position is outside stage limits. Please select a position within the stage limits.", "warning")
            return
        self.move_to_stage_position(stage_position)

    def _on_canvas_right_clicked(self, x: float, y: float, modifiers) -> None:
        """Right-click → context menu: add a new position or move the selected one."""
        point = self._point_to_microscope_coordinates(x, y)
        if point is None:
            return
        stage_position = self.microscope.project_stable_move(
            dx=point.x, dy=point.y,
            beam_type=self.image.metadata.image_settings.beam_type,
            base_position=self.image.metadata.stage_position)
        if not stage_position.is_within_limits(self.microscope._stage.limits, axes=["x", "y"]):
            notification_service.show_toast("Position is outside stage limits. Please select a position within the stage limits.", "warning")
            return

        # Build context menu
        config = ContextMenuConfig()
        config.add_action(
            "Add New Position Here",
            callback=lambda: self._add_position_at_stage_position(stage_position),
        )

        # Only show "Move Selected Position" if there are positions to move
        if len(self.lamellas) > 0:
            selected_name = self.lamella_list.selected_name
            config.add_action(
                f"Move Selected Position Here ({selected_name})",
                callback=lambda: self._update_selected_position(stage_position),
            )

        menu = ContextMenu(config, parent=self)
        menu.show_at_cursor()

    def _add_position_at_stage_position(self, stage_position: FibsemStagePosition) -> None:
        """Add a new position at the given stage position."""
        if self.parent_widget is None or self.parent_widget.experiment is None:
            return

        self.parent_widget.add_new_lamella(stage_position)
        # NOTE: PY_38 doesnt support callback for experiment.events required to refresh the display
        if sys.version_info < (3, 9):
            self._update_position_display()

    def _update_selected_position(self, stage_position: FibsemStagePosition) -> None:
        """Update the currently selected position to the given stage position."""
        if self.parent_widget is None or self.parent_widget.experiment is None:
            return

        idx = self.lamella_list.selected_index
        if idx == -1:
            logging.debug("No position selected to update.")
            return

        self.parent_widget.experiment.positions[idx].stage_position = stage_position
        self.parent_widget.experiment.save()
        self._update_position_display()

    def update_current_selected_position(self, _lamella=None):
        """Update the currently selected position."""
        lam = self.selected_lamella
        if lam is None:
            return

        self.label_position_info.setText(f"{lam.name}: {lam.stage_position.pretty_string}")

        # redraw the positions to show the selected one
        self.draw_current_stage_position()

    def update_positions_combobox(self):
        """Update the positions combobox with the current positions."""

        lamellas = self.lamellas
        has_positions = len(lamellas) > 0
        self.positions_panel.setEnabled(has_positions)
        if not has_positions:
            self.positions_panel.setToolTip("No positions available. Please add a position via Right Click on the image.")
        else:
            self.positions_panel.setToolTip("")

        self.lamella_list.set_lamella(lamellas)

    def _on_move_to_requested(self, lamella):
        """Handle move-to request from the list row's actions menu."""
        if lamella is None:
            return
        self.move_to_stage_position(lamella.stage_position)

    def _on_remove_requested(self, lamella):
        """Handle removal from the list row's remove button (confirmation already handled)."""
        if self.parent_widget is None or self.parent_widget.experiment is None:
            return
        try:
            self.parent_widget.experiment.positions.remove(lamella)
        except ValueError:
            return
        self.parent_widget.experiment.save()
        self._update_position_display()

    def _update_position_display(self):
        """refresh the position display."""
        self.update_positions_combobox()
        self.update_viewer()

    def move_to_stage_position(self, stage_position: FibsemStagePosition) -> None:
        """Move the stage to the selected position via movement widget."""
        self.movement_widget.move_to_position(stage_position)

    @ensure_main_thread
    def _on_stage_position_changed(self, stage_position: FibsemStagePosition):
        """Callback for when the stage position is changed."""
        if self.is_acquiring:
            return # do not update while acquiring
        try:
            self.draw_current_stage_position(stage_position=stage_position)
            self._update_info_bar(stage_position=stage_position)
        except Exception as e:
            self.microscope.stage_position_changed.disconnect(self._on_stage_position_changed)
            logging.error(f"Error updating viewer on stage position change, signal disconnected: {e}")

    def _hide_overlay_layers(self):
        """Clear all overlay groups (e.g. during acquisition); repopulated on the next
        ``draw_current_stage_position``."""
        self._reference_overlay.clear()
        self._current_overlay.clear()
        self._lamella_overlay.clear()

    def draw_current_stage_position(self, stage_position: Optional[FibsemStagePosition] = None):
        """Redraw the stage-linked overlays (reference frame, current position, lamellas)."""
        if self.image is None or self.image.metadata is None:
            return

        if self.microscope is None:
            return

        if self.is_acquiring:
            self._hide_overlay_layers()
            return

        if stage_position is None:
            stage_position = self.microscope._stage_position

        self._draw_reference_frame()
        self._draw_current_position(stage_position)
        self._draw_lamella_markers()
        self._draw_milling_pattern_overlay()  # deferred: no-op stub

    def _draw_reference_frame(self) -> None:
        """Static reference geometry (ReferenceFrame overlay): origin + grid-slot
        crosshairs, stage-limit rect, grid-boundary circle, TEM rect (compustage only).
        Refreshes on image / holder change."""
        if self.image is None or self.image.metadata is None:
            self._reference_overlay.clear()
            return
        pixelsize = self.image.metadata.pixel_size.x
        specs: List[ShapeSpec] = []

        # origin crosshair
        stage_origin = FibsemStagePosition(name="Origin", x=0, y=0, z=0, r=0, t=0)
        origin_pt = tiled.reproject_stage_positions_onto_image2(self.image, [stage_origin])[0]
        specs.append(ShapeSpec(kind="crosshair", cx=origin_pt.x, cy=origin_pt.y,
                               color=CROSSHAIR_CONFIG["colors"]["origin"], label="Origin (0, 0)"))

        # grid-slot crosshairs
        grid_positions = [s.position for s in self.microscope._stage.holder.slots.values()]
        if grid_positions:
            grid_points = tiled.reproject_stage_positions_onto_image2(self.image, grid_positions)
            for gp, sp in zip(grid_points, grid_positions):
                specs.append(ShapeSpec(kind="crosshair", cx=gp.x, cy=gp.y,
                                       color=CROSSHAIR_CONFIG["colors"]["grid"], label=sp.name))

        # stage limits / grid boundary / TEM (compustage only)
        if (self.show_stage_limits or self.show_circle_overlays) and self.microscope.stage_is_compustage:
            stage_limits = self.microscope._stage.limits
            xmin, xmax = stage_limits["x"].min, stage_limits["x"].max
            ymin, ymax = stage_limits["y"].min, stage_limits["y"].max
            centre_grid = FibsemStagePosition(name="Grid Centre", x=0, y=0, z=0, r=0, t=0)
            top_limit = FibsemStagePosition(name="Top Limit", x=0, y=ymin, z=0, r=0, t=0)
            bottom_limit = FibsemStagePosition(name="Bottom Limit", x=0, y=ymax, z=0, r=0, t=0)
            pts = tiled.reproject_stage_positions_onto_image2(
                self.image, [centre_grid, top_limit, bottom_limit])
            grid_centre = pts[0]
            width = (xmax - xmin) / pixelsize
            height = pts[1].y - pts[2].y
            if self.show_stage_limits:
                specs.append(ShapeSpec(kind="rect", cx=grid_centre.x, cy=grid_centre.y,
                                       width=width, height=height, color="yellow",
                                       label="Stage Limits"))
            if self.show_circle_overlays:
                specs.append(ShapeSpec(kind="circle", cx=grid_centre.x, cy=grid_centre.y,
                                       radius=1000e-6 / pixelsize, color="red",
                                       label="Grid Boundary"))
            if self.show_tem_stage_limits:
                size_px = 1600e-6 / pixelsize
                specs.append(ShapeSpec(kind="rect", cx=grid_centre.x, cy=grid_centre.y,
                                       width=size_px, height=size_px, color="orange",
                                       label="TEM Stage Limits"))

        self._reference_overlay.set_shapes(specs)

    def _draw_current_position(self, stage_position: FibsemStagePosition) -> None:
        """Current overview-acquisition FOV box + current-position crosshair
        (CurrentPosition overlay). Refreshes on every stage move."""
        if self.image is None or self.image.metadata is None:
            self._current_overlay.clear()
            return
        pixelsize = self.image.metadata.pixel_size.x
        sp = deepcopy(stage_position)
        sp.name = "Current Position"
        current_pt = tiled.reproject_stage_positions_onto_image2(self.image, [sp])[0]

        specs: List[ShapeSpec] = []
        if self.show_overview_fov:
            overview_settings = self.get_overview_settings()
            specs.append(ShapeSpec(
                kind="rect", cx=current_pt.x, cy=current_pt.y,
                width=overview_settings.total_fov_x / pixelsize,
                height=overview_settings.total_fov_y / pixelsize,
                color="magenta", label="Overview FoV"))
        specs.append(ShapeSpec(kind="crosshair", cx=current_pt.x, cy=current_pt.y,
                               color=CROSSHAIR_CONFIG["colors"]["current"], label="Stage Position"))
        self._current_overlay.set_shapes(specs)

    def _draw_lamella_markers(self) -> None:
        """Per-lamella FOV box + crosshair + name label, coloured by defect/selection
        (LamellaMarkers overlay). Refreshes on add/remove/select/defect change.

        The name label rides on the FOV box when saved-position FOVs are shown, else on
        the crosshair — the one cross-cutting quirk that was awkward to split across
        napari layers is trivial here (both live in this one overlay)."""
        if self.image is None or self.image.metadata is None:
            self._lamella_overlay.clear()
            return
        lamellas = self.lamellas
        if not lamellas:
            self._lamella_overlay.clear()
            return

        pixelsize = self.image.metadata.pixel_size.x
        points = tiled.reproject_stage_positions_onto_image2(self.image, self.positions)
        selected_index = self.lamella_list.selected_index
        show_fov = self.show_saved_positions_fov
        fov_w = 80e-6 / pixelsize  # TODO: make this match the milling fov
        fov_h = 1024 / 1536 * fov_w

        specs: List[ShapeSpec] = []
        for i, (lam, point) in enumerate(zip(lamellas, points)):
            color = self._lamella_color(lam, selected=(i == selected_index))
            if show_fov:
                specs.append(ShapeSpec(kind="rect", cx=point.x, cy=point.y,
                                       width=fov_w, height=fov_h, color=color, label=lam.name))
                specs.append(ShapeSpec(kind="crosshair", cx=point.x, cy=point.y, color=color))
            else:
                specs.append(ShapeSpec(kind="crosshair", cx=point.x, cy=point.y,
                                       color=color, label=lam.name))
        self._lamella_overlay.set_shapes(specs)

    def _draw_milling_pattern_overlay(self):
        """Draw the selected task's milling pattern reprojected onto all saved positions.
        Deferred (see docs/design/overview-minimap-cutover.md): re-add later by feeding
        the reprojected stage list to a MillingPatternOverlay on the canvas."""
        return

    def add_correlation_image(self, image: FibsemImage, is_gridbar: bool = False):
        """Disabled pending the correlation/gridbar rework
        (see docs/design/minimap-correlation-gridbar-rework.md). The composite-based
        approach was reverted: correlation needs a real, persisted alignment transform,
        not a stretch-to-fit composite."""
        return

    def update_correlation_ui(self):
        """Disabled pending the correlation/gridbar rework."""
        return

    def _toggle_correlation_mode(self, event=None):
        """Disabled pending the correlation/gridbar rework."""
        return

    def set_active_layer_for_movement(self) -> None:
        """No-op on the matplotlib canvas — there is no layer-selection concept; the
        canvas always handles pan/zoom + clicks. Kept for call-site parity."""
        return
