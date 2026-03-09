from __future__ import annotations

import copy
import logging
from typing import List, Optional, TYPE_CHECKING

import napari
import napari.utils.notifications
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from fibsem import conversions
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import LinePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import FibsemImage, Point
from fibsem.ui.napari.patterns import draw_milling_patterns_in_napari, is_pattern_placement_valid, MILLING_ALIGNMENT_AREA_LAYER_NAME
from fibsem.ui.napari.utilities import is_position_inside_layer
from fibsem.ui.widgets.custom_widgets import ContextMenu, ContextMenuConfig
from fibsem.ui.widgets.milling_task_config_widget2 import MillingTaskConfigWidget2
from fibsem.ui.widgets.milling_widget import FibsemMillingWidget2

if TYPE_CHECKING:
    from fibsem.ui import FibsemImageSettingsWidget

def _apply_diff_to_pattern(pattern, diff: Point) -> None:
    """Shift a pattern's position by diff (in-place). Handles LinePattern start/end offsets."""
    pattern.point = pattern.point + diff
    if isinstance(pattern, LinePattern):
        pattern.start_x += diff.x
        pattern.start_y += diff.y
        pattern.end_x += diff.x
        pattern.end_y += diff.y


class MillingTaskViewerWidget(QWidget):
    """MillingTaskConfigWidget2 + napari pattern visualization + milling execution.

    Layout (top → bottom):
        MillingTaskConfigWidget2   — collapsible config panels (Task / Alignment / Acquisition / Stages)
        FibsemMillingWidget2       — Run / Pause / Stop + progress bars (hidden when milling_enabled=False)

    Pattern visualization is driven by ``settings_changed`` — whenever the config changes the
    milling pattern Shapes layers in the napari viewer are redrawn.  A FIB image layer must be
    available (injected via ``set_fib_image()`` or discovered from the parent's ``image_widget``).

    Right-click on the FIB image layer shows a context menu to move patterns.
    """

    settings_changed = pyqtSignal(FibsemMillingTaskConfig)
    milling_progress_signal = pyqtSignal(dict)           # reserved for future use

    def __init__(
        self,
        microscope: FibsemMicroscope,
        viewer: Optional[napari.Viewer] = None,
        milling_task_config: Optional[FibsemMillingTaskConfig] = None,
        milling_enabled: bool = True,
        image_widget: Optional["FibsemImageSettingsWidget"] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.microscope = microscope
        self.viewer: Optional[napari.Viewer] = viewer or napari.current_viewer()
        self._milling_enabled = milling_enabled
        self._image_widget = image_widget

        # Viewer / pattern state
        self._fib_image: Optional[FibsemImage] = None
        self._fib_image_layer: Optional[napari.layers.Image] = None
        self._pattern_layer_names: List[str] = []
        self._background_milling_stages: List[FibsemMillingStage] = []

        # Right-click callback reference (kept for unregistration)
        self._right_click_callback = None

        self._setup_ui(milling_task_config)
        self._connect_signals()
        self._setup_viewer_integration()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self, milling_task_config: Optional[FibsemMillingTaskConfig]) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Config widget (owns scroll area internally)
        self.config_widget = MillingTaskConfigWidget2(
            microscope=self.microscope,
            milling_task_config=milling_task_config,
            parent=self,
        )
        layout.addWidget(self.config_widget)

        # Milling widget
        self.milling_widget = FibsemMillingWidget2(
            microscope=self.microscope,
            parent=self,
        )
        self.milling_widget.setVisible(self._milling_enabled)
        layout.addWidget(self.milling_widget)

    def _connect_signals(self) -> None:
        self.config_widget.settings_changed.connect(self._on_settings_changed)

    def _setup_viewer_integration(self) -> None:
        """Connect to image_widget (injected or discovered from parent chain)."""
        iw = self._image_widget
        if iw is not None:
            try:
                self._fib_image = iw.ib_image
                self._fib_image_layer = iw.ib_layer
                iw.viewer_update_signal.connect(self._on_viewer_image_updated)
            except Exception:
                pass
        self._register_right_click_callback()

    def closeEvent(self, event) -> None:
        if self.viewer is not None and self._right_click_callback is not None:
            try:
                self.viewer.mouse_drag_callbacks.remove(self._right_click_callback)
            except ValueError:
                pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Right-click pattern movement
    # ------------------------------------------------------------------

    def _register_right_click_callback(self) -> None:
        if self.viewer is None:
            return
        if self._right_click_callback is not None:
            try:
                self.viewer.mouse_drag_callbacks.remove(self._right_click_callback)
            except ValueError:
                pass
        self._right_click_callback = self._on_right_click
        self.viewer.mouse_drag_callbacks.append(self._right_click_callback)

    def _on_right_click(self, viewer: napari.Viewer, event) -> None:
        if event.button != 2 or event.type != "mouse_press":
            return
        if self._fib_image_layer is None or self._fib_image is None:
            return
        if self._fib_image.metadata is None:
            return
        stages = self.config_widget.milling_stages_widget.get_stages()
        if not stages:
            return
        if not is_position_inside_layer(event.position, self._fib_image_layer):
            return

        event.handled = True

        coords = self._fib_image_layer.world_to_data(event.position)
        point_clicked = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=coords[1], y=coords[0]),
            image=self._fib_image.data,
            pixelsize=self._fib_image.metadata.pixel_size.x,
        )

        selected = self.config_widget.milling_stages_widget._list._selected_stage
        selected_name = selected.name if selected is not None else stages[0].name

        cfg = ContextMenuConfig()
        cfg.add_action(
            "Move All Patterns Here",
            callback=lambda: self._move_patterns(point_clicked, move_all=True),
        )
        cfg.add_action(
            f"Move '{selected_name}' Here",
            callback=lambda: self._move_patterns(point_clicked, move_all=False),
        )
        ContextMenu(cfg, parent=self).show_at_cursor()

    def _move_patterns(self, point: Point, move_all: bool) -> None:
        """Move patterns to point. move_all=True shifts all relative to selected; False moves only selected."""
        stages = self.config_widget.milling_stages_widget.get_stages()
        if not stages or self._fib_image is None:
            return

        selected = self.config_widget.milling_stages_widget._list._selected_stage
        ref_idx = 0
        if selected is not None:
            for i, s in enumerate(stages):
                if s is selected:
                    ref_idx = i
                    break

        diff = point - stages[ref_idx].pattern.point

        # Validate all moves before applying any
        for idx, stage in enumerate(stages):
            if not move_all and idx != ref_idx:
                continue
            pattern_copy = copy.deepcopy(stage.pattern)
            _apply_diff_to_pattern(pattern_copy, diff)
            if not is_pattern_placement_valid(pattern_copy, self._fib_image):
                msg = f"'{stage.name}' pattern would be outside the FIB image."
                logging.warning(msg)
                napari.utils.notifications.show_warning(msg)
                return

        # Apply to live stage objects
        for idx, stage in enumerate(stages):
            if not move_all and idx != ref_idx:
                continue
            _apply_diff_to_pattern(stage.pattern, diff)

        # Refresh row display (inline spinboxes/labels) without full rebuild
        self.config_widget.milling_stages_widget._list.refresh_all()
        # Re-sync the pattern settings panel to the selected stage
        sw = self.config_widget.milling_stages_widget
        if sw._selected_stage is not None:
            sw._sync_panels_from_stage(sw._selected_stage)
        self._update_pattern_display()
        self.settings_changed.emit(self.config_widget.get_settings())

    # ------------------------------------------------------------------
    # Viewer / pattern display
    # ------------------------------------------------------------------

    def set_fib_image(self, image: FibsemImage, image_layer: Optional[napari.layers.Image]) -> None:
        """Inject a FIB image and its napari layer for pattern display."""
        self._fib_image = image
        self._fib_image_layer = image_layer
        self._update_pattern_display()

    def _on_viewer_image_updated(self) -> None:
        iw = self._image_widget
        if iw is None:
            try:
                iw = self.parent().image_widget  # type: ignore[attr-defined]
            except Exception:
                return
        try:
            self._fib_image = iw.ib_image
            self._fib_image_layer = iw.ib_layer
            self._update_pattern_display()
        except Exception as e:
            logging.error(f"MillingTaskViewerWidget: viewer image update error: {e}")

    def _update_pattern_display(self) -> None:
        if self.viewer is None or self._fib_image_layer is None:
            return
        if self._fib_image is None or self._fib_image.metadata is None:
            return
        config = self.config_widget.get_settings()

        if not config.stages:
            self._clear_pattern_display()
            return

        pixelsize = self._fib_image.metadata.pixel_size.x
        alignment_area = config.alignment.rect if config.alignment.enabled else None
        try:
            self._pattern_layer_names = draw_milling_patterns_in_napari(
                viewer=self.viewer,
                image_layer=self._fib_image_layer,
                milling_stages=config.stages,
                pixelsize=pixelsize,
                draw_crosshair=True,
                background_milling_stages=self._background_milling_stages,
                alignment_area=alignment_area,
            )
        except Exception as e:
            logging.error(f"MillingTaskViewerWidget: pattern display error: {e}")

    def _clear_pattern_display(self) -> None:
        """Remove milling pattern layers from the viewer."""
        if self.viewer is None:
            return
        try:
            for name in self._pattern_layer_names:
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name) # type: ignore
            if MILLING_ALIGNMENT_AREA_LAYER_NAME in self.viewer.layers:
                self.viewer.layers.remove(MILLING_ALIGNMENT_AREA_LAYER_NAME)  # type: ignore
        except Exception as e:
            logging.debug(f"MillingTaskViewerWidget: error removing layers: {e}")
        self._pattern_layer_names = []

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_settings_changed(self, config: FibsemMillingTaskConfig) -> None:
        self.settings_changed.emit(config)
        self._update_pattern_display()

    # ------------------------------------------------------------------
    # Public API — required by FibsemMillingWidget2
    # ------------------------------------------------------------------

    def get_config(self) -> FibsemMillingTaskConfig:
        return self.config_widget.get_config()

    # ------------------------------------------------------------------
    # Public API — mirrors MillingTaskConfigWidget
    # ------------------------------------------------------------------

    def get_settings(self) -> FibsemMillingTaskConfig:
        return self.config_widget.get_settings()

    def set_config(self, config: FibsemMillingTaskConfig) -> None:
        self.config_widget.set_config(config)
        self._update_pattern_display()

    def update_from_settings(self, settings: FibsemMillingTaskConfig) -> None:
        self.config_widget.update_from_settings(settings)
        self._update_pattern_display()

    def clear(self) -> None:
        self.config_widget.clear()
        self._update_pattern_display()

    def set_background_milling_stages(self, stages: List[FibsemMillingStage]) -> None:
        self._background_milling_stages = stages
        self._update_pattern_display()

    def set_manufacturer(self, manufacturer: Optional[str]) -> None:
        self.config_widget.milling_stages_widget.set_manufacturer(manufacturer)
