from __future__ import annotations

import copy
import logging
from typing import Callable, List, Optional, TYPE_CHECKING

import napari
from napari.layers import Image as NapariImageLayer
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from fibsem import conversions
from fibsem.ui import notification_service
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import LinePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import BeamType, FibsemImage, Point
from fibsem.ui.napari.patterns import (
    draw_milling_patterns_in_napari,
    draw_milling_fov_rect,
    is_pattern_placement_valid,
    MILLING_PATTERN_LAYER_NAME,
    MILLING_FOV_LAYER_NAME,
)
from fibsem.ui.napari.utilities import is_position_inside_layer
from fibsem.ui.widgets.custom_widgets import ContextMenu, ContextMenuConfig
from fibsem.ui.widgets.milling_task_config_widget2 import MillingTaskConfigWidget2
from fibsem.ui.widgets.milling_widget import FibsemMillingWidget2
from fibsem.ui.widgets.canvas_state import MillingSpec

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

    Right-click on the FIB image layer shows a context menu to move patterns
    and any extra actions injected by the parent widget.
    """

    settings_changed = pyqtSignal(FibsemMillingTaskConfig)

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
        self._show_alignment_area: bool = True
        self.parent_widget = parent

        self._fib_image: Optional[FibsemImage] = None
        self._fib_image_layer: Optional[NapariImageLayer] = None
        self._pattern_layer_names: List[str] = []
        # Quad-view path (set in _init_canvas_overlay when a controller exists)
        self._controller = None
        self._fib_canvas = None
        self._background_milling_stages: List[FibsemMillingStage] = []
        self._patterns_visible = True  # eye-toggle state (mirrored onto MillingSpec.visible)
        self._pattern_update_inflight = False
        self._pattern_update_pending = False
        self._settings_emit_pending: bool = False
        self._pending_settings: Optional[FibsemMillingTaskConfig] = None
        self._right_click_menu_action_provider: Optional[
            Callable[[ContextMenuConfig, Point], None]
        ] = None

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

        self.config_widget = MillingTaskConfigWidget2(
            microscope=self.microscope,
            milling_task_config=milling_task_config,
            parent=self,
        )
        layout.addWidget(self.config_widget)

        self.milling_widget = FibsemMillingWidget2(
            microscope=self.microscope,
            parent=self,
        )
        self.milling_widget.setVisible(self._milling_enabled)
        layout.addWidget(self.milling_widget)

    def _connect_signals(self) -> None:
        self.config_widget.settings_changed.connect(self._on_settings_changed)
        self.config_widget.eye_toggled.connect(self._on_eye_toggled)

    # ------------------------------------------------------------------
    # Section visibility API
    # ------------------------------------------------------------------

    def set_parameters_visible(self, visible: bool) -> None:
        """Show or hide the Milling Parameters panel."""
        self.config_widget.core_panel.setVisible(visible)

    def set_alignment_visible(self, visible: bool) -> None:
        """Show or hide the Alignment panel."""
        self.config_widget.alignment_panel.setVisible(visible)

    def set_acquisition_visible(self, visible: bool) -> None:
        """Show or hide the Acquisition panel."""
        self.config_widget.acquisition_panel.setVisible(visible)

    def _setup_viewer_integration(self) -> None:
        """Connect to image_widget (injected or discovered from parent chain)."""
        iw = self._image_widget
        if iw is not None:
            try:
                self._fib_image = iw.ib_image
                iw.viewer_update_signal.connect(self._on_viewer_image_updated)
            except Exception:
                pass
        self._init_canvas_overlay()
        self._register_right_click_callback()

    def _init_canvas_overlay(self) -> None:
        """Wire this widget to the quad-view controller when one is available.

        Milling patterns and the read-only alignment-area display are reducer-owned
        (pushed via ``controller.set_overlay`` / ``set_alignment_display``); this just
        stores the controller + FIB canvas. Only the main microscope tab injects an
        image_widget tied to the controller, so only it takes this path. Every other
        caller — including the napari-based Lamella Editor — leaves ``_controller``
        None and keeps the existing napari pattern path.
        """
        self._controller = None
        self._fib_canvas = None
        controller = self._view_controller()
        if controller is None:
            return
        self._controller = controller
        self._fib_canvas = controller.fib_canvas

    def _view_controller(self):
        """Return the quad-view controller via the injected image widget, or None."""
        iw = self._image_widget
        if iw is None:
            return None
        try:
            return iw._view_controller()
        except Exception:
            return None

    def set_controller(self, controller) -> None:
        """Directly attach a :class:`MicroscopeViewController` for callers that have no
        ``image_widget`` to discover one through (e.g. the Lamella Editor).

        Switches this widget onto the canvas pattern path *additively* — patterns and
        the read-only alignment display go through the reducer, and right-click
        reposition moves to the FIB canvas signal. The napari path is left intact for
        any caller that never sets a controller.
        """
        # tear down the napari right-click callback if it was wired at construction
        if self.viewer is not None and self._right_click_callback is not None:
            try:
                self.viewer.mouse_drag_callbacks.remove(self._right_click_callback)
            except ValueError:
                pass
            self._right_click_callback = None

        self._controller = controller
        self._fib_canvas = (
            controller.get_canvas(BeamType.ION) if controller is not None else None
        )
        if self._fib_canvas is not None:
            self._fib_canvas.canvas_right_clicked.connect(self._on_canvas_right_click)
        self._schedule_pattern_update()

    def closeEvent(self, event) -> None:
        if self._controller is not None and self._fib_canvas is not None:
            try:
                self._fib_canvas.canvas_right_clicked.disconnect(self._on_canvas_right_click)
            except Exception:
                pass
            try:
                self._controller.remove_overlay(BeamType.ION, "milling")
            except Exception:
                pass
            try:
                self._controller.set_alignment_display(BeamType.ION, None, False)
            except Exception:
                pass
        if self.viewer is not None and self._right_click_callback is not None:
            try:
                self.viewer.mouse_drag_callbacks.remove(self._right_click_callback)
            except ValueError:
                pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Right-click pattern movement
    # ------------------------------------------------------------------

    def set_right_click_menu_actions(
        self, action_provider: Optional[Callable[[ContextMenuConfig, Point], None]]
    ) -> None:
        """
        Register a callable that appends additional actions to the right-click context menu.

        The provider receives the context menu config and the click point in
        microscope image coordinates. Pass ``None`` to remove all custom actions.
        """
        self._right_click_menu_action_provider = action_provider

    def _register_right_click_callback(self) -> None:
        if self._controller is not None:
            # quad-view: reposition via the FIB canvas right-click signal
            self._fib_canvas.canvas_right_clicked.connect(self._on_canvas_right_click)
            return
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
        stages = self.config_widget.milling_stages_widget.get_enabled_stages()
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

        self._show_reposition_menu(point_clicked, stages)

    def _on_canvas_right_click(self, x: float, y: float, modifiers) -> None:
        """Quad-view FIB canvas right-click → pattern reposition menu.

        Mirrors ``_on_right_click`` but takes pixel coords straight from the canvas
        signal (no napari ``world_to_data`` / layer hit-test). Reuses ``_move_patterns``.
        """
        if self._fib_image is None or self._fib_image.metadata is None:
            return
        stages = self.config_widget.milling_stages_widget.get_enabled_stages()
        if not stages:
            return
        h, w = self._fib_image.data.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        point_clicked = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=x, y=y),
            image=self._fib_image.data,
            pixelsize=self._fib_image.metadata.pixel_size.x,
        )

        self._show_reposition_menu(point_clicked, stages)

    def _show_reposition_menu(self, point_clicked: Point, stages: list) -> None:
        """Build + show the pattern-reposition context menu at the cursor.

        Shared by the napari (:meth:`_on_right_click`) and canvas
        (:meth:`_on_canvas_right_click`) entry points, which differ only in how
        they derive ``point_clicked``.
        """
        selected = self.config_widget.milling_stages_widget._list._selected_stage
        # Fall back to first enabled stage if selected stage is disabled
        if selected is None or not selected.enabled:
            selected = stages[0]
        selected_name = selected.name

        cfg = ContextMenuConfig()
        if self._right_click_menu_action_provider is not None:
            try:
                self._right_click_menu_action_provider(cfg, point_clicked)
            except Exception:
                logging.exception("Failed to add custom context-menu actions.")
                notification_service.show_toast(
                    "Failed to add point-of-interest menu action; pattern movement options will still be shown.",
                    "warning",
                )
        if len(stages) > 1:
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
        stages = self.config_widget.milling_stages_widget.get_enabled_stages()
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

        for idx, stage in enumerate(stages):
            if not move_all and idx != ref_idx:
                continue
            pattern_copy = copy.deepcopy(stage.pattern)
            _apply_diff_to_pattern(pattern_copy, diff)
            if not is_pattern_placement_valid(pattern_copy, self._fib_image):
                msg = f"'{stage.name}' pattern would be outside the FIB image."
                logging.warning(msg)
                notification_service.show_toast(msg, "warning")
                return

        for idx, stage in enumerate(stages):
            if not move_all and idx != ref_idx:
                continue
            _apply_diff_to_pattern(stage.pattern, diff)

        self.config_widget.milling_stages_widget._list.refresh_all()
        sw = self.config_widget.milling_stages_widget
        if sw._selected_stage is not None:
            sw._sync_panels_from_stage(sw._selected_stage)
        self._on_settings_changed(self.config_widget.get_settings())

    # ------------------------------------------------------------------
    # Viewer / pattern display
    # ------------------------------------------------------------------

    def set_fib_image(
        self, image: FibsemImage, image_layer: Optional[NapariImageLayer]
    ) -> None:
        """Inject a FIB image and its napari layer for pattern display."""
        self._fib_image = image
        self._fib_image_layer = image_layer
        self._schedule_pattern_update()

    def set_alignment_area_visible(self, visible: bool) -> None:
        """Show/hide the alignment area rectangle in the viewer."""
        self._show_alignment_area = visible
        self._schedule_pattern_update()

    def _on_viewer_image_updated(self) -> None:
        iw = self._image_widget
        if iw is None:
            try:
                iw = self.parent_widget.image_widget  # type: ignore[attr-defined]
            except Exception:
                return
        try:
            self._fib_image = iw.ib_image
            self._schedule_pattern_update()
        except Exception as e:
            logging.error(f"MillingTaskViewerWidget: viewer image update error: {e}")

    def _update_pattern_display(self) -> None:
        if self._pattern_update_inflight:
            return
        self._pattern_update_pending = False

        # quad-view canvas path: render on the FIB canvas overlay, skip napari
        if self._controller is not None:
            self._update_canvas_patterns()
            return

        if self.viewer is None or self._fib_image_layer is None:
            return
        if self._fib_image is None or self._fib_image.metadata is None:
            return

        self._pattern_update_inflight = True
        config = self.config_widget.get_settings()
        stages = config.enabled_stages

        if not stages:
            self._clear_pattern_display()
            self._pattern_update_inflight = False
            return

        pixelsize = self._fib_image.metadata.pixel_size.x

        alignment_area = None
        if config.alignment.enabled and self._show_alignment_area:
            alignment_area = config.alignment.rect
        try:
            self._pattern_layer_names = draw_milling_patterns_in_napari(
                viewer=self.viewer,
                image_layer=self._fib_image_layer,
                milling_stages=stages,
                pixelsize=pixelsize,
                draw_crosshair=True,
                background_milling_stages=self._background_milling_stages,
                alignment_area=alignment_area,
            )
            draw_milling_fov_rect(
                viewer=self.viewer,
                image_layer=self._fib_image_layer,
                field_of_view=config.field_of_view,
                pixelsize=pixelsize,
            )
        except Exception as e:
            logging.error(f"MillingTaskViewerWidget: pattern display error: {e}")
        finally:
            self._pattern_update_inflight = False
            if self._pattern_update_pending:
                self._schedule_pattern_update()

    def _update_canvas_patterns(self) -> None:
        """Push the current enabled stages to the FIB canvas via the reducer (quad-view)."""
        if self._controller is None:
            return
        if self._fib_image is None or self._fib_image.metadata is None:
            return
        config = self.config_widget.get_settings()
        stages = config.enabled_stages
        if not stages:
            self._controller.remove_overlay(BeamType.ION, "milling")
            self._update_canvas_alignment(None)  # no patterns → no alignment (match napari)
            return
        self._update_canvas_alignment(config)
        selected = self.config_widget.milling_stages_widget._list._selected_stage
        selected_index = next((i for i, s in enumerate(stages) if s is selected), None)
        self._controller.set_overlay(
            BeamType.ION,
            MillingSpec(
                stages=stages,
                background_stages=self._background_milling_stages,
                selected_index=selected_index,
                visible=self._patterns_visible,
            ),
        )

    def _update_canvas_alignment(self, config) -> None:
        """Push the read-only alignment-area display to the controller (quad-view).

        Mirrors napari: only shown alongside patterns (i.e. when stages exist);
        pass ``config=None`` to hide. Yields to an active edit in the reducer.
        """
        if self._controller is None:
            return
        show = (
            config is not None
            and config.alignment.enabled
            and self._show_alignment_area
        )
        rect = config.alignment.rect if show else None
        self._controller.set_alignment_display(BeamType.ION, rect, show)

    def _clear_pattern_display(self) -> None:
        """Remove milling pattern layers from the viewer."""
        if self.viewer is None:
            return
        try:
            for name in self._pattern_layer_names:
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)  # type: ignore
            if MILLING_FOV_LAYER_NAME in self.viewer.layers:
                self.viewer.layers.remove(MILLING_FOV_LAYER_NAME)  # type: ignore
        except Exception as e:
            logging.debug(f"MillingTaskViewerWidget: error removing layers: {e}")
        self._pattern_layer_names = []

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_settings_changed(self, config: FibsemMillingTaskConfig) -> None:
        self._pending_settings = config
        if self._settings_emit_pending:
            return
        self._settings_emit_pending = True
        QTimer.singleShot(0, self._flush_settings_changed)
        self._schedule_pattern_update()

    def _flush_settings_changed(self) -> None:
        self._settings_emit_pending = False
        config = self._pending_settings
        self._pending_settings = None
        if config is None:
            return
        self.settings_changed.emit(config)

    def _schedule_pattern_update(self) -> None:
        if self._pattern_update_pending:
            return
        self._pattern_update_pending = True
        QTimer.singleShot(0, self._update_pattern_display)

    def _on_eye_toggled(self, visible: bool) -> None:
        self._patterns_visible = visible
        if self._controller is not None:
            # quad-view: toggle the milling overlay's visibility through the reducer
            self._controller.set_overlay_visible(BeamType.ION, "milling", visible)
            return
        if self.viewer is None:
            return
        if MILLING_PATTERN_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[MILLING_PATTERN_LAYER_NAME].visible = visible

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
        self._schedule_pattern_update()

    def update_from_settings(self, settings: FibsemMillingTaskConfig) -> None:
        self.config_widget.update_from_settings(settings)
        self._schedule_pattern_update()

    def clear(self) -> None:
        self.config_widget.clear()
        self._schedule_pattern_update()

    def set_background_milling_stages(self, stages: List[FibsemMillingStage]) -> None:
        self._background_milling_stages = stages
        self._schedule_pattern_update()

    def set_manufacturer(self, manufacturer: Optional[str]) -> None:
        self.config_widget.milling_stages_widget.set_manufacturer(manufacturer)
