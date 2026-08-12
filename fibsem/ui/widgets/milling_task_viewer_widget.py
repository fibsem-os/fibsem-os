from __future__ import annotations

import copy
import logging
from typing import Callable, List, Optional, TYPE_CHECKING

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from fibsem import conversions
from fibsem.ui import notification_service
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import LinePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import BeamType, FibsemImage, Point
# `is_pattern_placement_valid` is pure geometry despite living under `fibsem/ui/napari/`
# — it validates a pattern against the image bounds and never touches a Viewer or Layer.
# It is the last thing keeping this module coupled to that package, and it moves out with
# the geometry extraction tracked on FIB-407 §1.
from fibsem.ui.napari.patterns import is_pattern_placement_valid
from fibsem.ui.widgets.custom_widgets import ContextMenu, ContextMenuConfig
from fibsem.ui.widgets.milling_task_config_widget2 import MillingTaskConfigWidget2
from fibsem.ui.widgets.milling_widget import FibsemMillingWidget2
from fibsem.ui.widgets.canvas.canvas_state import MillingSpec

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
    """MillingTaskConfigWidget2 + canvas pattern visualization + milling execution.

    Layout (top → bottom):
        MillingTaskConfigWidget2   — collapsible config panels (Task / Alignment / Acquisition / Stages)
        FibsemMillingWidget2       — Run / Pause / Stop + progress bars (hidden when milling_enabled=False)

    Pattern visualization is driven by ``settings_changed``: whenever the config changes the
    stages are pushed to the FIB canvas through the reducer as a ``MillingSpec``. This needs
    a :class:`MicroscopeViewController` — supplied directly via :meth:`set_controller`, or
    discovered from an injected ``image_widget`` — and a FIB image, from
    :meth:`set_fib_image` or the same image widget. Without a controller the widget is a
    config editor with no display, which is how the coincidence viewer uses it (it draws
    its own overlay on its own canvas).

    Right-click on the FIB canvas shows a context menu to move patterns and any extra
    actions injected by the parent widget.
    """

    settings_changed = pyqtSignal(FibsemMillingTaskConfig)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        milling_task_config: Optional[FibsemMillingTaskConfig] = None,
        milling_enabled: bool = True,
        image_widget: Optional["FibsemImageSettingsWidget"] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.microscope = microscope
        self._milling_enabled = milling_enabled
        self._image_widget = image_widget
        self._show_alignment_area: bool = True
        self.parent_widget = parent

        self._fib_image: Optional[FibsemImage] = None
        # Quad-view path (set in _init_canvas_overlay when a controller exists)
        self._controller = None
        self._fib_canvas = None
        self._background_milling_stages: List[FibsemMillingStage] = []
        self._patterns_visible = True  # eye-toggle state (mirrored onto MillingSpec.visible)
        self._pattern_update_pending = False
        self._settings_emit_pending: bool = False
        self._pending_settings: Optional[FibsemMillingTaskConfig] = None
        self._right_click_menu_action_provider: Optional[
            Callable[[ContextMenuConfig, Point], None]
        ] = None

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
                # NOTE this connect sits inside the try with a load-bearing statement
                # after it — an exception on the line above silently skips both it and
                # the wiring below, and patterns quietly stop following the live image.
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
        image_widget tied to the controller, so only it takes this path. Callers with no
        image widget attach one themselves via :meth:`set_controller` (the Lamella
        Editor), or leave ``_controller`` None and display nothing (the coincidence
        viewer, which renders its own overlay).
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

        Switches this widget onto the canvas pattern path — patterns and the read-only
        alignment display go through the reducer, and right-click reposition is driven by
        the FIB canvas signal.
        """
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
        if self._controller is None:
            return
        # quad-view: reposition via the FIB canvas right-click signal
        self._fib_canvas.canvas_right_clicked.connect(self._on_canvas_right_click)

    def _on_canvas_right_click(self, x: float, y: float, modifiers) -> None:
        """FIB canvas right-click → pattern reposition menu.

        Takes pixel coords straight from the canvas signal and reuses
        ``_move_patterns``.
        """
        # The canvas is not part of the editor, so disabling that panel does not reach
        # this menu -- it has to be turned away itself while a mill is running
        # (FIB-580). Offering "Move Patterns Here" over a lamella being milled invites
        # exactly the edit the lock exists to prevent.
        if self.is_milling:
            return
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
        """Build + show the pattern-reposition context menu at the cursor."""
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
        # Guarded as well as the menu that opens it: this is the mutation, and a menu
        # already on screen when a run starts would otherwise still act on it.
        if self.is_milling:
            return
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
    # Pattern display
    # ------------------------------------------------------------------

    def set_fib_image(self, image: FibsemImage) -> None:
        """Inject the FIB image that patterns are drawn against."""
        self._fib_image = image
        self._schedule_pattern_update()

    def set_alignment_area_visible(self, visible: bool) -> None:
        """Show/hide the alignment area rectangle on the canvas."""
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
        """Debounced entry point: render the current stages onto the FIB canvas.

        Clears the pending flag first, so a change arriving while the reducer call is in
        flight schedules a fresh pass rather than being swallowed.
        """
        self._pattern_update_pending = False
        if self._controller is not None:
            self._update_canvas_patterns()

    def _update_canvas_patterns(self) -> None:
        """Push the current enabled stages to the FIB canvas via the reducer."""
        if self._controller is None:
            return
        if self._fib_image is None or self._fib_image.metadata is None:
            return
        config = self.config_widget.get_settings()
        stages = config.enabled_stages
        if not stages:
            self._controller.remove_overlay(BeamType.ION, "milling")
            self._update_canvas_alignment(None)  # no patterns → no alignment
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

        Only shown alongside patterns (i.e. when stages exist); pass ``config=None`` to
        hide. Yields to an active edit in the reducer.
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
            # toggle the milling overlay's visibility through the reducer
            self._controller.set_overlay_visible(BeamType.ION, "milling", visible)

    # ------------------------------------------------------------------
    # Public API — required by FibsemMillingWidget2
    # ------------------------------------------------------------------

    @property
    def is_milling(self) -> bool:
        """Whether a milling task is currently running.

        Delegates to the embedded run controls, which own the milling thread. Hosts hold
        *this* widget as ``milling_widget``, so callers asking "is it milling?" land here
        rather than on ``FibsemMillingWidget2`` — ``FibsemMovementWidget`` blocks
        click-to-move on it. That guard was written against the widget this class
        replaced and had been raising ``AttributeError`` in ``FibsemUI`` ever since,
        which is what broke click-to-move there.
        """
        return self.milling_widget.is_milling

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
