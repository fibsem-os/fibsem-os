from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from fibsem.ui import stylesheets
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import FibsemMillingStage, MillingStrategy, get_strategy
from fibsem.milling.patterning import get_pattern
from fibsem.milling.patterning.patterns2 import BasePattern
from fibsem.structures import BeamType, FibsemMillingSettings
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel
from fibsem.ui.widgets.milling_settings_widget import FibsemMillingSettingsWidget
from fibsem.ui.widgets.milling_stage_list_widget import MillingStageListWidget
from fibsem.ui.widgets.pattern_settings_widget import FibsemPatternSettingsWidget
from fibsem.ui.widgets.strategy_settings_widget import FibsemStrategySettingsWidget


class FibsemMillingStagesWidget(QWidget):
    """Composes MillingStageListWidget + FibsemMillingSettingsWidget + FibsemPatternSettingsWidget.

    Selecting a stage in the list shows its milling and pattern settings below.
    """

    stages_changed = pyqtSignal(list)  # List[FibsemMillingStage]
    eye_toggled = pyqtSignal(bool)     # True = patterns visible

    def __init__(
        self,
        microscope: FibsemMicroscope,
        stages: List[FibsemMillingStage],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.microscope = microscope
        self._selected_stage: Optional[FibsemMillingStage] = None
        self._pending_inline_stage: Optional[FibsemMillingStage] = None
        self._pending_inline_update: bool = False
        self._pending_selected_stage: Optional[FibsemMillingStage] = None
        self._pending_selection_update: bool = False

        self._setup_ui()
        self._connect_signals()
        self.set_stages(stages)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Stage list
        _current_values = self.microscope.get_available_values_cached("current", BeamType.ION)
        self._list = MillingStageListWidget(current_values=_current_values)
        layout.addWidget(self._list)

        # Detail panel (hidden until a stage is selected)
        self._detail_widget = QWidget()
        detail_layout = QVBoxLayout(self._detail_widget)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(4)

        # Milling settings
        self._milling_widget = FibsemMillingSettingsWidget(
            microscope=self.microscope,
            settings=FibsemMillingSettings(),
        )
        self._btn_advanced = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )

        milling_panel = TitledPanel("Milling", content=self._milling_widget)
        milling_panel.add_header_widget(self._btn_advanced)
        milling_panel._btn_collapse.setChecked(False)
        detail_layout.addWidget(milling_panel)

        # Pattern settings
        self._pattern_widget = FibsemPatternSettingsWidget(
            microscope=self.microscope,
            pattern=get_pattern("Rectangle"),
        )
        self._btn_advanced_pattern = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )

        pattern_panel = TitledPanel("Pattern", content=self._pattern_widget)
        pattern_panel.add_header_widget(self._btn_advanced_pattern)
        pattern_panel._btn_collapse.setChecked(False)
        detail_layout.addWidget(pattern_panel)

        # Strategy settings
        self._strategy_widget = FibsemStrategySettingsWidget(
            strategy=get_strategy("Standard"),
        )
        self._btn_advanced_strategy = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )

        strategy_panel = TitledPanel("Strategy", content=self._strategy_widget)
        strategy_panel.add_header_widget(self._btn_advanced_strategy)
        strategy_panel._btn_collapse.setChecked(False)
        detail_layout.addWidget(strategy_panel)

        layout.addWidget(self._detail_widget)
        self._detail_widget.setVisible(False)

    def _connect_signals(self) -> None:
        self._list.stage_selected.connect(self._on_row_selected)
        self._list.stage_added.connect(lambda _: self.stages_changed.emit(self._list.get_stages()))
        self._list.stage_removed.connect(self._on_stage_removed)
        self._list.order_changed.connect(self.stages_changed.emit)
        self._list.enabled_changed.connect(lambda _: self.stages_changed.emit(self._list.get_stages()))
        self._list.stage_changed.connect(self._on_inline_stage_changed)
        self._list.eye_toggled.connect(self.eye_toggled)
        self._milling_widget.settings_changed.connect(self._on_milling_settings_changed)
        self._btn_advanced.toggled.connect(self._on_advanced_toggled)
        self._pattern_widget.pattern_changed.connect(self._on_pattern_changed)
        self._btn_advanced_pattern.toggled.connect(self._on_advanced_pattern_toggled)
        self._strategy_widget.strategy_changed.connect(self._on_strategy_changed)
        self._btn_advanced_strategy.toggled.connect(self._on_advanced_strategy_toggled)

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _on_stage_selected(self, stage: FibsemMillingStage) -> None:
        self._selected_stage = stage
        self._milling_widget.set_settings(stage.milling)
        self._pattern_widget.set_pattern(stage.pattern)
        self._strategy_widget.set_strategy(stage.strategy)
        self._detail_widget.setVisible(True)

    def _on_row_selected(self, stage: Optional[FibsemMillingStage]) -> None:
        if stage is None:
            return
        self._pending_selected_stage = stage
        if self._pending_selection_update:
            return
        self._pending_selection_update = True
        QTimer.singleShot(0, self._flush_row_selection)

    def _flush_row_selection(self) -> None:
        self._pending_selection_update = False
        stage = self._pending_selected_stage
        self._pending_selected_stage = None
        if stage is None:
            return
        if stage not in self._list.get_stages():
            return
        self._on_stage_selected(stage)

    def _on_milling_settings_changed(self, settings: FibsemMillingSettings) -> None:
        if self._selected_stage is not None:
            self._selected_stage.milling = settings
            self._list.refresh_stage(self._selected_stage)
            self.stages_changed.emit(self._list.get_stages())

    def _on_pattern_changed(self, pattern: BasePattern) -> None:
        if self._selected_stage is not None:
            self._selected_stage.pattern = pattern
            self._list.refresh_stage(self._selected_stage)
            self.stages_changed.emit(self._list.get_stages())

    def _on_advanced_toggled(self, checked: bool) -> None:
        self._milling_widget.set_advanced_visible(checked)

    def _on_advanced_pattern_toggled(self, checked: bool) -> None:
        self._pattern_widget.set_advanced_visible(checked)

    def _on_strategy_changed(self, strategy: MillingStrategy) -> None:
        if self._selected_stage is not None:
            self._selected_stage.strategy = strategy
            self._list.refresh_stage(self._selected_stage)
            self.stages_changed.emit(self._list.get_stages())

    def _on_advanced_strategy_toggled(self, checked: bool) -> None:
        self._strategy_widget.set_advanced_visible(checked)

    def _on_inline_stage_changed(self, stage: FibsemMillingStage) -> None:
        """Handle an inline field edit from the row widget and sync detail panels if selected."""
        self._pending_inline_stage = stage
        if self._pending_inline_update:
            return
        self._pending_inline_update = True
        QTimer.singleShot(0, self._flush_inline_stage_changed)

    def _flush_inline_stage_changed(self) -> None:
        self._pending_inline_update = False
        stage = self._pending_inline_stage
        self._pending_inline_stage = None
        if stage is None:
            return
        if stage not in self._list.get_stages():
            return
        if self._selected_stage is stage:
            self._sync_panels_from_stage(stage)
        self.stages_changed.emit(self._list.get_stages())

    def _sync_panels_from_stage(self, stage: FibsemMillingStage) -> None:
        """Reload detail panels from stage without triggering their change signals."""
        for w in (self._milling_widget, self._pattern_widget, self._strategy_widget):
            w.blockSignals(True)
        self._milling_widget.set_settings(stage.milling)
        self._pattern_widget.set_pattern(stage.pattern)
        self._strategy_widget.set_strategy(stage.strategy)
        for w in (self._milling_widget, self._pattern_widget, self._strategy_widget):
            w.blockSignals(False)

    def _on_stage_removed(self, stage: FibsemMillingStage) -> None:
        if self._selected_stage is stage:
            self._selected_stage = None
            self._detail_widget.setVisible(False)
        self.stages_changed.emit(self._list.get_stages())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_stages(self, stages: List[FibsemMillingStage]) -> None:
        self._list.set_stages(stages)
        if stages:
            self._list.select_stage(stages[0])
        else:
            self._selected_stage = None
            self._detail_widget.setVisible(False)

    def get_stages(self) -> List[FibsemMillingStage]:
        return self._list.get_stages()

    def get_enabled_stages(self) -> List[FibsemMillingStage]:
        return self._list.get_enabled_stages()

    def set_manufacturer(self, manufacturer: Optional[str]) -> None:
        self._milling_widget.set_manufacturer(manufacturer)

    def set_advanced_visible(self, show: bool) -> None:
        self._btn_advanced.blockSignals(True)
        self._btn_advanced.setChecked(show)
        self._btn_advanced.blockSignals(False)
        self._btn_advanced.set_icon_state(show)
        self._milling_widget.set_advanced_visible(show)
