from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.ui import stylesheets
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import FibsemMillingStage, MillingStrategy, get_strategy
from fibsem.milling.patterning import get_pattern
from fibsem.milling.patterning.patterns2 import BasePattern
from fibsem.structures import FibsemMillingSettings
from fibsem.ui.widgets.custom_widgets import TitledPanel
from fibsem.ui.widgets.milling_settings_widget import FibsemMillingSettingsWidget
from fibsem.ui.widgets.milling_stage_list_widget import MillingStageListWidget
from fibsem.ui.widgets.pattern_settings_widget import FibsemPatternSettingsWidget
from fibsem.ui.widgets.strategy_settings_widget import FibsemStrategySettingsWidget


class FibsemMillingStagesWidget(QWidget):
    """Composes MillingStageListWidget + FibsemMillingSettingsWidget + FibsemPatternSettingsWidget.

    Selecting a stage in the list shows its milling and pattern settings below.
    """

    stages_changed = pyqtSignal(list)  # List[FibsemMillingStage]

    def __init__(
        self,
        microscope: FibsemMicroscope,
        stages: List[FibsemMillingStage],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.microscope = microscope
        self._selected_stage: Optional[FibsemMillingStage] = None

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
        self._list = MillingStageListWidget()
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
        self._btn_advanced = QToolButton()
        self._btn_advanced.setIcon(QIconifyIcon("mdi:tune", color=stylesheets.GRAY_ICON_COLOR))
        self._btn_advanced.setToolTip("Show advanced settings")
        self._btn_advanced.setCheckable(True)
        self._btn_advanced.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)

        milling_panel = TitledPanel("Milling", content=self._milling_widget)
        milling_panel.add_header_widget(self._btn_advanced)
        milling_panel._btn_collapse.setChecked(False)
        detail_layout.addWidget(milling_panel)

        # Pattern settings
        self._pattern_widget = FibsemPatternSettingsWidget(
            microscope=self.microscope,
            pattern=get_pattern("Rectangle"),
        )
        self._btn_advanced_pattern = QToolButton()
        self._btn_advanced_pattern.setIcon(QIconifyIcon("mdi:tune", color=stylesheets.GRAY_ICON_COLOR))
        self._btn_advanced_pattern.setToolTip("Show advanced settings")
        self._btn_advanced_pattern.setCheckable(True)
        self._btn_advanced_pattern.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)

        pattern_panel = TitledPanel("Pattern", content=self._pattern_widget)
        pattern_panel.add_header_widget(self._btn_advanced_pattern)
        pattern_panel._btn_collapse.setChecked(False)
        detail_layout.addWidget(pattern_panel)

        # Strategy settings
        self._strategy_widget = FibsemStrategySettingsWidget(
            strategy=get_strategy("Standard"),
        )
        self._btn_advanced_strategy = QToolButton()
        self._btn_advanced_strategy.setIcon(QIconifyIcon("mdi:tune", color=stylesheets.GRAY_ICON_COLOR))
        self._btn_advanced_strategy.setToolTip("Show advanced settings")
        self._btn_advanced_strategy.setCheckable(True)
        self._btn_advanced_strategy.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)

        strategy_panel = TitledPanel("Strategy", content=self._strategy_widget)
        strategy_panel.add_header_widget(self._btn_advanced_strategy)
        strategy_panel._btn_collapse.setChecked(False)
        detail_layout.addWidget(strategy_panel)

        layout.addWidget(self._detail_widget)
        self._detail_widget.setVisible(False)

    def _connect_signals(self) -> None:
        self._list.stage_selected.connect(self._on_stage_selected)
        self._list.stage_added.connect(lambda _: self.stages_changed.emit(self._list.get_stages()))
        self._list.stage_removed.connect(self._on_stage_removed)
        self._list.order_changed.connect(self.stages_changed.emit)
        self._list.enabled_changed.connect(lambda _: self.stages_changed.emit(self._list.get_stages()))
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
        icon = "mdi:tune-variant" if checked else "mdi:tune"
        color = stylesheets.PRIMARY_COLOR if checked else stylesheets.GRAY_ICON_COLOR
        self._btn_advanced.setIcon(QIconifyIcon(icon, color=color))
        self._btn_advanced.setToolTip("Hide advanced settings" if checked else "Show advanced settings")
        self._milling_widget.set_advanced_visible(checked)

    def _on_advanced_pattern_toggled(self, checked: bool) -> None:
        icon = "mdi:tune-variant" if checked else "mdi:tune"
        color = stylesheets.PRIMARY_COLOR if checked else stylesheets.GRAY_ICON_COLOR
        self._btn_advanced_pattern.setIcon(QIconifyIcon(icon, color=color))
        self._btn_advanced_pattern.setToolTip("Hide advanced settings" if checked else "Show advanced settings")
        self._pattern_widget.set_advanced_visible(checked)

    def _on_strategy_changed(self, strategy: MillingStrategy) -> None:
        if self._selected_stage is not None:
            self._selected_stage.strategy = strategy
            self._list.refresh_stage(self._selected_stage)
            self.stages_changed.emit(self._list.get_stages())

    def _on_advanced_strategy_toggled(self, checked: bool) -> None:
        icon = "mdi:tune-variant" if checked else "mdi:tune"
        color = stylesheets.PRIMARY_COLOR if checked else stylesheets.GRAY_ICON_COLOR
        self._btn_advanced_strategy.setIcon(QIconifyIcon(icon, color=color))
        self._btn_advanced_strategy.setToolTip("Hide advanced settings" if checked else "Show advanced settings")
        self._strategy_widget.set_advanced_visible(checked)

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
        self._on_advanced_toggled(show)
