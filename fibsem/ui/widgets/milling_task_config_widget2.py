from __future__ import annotations

import copy
from typing import Optional, TYPE_CHECKING, Union

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem import constants
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel, WheelBlocker
from fibsem.ui.widgets.milling_alignment_widget import FibsemMillingAlignmentWidget
from fibsem.ui.widgets.milling_stages_widget import FibsemMillingStagesWidget
from fibsem.ui.widgets.milling_task_acquisition_settings_widget import (
    FibsemMillingTaskAcquisitionSettingsWidget,
)

if TYPE_CHECKING:
    from fibsem.ui.widgets.milling_task_widget import FibsemMillingTaskWidget

_WIDGET_CONFIG = {
    "name": {"default": "Milling Task", "placeholder": "Enter task name..."},
    "field_of_view": {
        "range": (0.001, 10000),
        "decimals": 1,
        "step": 5.0,
        "default": 150.0,
        "suffix": " μm",
        "tooltip": "Field of view in micrometers (μm)",
        "keyboard_tracking": False,
    },
}

_INSTRUCTIONS_TEXT = "Instructions: Right Click to Move Current Pattern"


class MillingTaskConfigWidget2(QWidget):
    """MillingTaskConfigWidget using TitledPanel + FibsemMillingStagesWidget.

    Layout: Task (core) → Alignment → Acquisition → Milling stages.
    Alignment and acquisition panels start collapsed.
    No viewer/napari/correlation dependencies.
    """

    settings_changed = pyqtSignal(FibsemMillingTaskConfig)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        milling_task_config: Optional[FibsemMillingTaskConfig] = None,
        milling_enabled: bool = True,
        correlation_enabled: bool = True,
        parent: Optional[Union[QWidget, "FibsemMillingTaskWidget"]] = None,
    ) -> None:
        super().__init__(parent)
        self.microscope = microscope
        self._settings = milling_task_config or FibsemMillingTaskConfig()
        self.parent_widget = parent

        self._setup_ui()
        self.update_from_settings(self._settings)
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(4)

        use_scroll_area = True
        try:
            from fibsem.ui.widgets.milling_task_widget import FibsemMillingTaskWidget
            use_scroll_area = not isinstance(self.parent_widget, FibsemMillingTaskWidget)
        except Exception:
            use_scroll_area = True

        if use_scroll_area:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll_area.setWidget(content_widget)
            main_layout.addWidget(scroll_area)
        else:
            layout.setContentsMargins(0, 0, 0, 0)
            content_widget.setContentsMargins(0, 0, 0, 0)
            main_layout.addWidget(content_widget)

        # ── Core panel ──────────────────────────────────────────────
        core_content = QWidget()
        core_grid = QGridLayout(core_content)
        core_grid.setContentsMargins(4, 4, 4, 4)

        self.name_edit = QLineEdit()
        self.name_edit.setText(_WIDGET_CONFIG["name"]["default"])
        self.name_edit.setPlaceholderText(_WIDGET_CONFIG["name"]["placeholder"])

        fov = _WIDGET_CONFIG["field_of_view"]
        self.field_of_view_spinbox = QDoubleSpinBox()
        self.field_of_view_spinbox.setRange(*fov["range"])
        self.field_of_view_spinbox.setDecimals(fov["decimals"])
        self.field_of_view_spinbox.setSingleStep(fov["step"])
        self.field_of_view_spinbox.setValue(fov["default"])
        self.field_of_view_spinbox.setSuffix(fov["suffix"])
        self.field_of_view_spinbox.setToolTip(fov["tooltip"])
        self.field_of_view_spinbox.setKeyboardTracking(fov["keyboard_tracking"])
        self.field_of_view_spinbox.installEventFilter(
            WheelBlocker(self.field_of_view_spinbox)
        )

        self.label_instructions = QLabel(_INSTRUCTIONS_TEXT)
        self.label_instructions.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)

        core_grid.addWidget(QLabel("Name"), 0, 0)
        core_grid.addWidget(self.name_edit, 0, 1)
        core_grid.addWidget(QLabel("Field of View"), 1, 0)
        core_grid.addWidget(self.field_of_view_spinbox, 1, 1)
        core_grid.addWidget(self.label_instructions, 2, 0, 1, 2)
        core_grid.setColumnStretch(1, 1)

        core_panel = TitledPanel("Task", content=core_content)
        core_panel._btn_collapse.setChecked(True)
        layout.addWidget(core_panel)

        # ── Alignment panel ──────────────────────────────────────────
        self.alignment_widget = FibsemMillingAlignmentWidget(
            parent=self, show_advanced=False
        )
        self.alignment_widget.image_settings_widget.set_show_advanced_button(False)

        self._btn_enable_alignment = IconToolButton(
            icon="mdi:checkbox-blank-outline",
            checked_icon="mdi:checkbox-marked-outline",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Enable alignment",
            checked_tooltip="Disable alignment",
            checked=True,
        )
        self._btn_advanced_alignment = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )

        alignment_panel = TitledPanel("Alignment", content=self.alignment_widget)
        alignment_panel.add_header_widget(self._btn_enable_alignment)
        alignment_panel.add_header_widget(self._btn_advanced_alignment)
        alignment_panel._btn_collapse.setChecked(False)
        layout.addWidget(alignment_panel)

        # ── Acquisition panel ────────────────────────────────────────
        self.acquisition_widget = FibsemMillingTaskAcquisitionSettingsWidget(
            parent=self, show_advanced=False
        )
        self.acquisition_widget.image_settings_widget.set_show_advanced_button(False)

        self._btn_enable_acquisition = IconToolButton(
            icon="mdi:checkbox-blank-outline",
            checked_icon="mdi:checkbox-marked-outline",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Enable acquisition",
            checked_tooltip="Disable acquisition",
            checked=True,
        )
        self._btn_advanced_acquisition = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )

        acquisition_panel = TitledPanel("Acquisition", content=self.acquisition_widget)
        acquisition_panel.add_header_widget(self._btn_enable_acquisition)
        acquisition_panel.add_header_widget(self._btn_advanced_acquisition)
        acquisition_panel._btn_collapse.setChecked(False)
        layout.addWidget(acquisition_panel)

        # ── Milling stages ───────────────────────────────────────────
        self.milling_stages_widget = FibsemMillingStagesWidget(
            microscope=self.microscope, stages=[]
        )
        self._btn_stage_count = IconToolButton(icon="mdi:numeric-0-box-outline", size=32)
        self._btn_stage_count.setEnabled(False)

        milling_panel = TitledPanel("Milling Stages", content=self.milling_stages_widget)
        milling_panel.add_header_widget(self._btn_stage_count)
        milling_panel._btn_collapse.setChecked(True)
        layout.addWidget(milling_panel)

        self._update_stage_count_icon(0)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.name_edit.textChanged.connect(self._emit_settings_changed)
        self.field_of_view_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.alignment_widget.settings_changed.connect(self._emit_settings_changed)
        self.acquisition_widget.settings_changed.connect(self._emit_settings_changed)
        self.milling_stages_widget.stages_changed.connect(self._emit_settings_changed)
        self.milling_stages_widget.stages_changed.connect(
            lambda stages: self._update_stage_count_icon(len(stages))
        )
        self._btn_enable_alignment.toggled.connect(self._on_enable_alignment_toggled)
        self._btn_advanced_alignment.toggled.connect(self._on_advanced_alignment_toggled)
        self._btn_enable_acquisition.toggled.connect(self._on_enable_acquisition_toggled)
        self._btn_advanced_acquisition.toggled.connect(self._on_advanced_acquisition_toggled)
        self.alignment_widget.enabled_checkbox.toggled.connect(self._on_alignment_checkbox_changed)
        self.acquisition_widget.acquire_sem_checkbox.toggled.connect(self._on_acquisition_checkbox_changed)
        self.acquisition_widget.acquire_fib_checkbox.toggled.connect(self._on_acquisition_checkbox_changed)

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _update_stage_count_icon(self, n: int) -> None:
        icon_name = "mdi:numeric-9-plus-box-outline" if n > 9 else f"mdi:numeric-{n}-box-outline"
        self._btn_stage_count.setIcon(QIconifyIcon(icon_name, color=stylesheets.GRAY_ICON_COLOR))
        self._btn_stage_count.setToolTip(f"{n} stage{'s' if n != 1 else ''}")

    def _emit_settings_changed(self) -> None:
        self.settings_changed.emit(self.get_settings())

    def _on_enable_alignment_toggled(self, checked: bool) -> None:
        self.alignment_widget.enabled_checkbox.blockSignals(True)
        self.alignment_widget.enabled_checkbox.setChecked(checked)
        self.alignment_widget.enabled_checkbox.blockSignals(False)
        self.alignment_widget._update_controls_enabled()

    def _on_alignment_checkbox_changed(self, checked: bool) -> None:
        self._btn_enable_alignment.blockSignals(True)
        self._btn_enable_alignment.setChecked(checked)
        self._btn_enable_alignment.blockSignals(False)
        self._btn_enable_alignment.set_icon_state(checked)

    def _on_advanced_alignment_toggled(self, checked: bool) -> None:
        self.alignment_widget.set_show_advanced(checked)

    def _on_enable_acquisition_toggled(self, checked: bool) -> None:
        aw = self.acquisition_widget
        aw.acquire_sem_checkbox.blockSignals(True)
        aw.acquire_fib_checkbox.blockSignals(True)
        aw.acquire_sem_checkbox.setChecked(checked)
        aw.acquire_fib_checkbox.setChecked(checked)
        aw.acquire_sem_checkbox.blockSignals(False)
        aw.acquire_fib_checkbox.blockSignals(False)
        aw._update_image_settings_enabled()

    def _on_acquisition_checkbox_changed(self) -> None:
        aw = self.acquisition_widget
        any_enabled = aw.acquire_sem_checkbox.isChecked() or aw.acquire_fib_checkbox.isChecked()
        self._btn_enable_acquisition.blockSignals(True)
        self._btn_enable_acquisition.setChecked(any_enabled)
        self._btn_enable_acquisition.blockSignals(False)
        self._btn_enable_acquisition.set_icon_state(any_enabled)

    def _on_advanced_acquisition_toggled(self, checked: bool) -> None:
        self.acquisition_widget.set_show_advanced(checked)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_settings(self) -> FibsemMillingTaskConfig:
        self._settings.name = self.name_edit.text()
        self._settings.field_of_view = (
            self.field_of_view_spinbox.value() * constants.MICRO_TO_SI
        )
        self._settings.alignment = self.alignment_widget.get_settings()
        self._settings.acquisition = self.acquisition_widget.get_settings()
        self._settings.stages = self.milling_stages_widget.get_stages()
        return copy.deepcopy(self._settings)

    def get_config(self) -> FibsemMillingTaskConfig:
        return self.get_settings()

    def update_from_settings(self, settings: FibsemMillingTaskConfig) -> None:
        self.blockSignals(True)
        self._settings = copy.deepcopy(settings)
        self.name_edit.setText(settings.name)
        self.field_of_view_spinbox.setValue(
            settings.field_of_view * constants.SI_TO_MICRO
        )
        self.alignment_widget.update_from_settings(settings.alignment)
        self.acquisition_widget.update_from_settings(settings.acquisition)
        self.milling_stages_widget.set_stages(settings.stages)
        self._update_stage_count_icon(len(settings.stages))
        self.blockSignals(False)
        self._on_alignment_checkbox_changed(settings.alignment.enabled)  # sync button state to loaded values
        self._on_acquisition_checkbox_changed()  # sync button state to loaded values

    def set_config(self, config: FibsemMillingTaskConfig) -> None:
        self.update_from_settings(config)

    def clear(self) -> None:
        self.update_from_settings(FibsemMillingTaskConfig())

