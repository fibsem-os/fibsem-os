from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from typing import Optional

from fibsem import constants
from fibsem.config import AVAILABLE_RESOLUTIONS_ZIP, DEFAULT_SQUARE_RESOLUTION
from fibsem.structures import AutoFocusMode, AutoFocusSettings, BeamType, FocusStackSettings, ImageSettings, OverviewAcquisitionSettings, TileOrderStrategy
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel, ValueComboBox, ValueSpinBox
from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget


class OverviewAcquisitionSettingsWidget(QWidget):
    """Widget for editing OverviewAcquisitionSettings.

    Composes a tile-grid section (beam type, rows, cols, total FOV) on top of
    an embedded ImageSettingsWidget (resolution, dwell time, tile FOV,
    autocontrast, autogamma, save, path, filename).
    """

    settings_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._show_advanced = False
        self._setup_ui()
        self._connect_signals()
        self._update_total_fov_label()
        self._update_advanced_visibility()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # ── Overview Acquisition panel ────────────────────────────────
        grid_content = QWidget()
        grid_layout = QGridLayout(grid_content)
        grid_layout.setContentsMargins(4, 4, 4, 4)

        # Beam type
        grid_layout.addWidget(QLabel("Beam Type"), 0, 0)
        self.beam_type_combo = ValueComboBox(items=list(BeamType), format_fn=lambda bt: bt.name)
        grid_layout.addWidget(self.beam_type_combo, 0, 1, 1, 2)

        # Rows / cols
        grid_layout.addWidget(QLabel("Tiles (rows x cols)"), 1, 0)
        self.nrows_spinbox = ValueSpinBox(minimum=1, maximum=15, step=1, decimals=0)
        self.nrows_spinbox.setValue(3)
        self.ncols_spinbox = ValueSpinBox(minimum=1, maximum=15, step=1, decimals=0)
        self.ncols_spinbox.setValue(3)

        _tiles_row = QWidget()
        _tiles_layout = QHBoxLayout(_tiles_row)
        _tiles_layout.setContentsMargins(0, 0, 0, 0)
        _tiles_layout.addWidget(self.nrows_spinbox)
        _label_x = QLabel("x")
        _label_x.setAlignment(Qt.AlignCenter)  # type: ignore
        _tiles_layout.addWidget(_label_x)
        _tiles_layout.addWidget(self.ncols_spinbox)
        grid_layout.addWidget(_tiles_row, 1, 1, 1, 2)

        # Overlap
        grid_layout.addWidget(QLabel("Overlap (%)"), 2, 0)
        self.overlap_spinbox = ValueSpinBox(suffix="%", minimum=0.0, maximum=50.0, step=5.0, decimals=0)
        grid_layout.addWidget(self.overlap_spinbox, 2, 1, 1, 2)

        # Total FOV label (read-only, auto-updated)
        self._label_total_fov = QLabel("Total FOV: —")
        self._label_total_fov.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore
        grid_layout.addWidget(self._label_total_fov, 3, 0, 1, 3)

        # Focus stack (advanced)
        self._label_focus_stack = QLabel("Focus Stack")
        grid_layout.addWidget(self._label_focus_stack, 4, 0)
        self.focus_stack_enabled = QCheckBox()
        grid_layout.addWidget(self.focus_stack_enabled, 4, 1, 1, 2)

        self._label_focus_stack_steps = QLabel("Focus Stack Steps")
        grid_layout.addWidget(self._label_focus_stack_steps, 5, 0)
        self.focus_stack_steps = ValueSpinBox(minimum=1, maximum=10, step=1, decimals=0)
        self.focus_stack_steps.setValue(3)
        grid_layout.addWidget(self.focus_stack_steps, 5, 1, 1, 2)

        self._label_focus_stack_autofocus = QLabel("Focus Stack Autofocus")
        grid_layout.addWidget(self._label_focus_stack_autofocus, 6, 0)
        self.focus_stack_autofocus = QCheckBox()
        self.focus_stack_autofocus.setChecked(True)
        grid_layout.addWidget(self.focus_stack_autofocus, 6, 1, 1, 2)

        # Auto Focus mode (advanced)
        self._label_autofocus = QLabel("Auto Focus")
        grid_layout.addWidget(self._label_autofocus, 7, 0)
        self.autofocus_combo = ValueComboBox(
            items=list(AutoFocusMode),
            format_fn=lambda m: m.name.replace("_", " ").title(),
        )
        grid_layout.addWidget(self.autofocus_combo, 7, 1, 1, 2)

        # Tile order strategy (advanced)
        self._label_tile_order = QLabel("Tile Order")
        grid_layout.addWidget(self._label_tile_order, 8, 0)
        self.tile_order_combo = ValueComboBox(
            items=list(TileOrderStrategy),
            format_fn=lambda s: s.value.title(),
        )
        grid_layout.addWidget(self.tile_order_combo, 8, 1, 1, 2)

        self._adv_widgets = [
            (self._label_focus_stack,          self.focus_stack_enabled),
            (self._label_focus_stack_steps,    self.focus_stack_steps),
            (self._label_focus_stack_autofocus, self.focus_stack_autofocus),
            (self._label_autofocus,            self.autofocus_combo),
            (self._label_tile_order,           self.tile_order_combo),
        ]

        self._btn_advanced = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )

        # ── Tile Image Settings + combined panel ─────────────────────────────────
        self.image_settings_widget = ImageSettingsWidget(
            show_advanced=False,
            always_save=True,
        )
        self.image_settings_widget.hfw_label.setText("Field of View")
        self.image_settings_widget.set_show_advanced_button(False)

        # All standard + square resolutions are supported (non-square aspect handled in acquisition)
        self.image_settings_widget.set_available_resolutions(AVAILABLE_RESOLUTIONS_ZIP, default=DEFAULT_SQUARE_RESOLUTION)

        self._btn_advanced_imaging = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced image settings",
            checked_tooltip="Hide advanced image settings",
        )

        self._image_panel = TitledPanel("Tile Image Settings", content=self.image_settings_widget)
        self._image_panel.add_header_widget(self._btn_advanced_imaging)
        self._image_panel._btn_collapse.setChecked(True)

        # combine grid controls + image settings panel into one collapsible panel
        combined = QWidget()
        combined_layout = QVBoxLayout(combined)
        combined_layout.setContentsMargins(0, 0, 0, 0)
        combined_layout.setSpacing(0)
        combined_layout.addWidget(grid_content)
        combined_layout.addWidget(self._image_panel)

        grid_panel = TitledPanel("Overview Acquisition", content=combined)
        grid_panel.add_header_widget(self._btn_advanced)
        grid_panel._btn_collapse.setChecked(True)
        outer.addWidget(grid_panel)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._btn_advanced.toggled.connect(self.set_show_advanced)
        self.beam_type_combo.currentIndexChanged.connect(self._on_changed)
        self.nrows_spinbox.valueChanged.connect(self._on_changed)
        self._btn_advanced_imaging.toggled.connect(self.image_settings_widget.set_show_advanced)
        self.ncols_spinbox.valueChanged.connect(self._on_changed)
        self.overlap_spinbox.valueChanged.connect(self._on_changed)
        self.focus_stack_enabled.toggled.connect(self._on_changed)
        self.focus_stack_steps.valueChanged.connect(self._on_changed)
        self.focus_stack_autofocus.toggled.connect(self._on_changed)
        self.autofocus_combo.currentIndexChanged.connect(self._on_changed)
        self.tile_order_combo.currentIndexChanged.connect(self._on_changed)
        self.image_settings_widget.settings_changed.connect(self._on_changed)

    def _on_changed(self):
        self._update_total_fov_label()
        self.settings_changed.emit()

    # ------------------------------------------------------------------
    # Advanced visibility
    # ------------------------------------------------------------------

    def set_show_advanced(self, show: bool):
        self._show_advanced = show
        self._btn_advanced.blockSignals(True)
        self._btn_advanced.setChecked(show)
        self._btn_advanced.blockSignals(False)
        self._btn_advanced.set_icon_state(show)
        self._update_advanced_visibility()

    def _update_advanced_visibility(self):
        for label, widget in self._adv_widgets:
            label.setVisible(self._show_advanced)
            widget.setVisible(self._show_advanced)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_total_fov_label(self):
        settings = self.get_settings()
        total_w = settings.total_fov_x * constants.SI_TO_MICRO
        total_h = settings.total_fov_y * constants.SI_TO_MICRO
        sym = constants.MICRON_SYMBOL
        self._label_total_fov.setText(
            f"Total FOV: {total_w:.0f} × {total_h:.0f} {sym}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_settings(self) -> OverviewAcquisitionSettings:
        """Read current widget values and return OverviewAcquisitionSettings."""
        image_settings: ImageSettings = self.image_settings_widget.get_settings()
        image_settings.beam_type = self.beam_type_combo.value()
        return OverviewAcquisitionSettings(
            image_settings=image_settings,
            nrows=int(self.nrows_spinbox.value()),
            ncols=int(self.ncols_spinbox.value()),
            overlap=self.overlap_spinbox.value() / 100.0,
            focus_stack_settings=FocusStackSettings(
                enabled=self.focus_stack_enabled.isChecked(),
                n_steps=int(self.focus_stack_steps.value()),
                auto_focus=self.focus_stack_autofocus.isChecked(),
            ),
            autofocus_settings=AutoFocusSettings(mode=self.autofocus_combo.value()),
            tile_order=self.tile_order_combo.value(),
        )

    def update_from_settings(self, settings: OverviewAcquisitionSettings):
        """Populate all widgets from an OverviewAcquisitionSettings object."""
        # Block tile-grid signals to prevent cascading updates
        for w in [self.beam_type_combo, self.nrows_spinbox, self.ncols_spinbox,
                  self.overlap_spinbox, self.focus_stack_enabled, self.focus_stack_steps,
                  self.focus_stack_autofocus, self.autofocus_combo, self.tile_order_combo]:
            w.blockSignals(True)

        self.beam_type_combo.set_value(settings.image_settings.beam_type)
        self.nrows_spinbox.setValue(settings.nrows)
        self.ncols_spinbox.setValue(settings.ncols)
        self.overlap_spinbox.setValue(settings.overlap * 100.0)
        self.focus_stack_enabled.setChecked(settings.focus_stack_settings.enabled)
        self.focus_stack_steps.setValue(settings.focus_stack_settings.n_steps)
        self.focus_stack_autofocus.setChecked(settings.focus_stack_settings.auto_focus)
        self.autofocus_combo.set_value(settings.autofocus_settings.mode)
        self.tile_order_combo.set_value(settings.tile_order)

        for w in [self.beam_type_combo, self.nrows_spinbox, self.ncols_spinbox,
                  self.overlap_spinbox, self.focus_stack_enabled, self.focus_stack_steps,
                  self.focus_stack_autofocus, self.autofocus_combo, self.tile_order_combo]:
            w.blockSignals(False)

        self.image_settings_widget.update_from_settings(settings.image_settings)
        self._update_total_fov_label()


# ---------------------------------------------------------------------------
# Quick test harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QPushButton

    app = QApplication(sys.argv)
    w = QWidget()
    layout = QVBoxLayout(w)

    widget = OverviewAcquisitionSettingsWidget()
    layout.addWidget(widget)

    btn = QPushButton("Print settings")

    def _print():
        s = widget.get_settings()
        print(s)

    btn.clicked.connect(_print)
    layout.addWidget(btn)

    w.setWindowTitle("OverviewAcquisitionSettingsWidget test")
    w.show()
    sys.exit(app.exec_())
