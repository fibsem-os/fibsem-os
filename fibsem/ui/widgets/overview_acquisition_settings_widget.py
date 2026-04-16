from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from typing import Optional

from fibsem import constants
from fibsem.config import AVAILABLE_RESOLUTIONS_ZIP, DEFAULT_STANDARD_RESOLUTION
from fibsem.structures import AutoFocusMode, AutoFocusSettings, BeamType, ImageSettings, OverviewAcquisitionSettings, TileOrderStrategy
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel, ValueSpinBox, WheelBlocker
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
        self._setup_ui()
        self._connect_signals()
        self._update_total_fov_label()

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
        self.beam_type_combo = QComboBox()
        for bt in BeamType:
            self.beam_type_combo.addItem(bt.name, bt)
        self.beam_type_combo.installEventFilter(WheelBlocker(parent=self.beam_type_combo))
        grid_layout.addWidget(self.beam_type_combo, 0, 1, 1, 2)

        # Rows / cols
        grid_layout.addWidget(QLabel("Tiles (rows x cols)"), 1, 0)
        self.nrows_spinbox = QSpinBox()
        self.nrows_spinbox.setRange(1, 15)
        self.nrows_spinbox.setValue(3)
        self.nrows_spinbox.setKeyboardTracking(False)
        self.nrows_spinbox.installEventFilter(WheelBlocker(parent=self.nrows_spinbox))
        self.ncols_spinbox = QSpinBox()
        self.ncols_spinbox.setRange(1, 15)
        self.ncols_spinbox.setValue(3)
        self.ncols_spinbox.setKeyboardTracking(False)
        self.ncols_spinbox.installEventFilter(WheelBlocker(parent=self.ncols_spinbox))

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

        # Use focus stack
        grid_layout.addWidget(QLabel("Use Focus Stack"), 4, 0)
        self.use_focus_stack = QCheckBox()
        grid_layout.addWidget(self.use_focus_stack, 4, 1, 1, 2)

        # Auto Focus mode
        grid_layout.addWidget(QLabel("Auto Focus"), 5, 0)
        self.autofocus_combo = QComboBox()
        for mode in AutoFocusMode:
            self.autofocus_combo.addItem(mode.name.replace("_", " ").title(), mode)
        self.autofocus_combo.installEventFilter(WheelBlocker(parent=self.autofocus_combo))
        grid_layout.addWidget(self.autofocus_combo, 5, 1, 1, 2)

        # Tile order strategy
        grid_layout.addWidget(QLabel("Tile Order"), 6, 0)
        self.tile_order_combo = QComboBox()
        for strategy in TileOrderStrategy:
            self.tile_order_combo.addItem(strategy.value.title(), strategy)
        self.tile_order_combo.installEventFilter(WheelBlocker(parent=self.tile_order_combo))
        grid_layout.addWidget(self.tile_order_combo, 6, 1, 1, 2)

        grid_panel = TitledPanel("Overview Acquisition", content=grid_content)
        grid_panel._btn_collapse.setChecked(True)
        outer.addWidget(grid_panel)

        # ── Tile Image Settings panel ─────────────────────────────────
        self.image_settings_widget = ImageSettingsWidget(
            show_advanced=False,
            always_save=True,
        )
        self.image_settings_widget.hfw_label.setText("Field of View")
        self.image_settings_widget.set_show_advanced_button(False)

        # All standard + square resolutions are supported (non-square aspect handled in acquisition)
        self.image_settings_widget.resolution_combo.clear()
        for res_str, res in AVAILABLE_RESOLUTIONS_ZIP:
            self.image_settings_widget.resolution_combo.addItem(res_str, res)
        default_idx = self.image_settings_widget.resolution_combo.findText(DEFAULT_STANDARD_RESOLUTION)
        if default_idx >= 0:
            self.image_settings_widget.resolution_combo.setCurrentIndex(default_idx)

        self._btn_advanced_imaging = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )

        self._image_panel = TitledPanel("Tile Image Settings", content=self.image_settings_widget)
        self._image_panel.add_header_widget(self._btn_advanced_imaging)
        self._image_panel._btn_collapse.setChecked(True)
        outer.addWidget(self._image_panel)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.beam_type_combo.currentIndexChanged.connect(self._on_changed)
        self.nrows_spinbox.valueChanged.connect(self._on_changed)
        self._btn_advanced_imaging.toggled.connect(self.image_settings_widget.set_show_advanced)
        self.ncols_spinbox.valueChanged.connect(self._on_changed)
        self.overlap_spinbox.valueChanged.connect(self._on_changed)
        self.use_focus_stack.toggled.connect(self._on_changed)
        self.autofocus_combo.currentIndexChanged.connect(self._on_changed)
        self.tile_order_combo.currentIndexChanged.connect(self._on_changed)
        self.image_settings_widget.settings_changed.connect(self._on_changed)

    def _on_changed(self):
        self._update_total_fov_label()
        self.settings_changed.emit()

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
        image_settings.beam_type = self.beam_type_combo.currentData()
        return OverviewAcquisitionSettings(
            image_settings=image_settings,
            nrows=self.nrows_spinbox.value(),
            ncols=self.ncols_spinbox.value(),
            overlap=self.overlap_spinbox.value() / 100.0,
            use_focus_stack=self.use_focus_stack.isChecked(),
            autofocus_settings=AutoFocusSettings(mode=self.autofocus_combo.currentData()),
            tile_order=self.tile_order_combo.currentData(),
        )

    def update_from_settings(self, settings: OverviewAcquisitionSettings):
        """Populate all widgets from an OverviewAcquisitionSettings object."""
        # Block tile-grid signals to prevent cascading updates
        self.beam_type_combo.blockSignals(True)
        self.nrows_spinbox.blockSignals(True)
        self.ncols_spinbox.blockSignals(True)
        self.overlap_spinbox.blockSignals(True)
        self.use_focus_stack.blockSignals(True)
        self.autofocus_combo.blockSignals(True)
        self.tile_order_combo.blockSignals(True)

        idx = self.beam_type_combo.findData(settings.image_settings.beam_type)
        if idx >= 0:
            self.beam_type_combo.setCurrentIndex(idx)
        self.nrows_spinbox.setValue(settings.nrows)
        self.ncols_spinbox.setValue(settings.ncols)
        self.overlap_spinbox.setValue(settings.overlap * 100.0)
        self.use_focus_stack.setChecked(settings.use_focus_stack)
        idx = self.autofocus_combo.findData(settings.autofocus_settings.mode)
        if idx >= 0:
            self.autofocus_combo.setCurrentIndex(idx)
        idx = self.tile_order_combo.findData(settings.tile_order)
        if idx >= 0:
            self.tile_order_combo.setCurrentIndex(idx)

        self.beam_type_combo.blockSignals(False)
        self.nrows_spinbox.blockSignals(False)
        self.ncols_spinbox.blockSignals(False)
        self.overlap_spinbox.blockSignals(False)
        self.use_focus_stack.blockSignals(False)
        self.autofocus_combo.blockSignals(False)
        self.tile_order_combo.blockSignals(False)

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
