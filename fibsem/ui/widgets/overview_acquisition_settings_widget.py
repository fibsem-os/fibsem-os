from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon
from typing import Optional

from fibsem import constants
from fibsem.config import SQUARE_RESOLUTIONS_ZIP, DEFAULT_SQUARE_RESOLUTION
from fibsem.structures import BeamType, ImageSettings, OverviewAcquisitionSettings
from fibsem.ui.widgets.custom_widgets import WheelBlocker
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

        # --- Tile grid group ---
        self._grid_group = QGroupBox("Overview Acquisition")
        grid_layout = QGridLayout(self._grid_group)
        grid_layout.setContentsMargins(6, 6, 6, 6)

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

        # Total FOV label (read-only, auto-updated)
        self._label_total_fov = QLabel("Total FOV: —")
        self._label_total_fov.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore
        grid_layout.addWidget(self._label_total_fov, 2, 0, 1, 3)

        outer.addWidget(self._grid_group)

        # --- Per-tile image settings ---
        self._image_group = QGroupBox("Tile Image Settings")
        _img_layout = QVBoxLayout(self._image_group)
        _img_layout.setContentsMargins(6, 6, 6, 6)

        self.image_settings_widget = ImageSettingsWidget(
            show_advanced=False,
            always_save=True,
        )
        # Rename the HFW label to make clear it's the per-tile FOV
        self.image_settings_widget.hfw_label.setText("Field of View")

        # Tiled acquisition requires square resolutions (equal x/y pixel size for stitching)
        self.image_settings_widget.resolution_combo.clear()
        for res_str, res in SQUARE_RESOLUTIONS_ZIP:
            self.image_settings_widget.resolution_combo.addItem(res_str, res)
        default_idx = self.image_settings_widget.resolution_combo.findText(DEFAULT_SQUARE_RESOLUTION)
        if default_idx >= 0:
            self.image_settings_widget.resolution_combo.setCurrentIndex(default_idx)

        _img_layout.addWidget(self.image_settings_widget)
        outer.addWidget(self._image_group)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.beam_type_combo.currentIndexChanged.connect(self._on_changed)
        self.nrows_spinbox.valueChanged.connect(self._on_changed)
        self.ncols_spinbox.valueChanged.connect(self._on_changed)
        self.image_settings_widget.settings_changed.connect(self._on_changed)

    def _on_changed(self):
        self._update_total_fov_label()
        self.settings_changed.emit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_total_fov_label(self):
        tile_fov_um = self.image_settings_widget.hfw_spinbox.value()  # already in µm
        nrows = self.nrows_spinbox.value()
        ncols = self.ncols_spinbox.value()
        total_w = ncols * tile_fov_um
        total_h = nrows * tile_fov_um
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
            overlap=0.0,
        )

    def update_from_settings(self, settings: OverviewAcquisitionSettings):
        """Populate all widgets from an OverviewAcquisitionSettings object."""
        # Block tile-grid signals to prevent cascading updates
        self.beam_type_combo.blockSignals(True)
        self.nrows_spinbox.blockSignals(True)
        self.ncols_spinbox.blockSignals(True)

        idx = self.beam_type_combo.findData(settings.image_settings.beam_type)
        if idx >= 0:
            self.beam_type_combo.setCurrentIndex(idx)
        self.nrows_spinbox.setValue(settings.nrows)
        self.ncols_spinbox.setValue(settings.ncols)

        self.beam_type_combo.blockSignals(False)
        self.nrows_spinbox.blockSignals(False)
        self.ncols_spinbox.blockSignals(False)

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
