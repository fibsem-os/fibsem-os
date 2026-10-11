from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QWidget,
)

from fibsem.fm.structures import ZParameters, ZStackOrder
from fibsem.structures import get_fields_with_metadata
from fibsem.ui.tokens import (
    NEUTRAL_650,
)
from fibsem.ui.widgets.custom_widgets import (
    ValueComboBox,
    ValueSpinBox,
)
from fibsem.ui.widgets.form_builder import configure_spinbox

# How each value is shown, and its bounds: ZParameters' field metadata.
_META = get_fields_with_metadata(ZParameters)


class ZParametersWidget(QWidget):
    settings_changed = pyqtSignal(ZParameters)

    def __init__(self, z_parameters: ZParameters, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._z_parameters = z_parameters
        self.parent_widget = parent
        self.initUI()

    def initUI(self):

        # Z minimum
        self.label_zmin = QLabel(_META["zmin"]["label"], self)
        self.doubleSpinBox_zmin = ValueSpinBox(parent=self)
        configure_spinbox(self.doubleSpinBox_zmin, _META["zmin"])
        self.doubleSpinBox_zmin.setValue(self._shown("zmin"))
        self.doubleSpinBox_zmin.setKeyboardTracking(False)

        # Z maximum
        self.label_zmax = QLabel(_META["zmax"]["label"], self)
        self.doubleSpinBox_zmax = ValueSpinBox(parent=self)
        configure_spinbox(self.doubleSpinBox_zmax, _META["zmax"])
        self.doubleSpinBox_zmax.setValue(self._shown("zmax"))
        self.doubleSpinBox_zmax.setKeyboardTracking(False)

        # Z step
        self.label_zstep = QLabel(_META["zstep"]["label"], self)
        self.doubleSpinBox_zstep = ValueSpinBox(parent=self)
        configure_spinbox(self.doubleSpinBox_zstep, _META["zstep"])
        self.doubleSpinBox_zstep.setValue(self._shown("zstep"))
        self.doubleSpinBox_zstep.setKeyboardTracking(False)

        # Number of planes (calculated, read-only)
        self.label_num_planes = QLabel("Planes", self)
        self.label_num_planes_value = QLabel(self._calculate_num_planes(), self)
        self.label_num_planes_value.setStyleSheet(f"QLabel {{ color: {NEUTRAL_650}; }}")

        # Acquisition order
        self.label_order = QLabel("Order", self)
        self.combo_order = ValueComboBox(parent=self)
        self.combo_order.addItem("Channel-wise", ZStackOrder.CHANNEL)
        self.combo_order.addItem("Z-level-wise", ZStackOrder.Z_LEVEL)
        self.combo_order.setCurrentIndex(
            0 if self._z_parameters.order == ZStackOrder.CHANNEL else 1
        )
        self.combo_order.setToolTip(
            "Channel-wise: acquire all z-planes per channel\n"
            "Z-level-wise: acquire all channels per z-plane"
        )

        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.label_zmin, 0, 0)
        layout.addWidget(self.doubleSpinBox_zmin, 0, 1)
        layout.addWidget(self.label_zmax, 1, 0)
        layout.addWidget(self.doubleSpinBox_zmax, 1, 1)
        layout.addWidget(self.label_zstep, 2, 0)
        layout.addWidget(self.doubleSpinBox_zstep, 2, 1)
        layout.addWidget(self.label_num_planes, 3, 0)
        layout.addWidget(self.label_num_planes_value, 3, 1)
        layout.addWidget(self.label_order, 4, 0)
        layout.addWidget(self.combo_order, 4, 1)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the grid layout
        self.setLayout(layout)

        # connect signals
        self.doubleSpinBox_zmin.valueChanged.connect(self._on_zmin_changed)
        self.doubleSpinBox_zmax.valueChanged.connect(self._on_zmax_changed)
        self.doubleSpinBox_zstep.valueChanged.connect(self._on_zstep_changed)
        self.combo_order.currentIndexChanged.connect(self._on_order_changed)

    def _shown(self, name: str) -> float:
        """The parameter ``name`` as its box shows it (µm)."""
        return getattr(self._z_parameters, name) * _META[name]["scale"]

    @property
    def z_parameters(self) -> ZParameters:
        """Get the ZParameters instance."""
        return self._z_parameters

    @z_parameters.setter
    def z_parameters(self, value: ZParameters):
        """Set the ZParameters instance and update the display."""
        self._z_parameters = value
        self._update_num_planes_display()

        # Update spin boxes to reflect new parameters
        # block signals to prevent recursive updates
        self.doubleSpinBox_zmin.blockSignals(True)
        self.doubleSpinBox_zmax.blockSignals(True)
        self.doubleSpinBox_zstep.blockSignals(True)
        self.combo_order.blockSignals(True)
        self.doubleSpinBox_zmin.setValue(self._shown("zmin"))
        self.doubleSpinBox_zmax.setValue(self._shown("zmax"))
        self.doubleSpinBox_zstep.setValue(self._shown("zstep"))
        self.combo_order.setCurrentIndex(
            0 if self.z_parameters.order == ZStackOrder.CHANNEL else 1
        )
        self.doubleSpinBox_zmin.blockSignals(False)
        self.doubleSpinBox_zmax.blockSignals(False)
        self.doubleSpinBox_zstep.blockSignals(False)
        self.combo_order.blockSignals(False)

    def _calculate_num_planes(self) -> str:
        """Calculate the number of planes based on current parameters."""
        try:
            num_planes = self.z_parameters.num_planes
            if num_planes <= 0:
                return "Invalid"
            return f"{num_planes}"
        except (ValueError, ZeroDivisionError):
            return "Invalid"

    def _update_num_planes_display(self):
        """Update the number of planes display."""
        self.label_num_planes_value.setText(self._calculate_num_planes())

    def _on_zmin_changed(self, value: float):
        """Handle Z min value change."""
        self.z_parameters.zmin = value / _META["zmin"]["scale"]
        self._update_num_planes_display()

        # Ensure zmin <= zmax
        if self.z_parameters.zmin > self.z_parameters.zmax:
            self.doubleSpinBox_zmax.setValue(value)
        self.settings_changed.emit(self.z_parameters)

    def _on_zmax_changed(self, value: float):
        """Handle Z max value change."""
        self.z_parameters.zmax = value / _META["zmax"]["scale"]
        self._update_num_planes_display()

        # Ensure zmax >= zmin
        if self.z_parameters.zmax < self.z_parameters.zmin:
            self.doubleSpinBox_zmin.setValue(value)
        self.settings_changed.emit(self.z_parameters)

    def _on_zstep_changed(self, value: float):
        """Handle Z step value change."""
        self.z_parameters.zstep = value / _META["zstep"]["scale"]
        self._update_num_planes_display()
        self.settings_changed.emit(self.z_parameters)

    def _on_order_changed(self, index: int):
        """Handle acquisition order change."""
        self.z_parameters.order = self.combo_order.itemData(index)
        self.settings_changed.emit(self.z_parameters)
