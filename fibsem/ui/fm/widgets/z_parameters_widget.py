
from typing import TYPE_CHECKING, Optional

from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QWidget,
)
from fibsem.fm.structures import ZParameters, ZStackOrder
from PyQt5.QtCore import pyqtSignal

Z_PARAMETERS_CONFIG = {
    "step_size": 0.1,  # µm
    "decimals": 2,  # number of decimal places
    "suffix": " µm",  # unit suffix
    "tooltips": {
        "zmin": "Minimum Z position relative to current position",
        "zmax": "Maximum Z position relative to current position", 
        "zstep": "Step size between Z positions",
    },
    "range": {
        "zmin": (-100.0, -0.25),  # µm
        "zmax": (0.25, 100.0),    # µm
        "zstep": (0.1, 10.0),     # µm
    },
}

if TYPE_CHECKING:
    from fibsem.ui.FMAcquisitionWidget import FMAcquisitionWidget

class ZParametersWidget(QWidget):

    settings_changed = pyqtSignal(ZParameters)

    def __init__(self, z_parameters: ZParameters, parent: Optional['FMAcquisitionWidget'] = None):
        super().__init__(parent)
        self._z_parameters = z_parameters
        self.parent_widget = parent
        self.initUI()

    def initUI(self):

        # Z minimum
        self.label_zmin = QLabel("Z Min", self)
        self.doubleSpinBox_zmin = QDoubleSpinBox(self)
        self.doubleSpinBox_zmin.setRange(*Z_PARAMETERS_CONFIG["range"]["zmin"])  # ±100 µm range
        self.doubleSpinBox_zmin.setValue(self.z_parameters.zmin * 1e6)  # Convert m to µm
        self.doubleSpinBox_zmin.setSingleStep(Z_PARAMETERS_CONFIG["step_size"])
        self.doubleSpinBox_zmin.setDecimals(Z_PARAMETERS_CONFIG["decimals"])
        self.doubleSpinBox_zmin.setSuffix(Z_PARAMETERS_CONFIG["suffix"])
        self.doubleSpinBox_zmin.setToolTip(Z_PARAMETERS_CONFIG["tooltips"]["zmin"])
        self.doubleSpinBox_zmin.setKeyboardTracking(False)

        # Z maximum
        self.label_zmax = QLabel("Z Max", self)
        self.doubleSpinBox_zmax = QDoubleSpinBox(self)
        self.doubleSpinBox_zmax.setRange(*Z_PARAMETERS_CONFIG["range"]["zmax"])  # ±100 µm range
        self.doubleSpinBox_zmax.setValue(self.z_parameters.zmax * 1e6)  # Convert m to µm
        self.doubleSpinBox_zmax.setSingleStep(Z_PARAMETERS_CONFIG["step_size"])
        self.doubleSpinBox_zmax.setDecimals(Z_PARAMETERS_CONFIG["decimals"])
        self.doubleSpinBox_zmax.setSuffix(Z_PARAMETERS_CONFIG["suffix"])
        self.doubleSpinBox_zmax.setToolTip(Z_PARAMETERS_CONFIG["tooltips"]["zmax"])
        self.doubleSpinBox_zmax.setKeyboardTracking(False)

        # Z step
        self.label_zstep = QLabel("Z Step", self)
        self.doubleSpinBox_zstep = QDoubleSpinBox(self)
        self.doubleSpinBox_zstep.setRange(*Z_PARAMETERS_CONFIG["range"]["zstep"])  # 0.1 to 10 µm range
        self.doubleSpinBox_zstep.setValue(self.z_parameters.zstep * 1e6)  # Convert m to µm
        self.doubleSpinBox_zstep.setSingleStep(Z_PARAMETERS_CONFIG["step_size"])
        self.doubleSpinBox_zstep.setDecimals(Z_PARAMETERS_CONFIG["decimals"])
        self.doubleSpinBox_zstep.setSuffix(Z_PARAMETERS_CONFIG["suffix"])
        self.doubleSpinBox_zstep.setToolTip(Z_PARAMETERS_CONFIG["tooltips"]["zstep"])
        self.doubleSpinBox_zstep.setKeyboardTracking(False)

        # Number of planes (calculated, read-only)
        self.label_num_planes = QLabel("Planes", self)
        self.label_num_planes_value = QLabel(self._calculate_num_planes(), self)
        self.label_num_planes_value.setStyleSheet("QLabel { color: #666666; }")

        # Acquisition order
        self.label_order = QLabel("Order", self)
        self.combo_order = QComboBox(self)
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
        self.doubleSpinBox_zmin.setValue(self.z_parameters.zmin * 1e6)  # Convert m to µm
        self.doubleSpinBox_zmax.setValue(self.z_parameters.zmax * 1e6)  # Convert m to µm
        self.doubleSpinBox_zstep.setValue(self.z_parameters.zstep * 1e6)  # Convert m to µm
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
        self.z_parameters.zmin = value * 1e-6  # Convert µm to m
        self._update_num_planes_display()
        
        # Ensure zmin <= zmax
        if self.z_parameters.zmin > self.z_parameters.zmax:
            self.doubleSpinBox_zmax.setValue(value)
        self.settings_changed.emit(self.z_parameters)

    def _on_zmax_changed(self, value: float):
        """Handle Z max value change."""
        self.z_parameters.zmax = value * 1e-6  # Convert µm to m
        self._update_num_planes_display()
        
        # Ensure zmax >= zmin
        if self.z_parameters.zmax < self.z_parameters.zmin:
            self.doubleSpinBox_zmin.setValue(value)
        self.settings_changed.emit(self.z_parameters)

    def _on_zstep_changed(self, value: float):
        """Handle Z step value change."""
        self.z_parameters.zstep = value * 1e-6  # Convert µm to m
        self._update_num_planes_display()
        self.settings_changed.emit(self.z_parameters)

    def _on_order_changed(self, index: int):
        """Handle acquisition order change."""
        self.z_parameters.order = self.combo_order.itemData(index)
        self.settings_changed.emit(self.z_parameters)