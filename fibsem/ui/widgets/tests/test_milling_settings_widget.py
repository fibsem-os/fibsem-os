"""Standalone test script for FibsemMillingSettingsWidget.

Run directly:
    python fibsem/ui/widgets/tests/test_milling_settings_widget.py
"""
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import utils
from fibsem.structures import FibsemMillingSettings
from fibsem.ui.widgets.milling_settings_widget import FibsemMillingSettingsWidget


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    settings = FibsemMillingSettings()

    win = QWidget()
    win.setWindowTitle("FibsemMillingSettingsWidget — test")
    win.setStyleSheet("background: #2b2d31; color: #d1d2d4;")

    root = QVBoxLayout(win)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(12)

    # --- controls row ---
    controls = QWidget()
    ctrl_layout = QFormLayout(controls)
    ctrl_layout.setContentsMargins(0, 0, 0, 0)

    mfr_combo = QComboBox()
    mfr_combo.addItems(["Demo (none)", "ThermoFisher", "Tescan"])
    ctrl_layout.addRow("Manufacturer:", mfr_combo)

    adv_check = QCheckBox("Show advanced")
    ctrl_layout.addRow("Advanced:", adv_check)

    root.addWidget(controls)

    # --- the widget under test ---
    widget = FibsemMillingSettingsWidget(microscope=microscope, settings=settings)
    root.addWidget(widget)

    # --- status ---
    status = QLabel("Change any control to see settings_changed.")
    status.setStyleSheet("color: #909090; font-style: italic;")
    root.addWidget(status)

    btn_get = QPushButton("Print get_settings()")
    root.addWidget(btn_get)

    # --- connections ---
    def on_mfr_changed(idx: int) -> None:
        mfr_map = {0: "Demo", 1: "ThermoFisher", 2: "Tescan"}
        widget.set_manufacturer(mfr_map[idx])

    def on_adv_changed(checked: bool) -> None:
        widget.set_advanced_visible(checked)

    def on_settings_changed(s: FibsemMillingSettings) -> None:
        status.setText(
            f"current={s.milling_current*1e12:.1f} pA  |  "
            f"voltage={s.milling_voltage/1e3:.1f} kV  |  "
            f"mode={s.patterning_mode}"
        )
        print("Settings changed:", s.to_dict())

    def on_print() -> None:
        s = widget.get_settings()
        print(s)

    mfr_combo.currentIndexChanged.connect(on_mfr_changed)
    adv_check.toggled.connect(on_adv_changed)
    widget.settings_changed.connect(on_settings_changed)
    btn_get.clicked.connect(on_print)

    # start with ThermoFisher selected
    mfr_combo.setCurrentIndex(1)

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
