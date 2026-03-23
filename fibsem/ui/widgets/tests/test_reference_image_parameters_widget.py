"""Standalone test script for ReferenceImageParametersWidget.

Run directly:
    python fibsem/ui/widgets/tests/test_reference_image_parameters_widget.py
"""
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.structures import ReferenceImageParameters
from fibsem.ui.widgets.reference_image_parameters_widget import (
    ReferenceImageParametersWidget,
)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = QWidget()
    win.setWindowTitle("ReferenceImageParametersWidget — test")
    win.setStyleSheet("background: #2b2d31; color: #d1d2d4;")
    win.resize(500, 600)

    outer = QVBoxLayout(win)
    outer.setContentsMargins(12, 12, 12, 12)
    outer.setSpacing(8)

    widget = ReferenceImageParametersWidget()
    outer.addWidget(widget)

    status = QLabel("Change a setting to see settings_changed output.")
    status.setStyleSheet("color: #909090; font-style: italic;")
    status.setWordWrap(True)
    outer.addWidget(status)

    btn_row = QWidget()
    btn_layout = QHBoxLayout(btn_row)
    btn_layout.setContentsMargins(0, 0, 0, 0)
    btn_print = QPushButton("Print get_settings()")
    btn_load = QPushButton("Load defaults")
    btn_layout.addWidget(btn_print)
    btn_layout.addWidget(btn_load)
    outer.addWidget(btn_row)

    def on_settings_changed(s: ReferenceImageParameters) -> None:
        status.setText(
            f"settings_changed: sem={s.acquire_sem}, fib={s.acquire_fib}, "
            f"img1={s.acquire_image1} ({s.field_of_view1*1e6:.0f}µm), "
            f"img2={s.acquire_image2} ({s.field_of_view2*1e6:.0f}µm)"
        )

    def on_print() -> None:
        s = widget.get_settings()
        print(f"acquire_sem={s.acquire_sem}  acquire_fib={s.acquire_fib}")
        print(f"acquire_image1={s.acquire_image1}  fov1={s.field_of_view1*1e6:.1f} µm")
        print(f"acquire_image2={s.acquire_image2}  fov2={s.field_of_view2*1e6:.1f} µm")
        print(f"imaging={s.imaging}")

    def on_load() -> None:
        widget.update_from_settings(ReferenceImageParameters())
        status.setText("Loaded default ReferenceImageParameters")

    widget.settings_changed.connect(on_settings_changed)
    btn_print.clicked.connect(on_print)
    btn_load.clicked.connect(on_load)

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
