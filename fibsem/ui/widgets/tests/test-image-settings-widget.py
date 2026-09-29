"""
Quick test script for FibsemImageSettingsWidget with the new sub-widgets.

Run with:
    python fibsem/ui/widgets/tests/test-image-settings-widget.py
"""
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fibsem import utils
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController


class _MockMovementWidget:
    """Minimal stand-in for movement_widget used in _toggle_interactions."""
    def _toggle_interactions(self, enable: bool, caller: str = None):
        print(f"[movement_widget] _toggle_interactions(enable={enable}, caller={caller})")


class MockParent(QWidget):
    """Minimal parent that satisfies FibsemImageSettingsWidget requirements.

    The widget resolves its display through ``view_controller`` on the parent — the
    shape standalone FibsemUI presents. It has no napari path left.
    """
    def __init__(self):
        super().__init__()
        self.view_controller = MicroscopeViewController(parent=self)
        self.movement_widget = _MockMovementWidget()


def main():
    app = QApplication(sys.argv)

    microscope, settings = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    image_settings = settings.image

    parent = MockParent()

    widget = FibsemImageSettingsWidget(
        microscope=microscope,
        image_settings=image_settings,
        parent=parent,
    )

    # --- Diagnostic buttons ---
    panel = QWidget()
    layout = QVBoxLayout(panel)

    def print_image_settings():
        s = widget._get_image_settings_from_ui()
        print("\n--- ImageSettings ---")
        print(f"  beam_type:    {s.beam_type}")
        print(f"  resolution:   {s.resolution}")
        print(f"  dwell_time:   {s.dwell_time*1e6:.2f} µs")
        print(f"  hfw:          {s.hfw*1e6:.1f} µm")
        print(f"  save:         {s.save}")
        print(f"  autocontrast: {s.autocontrast}")
        print(f"  line_int:     {s.line_integration}")

    def print_beam_settings():
        s = widget._get_beam_settings_from_ui()
        print(f"\n--- BeamSettings ({widget.dual_beam_widget.beam_type.name}) ---")
        for k, v in s.to_dict().items():
            print(f"  {k}: {v}")

    def print_detector_settings():
        s = widget._get_detector_settings_from_ui()
        print(f"\n--- DetectorSettings ({widget.dual_beam_widget.beam_type.name}) ---")
        print(f"  type:       {s.type}")
        print(f"  mode:       {s.mode}")
        print(f"  brightness: {s.brightness:.2f}")
        print(f"  contrast:   {s.contrast:.2f}")

    btn_img = QPushButton("Print ImageSettings")
    btn_img.clicked.connect(print_image_settings)
    layout.addWidget(btn_img)

    btn_beam = QPushButton("Print BeamSettings")
    btn_beam.clicked.connect(print_beam_settings)
    layout.addWidget(btn_beam)

    btn_det = QPushButton("Print DetectorSettings")
    btn_det.clicked.connect(print_detector_settings)
    layout.addWidget(btn_det)

    # quad view (left) | widget + diagnostics (right), as the real host lays it out
    right = QWidget()
    right_layout = QVBoxLayout(right)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.addWidget(widget)
    right_layout.addWidget(panel)

    window = QSplitter(Qt.Horizontal)
    window.addWidget(parent.view_controller.widget)
    window.addWidget(right)
    window.setSizes([720, 460])
    window.resize(1280, 800)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
