"""
Quick test script for FibsemImageSettingsWidget with the new sub-widgets.

Run with:
    python fibsem/ui/widgets/tests/test-image-settings-widget.py
"""
import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

import napari
from fibsem import utils
from fibsem.structures import ImageSettings, BeamType
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget


class _MockMovementWidget:
    """Minimal stand-in for movement_widget used in _toggle_interactions."""
    def _toggle_interactions(self, enable: bool, caller: str = None):
        print(f"[movement_widget] _toggle_interactions(enable={enable}, caller={caller})")


class MockParent(QWidget):
    """Minimal parent that satisfies FibsemImageSettingsWidget requirements."""
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.movement_widget = _MockMovementWidget()


def main():
    app = QApplication(sys.argv)

    microscope, settings = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    image_settings = settings.image

    viewer = napari.Viewer()

    parent = MockParent(viewer=viewer)

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
        print(f"\n--- ImageSettings ---")
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

    viewer.window.add_dock_widget(widget, area="right", name="Image Settings")
    viewer.window.add_dock_widget(panel, area="right", name="Test Controls")

    napari.run()


if __name__ == "__main__":
    main()
