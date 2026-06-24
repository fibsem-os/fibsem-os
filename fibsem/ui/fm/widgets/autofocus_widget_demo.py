"""Standalone demo for the dynamic multi-pass AutofocusWidget.

Run directly:
    python fibsem/ui/fm/widgets/autofocus_widget_demo.py

A checkbox toggles add/remove editing (hidden by default). A live readout
shows the current AutoFocusSettings as the user edits.
"""
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem.fm.structures import ChannelSettings
from fibsem.ui.fm.widgets.autofocus_widget import AutofocusWidget


def _format(settings) -> str:
    lines = [f"method = {settings.method.value}", f"channel = {settings.channel_name}"]
    for i, p in enumerate(settings.passes):
        state = "on " if p.enabled else "off"
        lines.append(
            f"  pass {i}: [{state}] range={p.search_range * 1e6:.1f} µm  "
            f"step={p.step_size * 1e6:.2f} µm  (n_steps={p.n_steps})"
        )
    return "\n".join(lines)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    channels = [
        ChannelSettings(name="Reflection", excitation_wavelength=550, emission_wavelength=None),
        ChannelSettings(name="GFP", excitation_wavelength=488, emission_wavelength="FLUORESCENCE"),
        ChannelSettings(name="mCherry", excitation_wavelength=550, emission_wavelength="FLUORESCENCE"),
    ]

    win = QWidget()
    win.setWindowTitle("AutofocusWidget — demo")
    win.setStyleSheet("background: #2b2d31; color: #d1d2d4;")
    root = QVBoxLayout(win)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(10)

    widget = AutofocusWidget(channel_settings=channels)
    root.addWidget(widget)

    edit_toggle = QCheckBox("Enable pass editing (show + / trash)")
    root.addWidget(edit_toggle)

    readout = QLabel()
    readout.setStyleSheet("font-family: monospace; color: #9fd3a0;")
    root.addWidget(readout)

    def refresh(*_):
        readout.setText(_format(widget.get_autofocus_settings()))

    edit_toggle.toggled.connect(widget.set_pass_editing_enabled)
    widget.settings_changed.connect(refresh)
    refresh()

    win.resize(460, 360)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
