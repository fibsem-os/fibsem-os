"""Standalone test script for FluorescenceMultiChannelWidget.

Run directly:
    python fibsem/ui/fm/widgets/test_channel_list_widget.py
"""
import sys

from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings
from fibsem.ui.fm.widgets.fm_multi_channel_widget import FluorescenceMultiChannelWidget


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    fm = FluorescenceMicroscope()

    channels = [
        ChannelSettings(name="DAPI",     excitation_wavelength=365, emission_wavelength=None, power=0.05, exposure_time=0.050, color="blue",   gain=0.5),
        ChannelSettings(name="GFP",      excitation_wavelength=450, emission_wavelength="Fluorescence", power=0.10, exposure_time=0.100, color="cyan",   gain=0.6),
        ChannelSettings(name="mCherry",  excitation_wavelength=550, emission_wavelength=None, power=0.20, exposure_time=0.200, color="red",    gain=0.7),
    ]

    win = QWidget()
    win.setWindowTitle("FluorescenceMultiChannelWidget — test")
    win.setStyleSheet("background: #2b2d31; color: #d1d2d4;")

    root = QVBoxLayout(win)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(12)

    widget = FluorescenceMultiChannelWidget(fm=fm, channel_settings=channels)
    root.addWidget(widget)

    status = QLabel("Click a row to select a channel")
    status.setStyleSheet("color: #909090; font-style: italic;")
    root.addWidget(status)

    def on_field_changed(channel: ChannelSettings, field: str, value) -> None:
        print(f"Field changed: {channel.name}  field={field}  value={value}")
        status.setText(f"Changed: {channel.name}.{field} = {value}")

    def on_removed(channel: ChannelSettings) -> None:
        status.setText(f"Removed: {channel.name}")

    def on_added(channel: ChannelSettings) -> None:
        status.setText(f"Added: {channel.name}")

    def on_enabled(enabled_channels) -> None:
        print("Enabled:", [c.name for c in enabled_channels])

    def on_order(ordered_channels) -> None:
        print("Order:", [c.name for c in ordered_channels])

    widget.channel_field_changed.connect(on_field_changed)
    widget.channel_removed.connect(on_removed)
    widget.channel_added.connect(on_added)
    widget.enabled_changed.connect(on_enabled)
    widget.order_changed.connect(on_order)

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
