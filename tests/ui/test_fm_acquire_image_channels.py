"""Acquire Image takes every channel in the list, one plane (FIB-943).

It used to take the selected channel only, so the one way to a single image with
all its channels in it was a two-plane z-stack. The selected channel is what the
live view streams; the list is what an acquisition takes, image or stack alike.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.fm.structures import ChannelSettings  # noqa: E402
from fibsem.ui.widgets import fluorescence_control_widget as module  # noqa: E402
from fibsem.ui.widgets.fluorescence_control_widget import (  # noqa: E402
    FMControlWidget,
    _DemoHost,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def widget(qapp):
    from fibsem.ui.fm.overview_app import build_microscope

    host = _DemoHost()
    w = FMControlWidget(microscope=build_microscope(), parent=host)
    w.fm.objective.insert()
    w.channelSettingsWidget.channel_settings = [
        ChannelSettings(name="one", excitation_wavelength=365.0),
        ChannelSettings(name="two", excitation_wavelength=450.0),
        ChannelSettings(name="three", excitation_wavelength=550.0),
    ]
    yield w
    w.close()
    w.deleteLater()
    qapp.processEvents()


@pytest.fixture()
def started(monkeypatch, widget):
    """Capture what the acquisition worker would be started with, without running it."""
    calls = []

    class _Worker:
        def __init__(self, fn, *args):
            calls.append(args)

        def start(self):
            widget.fm.set_acquiring(False)

    monkeypatch.setattr(module, "FunctionWorker", _Worker)
    monkeypatch.setattr(
        widget, "_generate_acquisition_filename", lambda name: f"/tmp/{name}.ome.tiff"
    )
    return calls


def test_acquire_image_takes_every_channel_on_one_plane(widget, started):
    widget.pushButton_acquire_single_image.click()
    assert len(started) == 1
    channels, zparams, filename = started[0]
    assert [c.name for c in channels] == ["one", "two", "three"]
    assert zparams is None
    assert filename.endswith("image.ome.tiff")


def test_acquire_z_stack_takes_the_same_channels_with_planes(widget, started):
    widget.pushButton_acquire_zstack.click()
    channels, zparams, filename = started[0]
    assert [c.name for c in channels] == ["one", "two", "three"]
    assert zparams is not None
    assert filename.endswith("z-stack.ome.tiff")


def test_the_live_view_still_streams_the_selected_channel(widget, monkeypatch):
    streamed = []
    monkeypatch.setattr(
        widget.fm,
        "start_acquisition",
        lambda channel_settings: streamed.append(channel_settings),
    )
    channels = widget.channelSettingsWidget.channel_settings
    widget.channelSettingsWidget._list._set_selected(channels[1])
    widget.toggle_acquisition()
    assert [c.name for c in streamed] == ["two"]
