"""Acquire Image takes every channel in the list, one plane (FIB-943).

It used to take the selected channel only, so the one way to a single image with
all its channels in it was a two-plane z-stack. The selected channel is what the
live view streams; the list is what an acquisition takes, image or stack alike.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")


from fibsem.fm.structures import ChannelSettings  # noqa: E402
from fibsem.ui.widgets import fluorescence_control_widget as module  # noqa: E402
from fibsem.ui.widgets.fluorescence_control_widget import (  # noqa: E402
    FMControlWidget,
)
from tests.fixtures.fm_host import FMHost  # noqa: E402


@pytest.fixture()
def widget(qapp):
    from tests.fixtures.fm_microscope import build_microscope

    host = FMHost()
    w = FMControlWidget(
        microscope=build_microscope(),
        parent=host,
        view_controller=lambda: host.view_controller,
    )
    w.fm.objective.insert()
    w.acquisitionSettingsWidget.channel_settings = [
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

        def start(self, **_):
            widget.fm.set_acquiring(False)
            return True

    monkeypatch.setattr(module, "FunctionWorker", _Worker)
    monkeypatch.setattr(widget, "_acquisition_destination", lambda name: ("/tmp", name))
    return calls


def test_acquire_image_takes_every_channel_on_one_plane(widget, started):
    widget.pushButton_acquire_single_image.click()
    assert len(started) == 1
    channels, zparams, path, name = started[0]
    assert [c.name for c in channels] == ["one", "two", "three"]
    assert zparams is None
    assert name == "image"


def test_acquire_z_stack_takes_the_same_channels_with_planes(widget, started):
    widget.pushButton_acquire_zstack.click()
    channels, zparams, path, name = started[0]
    assert [c.name for c in channels] == ["one", "two", "three"]
    assert zparams is not None
    assert name == "z-stack"


def test_the_live_view_still_streams_the_selected_channel(widget, monkeypatch):
    streamed = []
    monkeypatch.setattr(
        widget.fm,
        "start_acquisition",
        lambda channel_settings: streamed.append(channel_settings),
    )
    channels = widget.acquisitionSettingsWidget.channel_settings
    widget.acquisitionSettingsWidget.channels_widget._list._set_selected(channels[1])
    widget.toggle_acquisition()
    assert [c.name for c in streamed] == ["two"]


def test_an_image_is_acquired_and_saved_through_the_imaging_service(
    widget, tmp_path, monkeypatch
):
    """FIB-1447: the button's worker is the service's acquisition, a run the agent
    can see, and the FM is free again after."""
    monkeypatch.setattr(
        widget, "_acquisition_destination", lambda name: (str(tmp_path), name)
    )
    finished = []
    widget.acquisition_finished_signal.connect(lambda: finished.append(True))

    widget._image_acquistion_worker(
        widget.acquisitionSettingsWidget.channel_settings[:1],
        None,
        str(tmp_path),
        "image",
    )

    assert [p.name for p in tmp_path.glob("image-*.ome.tiff")]
    assert widget.fm.is_acquiring is False
    assert widget.microscope.imaging.runs.latest.command == "fm_acquire_image"
    assert finished


def test_cancel_stops_the_service_run(widget, monkeypatch):
    stopped = []
    monkeypatch.setattr(widget.microscope.imaging, "stop", lambda: stopped.append(1))

    class _Alive:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    widget._acquisition_thread = _Alive()
    widget.cancel_acquisition()

    assert stopped == [1]
