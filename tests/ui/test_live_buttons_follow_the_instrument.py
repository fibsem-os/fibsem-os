"""The main window's live buttons follow the instrument, and Stop never holds the window.

Pausing a view in xT ends fibsem's live view on that view's own thread. The SEM/FIB
Start/Stop Acquisition button kept saying "Stop Acquisition", and clicking it then
started live view again. Stop also waited for the frame in flight on the GUI thread, as
the FM panel's Stop and Cancel did.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_live_buttons_follow_the_instrument.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")

import fibsem.config as _fibsem_config
from fibsem import utils
from fibsem.structures.settings import BeamType

_ARCTIS_CONFIG = os.path.join(
    os.path.dirname(_fibsem_config.__file__), "config", "sim-arctis-configuration.yaml"
)


def _wait_for(qapp, condition, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def image_widget(qapp):
    from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget

    microscope, settings = utils.setup_session(manufacturer="Demo")
    widget = FibsemImageSettingsWidget(
        microscope=microscope, image_settings=settings.image
    )
    widget.dual_beam_widget.sem_radio.setChecked(True)
    yield widget
    microscope.stop_acquisition()
    widget.close()


@pytest.fixture
def fm_widget(qapp):
    from PyQt5.QtWidgets import QWidget

    from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget

    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=_ARCTIS_CONFIG)
    host = QWidget()
    widget = FMControlWidget(microscope=microscope, parent=host)
    yield widget
    microscope.fm.stop_acquisition()
    widget.close()


class TestTheBeamButton:
    def test_it_follows_a_stop_made_elsewhere(self, qapp, image_widget):
        button = image_widget.pushButton_start_acquisition
        image_widget.toggle_live_acquisition()
        assert button.text() == "Stop Acquisition"
        beam = image_widget.microscope.beams[BeamType.ELECTRON]
        assert _wait_for(qapp, lambda: beam.is_live)

        # Ended off the GUI thread, as live view ends after a pause in xT.
        threading.Thread(target=beam.stop_live).start()

        assert _wait_for(qapp, lambda: button.text() == "Start Acquisition")
        assert image_widget.pushButton_acquire_sem_image.isEnabled()

        # So the next click starts live view, as the button says.
        image_widget.toggle_live_acquisition()
        assert button.text() == "Stop Acquisition"

    def test_stop_does_not_hold_the_gui_thread(self, qapp, image_widget, monkeypatch):
        microscope = image_widget.microscope
        button = image_widget.pushButton_start_acquisition
        image_widget.toggle_live_acquisition()
        assert _wait_for(qapp, lambda: microscope.is_acquiring)

        stop = microscope.stop_acquisition

        def slow_stop():
            time.sleep(1.0)  # a long frame in flight
            stop()

        monkeypatch.setattr(microscope, "stop_acquisition", slow_stop)

        started = time.monotonic()
        button.click()
        assert time.monotonic() - started < 0.5
        assert button.text() == "Stopping…"
        assert not button.isEnabled()

        assert _wait_for(qapp, lambda: button.text() == "Start Acquisition")
        assert button.isEnabled()
        assert not microscope.is_acquiring


def test_a_beam_signals_live_view_starting_and_ending():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    beam = microscope.beams[BeamType.ELECTRON]
    seen = []
    beam.changed.connect(lambda name, value: name == "live" and seen.append(value))

    beam.start_live()
    beam.stop_live()
    beam._live_thread.join(timeout=30)

    assert seen == [True, False]


class TestTheFMPanel:
    def test_stop_does_not_hold_the_gui_thread(self, qapp, fm_widget, monkeypatch):
        fm = fm_widget.fm
        button = fm_widget.pushButton_toggle_acquisition
        fm.start_acquisition(channel_settings=fm_widget.channel_settings)
        assert _wait_for(qapp, lambda: button.text() == "Stop Acquisition")

        stop = fm.stop_acquisition

        def slow_stop():
            time.sleep(1.0)
            stop()

        monkeypatch.setattr(fm, "stop_acquisition", slow_stop)

        started = time.monotonic()
        fm_widget.stop_acquisition()
        assert time.monotonic() - started < 0.5
        assert button.text() == "Stopping…"
        assert not button.isEnabled()

        assert _wait_for(qapp, lambda: button.text() == "Start Acquisition")
        assert not fm.is_streaming

    def test_cancel_does_not_wait_for_the_run(self, fm_widget, monkeypatch):
        stopped = []
        monkeypatch.setattr(
            fm_widget.microscope.imaging, "stop", lambda: stopped.append(1)
        )

        class _SlowRun:
            def is_alive(self):
                return True

            def join(self, timeout=None):
                time.sleep(timeout or 0)

        fm_widget._acquisition_thread = _SlowRun()
        started = time.monotonic()
        fm_widget.cancel_acquisition()

        assert stopped == [1]
        assert time.monotonic() - started < 0.5
