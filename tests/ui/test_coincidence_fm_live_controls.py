"""The Coincidence Milling Viewer's live FM controls follow the stream, not the last click.

Pausing the FM in xT ends fibsem's live view from the acquisition thread. The viewer's
button kept saying "Stop Acquisition", and clicking it then started live view again,
unpausing the FM in xT. Stopping also waited for the frame in flight on the GUI thread,
and live frames updated the timelapse slider from the acquisition thread.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_coincidence_fm_live_controls.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt5")


@pytest.fixture
def viewer(qapp, tmp_path):
    from fibsem import config as cfg
    from fibsem import utils
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.ui.coincidence.viewer import (
        FluorescenceCoincidenceViewerWidget,
    )

    microscope, _ = utils.setup_session(
        config_path=os.path.join(
            cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
        )
    )
    widget = FluorescenceCoincidenceViewerWidget(
        microscope=microscope,
        experiment=Experiment(path=str(tmp_path), name="fm-live-controls-test"),
    )
    yield widget
    microscope.fm.stop_acquisition()
    widget.close()


def _wait_for(qapp, condition, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_the_button_follows_a_stop_made_elsewhere(qapp, viewer):
    fm = viewer.microscope.fm
    button = viewer.btn_toggle_fm_acquisition

    fm.start_acquisition()
    assert _wait_for(qapp, lambda: button.text() == "Stop Acquisition")

    # Ended off the GUI thread, as the acquisition thread does after a pause in xT.
    threading.Thread(target=fm.stop_acquisition).start()

    assert _wait_for(qapp, lambda: not fm.is_streaming)
    assert _wait_for(qapp, lambda: button.text() == "Start Acquisition")
    assert viewer._act_pause_acquisition.text() == "Resume Acquisition"


def test_stop_does_not_hold_the_gui_thread(qapp, viewer, monkeypatch):
    fm = viewer.microscope.fm
    button = viewer.btn_toggle_fm_acquisition
    fm.start_acquisition()
    assert _wait_for(qapp, lambda: button.text() == "Stop Acquisition")

    stop = fm.stop_acquisition

    def slow_stop():
        time.sleep(1.0)  # a long frame in flight
        stop()

    monkeypatch.setattr(fm, "stop_acquisition", slow_stop)

    started = time.monotonic()
    button.click()
    assert time.monotonic() - started < 0.5
    assert button.text() == "Stopping…"
    assert not button.isEnabled()

    button.click()  # a second click while stopping does nothing
    assert fm.is_streaming

    assert _wait_for(qapp, lambda: button.text() == "Start Acquisition")
    assert button.isEnabled()
    assert not fm.is_streaming


def test_live_frames_reach_the_widgets_on_the_gui_thread(qapp, viewer, monkeypatch):
    seen = []
    monkeypatch.setattr(
        viewer,
        "_maybe_accumulate_timelapse",
        lambda image: seen.append(threading.current_thread()),
    )
    frame = SimpleNamespace(data=np.zeros((16, 16), np.float32))

    worker = threading.Thread(target=viewer._on_fm_live_image, args=(frame,))
    worker.start()
    worker.join()

    assert _wait_for(qapp, lambda: bool(seen))
    assert seen == [threading.main_thread()]
