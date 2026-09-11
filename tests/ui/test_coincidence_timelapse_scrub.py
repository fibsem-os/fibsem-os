"""The timelapse scrubber stays where the operator put it while a run accumulates frames.

FIB-968: the viewer adds a frame to its timelapse every TIMELAPSE_INTERVAL seconds. The
display path already stood aside while the operator was scrubbing back through earlier
frames; the accumulation path did not, and every interval it moved the slider to the end
and rewrote the label with the newest timestamp, while the canvas kept showing the
earlier frame. The slider's range still has to grow so the new frame is reachable.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_coincidence_timelapse_scrub.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt5")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(qapp, tmp_path):
    from fibsem import config as cfg
    from fibsem import utils
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.ui.fluorescence_coincidence_viewer_widget import (
        FluorescenceCoincidenceViewerWidget,
    )

    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    )
    widget = FluorescenceCoincidenceViewerWidget(
        microscope=microscope,
        experiment=Experiment(path=str(tmp_path), name="timelapse-scrub-test"),
    )
    yield widget
    widget.close()


def _accumulate(viewer):
    viewer._last_timelapse_time = 0.0  # make the interval elapse
    frame = SimpleNamespace(data=np.zeros((16, 16), np.float32))
    viewer._maybe_accumulate_timelapse(frame)


def test_scrubbing_holds_the_slider_and_label_while_frames_accumulate(viewer):
    for _ in range(3):
        _accumulate(viewer)
    viewer.fm_canvas.time_slider.setValue(0)  # drag the scrubber back
    assert viewer._is_scrubbing
    label_before = viewer.fm_canvas.frame_label.text()

    _accumulate(viewer)

    assert len(viewer._timelapse_frames) == 4
    assert viewer.fm_canvas.time_slider.maximum() == 3  # the new frame is reachable
    assert viewer.fm_canvas.time_slider.value() == 0
    assert viewer.fm_canvas.frame_label.text() == label_before


def test_live_view_still_follows_the_newest_frame(viewer):
    for _ in range(3):
        _accumulate(viewer)
    assert not viewer._is_scrubbing
    assert viewer.fm_canvas.time_slider.value() == 2
    assert viewer.fm_canvas.frame_label.text().endswith("(2/2)")
