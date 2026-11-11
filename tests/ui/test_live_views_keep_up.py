"""Live FM views draw the newest frame at a steady rate instead of queueing every one.

During coincidence milling the FM live view sent up to 20 frames a second, and each one
was queued onto the GUI thread with the strategy's stats. A slow redraw built a backlog
that froze the window, Stop Milling with it.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_live_views_keep_up.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from fibsem.ui.qt.coalesce import Coalescer


def _wait_for(qapp, condition, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


class TestTheCoalescer:
    def test_only_the_newest_value_is_delivered(self, qapp):
        seen = []
        coalescer = Coalescer(seen.append, interval_ms=10_000)
        for value in range(5):
            coalescer.put(value)
        coalescer.flush()
        coalescer.flush()  # nothing new: nothing delivered
        assert seen == [4]
        coalescer.stop()

    def test_keep_all_delivers_every_value_in_order(self, qapp):
        seen = []
        coalescer = Coalescer(seen.append, interval_ms=10_000, keep_all=True)
        for value in range(3):
            coalescer.put(value)
        coalescer.flush()
        coalescer.flush()
        assert seen == [[0, 1, 2]]
        coalescer.stop()

    def test_values_from_another_thread_arrive_on_the_gui_thread(self, qapp):
        threads = []
        coalescer = Coalescer(
            lambda value: threads.append(threading.current_thread()), interval_ms=10
        )
        worker = threading.Thread(target=coalescer.put, args=(1,))
        worker.start()
        worker.join()
        assert _wait_for(qapp, lambda: bool(threads))
        assert threads == [threading.main_thread()]
        coalescer.stop()

    def test_a_failing_draw_does_not_raise_into_qt(self, qapp):
        def fail(value):
            raise ValueError("bad frame")

        coalescer = Coalescer(fail, interval_ms=10_000)
        coalescer.put(1)
        coalescer.flush()
        coalescer.stop()

    def test_stop_drops_what_is_waiting(self, qapp):
        seen = []
        coalescer = Coalescer(seen.append, interval_ms=10_000)
        coalescer.put(1)
        coalescer.stop()
        coalescer.flush()
        assert seen == []


def test_the_fm_live_view_pulls_at_most_once_per_exposure():
    from fibsem.drivers.autoscript.devices import AutoscriptFM

    fm = SimpleNamespace(
        camera=SimpleNamespace(exposure_time=SimpleNamespace(cached=0.1)),
        _last_live_pull=0.0,
    )
    pulls = []
    for _ in range(4):
        AutoscriptFM._pace_live(fm)
        pulls.append(time.monotonic())

    gaps = [later - earlier for earlier, later in zip(pulls, pulls[1:])]
    assert all(gap >= 0.095 for gap in gaps)


def test_a_long_exposure_still_hears_a_stop_within_a_second():
    from fibsem.drivers.autoscript.devices import LIVE_MAX_WAIT_S, AutoscriptFM

    fm = SimpleNamespace(
        camera=SimpleNamespace(exposure_time=SimpleNamespace(cached=30.0)),
        _last_live_pull=time.monotonic(),
    )
    started = time.monotonic()
    AutoscriptFM._pace_live(fm)
    assert time.monotonic() - started <= LIVE_MAX_WAIT_S + 0.1


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
        experiment=Experiment(path=str(tmp_path), name="live-views-keep-up-test"),
    )
    yield widget
    widget.close()


def _stats(value, **extra):
    return {
        "value": value,
        "drop_fraction": 1.0,
        "threshold_fraction": 0.5,
        "warmup_complete": True,
        **extra,
    }


class TestTheCoincidenceViewerStats:
    def test_every_value_reaches_the_plot(self, qapp, viewer, monkeypatch):
        plotted = []
        monkeypatch.setattr(viewer.line_plot_widget, "append_value", plotted.append)
        viewer._apply_intensity_stats([_stats(1.0), _stats(2.0), _stats(3.0)])
        assert plotted == [1.0, 2.0, 3.0]

    def test_a_drop_in_an_earlier_frame_still_shows(self, qapp, viewer, monkeypatch):
        states = []
        monkeypatch.setattr(viewer.run_border, "set_state", states.append)
        viewer._apply_intensity_stats([_stats(1.0, drop_detected=True), _stats(2.0)])
        assert states == ["waiting"]

    def test_the_chip_is_restyled_only_when_it_changes(self, qapp, viewer, monkeypatch):
        chip = viewer.label_threshold_chip
        viewer._on_intensity_stats(_stats(1.0))
        restyled = []
        set_style = chip.setStyleSheet
        monkeypatch.setattr(
            chip, "setStyleSheet", lambda s: (restyled.append(s), set_style(s))
        )

        viewer._on_intensity_stats(_stats(1.0))
        assert restyled == []

        viewer._on_intensity_stats(_stats(1.0, drop_fraction=0.4))
        assert len(restyled) == 1


def test_a_slow_draw_leaves_the_window_as_long_free(qapp):
    """A 15 ms exposure gives a new frame every tick; a frame that takes 300 ms to
    draw must not make the next draw start at once, or clicks wait behind draws."""

    def slow_draw(value):
        time.sleep(0.3)

    coalescer = Coalescer(slow_draw, interval_ms=100)
    coalescer.put(1)
    coalescer.flush()
    assert coalescer._timer.interval() >= 300

    coalescer._deliver = lambda value: None
    coalescer.put(2)
    coalescer.flush()
    assert coalescer._timer.interval() == 100  # fast again: back to the base rate
    coalescer.stop()


def _fm_frame(shape, pixel_size=1e-7):
    from fibsem.fm.structures import (
        FluorescenceChannelMetadata,
        FluorescenceImage,
        FluorescenceImageMetadata,
    )

    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    metadata = FluorescenceImageMetadata(
        pixel_size_x=pixel_size,
        pixel_size_y=pixel_size,
        resolution=(shape[-1], shape[-2]),
        channels=[FluorescenceChannelMetadata(name="reflection")],
    )
    return FluorescenceImage(data=data, metadata=metadata)


class TestLiveFramesAreBinnedForDisplay:
    def test_a_full_frame_is_drawn_at_most_1024_across(self):
        from fibsem.ui.widgets.fluorescence_control_widget import binned_for_display

        frame = _fm_frame((1, 1, 2048, 2048))
        shown = binned_for_display(frame)

        assert shown.data.shape == (1, 1, 1024, 1024)
        assert shown.data.dtype == np.uint16
        # The same field of view: the scalebar and click-to-move stay right.
        assert shown.metadata.pixel_size_x == pytest.approx(2e-7)
        assert shown.metadata.pixel_size_y == pytest.approx(2e-7)
        assert shown.metadata.resolution == (1024, 1024)
        # Mean binning keeps the brightness.
        assert float(shown.data.mean()) == pytest.approx(
            float(frame.data.mean()), rel=1e-3
        )
        # The acquired frame is untouched.
        assert frame.data.shape == (1, 1, 2048, 2048)
        assert frame.metadata.pixel_size_x == pytest.approx(1e-7)

    def test_small_frames_and_z_stacks_are_shown_as_they_are(self):
        from fibsem.ui.widgets.fluorescence_control_widget import binned_for_display

        small = _fm_frame((1, 1, 512, 512))
        stack = _fm_frame((1, 5, 2048, 2048))
        assert binned_for_display(small) is small
        assert binned_for_display(stack) is stack

    def test_an_odd_size_is_cropped_to_whole_bins(self):
        from fibsem.ui.widgets.fluorescence_control_widget import binned_for_display

        shown = binned_for_display(_fm_frame((1, 1, 2049, 3001)))
        assert shown.data.shape == (1, 1, 683, 1000)
        assert shown.metadata.pixel_size_x == pytest.approx(3e-7)


def test_the_fm_panel_draws_the_binned_frame_and_clicks_by_it(qapp, monkeypatch):
    from PyQt5.QtWidgets import QWidget

    import fibsem.config as _fibsem_config
    from fibsem import utils
    from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget

    config = os.path.join(
        os.path.dirname(_fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=config)
    host = QWidget()
    widget = FMControlWidget(microscope=microscope, parent=host)
    drawn = []
    monkeypatch.setattr(
        widget,
        "_view_controller",
        lambda: SimpleNamespace(set_fm_image=drawn.append),
    )
    try:
        widget.update_image(_fm_frame((1, 1, 2048, 2048)))
        assert drawn[0].data.shape[-2:] == (1024, 1024)
        # Double-click maps canvas pixels with the shape and pixel size drawn.
        assert widget._fm_image_shape == (1024, 1024)
        assert widget._fm_pixelsize == pytest.approx(2e-7)
    finally:
        monkeypatch.undo()  # closing asks the real controller
        widget.close()
