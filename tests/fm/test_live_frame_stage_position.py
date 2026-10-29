"""Live frames take the stage position and channel settings from caches; acquired
images read them live."""

import threading
import traceback

import pytest

from tests.fixtures.fm_microscope import build_microscope


@pytest.fixture
def counted():
    microscope = build_microscope()
    microscope.get_stage_position()  # fills the cache
    reads = []
    read = microscope.get_stage_position

    def counting():
        # The simulator reads the stage to render its scene; only the metadata's
        # reads count.
        if any(f.name == "_stage_position" for f in traceback.extract_stack()):
            reads.append(threading.current_thread().name)
        return read()

    microscope.get_stage_position = counting
    return microscope, reads


def test_a_live_frame_reads_no_stage_position(counted):
    """A read per frame was a stage call at the camera's frame rate (about 20 a second
    on a METEOR, 2026-10-29) for frames that are only displayed."""
    microscope, reads = counted

    image = microscope.fm._acquire(live=True)

    assert reads == []
    assert image.metadata.stage_position == microscope.stage.position.cached


def test_an_acquired_image_still_reads_the_stage(counted):
    microscope, reads = counted

    image = microscope.fm.acquire_image()

    assert len(reads) == 1
    assert image.metadata.stage_position is not None


def test_live_view_reads_no_stage_position_per_frame(counted):
    microscope, reads = counted
    frames = []
    done = threading.Event()

    def on_frame(image):
        frames.append(image)
        if len(frames) >= 3:
            done.set()

    microscope.fm.acquisition_signal.connect(on_frame)
    microscope.fm.start_acquisition()
    try:
        assert done.wait(10.0), "live view produced no frames"
    finally:
        microscope.fm.stop_acquisition()

    assert reads == []


# The channel's settings: about ten more reads per live frame


@pytest.fixture
def parameter_reads(monkeypatch):
    from fibsem.devices.core import BoundParameter

    reads = []
    get_value = BoundParameter.get_value

    def counting(self):
        if any(f.name == "_metadata_for_frame" for f in traceback.extract_stack()):
            reads.append(f"{self.device.name}.{self.name}")
        return get_value(self)

    monkeypatch.setattr(BoundParameter, "get_value", counting)
    return reads


def test_a_live_frame_reads_no_channel_settings(counted, parameter_reads):
    microscope, _ = counted
    microscope.fm._acquire(live=True)  # the first takes the frame size
    parameter_reads.clear()

    image = microscope.fm._acquire(live=True)

    assert parameter_reads == []
    channel = image.metadata.channels[0]
    assert channel.exposure_time == microscope.fm.camera.exposure_time
    assert channel.objective_position == microscope.fm.objective.position


def test_an_acquired_image_takes_no_cached_settings(counted, monkeypatch):
    microscope, _ = counted
    monkeypatch.setattr(
        microscope.fm, "_cached_settings", lambda: pytest.fail("used the caches")
    )

    microscope.fm.acquire_image()


def test_a_live_frame_carries_a_setting_written_since(counted):
    microscope, _ = counted
    fm = microscope.fm
    fm._acquire(live=True)

    fm.camera.exposure_time = 0.2
    fm.objective.move_relative(1e-6)
    image = fm._acquire(live=True)

    channel = image.metadata.channels[0]
    assert channel.exposure_time == pytest.approx(0.2)
    assert channel.objective_position == pytest.approx(fm.objective.position)


def test_a_binning_change_re_reads_the_frame_size(counted, parameter_reads):
    microscope, _ = counted
    fm = microscope.fm
    fm._acquire(live=True)
    other = next(b for b in fm.camera.available_binnings if b != fm.camera.binning)

    fm.camera.binning = other
    parameter_reads.clear()
    image = fm._acquire(live=True)

    assert sorted(parameter_reads) == ["camera.pixel_size", "camera.resolution"]
    assert image.metadata.pixel_size_x == pytest.approx(fm.camera.pixel_size[0])
