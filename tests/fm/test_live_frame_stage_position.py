"""Live frames take the stage's cached position; acquired images read it live."""

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
