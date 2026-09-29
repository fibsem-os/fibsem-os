"""The event buffer: sequence/eviction semantics, the long-poll, and the taps
fed by real signals — the Demo microscope's psygnal emissions and a real
HookManager firing lifecycle events."""

import json
import os
import threading
import time

import pytest

from fibsem.applications.autolamella.server.events import (
    EventBuffer,
    attach_microscope_taps,
    make_lifecycle_hook,
    to_plain,
)


def test_sequences_are_monotonic_and_filterable():
    buffer = EventBuffer()
    for i in range(5):
        buffer.append("tick", {"i": i})
    result = buffer.events_since(3)
    assert [e["seq"] for e in result["events"]] == [4, 5]
    assert result["latest_seq"] == 5
    assert result["oldest_available"] == 1
    json.dumps(result)


def test_eviction_is_a_visible_gap_not_silent_continuity():
    buffer = EventBuffer(maxlen=3)
    for i in range(10):
        buffer.append("tick", {"i": i})
    result = buffer.events_since(0)
    # A client that was at seq 0 can SEE it missed 1..7: oldest_available says so.
    assert result["oldest_available"] == 8
    assert [e["seq"] for e in result["events"]] == [8, 9, 10]


def test_long_poll_wakes_on_append():
    buffer = EventBuffer()

    def late_append():
        time.sleep(0.15)
        buffer.append("news", {"n": 1})

    thread = threading.Thread(target=late_append, daemon=True)
    start = time.monotonic()
    thread.start()
    result = buffer.wait_for(since=0, timeout=5.0)
    waited = time.monotonic() - start
    assert [e["kind"] for e in result["events"]] == ["news"]
    assert waited < 2.0  # woke on the append, not the timeout


def test_long_poll_times_out_empty():
    buffer = EventBuffer()
    start = time.monotonic()
    result = buffer.wait_for(since=0, timeout=0.2)
    assert result["events"] == []
    assert 0.15 <= time.monotonic() - start < 2.0


def test_appends_from_many_threads_never_lose_or_duplicate_a_seq():
    buffer = EventBuffer(maxlen=5000)
    threads = [
        threading.Thread(
            target=lambda: [buffer.append("t", {}) for _ in range(200)], daemon=True
        )
        for _ in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    result = buffer.events_since(0)
    seqs = [e["seq"] for e in result["events"]]
    assert seqs == list(range(1, 1001))


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


def test_microscope_taps_capture_real_emissions(microscope):
    from fibsem.structures import BeamType

    buffer = EventBuffer()
    disposers = attach_microscope_taps(buffer, microscope)
    try:
        # A real image through the real signal. Note: one-shot acquire_image
        # does NOT emit on the acquisition signals today -- only the streaming
        # worker does (a documented upstream gap) -- so the test emits the way
        # the worker would, with a genuinely acquired image.
        image = microscope.acquire_image(beam_type=BeamType.ION)
        microscope.fib_acquisition_signal.emit(image)
        from fibsem.structures import FibsemStagePosition

        microscope.move_stage_relative(
            FibsemStagePosition(
                x=1e-6, y=0.0, z=0.0, r=0.0, t=0.0, coordinate_system="RAW"
            )
        )
        result = buffer.events_since(0)
        kinds = [e["kind"] for e in result["events"]]
        assert "fib_acquisition" in kinds
        assert "stage_position_changed" in kinds
        acquisition = next(
            e for e in result["events"] if e["kind"] == "fib_acquisition"
        )
        assert acquisition["payload"]["beam_type"] == "ION"
        assert acquisition["payload"]["shape"] is not None
        assert "image_b64" not in json.dumps(acquisition)  # metadata, never pixels
        json.dumps(result)
    finally:
        for dispose in disposers:
            dispose()


def test_disposers_detach_the_taps(microscope):
    from fibsem.structures import BeamType

    buffer = EventBuffer()
    for dispose in attach_microscope_taps(buffer, microscope):
        dispose()
    image = microscope.acquire_image(beam_type=BeamType.ION)
    microscope.fib_acquisition_signal.emit(image)
    assert buffer.events_since(0)["events"] == []


def test_typed_milling_progress_serializes_through_the_real_signal(microscope):
    from fibsem.milling.progress import MillingProgress, MillingProgressStatus

    buffer = EventBuffer()
    disposers = attach_microscope_taps(buffer, microscope)
    try:
        microscope.milling_progress_signal.emit(
            MillingProgress(
                status=MillingProgressStatus.STAGE_UPDATE,
                stage_name="Rough Mill 01",
                current_stage=0,
                total_stages=2,
                remaining_time=42.0,
            )
        )
        event = buffer.events_since(0)["events"][-1]
        assert event["kind"] == "milling_progress"
        assert event["payload"]["status"] == MillingProgressStatus.STAGE_UPDATE.value
        assert event["payload"]["stage_name"] == "Rough Mill 01"
        assert event["payload"]["remaining_time"] == 42.0
        json.dumps(event)
    finally:
        for dispose in disposers:
            dispose()


def test_lifecycle_hook_feeds_the_buffer_through_a_real_hook_manager():
    from fibsem.hooks import HookContext, HookEvent, HookManager

    buffer = EventBuffer()
    manager = HookManager()
    manager.register(make_lifecycle_hook(buffer))
    manager.fire(
        HookContext(
            event=HookEvent.TASK_COMPLETED.value,
            task_name="Rough Milling",
            item_name="sunny-mole",
            tasks_remaining=3,
        )
    )
    event = buffer.events_since(0)["events"][0]
    assert event["kind"] == "task_completed"
    assert event["payload"]["item_name"] == "sunny-mole"
    assert event["payload"]["tasks_remaining"] == 3
    assert "lamella_name" not in event["payload"]
    json.dumps(event)


def test_to_plain_handles_the_awkward_types():
    import numpy as np

    from fibsem.milling.progress import MillingProgressStatus

    assert (
        to_plain(MillingProgressStatus.TASK_STARTED)
        == MillingProgressStatus.TASK_STARTED.value
    )
    from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

    assert to_plain(AutoLamellaTaskStatus.InProgress) == "InProgress"
    assert to_plain(np.float64(1.5)) == 1.5
    assert to_plain({"a": (1, 2)}) == {"a": [1, 2]}
