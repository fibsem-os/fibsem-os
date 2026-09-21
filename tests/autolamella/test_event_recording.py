"""The app's event stream and its record on disk (FIB-1031).

The rules that matter are the ones a failure would hide: a slow disk must never
slow the thread emitting the event (often the milling thread), a record lands in
the file of the experiment it belongs to, and a write cut short is detected
rather than half-read.
"""

import json
import os
import threading
import time
from datetime import datetime

import pytest

from fibsem import utils
from fibsem.applications.autolamella import event_recording
from fibsem.applications.autolamella.event_recording import (
    EVENTS_FILENAME,
    EventFileWriter,
    EventRecorder,
    read_events,
)
from fibsem.applications.autolamella.server.events import EventBuffer
from fibsem.structures import FibsemStagePosition


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    yield microscope
    microscope.disconnect()


def _move(microscope):
    microscope.get_stage_position()
    microscope.move_stage_relative(
        FibsemStagePosition(x=1e-6, y=0, z=0, r=0, t=0, coordinate_system="RAW")
    )


def _written(path):
    return list(read_events(path)) if path.exists() else []


# ── the buffer ───────────────────────────────────────────────────────────────


def test_every_record_says_when_with_its_utc_offset():
    buffer = EventBuffer()
    buffer.append("thing", {})
    (record,) = buffer.events_since(0)["events"]
    when = datetime.fromisoformat(record["t"])
    assert when.utcoffset() is not None
    assert abs(when.timestamp() - record["timestamp"]) < 1e-3


def test_a_stamp_adds_fields_but_cannot_replace_the_record_s_own():
    buffer = EventBuffer(stamp=lambda kind, payload: {"where": kind, "seq": -1})
    buffer.append("thing", {"a": 1})
    (record,) = buffer.events_since(0)["events"]
    assert record["where"] == "thing"
    assert record["seq"] == 1 and record["payload"] == {"a": 1}


def test_a_failing_stamp_costs_the_stamp_not_the_event():
    def stamp(kind, payload):
        raise RuntimeError("no context")

    buffer = EventBuffer(stamp=stamp)
    assert buffer.append("thing", {}) == 1
    assert buffer.events_since(0)["events"][0]["kind"] == "thing"


def test_subscribers_see_every_record_in_order_until_they_leave():
    buffer = EventBuffer()
    seen = []
    dispose = buffer.subscribe(lambda record: seen.append(record["seq"]))

    def failing(record):
        raise RuntimeError("a broken subscriber")

    buffer.subscribe(failing)
    threads = [
        threading.Thread(target=lambda: [buffer.append("x", {}) for _ in range(50)])
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == list(range(1, 201))
    dispose()
    buffer.append("x", {})
    assert len(seen) == 200


# ── the file ─────────────────────────────────────────────────────────────────


def test_records_land_in_the_file_that_was_current_when_they_were_written(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    writer = EventFileWriter()
    writer.write({"n": 0})  # no file yet: not written anywhere
    writer.set_path(first)
    writer.write({"n": 1})
    writer.set_path(second)
    writer.write({"n": 2})
    writer.close()
    assert [r["n"] for r in _written(first)] == [1]
    assert [r["n"] for r in _written(second)] == [2]


def test_a_stalled_disk_never_blocks_the_emitting_thread(tmp_path, monkeypatch):
    released = threading.Event()
    real_dumps = json.dumps

    def slow_dumps(*args, **kwargs):
        released.wait(10)
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(event_recording.json, "dumps", slow_dumps)
    path = tmp_path / EVENTS_FILENAME
    writer = EventFileWriter()
    writer.set_path(path)
    start = time.monotonic()
    for n in range(200):
        writer.write({"n": n})
    assert time.monotonic() - start < 0.5
    released.set()
    writer.close()
    assert [r["n"] for r in _written(path)] == list(range(200))


def test_when_the_disk_is_far_behind_records_are_dropped_not_waited_for(
    tmp_path, monkeypatch
):
    released = threading.Event()
    monkeypatch.setattr(event_recording, "_QUEUE_LIMIT", 5)
    real_dumps = json.dumps
    monkeypatch.setattr(
        event_recording.json,
        "dumps",
        lambda *a, **k: (released.wait(10), real_dumps(*a, **k))[1],
    )
    writer = EventFileWriter()
    writer.set_path(tmp_path / EVENTS_FILENAME)
    for n in range(50):
        writer.write({"n": n})
    assert writer.dropped > 0
    released.set()
    writer.close()


def test_a_file_that_cannot_be_written_is_logged_and_the_next_one_still_works(
    tmp_path, caplog
):
    good = tmp_path / EVENTS_FILENAME
    writer = EventFileWriter()
    writer.set_path(tmp_path / "missing-directory" / EVENTS_FILENAME)
    writer.write({"n": 1})
    writer.write({"n": 2})
    writer.set_path(good)
    writer.write({"n": 3})
    writer.close()
    assert [r["n"] for r in _written(good)] == [3]
    failures = [r for r in caplog.records if "could not record events" in r.message]
    assert len(failures) == 1  # once per file, not once per record


def test_a_write_cut_short_is_skipped_not_half_read(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    path.write_text('{"n": 1}\n\nnot json\n{"n": 2}\n{"n": 3, "cut', encoding="utf-8")
    assert [r["n"] for r in read_events(path)] == [1, 2]


# ── the recorder ──────────────────────────────────────────────────────────────


def test_the_recorder_writes_the_microscope_s_events_to_the_experiment(
    tmp_path, microscope
):
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        _move(microscope)
    finally:
        recorder.close()
    records = _written(tmp_path / EVENTS_FILENAME)
    moves = [r for r in records if r["kind"] == "stage_position_changed"]
    assert moves, [r["kind"] for r in records]
    assert all(r["session"] == recorder.session_id for r in records)


def test_each_event_says_where_in_the_run_it_happened(
    tmp_path, microscope, monkeypatch
):
    ref = microscope.experiment
    # What registering an experiment with the microscope sets.
    monkeypatch.setattr(ref, "id", "E1")
    monkeypatch.setattr(ref, "name", "an-experiment")
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        recorder.buffer.append("outside", {})
        ref.set_workflow_metadata(
            item_id="L1", item_name="01-lamella", task_id="T1", task_name="Mill"
        )
        recorder.buffer.append("inside", {})
        recorder.buffer.append("prompt_answered", {"answered_by": "agent"})
    finally:
        ref.clear_workflow_metadata()
        recorder.close()
    outside, inside, answered = _written(tmp_path / EVENTS_FILENAME)
    assert outside["item"] is None and outside["task"] is None
    assert outside["actor"] is None  # nothing says who: not a guess
    assert inside["item"] == {"id": "L1", "name": "01-lamella"}
    assert inside["task"] == {"id": "T1", "name": "Mill"}
    assert inside["actor"] == "task"
    assert answered["actor"] == "agent"
    assert inside["experiment"] == {"id": "E1", "name": "an-experiment"}


def test_the_lifecycle_hook_records_every_run_it_is_registered_for(
    tmp_path, microscope
):
    from fibsem.hooks import HookContext, HookEvent, HookManager

    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        for run in range(2):  # the app builds a new manager per run
            manager = HookManager()
            manager.register(recorder.lifecycle_hook)
            manager.fire(
                HookContext(event=HookEvent.TASK_STARTED.value, task_name=f"run {run}")
            )
    finally:
        recorder.close()
    started = [
        r["payload"]["task_name"]
        for r in _written(tmp_path / EVENTS_FILENAME)
        if r["kind"] == "task_started"
    ]
    assert started == ["run 0", "run 1"]


def test_a_task_that_has_finished_still_says_which_task_it_was(tmp_path, microscope):
    """task_completed fires after the task clears its context from the microscope;
    the event names the lamella and task from its own context instead."""
    from fibsem.hooks import HookContext, HookEvent, HookManager

    microscope.experiment.clear_workflow_metadata()
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        manager = HookManager()
        manager.register(recorder.lifecycle_hook)
        manager.fire(
            HookContext(
                event=HookEvent.TASK_COMPLETED.value,
                task_name="Mill Fiducial",
                task_id="T1",
                item_name="01-lamella",
                item_id="L1",
                experiment_id="E1",
                experiment_name="an-experiment",
            )
        )
    finally:
        recorder.close()
    (record,) = _written(tmp_path / EVENTS_FILENAME)
    assert record["item"] == {"id": "L1", "name": "01-lamella"}
    assert record["task"] == {"id": "T1", "name": "Mill Fiducial"}
    assert record["experiment"] == {"id": "E1", "name": "an-experiment"}
    assert record["actor"] == "task"


def test_switching_experiment_switches_the_file(tmp_path, microscope):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    recorder = EventRecorder(microscope)  # no experiment yet
    try:
        recorder.buffer.append("before any experiment", {})
        recorder.set_experiment(first)
        recorder.buffer.append("in the first", {})
        recorder.set_experiment(second)
        recorder.buffer.append("in the second", {})
    finally:
        recorder.close()
    assert [r["kind"] for r in _written(first / EVENTS_FILENAME)] == ["in the first"]
    assert [r["kind"] for r in _written(second / EVENTS_FILENAME)] == ["in the second"]


def test_closing_detaches_the_taps(tmp_path, microscope):
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    recorder.close()
    before = recorder.buffer.events_since(0)["latest_seq"]
    _move(microscope)
    assert recorder.buffer.events_since(0)["latest_seq"] == before
    assert not recorder.writer.alive
