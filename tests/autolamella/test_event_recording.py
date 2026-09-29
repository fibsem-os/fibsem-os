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
    # A task running is not this thread acting for it: only its mark says so.
    assert inside["actor"] is None
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


def test_how_a_run_ends_is_the_task_s_and_its_start_is_whoever_started_it(
    tmp_path, microscope
):
    """Workflow events name no task and fire outside any task's mark. The
    ending is the run's own, so the task's (FIB-1079); the start is whoever
    started the run: here the operator (unmarked, in the app), and the agent
    through the server."""
    from fibsem.acting import AGENT, OPERATOR, acting
    from fibsem.hooks import HookContext, HookEvent, HookManager

    recorder = EventRecorder(
        microscope, experiment_path=tmp_path, default_actor=OPERATOR
    )
    manager = HookManager()
    manager.register(recorder.lifecycle_hook)
    ends = (
        HookEvent.WORKFLOW_COMPLETED,
        HookEvent.WORKFLOW_CANCELLED,
        HookEvent.WORKFLOW_STALLED,
        HookEvent.EXPERIMENT_COMPLETED,
    )
    try:
        manager.fire(HookContext(event=HookEvent.WORKFLOW_STARTED.value))
        for event in ends:
            manager.fire(HookContext(event=event.value))
        with acting(AGENT):  # a run the agent started, and stopped
            manager.fire(HookContext(event=HookEvent.WORKFLOW_STARTED.value))
            manager.fire(HookContext(event=HookEvent.WORKFLOW_CANCELLED.value))
    finally:
        recorder.close()
    actors = [(r["kind"], r["actor"]) for r in _written(tmp_path / EVENTS_FILENAME)]
    assert actors == [
        ("workflow_started", "operator"),
        ("workflow_completed", "task"),
        ("workflow_cancelled", "task"),
        ("workflow_stalled", "task"),
        ("experiment_completed", "task"),
        ("workflow_started", "agent"),
        ("workflow_cancelled", "task"),
    ]


def test_a_thread_s_mark_says_who_acted_and_unmarked_is_the_default(
    tmp_path, microscope
):
    from fibsem.acting import AGENT, OPERATOR, TASK, acting

    recorder = EventRecorder(
        microscope, experiment_path=tmp_path, default_actor=OPERATOR
    )
    try:
        with acting(TASK):
            recorder.buffer.append("by_the_task", {})
            with acting(AGENT):
                recorder.buffer.append("by_the_agent", {})
        recorder.buffer.append("unmarked", {})
    finally:
        recorder.close()
    actors = {r["kind"]: r["actor"] for r in _written(tmp_path / EVENTS_FILENAME)}
    assert actors == {
        "by_the_task": "task",
        "by_the_agent": "agent",
        "unmarked": "operator",
    }


def test_a_move_made_beside_a_running_task_is_the_operator_s(
    tmp_path, microscope, monkeypatch
):
    """The reason for marking threads: while a task runs, the app-wide record of
    what is running names it, and a move from the movement widget read it."""
    from fibsem.acting import OPERATOR
    from fibsem.applications.autolamella.structures import Lamella
    from fibsem.applications.autolamella.workflows.tasks.select_position import (
        SelectMillingPositionTask,
        SelectMillingPositionTaskConfig,
    )

    running, done = threading.Event(), threading.Event()

    def _run(self):  # the real task, paused mid-run for a move beside it
        _move(self.microscope)
        running.set()
        assert done.wait(10)

    monkeypatch.setattr(SelectMillingPositionTask, "_run", _run)
    lamella = Lamella(path=tmp_path / "01-test", number=1, petname="test")
    lamella.path.mkdir(parents=True)
    task = SelectMillingPositionTask(
        microscope=microscope,
        config=SelectMillingPositionTaskConfig(use_autofocus=False),
        lamella=lamella,
    )
    recorder = EventRecorder(
        microscope, experiment_path=tmp_path, default_actor=OPERATOR
    )
    worker = threading.Thread(target=task.run)
    try:
        worker.start()
        assert running.wait(10)
        _move(microscope)  # the operator, on another thread
        done.set()
        worker.join(10)
    finally:
        recorder.close()
    records = _written(tmp_path / EVENTS_FILENAME)
    task_move, operator_move = _of_kind(records, "stage_moved")
    assert task_move["actor"] == "task"
    assert operator_move["actor"] == "operator"
    # Both happened during the task's run on its lamella: that is where, not who.
    assert (
        task_move["item"]
        == operator_move["item"]
        == {
            "id": lamella.id,
            "name": lamella.name,
        }
    )


def test_a_request_to_the_agent_server_is_the_agent_s(tmp_path, microscope):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from fibsem.acting import OPERATOR
    from fibsem.server import AuthConfig, build_server

    app = build_server(
        microscope, auth=AuthConfig.generate(arm_hardware=True, token="t")
    )
    recorder = EventRecorder(
        microscope, experiment_path=tmp_path, default_actor=OPERATOR
    )
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/move_stage_relative",
                headers={"Authorization": "Bearer t"},
                json={
                    "position": FibsemStagePosition(
                        x=1e-6, y=0, z=0, r=0, t=0, coordinate_system="RAW"
                    ).to_dict()
                },
            )
        assert response.status_code == 200, response.text
        _move(microscope)  # the app's own, after
    finally:
        recorder.close()
    agent_move, operator_move = _of_kind(
        _written(tmp_path / EVENTS_FILENAME), "stage_moved"
    )
    assert agent_move["actor"] == "agent"
    assert operator_move["actor"] == "operator"


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


# ── facts for the record (FIB-1032) ──────────────────────────────────────────


def _of_kind(records, kind):
    return [r for r in records if r["kind"] == kind]


def test_an_acquisition_records_the_file_it_was_saved_to(tmp_path, microscope):
    from fibsem import acquire
    from fibsem.structures import BeamType, ImageSettings

    settings = ImageSettings(
        beam_type=BeamType.ION, filename="ref_test", path=str(tmp_path), save=True
    )
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        saved = acquire.new_image(microscope, settings)
        settings.save = False
        acquire.new_image(microscope, settings)
    finally:
        recorder.close()
    first, second = _of_kind(_written(tmp_path / EVENTS_FILENAME), "image_acquired")
    assert first["payload"]["path"] == saved.filepath
    # the name actually written: suffix and extension, which the log never had
    assert os.path.basename(first["payload"]["path"]) == "ref_test_ib.tif"
    assert first["payload"]["beam_type"] == "ION"
    assert first["payload"]["filename"] == "ref_test"
    assert first["payload"]["hfw"] == saved.metadata.image_settings.hfw
    assert second["payload"]["path"] is None  # not saved: nothing to point at
    assert len(json.dumps(first)) < 2000  # metadata, never pixels


def test_a_failing_subscriber_costs_the_record_not_the_acquisition(
    tmp_path, microscope
):
    from fibsem import acquire
    from fibsem.structures import BeamType, ImageSettings

    def broken(kind, payload):
        raise RuntimeError("a broken subscriber")

    microscope.record_signal.connect(broken)
    try:
        image = acquire.new_image(
            microscope,
            ImageSettings(
                beam_type=BeamType.ELECTRON, filename="x", path=str(tmp_path), save=True
            ),
        )
    finally:
        microscope.record_signal.disconnect(broken)
    assert os.path.exists(image.filepath)


def test_a_tap_that_fails_never_reaches_the_emitting_thread(microscope):
    """psygnal hands a subscriber's exception back to the emitter -- for milling
    progress, the milling thread. The taps swallow their own failures."""
    from fibsem.applications.autolamella.server.events import attach_microscope_taps
    from fibsem.milling.progress import MillingProgress, MillingProgressStatus

    class BrokenBuffer(EventBuffer):
        def append(self, kind, payload):
            raise RuntimeError("the buffer is broken")

    disposers = attach_microscope_taps(BrokenBuffer(), microscope)
    try:
        microscope.milling_progress_signal.emit(
            MillingProgress(status=MillingProgressStatus.STAGE_UPDATE)
        )
        microscope.record_event("anything", {})
        _move(microscope)  # stage_position_changed
    finally:
        for dispose in disposers:
            dispose()


def test_a_task_records_its_steps_and_every_image_it_saved(tmp_path, microscope):
    from fibsem.applications.autolamella.structures import Lamella
    from fibsem.applications.autolamella.workflows.tasks.select_position import (
        SelectMillingPositionTask,
        SelectMillingPositionTaskConfig,
    )

    lamella = Lamella(path=tmp_path / "01-test", number=1, petname="test")
    lamella.path.mkdir(parents=True)
    lamella.milling_pose = microscope.get_microscope_state()
    task = SelectMillingPositionTask(
        microscope=microscope,
        config=SelectMillingPositionTaskConfig(use_autofocus=False),
        lamella=lamella,
    )
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        task.run()
    finally:
        recorder.close()
    records = _written(tmp_path / EVENTS_FILENAME)

    steps = _of_kind(records, "task_step")
    assert steps
    assert not {"STARTED", "FINISHED"} & {s["payload"]["step"] for s in steps}
    for step in steps:
        assert step["payload"]["item_type"] == "lamella"
        assert step["item"] == {"id": lamella.id, "name": lamella.name}
        assert step["task"]["name"] == task.task_name
        assert step["actor"] == "task"  # the task's thread says so; no default here

    recorded = {
        r["payload"]["path"]
        for r in _of_kind(records, "image_acquired")
        if r["payload"]["path"]
    }
    on_disk = {str(p) for p in lamella.path.rglob("*.tif")}
    assert on_disk and on_disk == recorded


def test_a_milling_stage_is_recorded_with_its_patterns_before_it_mills(
    tmp_path, microscope
):
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.tasks import FibsemMillingTask, FibsemMillingTaskConfig

    names = ["Rough Mill 01", "Rough Mill 02"]
    config = FibsemMillingTaskConfig.from_stages(
        stages=[FibsemMillingStage(name=name) for name in names], name="Rough Milling"
    )
    config.alignment.enabled = False
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        FibsemMillingTask(microscope, config).run()
    finally:
        recorder.close()
    records = _written(tmp_path / EVENTS_FILENAME)

    started = _of_kind(records, "milling_stage_started")
    assert [r["payload"]["stage"]["name"] for r in started] == names
    for record in started:
        stage = record["payload"]["stage"]
        assert stage["pattern"]["name"]  # the geometry that was milled
        assert stage["milling"]["hfw"] == config.field_of_view  # as the task set it
        # the end is the progress signal's; the start comes first
        (finished,) = [
            r
            for r in _of_kind(records, "milling_progress")
            if r["payload"]["status"] == "stage-finished"
            and r["payload"]["stage_name"] == stage["name"]
        ]
        assert record["seq"] < finished["seq"]


def test_a_spot_burn_records_its_points_and_field_of_view(
    tmp_path, microscope, monkeypatch
):
    from fibsem.imaging.spot import SpotBurnSettings
    from fibsem.structures import BeamType, Point

    monkeypatch.setattr(time, "sleep", lambda *_: None)  # the exposure countdown
    field_of_view = microscope.get_field_of_view(BeamType.ION)
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        microscope.run_spot_burn(
            settings=SpotBurnSettings(
                coordinates=[Point(0.25, 0.5), Point(0.75, 0.5), Point(1.5, 0.5)],
                exposure_time=1.0,
                milling_current=1e-10,
            ),
            beam_type=BeamType.ION,
        )
    finally:
        recorder.close()
    records = _written(tmp_path / EVENTS_FILENAME)

    (started,) = _of_kind(records, "spot_burn_started")
    assert started["payload"]["coordinates"] == [[0.25, 0.5], [0.75, 0.5]]
    assert started["payload"]["dropped"] == 1  # outside the field: not burned
    # fractions of the field: this is what places them on another image
    assert started["payload"]["field_of_view"] == field_of_view
    (ended,) = [
        r
        for r in _of_kind(records, "spot_burn_progress")
        if r["payload"]["status"] == "finished"
    ]
    assert started["seq"] < ended["seq"]


def test_a_stage_that_cannot_be_described_still_mills(
    tmp_path, microscope, monkeypatch
):
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.strategy.standard import StandardMillingStrategy
    from fibsem.milling.tasks import FibsemMillingTask, FibsemMillingTaskConfig

    def broken(self, short=False):
        raise RuntimeError("cannot describe this stage")

    milled = []
    real_run = StandardMillingStrategy.run

    def run(self, *args, **kwargs):
        milled.append(True)
        monkeypatch.setattr(FibsemMillingStage, "to_dict", real_to_dict)
        return real_run(self, *args, **kwargs)

    config = FibsemMillingTaskConfig.from_stages(
        stages=[FibsemMillingStage(name="Rough Mill 01")], name="Rough Milling"
    )
    config.alignment.enabled = False
    real_to_dict = FibsemMillingStage.to_dict
    # broken only until the mill starts: the log line after it describes the stage too
    monkeypatch.setattr(FibsemMillingStage, "to_dict", broken)
    monkeypatch.setattr(StandardMillingStrategy, "run", run)
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        FibsemMillingTask(microscope, config).run()
    finally:
        recorder.close()
    assert milled == [True]
    assert not _of_kind(_written(tmp_path / EVENTS_FILENAME), "milling_stage_started")


# ── acquisitions outside new_image ───────────────────────────────────────────


def test_an_image_taken_as_the_beam_is_set_records_the_exact_file(tmp_path, microscope):
    """Saved at the name given -- no `_ib` suffix -- as the coincidence strategy's
    before and after images always were."""
    from fibsem import acquire
    from fibsem.structures import BeamType

    target = tmp_path / "pre-milling-fib-image.tif"
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        image = acquire.acquire_current_image(
            microscope, BeamType.ION, path=str(target)
        )
        acquire.acquire_current_image(microscope, BeamType.ION)
    finally:
        recorder.close()
    saved, unsaved = _of_kind(_written(tmp_path / EVENTS_FILENAME), "image_acquired")
    assert target.exists() and saved["payload"]["path"] == image.filepath == str(target)
    assert saved["payload"]["beam_type"] == "ION"
    assert unsaved["payload"]["path"] is None


def test_milling_s_final_image_is_recorded_after_its_stages(tmp_path, microscope):
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.tasks import FibsemMillingTask, FibsemMillingTaskConfig

    config = FibsemMillingTaskConfig.from_stages(
        stages=[FibsemMillingStage(name="Rough Mill 01")], name="Rough Milling"
    )
    config.alignment.enabled = False
    assert config.acquisition.acquire_final_image and not config.acquisition.enabled
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        FibsemMillingTask(microscope, config).run()
    finally:
        recorder.close()
    records = _written(tmp_path / EVENTS_FILENAME)
    (started,) = _of_kind(records, "milling_stage_started")
    final = _of_kind(records, "image_acquired")[-1]
    assert final["seq"] > started["seq"]
    assert final["payload"]["beam_type"] == "ION"
    assert final["payload"]["reduced_area"] is None  # a full frame: it ends the overlay


def test_a_coincidence_check_is_recorded_and_still_acquires_as_given(
    tmp_path, microscope, monkeypatch
):
    """Through `acquire` now, and still never autocontrasted or saved, whatever
    the settings ask -- as when it called the microscope directly."""
    from fibsem.alignment.coincidence import _default_image_settings, check_coincidence

    settings = _default_image_settings()
    settings.autocontrast, settings.save, settings.path = True, True, str(tmp_path)
    autocontrasted = []
    monkeypatch.setattr(
        microscope, "autocontrast", lambda *a, **k: autocontrasted.append(a)
    )
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        check_coincidence(microscope, image_settings=settings)
    finally:
        recorder.close()
    pair = _of_kind(_written(tmp_path / EVENTS_FILENAME), "image_acquired")
    assert [r["payload"]["beam_type"] for r in pair] == ["ELECTRON", "ION"]
    assert all(r["payload"]["path"] is None for r in pair)
    assert autocontrasted == [] and not list(tmp_path.glob("*.tif"))
