"""Reading an experiment's log back as a replay timeline.

Two halves. The first runs the real producers on the Demo microscope and
replays what they wrote: the only way to notice a producer changing the shape
of a record the replay reads. The second pins the reader's own decisions --
what it refuses to evaluate, how it finds an image in a copied experiment,
what it does when a later image overwrote an earlier one.
"""

import math
from datetime import datetime, timedelta

import pytest

from fibsem.applications.autolamella.event_recording import (
    EVENTS_FILENAME,
    read_events,
)
from fibsem.applications.autolamella.tools.replay import (
    EventKind,
    load_replay,
    parse_record,
    read_log_records,
)
from fibsem.milling.base import FibsemMillingStage

from ._replay_experiment import (
    FM_CHANNELS,
    FM_PLANES,
    FM_STACK,
    LAMELLA,
    MILLING_STAGES,
    MILLING_TASK,
    SPOTS,
    record_demo_experiment,
)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    try:
        root = record_demo_experiment(
            tmp_path_factory.mktemp("replay") / "exp", monkeypatch
        )
    finally:
        monkeypatch.undo()
    return load_replay(root)


def _of(replay, kind):
    return [(i, e) for i, e in enumerate(replay.events) if e.kind == kind]


# ── what the real producers write ────────────────────────────────────────────


def test_every_record_the_replay_reads_is_readable(recorded):
    assert recorded.records_read > 0
    assert recorded.records_unreadable == 0


def test_the_task_is_replayed_step_by_step_with_its_lamella(recorded):
    steps = [e.step for _, e in _of(recorded, EventKind.TASK)]
    assert steps[0] == "STARTED" and steps[-1] == "FINISHED"
    assert "ACQUIRE_REFERENCE_IMAGES" in steps
    assert all(e.item == "test" for _, e in _of(recorded, EventKind.TASK))
    assert all(e.item_type == "lamella" for e in recorded.events if e.item)


def test_every_saved_image_is_found_on_disk(recorded):
    images = [e for _, e in _of(recorded, EventKind.IMAGE)]
    saved = [e for e in images if e.saved]
    assert saved, "the task saves its reference images"
    assert all(e.image_on_disk for e in saved), [
        e.summary for e in saved if not e.image_on_disk
    ]
    assert {e.beam for e in saved} == {"ELECTRON", "ION"}
    # Found under the lamella's directory, with the beam suffix the save adds.
    assert all(e.image_path.parent.name == LAMELLA for e in saved)
    assert all(e.image_path.stem.endswith(("_eb", "_ib")) for e in saved)


def test_a_stage_move_knows_where_the_stage_ended_up(recorded):
    moves = [e for _, e in _of(recorded, EventKind.STAGE)]
    assert moves
    assert all(e.position is not None for e in moves)


def test_actions_after_the_task_finished_belong_to_no_lamella(recorded):
    milling = [e for _, e in _of(recorded, EventKind.MILLING)]
    assert milling and all(e.item is None for e in milling)


def test_milling_is_replayed_over_the_last_full_frame_fib_image(recorded):
    mills = [(i, e) for i, e in _of(recorded, EventKind.MILLING) if "stage" in e.data]
    assert [e.data["stage"]["name"] for _, e in mills] == list(MILLING_STAGES)
    index, first = mills[0]
    scene = recorded.scene(index)
    assert scene.milling is first
    # Every stage of the milling task is drawn, not only the one being milled.
    assert [d["name"] for d in scene.milling_stages] == list(MILLING_STAGES)
    assert scene.fib is not None and scene.fib.image_on_disk and scene.fib.is_full_frame
    # What was logged is enough to rebuild the stage the overlay draws.
    stage = FibsemMillingStage.from_dict(scene.milling_stages[0])
    assert stage.name == MILLING_STAGES[0]
    assert stage.define_patterns()
    assert MILLING_TASK in first.summary


def test_spot_burns_accumulate_in_the_field_they_were_burnt_in(recorded):
    spots = [i for i, e in _of(recorded, EventKind.MILLING) if "spot" in e.data]
    assert len(spots) == len(SPOTS)
    first, last = recorded.scene(spots[0]), recorded.scene(spots[-1])
    assert len(first.spots) == 1 and len(last.spots) == len(SPOTS)
    # Fractions of the field the FIB last scanned, as metres from its centre:
    # the milling task re-imaged at its own field width before the burn.
    field = next(
        e
        for e in reversed(recorded.events[: spots[0]])
        if e.kind == EventKind.IMAGE and e.beam == "ION"
    ).field_size
    expected = [((p.x - 0.5) * field[0], (p.y - 0.5) * field[1]) for p in SPOTS]
    assert last.spots == pytest.approx(expected)
    # That new frame also ended the milling before it.
    assert last.milling is None


def test_a_saved_fm_z_stack_is_replayed_from_its_own_metadata(recorded):
    """The log has no record of an FM acquisition; the file places it."""
    ((index, event),) = _of(recorded, EventKind.FLUORESCENCE)
    assert event.image_path.name == FM_STACK and event.image_on_disk
    assert event.data["planes"] == FM_PLANES
    assert len(event.data["channels"]) == FM_CHANNELS
    assert recorded.scene(index).fm is event
    assert recorded.scene(len(recorded.events) - 1).fm is event
    assert index == 0 or recorded.scene(index - 1).fm is None


def test_events_are_in_time_order(recorded):
    times = [e.time for e in recorded.events]
    assert times == sorted(times)


# ── the reader's own decisions ───────────────────────────────────────────────


def test_records_are_read_as_data_not_evaluated():
    assert parse_record(
        "{'msg': 'x', 'a': np.float64(2e-09), 'b': array([1.5, 2.0]), 'c': np.True_,"
        " 'p': FibsemStagePosition(name=None, x=1.0, y=-2.0), 'd': -3}"
    ) == {
        "msg": "x",
        "a": 2e-09,
        "b": [1.5, 2.0],
        "c": True,
        "p": {"name": None, "x": 1.0, "y": -2.0},
        "d": -3,
    }
    # Anything that is not a literal, or a call wrapping one, is refused whole.
    assert parse_record("{'msg': __import__('os').getcwd()}") is None
    assert parse_record("{'msg': open('f', 'w')}") is None
    assert parse_record("{'msg': <BeamType.ION: 2>}") is None
    assert parse_record("not a record") is None


def _line(time: datetime, function: str, message: str, level: str = "DEBUG") -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S,") + f"{time.microsecond // 1000:03d}"
    return f"{stamp} — root — {level} — {function}:1 — {message}\n"


def _acquire(time, directory, filename, beam="ION", save=True):
    record = (
        "{'msg': 'acquire_image', 'metadata': {'image': {'beam_type': '%s', "
        "'resolution': [1536, 1024], 'hfw': 0.0001, 'save': %s, 'path': %r, "
        "'filename': %r, 'reduced_area': None}, 'microscope_state': {}}}"
        % (beam, save, directory, filename)
    )
    return _line(time, "acquire_image", record)


T0 = datetime(2026, 9, 13, 20, 0, 0)


def test_images_are_found_in_a_copied_and_renamed_experiment(tmp_path):
    """Recorded on the instrument PC as C:\\...\\AutoLamella-X\\01-a; here as copy/01-a."""
    root = tmp_path / "a-copy"
    (root / "01-a").mkdir(parents=True)
    (root / "01-a" / "ref_start_ib.tif").write_bytes(b"")
    recorded_dir = "C:\\Users\\User\\Desktop\\AutoLamella-X\\01-a"
    (root / "logfile.log").write_text(
        _acquire(T0, recorded_dir, "ref_start"), encoding="utf-8"
    )
    (event,) = load_replay(root).events
    assert event.image_path == root / "01-a" / "ref_start_ib.tif"


def test_an_overwritten_image_is_only_shown_for_the_acquisition_that_wrote_it(tmp_path):
    """`ref_alignment` is retaken by every task; the file holds only the last one."""
    root = tmp_path / "exp"
    (root / "01-a").mkdir(parents=True)
    (root / "01-a" / "ref_alignment_ib.tif").write_bytes(b"")
    lines = [
        _acquire(T0 + timedelta(seconds=s), str(root / "01-a"), "ref_alignment")
        for s in (0, 60)
    ]
    (root / "logfile.log").write_text("".join(lines), encoding="utf-8")
    first, last = load_replay(root).events
    assert not first.image_on_disk and last.image_on_disk


def test_unsaved_images_are_counted_rather_than_passed_off_as_current(tmp_path):
    root = tmp_path / "exp"
    (root / "01-a").mkdir(parents=True)
    (root / "01-a" / "ref_start_ib.tif").write_bytes(b"")
    lines = [_acquire(T0, str(root / "01-a"), "ref_start")] + [
        _acquire(T0 + timedelta(seconds=s), str(root / "01-a"), "live", save=False)
        for s in (1, 2)
    ]
    (root / "logfile.log").write_text("".join(lines), encoding="utf-8")
    replay = load_replay(root)
    scene = replay.scene(2)
    assert scene.fib is replay.events[0]
    assert scene.fib_unsaved_since == 2


def test_a_scene_scoped_to_an_item_shows_only_its_images(tmp_path):
    """Lamella a, then b, then a again: a's second step shows a's image, not b's."""
    root = tmp_path / "exp"
    status = "{'msg': 'status', 'lamella': '%s', 'task_name': 'T', 'task_step': '%s'}"
    lines = []
    for n, name in enumerate(("a", "b", "a")):
        (root / name).mkdir(parents=True, exist_ok=True)
        start = T0 + timedelta(minutes=10 * n)
        lines.append(_line(start, "log_status_message", status % (name, "STARTED")))
        if n < 2:  # the second visit to a acquires nothing
            (root / name / "ref_ib.tif").write_bytes(b"")
            lines.append(
                _acquire(start + timedelta(seconds=1), str(root / name), "ref")
            )
        lines.append(
            _line(
                start + timedelta(seconds=2),
                "log_status_message",
                status % (name, "FINISHED"),
            )
        )
    (root / "logfile.log").write_text("".join(lines), encoding="utf-8")
    replay = load_replay(root)
    back_to_a = max(i for i, e in enumerate(replay.events) if e.item == "a")
    assert replay.scene(back_to_a).fib.item == "b"  # the instrument
    assert replay.scene(back_to_a, item="a").fib.item == "a"  # the item
    assert replay.scene(0, item="b").fib is None  # nothing of b's yet


def test_a_milling_stage_is_placed_at_its_start(tmp_path):
    """It is logged when it finishes; the replay shows it from when it began."""
    root = tmp_path / "exp"
    root.mkdir()
    record = (
        "{'msg': 'milling_task', 'milling_task_id': 'm1', 'milling_task_name': 'Trench',"
        " 'idx': 0, 'stage': {'name': 'Rough', 'milling': {}, 'pattern': {}},"
        " 'start_time': 1000.0, 'end_time': 1090.0}"
    )
    (root / "logfile.log").write_text(
        _line(T0, "_mill_stage", record), encoding="utf-8"
    )
    (event,) = load_replay(root).events
    assert event.time == T0 - timedelta(seconds=90)
    assert event.duration == 90


def test_a_milling_task_s_stages_keep_the_order_they_were_milled_in(tmp_path):
    """Durations come from a clock coarser than the log's on Windows before 3.13.

    There ``time.time()`` ticks every ~16 ms: a stage milled inside one tick
    measures 0 s, the next a whole tick. Worked back from its end, the second
    would have started before the first. It started after the first finished.
    """
    root = tmp_path / "exp"
    root.mkdir()
    record = (
        "{'msg': 'milling_task', 'milling_task_id': 'm1', 'milling_task_name': 'Trench',"
        " 'idx': %d, 'stage': {'name': '%s', 'milling': {}, 'pattern': {}},"
        " 'start_time': %r, 'end_time': %r}"
    )
    lines = [
        _line(T0, "_mill_stage", record % (0, "Rough", 1000.0, 1000.0)),
        _line(
            T0 + timedelta(milliseconds=1),
            "_mill_stage",
            record % (1, "Polish", 1000.0, 1000.0156),
        ),
    ]
    (root / "logfile.log").write_text("".join(lines), encoding="utf-8")
    first, second = load_replay(root).events
    assert [first.data["stage"]["name"], second.data["stage"]["name"]] == [
        "Rough",
        "Polish",
    ]
    assert first.time == T0
    assert second.time == T0  # when the first finished, not before
    assert second.duration == pytest.approx(0.0156)


def test_a_cp1252_log_is_read(tmp_path):
    """Older Windows installs wrote the log in cp1252; the em-dash separator decides."""
    root = tmp_path / "exp"
    root.mkdir()
    (root / "logfile.log").write_bytes(
        _line(T0, "run", "Point fit failed", level="ERROR").encode("cp1252")
    )
    assert [r.message for r in read_log_records(root / "logfile.log")] == [
        "Point fit failed"
    ]
    (event,) = load_replay(root).events
    assert event.kind == EventKind.MESSAGE


def test_a_prompt_answer_says_who_answered(tmp_path):
    """The line QtResponder writes when a prompt is answered."""
    root = tmp_path / "exp"
    root.mkdir()
    message = "prompt answered: PickPOI response=True by=agent adjusted=True"
    (root / "logfile.log").write_text(
        _line(T0, "answer_confirm", message, level="INFO"), encoding="utf-8"
    )
    (event,) = load_replay(root).events
    assert event.kind == EventKind.PROMPT
    assert event.data == {
        "prompt": "PickPOI",
        "response": True,
        "answered_by": "agent",
        "adjusted": True,
    }


def test_a_grid_workflow_item_is_a_grid(tmp_path):
    """Grid tasks name their item under `grid`, not `lamella`."""
    root = tmp_path / "exp"
    root.mkdir()
    status = (
        "{'msg': 'status', 'grid': 'grid-1', 'grid_id': 'g1',"
        " 'task_name': 'SEM Overview', 'task_step': 'STARTED'}"
    )
    (root / "logfile.log").write_text(
        _line(T0, "log_status_message", status), encoding="utf-8"
    )
    (event,) = load_replay(root).events
    assert (event.item, event.item_type) == ("grid-1", "grid")


def test_an_fm_image_belongs_to_the_task_step_it_was_taken_in(tmp_path):
    from fibsem.fm.structures import FluorescenceImage

    root = tmp_path / "exp"
    (root / "01-a").mkdir(parents=True)
    status = (
        "{'msg': 'status', 'lamella': '01-a', 'task_name': 'Acquire FM',"
        " 'task_step': '%s'}"
    )
    lines = [
        _line(T0, "log_status_message", status % "ACQUIRE_FLUORESCENCE_IMAGE"),
        _line(T0 + timedelta(minutes=5), "log_status_message", status % "FINISHED"),
    ]
    (root / "logfile.log").write_text("".join(lines), encoding="utf-8")
    image = FluorescenceImage.generate_blank_image(resolution=(32, 32), zlevels=2)
    image.metadata.acquisition_date = (T0 + timedelta(minutes=1)).isoformat()
    image.save(str(root / "01-a" / "01-a-zstack.ome.tiff"))
    image.metadata.acquisition_date = (T0 + timedelta(minutes=10)).isoformat()
    image.save(str(root / "overview.ome.tiff"))

    in_task, afterwards = [
        e for e in load_replay(root).events if e.kind == EventKind.FLUORESCENCE
    ]
    assert (in_task.item, in_task.step) == ("01-a", "ACQUIRE_FLUORESCENCE_IMAGE")
    assert in_task.time == T0 + timedelta(minutes=1)
    assert afterwards.item is None


def test_a_directory_without_a_log_cannot_be_replayed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_replay(tmp_path)


# ── an experiment recorded with events.jsonl ─────────────────────────────────


@pytest.fixture(scope="module")
def recorded_twice(tmp_path_factory):
    """One Demo run, replayed from its events.jsonl and from its log."""
    from fibsem.applications.autolamella.tools.replay import _load_from_log

    monkeypatch = pytest.MonkeyPatch()
    try:
        root = record_demo_experiment(
            tmp_path_factory.mktemp("replay-events") / "exp",
            monkeypatch,
            record_events=True,
        )
    finally:
        monkeypatch.undo()
    return load_replay(root), _load_from_log(root)


def test_an_experiment_with_events_is_replayed_from_them(recorded_twice):
    events, _ = recorded_twice
    assert events.source == EVENTS_FILENAME
    assert events.records_read > 0 and events.records_unreadable == 0


def test_the_events_replay_what_the_log_replays(recorded_twice):
    """The same run, read both ways: the same steps, files, milling and spots."""
    events, log = recorded_twice

    def steps(replay):
        # STARTED / FINISHED reach events.jsonl from the lifecycle hook, which
        # the app registers per run; this bare task run has no task manager to
        # fire it. The reader's handling of them is pinned by hand below.
        return [
            (e.item, e.task, e.step)
            for _, e in _of(replay, EventKind.TASK)
            if e.step not in ("STARTED", "FINISHED")
        ]

    def files(replay):
        return {
            e.image_path for _, e in _of(replay, EventKind.IMAGE) if e.image_on_disk
        }

    def mills(replay):
        return [
            e.data["stage"]["name"]
            for _, e in _of(replay, EventKind.MILLING)
            if "stage" in e.data
        ]

    assert steps(events) == steps(log)
    assert files(events) and files(events) == files(log)
    assert mills(events) == mills(log) == list(MILLING_STAGES)
    assert all(e.item_type == "lamella" for e in events.events if e.item)
    spots = [i for i, e in _of(events, EventKind.MILLING) if "spot" in e.data]
    log_spots = [i for i, e in _of(log, EventKind.MILLING) if "spot" in e.data]
    assert events.scene(spots[-1]).spots == pytest.approx(
        log.scene(log_spots[-1]).spots
    )
    # milling's own final image is recorded, so the overlay ends where the log's does
    assert events.scene(spots[-1]).milling is None
    assert log.scene(log_spots[-1]).milling is None
    ((_, fm),) = _of(events, EventKind.FLUORESCENCE)
    assert fm.image_path.name == FM_STACK


def test_milling_is_placed_when_it_started_not_worked_back_from_its_end(
    recorded_twice,
):
    events, _ = recorded_twice
    for index, mill in _of(events, EventKind.MILLING):
        if "stage" not in mill.data:
            continue
        assert mill.duration is not None and mill.duration >= 0
        scene = events.scene(index)
        assert [d["name"] for d in scene.milling_stages] == list(MILLING_STAGES)
        assert FibsemMillingStage.from_dict(scene.milling_stages[0]).define_patterns()


# ── the events reader's own decisions ────────────────────────────────────────


def _record(t, kind, payload=None, item=None, task=None, task_id="T1"):
    return {
        "t": t,
        "kind": kind,
        "payload": payload or {},
        "item": {"id": "L1", "name": item} if item else None,
        "task": {"id": task_id, "name": task} if task else None,
    }


def _write_events(root, *records, torn=None):
    import json

    text = "".join(json.dumps(r) + "\n" for r in records)
    (root / EVENTS_FILENAME).write_text(text + (torn or ""), encoding="utf-8")


def test_event_times_are_the_instrument_s_wall_clock(tmp_path):
    """Read as the log and the FM files record time: not converted to this
    machine's zone, whatever it is."""
    _write_events(
        tmp_path, _record("2026-09-21T14:00:00.250+02:00", "task_started", task="Mill")
    )
    (event,) = load_replay(tmp_path).events
    assert event.time == datetime(2026, 9, 21, 14, 0, 0, 250000)


def test_a_record_cut_short_is_counted_not_read(tmp_path):
    _write_events(
        tmp_path,
        _record("2026-09-21T14:00:00.000+10:00", "task_started", task="Mill"),
        torn='{"t": "2026-09-21T14:00:01',
    )
    replay = load_replay(tmp_path)
    assert len(replay.events) == 1
    assert replay.records_unreadable == 1


def test_a_recorded_path_is_found_in_a_copied_and_renamed_experiment(tmp_path):
    (tmp_path / "01-test").mkdir()
    (tmp_path / "01-test" / "ref_ib.tif").write_bytes(b"")
    _write_events(
        tmp_path,
        _record(
            "2026-09-21T14:00:00.000+10:00",
            "image_acquired",
            {"path": "D:\\data\\old-name\\01-test\\ref_ib.tif", "beam_type": "ION"},
        ),
    )
    (image,) = load_replay(tmp_path).events
    assert image.image_path == tmp_path / "01-test" / "ref_ib.tif"


def test_milling_stages_started_in_the_same_clock_tick_keep_their_order(tmp_path):
    """Each stage is placed at its own start, so a coarse clock only makes a tie."""
    t = "2026-09-21T14:00:00.000+10:00"

    def started(idx, name):
        stage = {"name": name, "milling": {}, "pattern": {}}
        payload = {"task_id": "m1", "task_name": "Trench", "stage": stage}
        return _record(t, "milling_stage_started", {**payload, "stage_index": idx})

    def finished(name):
        payload = {"task_id": "m1", "stage_name": name, "status": "stage-finished"}
        return _record(t, "milling_progress", payload)

    _write_events(
        tmp_path,
        started(0, "Rough"),
        finished("Rough"),
        started(1, "Polish"),
        finished("Polish"),
    )
    replay = load_replay(tmp_path)
    assert [e.data["stage"]["name"] for e in replay.events] == ["Rough", "Polish"]
    assert [e.duration for e in replay.events] == [0.0, 0.0]


def test_a_cancelled_spot_burn_replays_only_the_points_it_reached(tmp_path):
    start = {
        "coordinates": [[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]],
        "field_of_view": 1e-4,
        "exposure_time": 10.0,
        "milling_current": 1e-10,
    }
    _write_events(
        tmp_path,
        _record("2026-09-21T14:00:00.000+10:00", "spot_burn_started", start),
        _record(
            "2026-09-21T14:00:12.000+10:00",
            "spot_burn_progress",
            {"status": "burning", "current_point": 2},
        ),
        _record(
            "2026-09-21T14:00:13.000+10:00",
            "spot_burn_progress",
            {"status": "cancelled", "current_point": 3},
        ),
    )
    spots = load_replay(tmp_path).events
    assert [e.data["spot"] for e in spots] == [(0.1, 0.5), (0.5, 0.5)]
    assert spots[1].time - spots[0].time == timedelta(seconds=10)
    assert all(e.data["field_of_view"] == 1e-4 for e in spots)


def test_a_prompt_is_replayed_from_when_it_was_asked(tmp_path):
    _write_events(
        tmp_path,
        _record(
            "2026-09-21T14:00:00.000+10:00",
            "prompt_raised",
            {"type": "PickPOI", "message": "Pick the point of interest"},
        ),
        _record(
            "2026-09-21T14:02:00.000+10:00",
            "prompt_answered",
            {"type": "PickPOI", "response": True, "answered_by": "operator"},
        ),
    )
    asked, answered = load_replay(tmp_path).events
    assert asked.summary == "PickPOI asked: Pick the point of interest"
    assert answered.summary == "PickPOI answered Yes by the operator"
    assert answered.time - asked.time == timedelta(minutes=2)


def test_a_failed_task_ends_its_context(tmp_path):
    """Warnings come from the log; each belongs to the task running when it was
    written, and a failed task is no longer running."""
    _write_events(
        tmp_path,
        _record(
            "2026-09-21T14:00:00.000+10:00", "task_started", item="01", task="Mill"
        ),
        _record(
            "2026-09-21T14:01:00.000+10:00",
            "task_failed",
            {"error": "stage limit"},
            item="01",
            task="Mill",
        ),
    )
    (tmp_path / "logfile.log").write_text(
        _line(datetime(2026, 9, 21, 14, 0, 30), "mill", "drift high", "WARNING")
        + _line(datetime(2026, 9, 21, 14, 2), "move", "limit reached", "WARNING"),
        encoding="utf-8",
    )
    started, during, failed, after = load_replay(tmp_path).events
    assert failed.step == "FAILED" and failed.summary == "Mill — Failed: stage limit"
    assert (during.kind, during.item, during.task) == (EventKind.MESSAGE, "01", "Mill")
    assert (after.kind, after.item, after.task) == (EventKind.MESSAGE, None, None)


def test_a_spot_is_placed_by_the_field_its_burn_recorded(tmp_path):
    """Not by the last frame's, which the log had to assume was the same."""
    _write_events(
        tmp_path,
        _record(
            "2026-09-21T14:00:00.000+10:00",
            "image_acquired",
            {"beam_type": "ION", "hfw": 2e-4, "shape": [1024, 1536], "path": None},
        ),
        _record(
            "2026-09-21T14:00:10.000+10:00",
            "spot_burn_started",
            {"coordinates": [[0.75, 0.5]], "field_of_view": 1e-4, "exposure_time": 1.0},
        ),
        _record(
            "2026-09-21T14:00:11.000+10:00",
            "spot_burn_progress",
            {"status": "finished"},
        ),
    )
    replay = load_replay(tmp_path)
    (spot,) = replay.scene(len(replay.events) - 1).spots
    assert spot == pytest.approx((0.25 * 1e-4, 0.0))


# ── moves, alignment, focus and FM, as the stream records them ───────────────


def test_each_stage_move_is_one_row_where_the_log_has_one_per_call(recorded_twice):
    """A safe move is up to three absolute moves, and the log has a line for
    each; the stream records the move once. Both end at the same place."""
    events, log = recorded_twice
    moves = [e for _, e in _of(events, EventKind.STAGE)]
    recorded = [
        r
        for r in read_events(events.root / EVENTS_FILENAME)
        if r["kind"] == "stage_moved"
    ]
    assert moves and len(moves) == len(recorded)
    assert len(moves) < len([e for _, e in _of(log, EventKind.STAGE)])
    assert all(e.data["move"] for e in moves)
    assert any(e.position is not None for e in moves)
    assert events.stage_position_at(events.end) == log.stage_position_at(log.end)


def _at(seconds):
    return f"2026-09-21T14:00:{seconds:02d}.000+10:00"


def test_stage_moves_say_what_they_were_and_where_they_ended(tmp_path):
    end = {"x": 1e-5, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.5}
    _write_events(
        tmp_path,
        _record(
            _at(0),
            "stage_moved",
            {
                "move": "stable_move",
                "request": {"dx": 2e-6, "dy": -1e-6, "beam_type": "ELECTRON"},
                "end": end,
            },
        ),
        _record(
            _at(1),
            "stage_moved",
            {"move": "move_to_orientation", "request": {"orientation": "FIB"}},
        ),
        _record(
            _at(2),
            "stage_moved",
            {"move": "move_to_milling_angle", "request": {"milling_angle": 0.2618}},
        ),
        _record(
            _at(3),
            "stage_moved",
            {
                "move": "move_to_device",
                "request": {"device": "FM", "orientation": None},
            },
        ),
        _record(
            _at(4),
            "stage_moved",
            {
                "move": "move_to_orientation",
                "request": {"orientation": "SIDEWAYS"},
                "error": "ValueError: Orientation SIDEWAYS not supported.",
            },
        ),
    )
    replay = load_replay(tmp_path)
    assert [e.summary for e in replay.events] == [
        "Stable move in the SEM dx=2.0 µm, dy=-1.0 µm",
        "Move to the FIB orientation",
        "Move to a milling angle of 15.0°",
        "Move to the FM",
        "Move to the SIDEWAYS orientation — failed: "
        "ValueError: Orientation SIDEWAYS not supported.",
    ]
    assert all(e.kind == EventKind.STAGE for e in replay.events)
    assert replay.events[0].position == end
    # a move without an end shows the last position known before it
    assert replay.scene(1).stage_position == end


def test_a_position_read_is_the_stage_track_not_a_row(tmp_path):
    read = {"x": 3e-6, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.0}
    _write_events(
        tmp_path,
        _record(_at(0), "stage_position_changed", {"position": read}),
        _record(_at(1), "task_started", task="Mill"),
    )
    replay = load_replay(tmp_path)
    (started,) = replay.events
    assert started.kind == EventKind.TASK
    assert replay.scene(0).stage_position == read


def test_beam_shifts_alignment_focus_and_coincidence_are_alignment_rows(tmp_path):
    _write_events(
        tmp_path,
        _record(_at(0), "beam_shifted", {"dx": 1e-7, "dy": -2e-7, "beam_type": "ION"}),
        _record(
            _at(1),
            "alignment",
            {
                "beam_type": "ION",
                "subsystem": "beam-shift",
                "results": [
                    {"shift": {"x": 4e-7, "y": 0.0}},
                    {"shift": {"x": 1e-7, "y": 0.0}},
                ],
                "validation": {"agreement": False, "max_disagreement_px": 6.4},
                "aborted": False,
            },
        ),
        _record(
            _at(2),
            "autofocus",
            {
                "beam_type": "ELECTRON",
                "initial_working_distance": 0.004,
                "working_distance": 0.00412,
            },
        ),
        _record(
            _at(3),
            "coincidence_measured",
            {
                "dx": 1e-7,
                "dy": 3e-6,
                "is_reliable": False,
                "refusal_reason": "rival_peak",
            },
        ),
    )
    replay = load_replay(tmp_path)
    assert all(e.kind == EventKind.ALIGNMENT for e in replay.events)
    assert [e.summary for e in replay.events] == [
        "FIB beam shift dx=100 nm, dy=-200 nm",
        "FIB alignment by beam shift, 2 steps, last shift dx=100 nm, dy=0 nm"
        " — the methods disagree by 6 px",
        "SEM autofocus: working distance 4.000 → 4.120 mm",
        "Coincidence measured dx=0.1 µm, dy=3.0 µm — refused (rival_peak)",
    ]


def _fm_file(path):
    from fibsem.fm.structures import FluorescenceImage

    FluorescenceImage.generate_blank_image(
        resolution=(32, 24), zlevels=2, n_channels=1, random=True
    ).save(str(path))
    return path


def test_a_recorded_fm_image_is_placed_when_it_started_and_where_its_record_says(
    tmp_path,
):
    (tmp_path / "01-test").mkdir()
    recorded = _fm_file(tmp_path / "01-test" / "zstack.ome.tiff")
    unrecorded = _fm_file(tmp_path / "01-test" / "saved-by-hand.ome.tiff")
    _write_events(
        tmp_path,
        _record(
            _at(30),
            "fm_image_acquired",
            {
                "path": "D:\\old-name\\01-test\\zstack.ome.tiff",
                "acquired_at": "2026-09-21T14:00:10",
                "channels": [{"name": "GFP"}],
                "z_positions": [0.0, 1e-6],
                "stage_position": {"x": 0.0, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.0},
            },
            item="01-test",
            task="Acquire FM",
        ),
    )
    replay = load_replay(tmp_path)
    fm = [e for e in replay.events if e.kind == EventKind.FLUORESCENCE]
    assert sorted(e.image_path.name for e in fm) == [
        unrecorded.name,
        recorded.name,
    ]  # each once: the recorded file is not found again on disk
    (from_record,) = [e for e in fm if e.image_path == recorded]
    assert from_record.time == datetime(2026, 9, 21, 14, 0, 10)  # when it started
    assert (from_record.item, from_record.task) == ("01-test", "Acquire FM")
    assert from_record.summary == "FM z-stack, 2 planes, GFP — zstack.ome.tiff"


def test_an_fm_file_written_twice_shows_only_for_its_last_write(tmp_path):
    _fm_file(tmp_path / "overview.ome.tiff")
    overview = {"path": str(tmp_path / "overview.ome.tiff"), "overview": {"rows": 2}}
    _write_events(
        tmp_path,
        _record(_at(0), "fm_image_acquired", overview),
        _record(_at(9), "fm_image_acquired", overview),
    )
    first, last = load_replay(tmp_path).events
    assert first.image_path is None and last.image_path is not None
    assert last.summary == "FM overview — overview.ome.tiff"


def test_fm_autofocus_and_the_objective_are_fm_rows(tmp_path):
    _write_events(
        tmp_path,
        _record(
            _at(0), "objective_state_changed", {"state": "Inserted", "position": 0.0}
        ),
        _record(
            _at(1), "fm_autofocus", {"initial_position": 1.2e-3, "position": 1.2015e-3}
        ),
    )
    replay = load_replay(tmp_path)
    assert all(e.kind == EventKind.FLUORESCENCE for e in replay.events)
    assert [e.summary for e in replay.events] == [
        "FM objective inserted",
        "FM autofocus: objective 1200.0 → 1201.5 µm",
    ]


def test_a_recorded_fm_image_outside_a_task_is_not_given_one_by_time(tmp_path):
    """A run without the lifecycle hook records no task end, so a task's steps
    never close. The file scan places FM images by the step they fall in; a
    recorded image says for itself that it belonged to no task."""
    _fm_file(tmp_path / "later.ome.tiff")
    _write_events(
        tmp_path,
        _record(_at(0), "task_step", {"step": "MILL"}, item="01", task="Mill"),
        _record(
            _at(5),
            "fm_image_acquired",
            {
                "path": str(tmp_path / "later.ome.tiff"),
                "acquired_at": "2026-09-21T14:00:05",
            },
        ),
    )
    _, fm = load_replay(tmp_path).events
    assert fm.kind == EventKind.FLUORESCENCE
    assert (fm.item, fm.task) == (None, None)


def _edit(t, payload, actor, **context):
    record = _record(t, "edit", payload, **context)
    record["actor"] = actor
    return record


def test_an_edit_says_what_changed_who_made_it_and_from_where(tmp_path):
    """On the lamella and task edited, not the ones the run was on. The values
    that changed are found in the whole objects recorded before and after."""
    points = [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}, {"x": 0.5, "y": 0.6}]
    _write_events(
        tmp_path,
        _record(_at(0), "task_step", {"step": "MILL"}, item="01", task="Mill Fiducial"),
        _edit(
            _at(1),
            {
                "item": {"id": "L2", "name": "02"},
                "task": "Rough Milling",
                "target": "milling.mill_rough",
                "before": {
                    "stages": [{"name": "Rough", "pattern": {"depth": 2e-6}}],
                    "acquisition": {"imaging": {"path": "/data/an-experiment/02-bear"}},
                },
                "after": {
                    "stages": [
                        {"name": "Rough", "pattern": {"depth": 3e-6, "passes": 2}}
                    ],
                    "acquisition": {
                        "imaging": {
                            "path": "/data/2026-09-22/an-experiment/01-solid-cow"
                        }
                    },
                },
                "via": "apply to other lamellae",
            },
            "operator",
            item="01",
            task="Mill Fiducial",
        ),
        _edit(
            _at(2),
            {
                "item": None,
                "task": "Rough Milling",
                "target": "protocol.parameters.sync_to_poi",
                "before": True,
                "after": False,
                "via": "protocol editor",
            },
            None,  # a recorder outside the app does not know who
        ),
        _edit(
            _at(3),
            {
                "item": {"id": "L1", "name": "01"},
                "task": "Spot Burn",
                "target": "parameters.coordinates",
                "before": points[:1],
                "after": [{"x": 0.1, "y": 0.25}] + points[1:] + [{"x": 0.7, "y": 0.8}],
                "via": "agent patch",
            },
            "agent",
        ),
    )
    _, *edits = load_replay(tmp_path).events
    assert all(e.kind == EventKind.EDIT for e in edits)
    assert [(e.item, e.task, e.step) for e in edits] == [
        ("02", "Rough Milling", None),
        (None, "Rough Milling", None),
        ("01", "Spot Burn", None),
    ]
    assert [e.summary for e in edits] == [
        "milling.mill_rough: stages.0.pattern.depth 2e-06 → 3e-06,"
        " stages.0.pattern.passes (none) → 2,"
        " acquisition.imaging.path /data/an-experiment/02-bear"
        " → …9-22/an-experiment/01-solid-cow"
        " — by the operator (apply to other lamellae)",
        "protocol.parameters.sync_to_poi: True → False (protocol editor)",
        "parameters.coordinates: 0.y 0.2 → 0.25, 1 (none) → {'x': 0.3, 'y': 0.4},"
        " 2 (none) → {'x': 0.5, 'y': 0.6}, 1 more — by the agent (agent patch)",
    ]


def test_an_edit_leaves_out_rounding_and_names_a_config_added_or_removed(tmp_path):
    """A field of view a widget showed in µm comes back in metres a bit off:
    not a change. A whole task config added or removed is named, not printed."""
    fov = 80 * 1e-6  # 80 µm, as a spin box hands it back
    assert fov != 8e-05
    config = {
        "parameters": {"sync_to_poi": True, "acquire_image1": True},
        "milling": {"mill_rough": {"field_of_view": 8e-05}},
    }

    def edit(t, target, before, after, via, item=None):
        payload = {"item": item, "task": "Rough Milling", "target": target}
        payload.update(before=before, after=after, via=via)
        return _edit(_at(t), payload, "operator")

    _write_events(
        tmp_path,
        edit(
            0,
            "protocol.milling.mill_rough",
            {"field_of_view": 8e-05, "stages": [{"pattern": {"depth": 6.5e-07}}]},
            {"field_of_view": fov, "stages": [{"pattern": {"depth": 1.3e-06}}]},
            "protocol editor",
        ),
        edit(
            1,
            "protocol.milling.mill_rough",
            {"field_of_view": 8e-05},
            {"field_of_view": fov},
            "protocol editor",
        ),
        edit(
            2,
            "protocol.milling.mill_rough",
            {"field_of_view": 1.00001e-4},
            {"field_of_view": 1.00002e-4},
            "protocol editor",
        ),
        edit(3, "task_config", None, config, "add task", {"id": "L1", "name": "01"}),
        edit(4, "protocol.task_config", config, None, "remove task"),
    )
    assert [e.summary for e in load_replay(tmp_path).events] == [
        "protocol.milling.mill_rough: stages.0.pattern.depth 6.5e-07 → 1.3e-06"
        " — by the operator (protocol editor)",
        "protocol.milling.mill_rough: rounding only — by the operator (protocol editor)",
        "protocol.milling.mill_rough: field_of_view 0.000100001 → 0.000100002"
        " — by the operator (protocol editor)",
        "task_config: added — by the operator (add task)",
        "protocol.task_config: removed — by the operator (remove task)",
    ]


def _proposal(t, event, actor, proposal, **payload):
    """A question or a decision, as the recorder writes it: on lamella 01's
    Setup, which need not be where the run is."""
    record = _record(
        _at(t),
        event,
        {
            "item": {"id": "L1", "name": "01"},
            "task": "Setup Lamella Position",
            "proposal_id": proposal,
            **payload,
        },
        item="02",
        task="Rough Milling",
    )
    record["actor"] = actor
    return record


def test_a_decision_says_what_was_decided_by_whom_and_what_they_changed(tmp_path):
    """A confirmation "as it stands" is one row, with the move the operator
    made before it once the task fills that in. A point or a position moved is
    a move in µm and degrees."""
    here = {"x": 1e-3, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.2, "name": "p"}
    moved = dict(here, x=1.002e-3, t=0.2 + math.radians(1.0))
    poi = {"x": 0.0, "y": 0.0}
    state = {"kind": "state", "proposed": {"stage_position": here}}
    confirmed = {"outcome": "Confirmed", "author": "human:op", "via": "workflow"}
    _write_events(
        tmp_path,
        _proposal(0, "proposal_asked", "task", "P1", **state, message="Tilt"),
        _proposal(
            1,
            "proposal_decided",
            "operator",
            "P1",
            **state,
            **confirmed,
            decision=0,
            decided={},
        ),
        _proposal(
            2,
            "proposal_decided",
            "operator",
            "P1",
            **state,
            **confirmed,
            decision=0,
            decided={"stage_position": moved},
            filled_in=True,
        ),
        _proposal(
            3,
            "proposal_decided",
            "operator",
            "P2",
            kind="point_of_interest",
            proposed={"poi": poi},
            outcome="Confirmed",
            author="human:op",
            via="review",
            decision=0,
            decided={"poi": {"x": 2e-6, "y": -1e-6}},
        ),
        _proposal(
            4,
            "proposal_decided",
            "agent",
            "P3",
            kind="point_of_interest",
            proposed={"poi": poi},
            outcome="Confirmed",
            author="agent:m",
            via="server",
            decision=0,
            decided={"poi": poi},
        ),
        _proposal(
            5,
            "proposal_decided",
            "task",
            "P4",
            kind="point_of_interest",
            proposed={"poi": poi},
            outcome="Unreviewed",
            author="auto:poi",
            via="workflow",
            reason="Rough Milling started",
            decision=0,
            decided={"poi": poi},
        ),
        _proposal(
            6,
            "proposal_decided",
            "task",
            "P5",
            kind="state",
            proposed={},
            outcome="Withdrawn",
            author="auto:workflow",
            via="workflow",
            reason="the run stopped",
            decision=0,
            decided={},
        ),
        _proposal(
            7,
            "proposal_decided",
            "operator",
            "P6",
            kind="task_result",
            proposed={},
            outcome="Confirmed",
            author="human:op",
            via="review",
            decision=1,
            decided={},
        ),
    )
    events = load_replay(tmp_path).events
    assert [(e.kind, e.item, e.task) for e in events] == [
        (EventKind.PROMPT, "01", "Setup Lamella Position")
    ] + [(EventKind.DECISION, "01", "Setup Lamella Position")] * 6
    assert [e.summary for e in events] == [
        "Position asked: Tilt",
        "Position confirmed: moved x +2.0 µm, t +1.0° — by the operator (workflow)",
        "Point of interest confirmed: moved x +2.0 µm, y -1.0 µm"
        " — by the operator (review)",
        "Point of interest confirmed, as proposed — by the agent (server)",
        "Point of interest used as proposed, unreviewed: Rough Milling started"
        " — by the task (workflow)",
        "Position withdrawn: the run stopped — by the task (workflow)",
        "Result confirmed — by the operator (review)",
    ]
    assert events[1].time == events[0].time + timedelta(seconds=1), (
        "the row is when it was confirmed, not when the position was read"
    )
    assert events[1].data["filled_in"] is True


def test_a_correlation_says_where_it_put_the_point_and_how_well_it_fits(tmp_path):
    """On the lamella it was for, not the one the run was on."""
    records = [
        _correlation(
            _at(1),
            {
                "item": {"id": "L2", "name": "02"},
                "poi": {"x": 5.95e-6, "y": -4.9e-6},
                "rms_px": 3.09,
                "rms_nm": 201.3,
                "fiducials": 9,
                "refractive_index": {"mode": "pre", "factor": 1.3},
                "verdict": "good",
                "seeded": True,
            },
            "operator",
            item="01",
            task="Mill Fiducial",
        ),
        _correlation(
            _at(2),
            {
                "item": {"id": "L1", "name": "01"},
                "poi": {"x": 1e-6, "y": 2e-6},
                "rms_px": 4.04,
                "rms_nm": None,  # the FIB image had no pixel size
                "fiducials": 6,
                "refractive_index": {"mode": "post", "factor": 1.25},
                "verdict": None,
                "seeded": False,
            },
            None,
        ),
        _correlation(_at(3), {"rms_nm": "not a number"}, None),
    ]
    _write_events(
        tmp_path,
        _record(_at(0), "task_step", {"step": "MILL"}, item="01", task="Mill Fiducial"),
        *records,
    )
    _, *rows = load_replay(tmp_path).events
    assert all(e.kind == EventKind.CORRELATION for e in rows)
    assert [(e.item, e.task, e.step) for e in rows] == [
        ("02", None, None),
        ("01", None, None),
        (None, None, None),
    ]
    assert [e.summary for e in rows] == [
        "Correlation: point of interest x=6.0 µm, y=-4.9 µm — RMS 201 nm over 9"
        " fiducials, good fit, refractive index ×1.30 before the fit"
        " — by the operator",
        "Correlation: point of interest x=1.0 µm, y=2.0 µm — RMS 4.0 px over 6"
        " fiducials, unseeded, refractive index ×1.25 after the fit",
        "Correlation: point of interest x=? µm, y=? µm",
    ]


def _correlation(t, payload, actor, **context):
    record = _record(t, "correlation", payload, **context)
    record["actor"] = actor
    return record
