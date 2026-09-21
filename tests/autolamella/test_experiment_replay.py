"""Reading an experiment's log back as a replay timeline.

Two halves. The first runs the real producers on the Demo microscope and
replays what they wrote: the only way to notice a producer changing the shape
of a record the replay reads. The second pins the reader's own decisions --
what it refuses to evaluate, how it finds an image in a copied experiment,
what it does when a later image overwrote an earlier one.
"""

from datetime import datetime, timedelta

import pytest

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
