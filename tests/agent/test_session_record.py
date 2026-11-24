"""The conversation with the agent is kept with its session, everything else
with the experiment: one live stream, two files, each record in one."""

import json

from fibsem.acting import AGENT, OPERATOR
from fibsem.ai.agent.recent import read_rows
from fibsem.server.agent import files as agent_files
from fibsem.workflows.actions import (
    ACTION,
    NOTE,
    action_record,
    is_session_record,
    note_record,
)
from fibsem.workflows.events.recording import (
    EVENTS_FILENAME,
    EventRecorder,
    read_events,
    read_events_after_in,
)
from tests.fixtures.microscopes import demo_microscope

microscope = demo_microscope(scope="function")


def _record(kind, payload, actor=None):
    return {"kind": kind, "payload": payload, "actor": actor}


def test_the_conversation_is_the_session_s():
    assert is_session_record(_record(NOTE, note_record("hi", to=AGENT, id="m1")))
    assert is_session_record(_record(NOTE, note_record("done", to=OPERATOR)))
    assert is_session_record(_record(NOTE, {"text": "x", "for_record": True}))
    assert is_session_record(
        _record(ACTION, action_record("Stop the agent", stops="agent"))
    )
    assert is_session_record(
        _record(ACTION, action_record("Interrupt the agent", interrupts="m1"))
    )
    assert is_session_record(_record("agent_level_set", {"level": "watch"}))


def test_notes_and_the_run_are_the_experiment_s():
    assert not is_session_record(_record(NOTE, {"text": "[run_state] grid 1"}))
    assert not is_session_record(
        _record(NOTE, note_record("[verdict] ok", item_id="i1", item_name="L1"))
    )
    assert not is_session_record(
        _record(ACTION, action_record("Stop the workflow", stops="workflow"))
    )
    for kind in ("task_started", "agent_session_joined", "agent_plan_edited"):
        assert not is_session_record(_record(kind, {}))


def _written(path):
    return list(read_events(path)) if path.exists() else []


def _say(recorder, text, **fields):
    recorder.buffer.append(
        NOTE, note_record(text, to=AGENT, id=text[:8], **fields), actor=OPERATOR
    )


def test_each_record_goes_to_one_file(microscope, tmp_path):
    experiment = tmp_path / "exp"
    experiment.mkdir()
    recorder = EventRecorder(microscope, experiment_path=experiment)
    _say(recorder, "before any session")
    recorder.set_agent_session("s1")
    _say(recorder, "hello")
    recorder.buffer.append(NOTE, note_record("[run_state] two left"), actor=AGENT)
    recorder.buffer.append(
        ACTION, action_record("Stop the agent", stops="agent"), actor=OPERATOR
    )
    recorder.set_agent_session(None)
    _say(recorder, "after it ended")
    recorder.close()

    session = agent_files.session_folder("s1") / EVENTS_FILENAME
    in_session = [r["payload"].get("text") or r["kind"] for r in _written(session)]
    in_experiment = [
        r["payload"].get("text") or r["kind"]
        for r in _written(experiment / EVENTS_FILENAME)
    ]
    assert in_session == ["hello", "action"]
    assert in_experiment == [
        "before any session",
        "[run_state] two left",
        "after it ended",
    ]


def test_a_message_with_no_experiment_open_is_kept(microscope):
    recorder = EventRecorder(microscope)
    recorder.set_agent_session("s1")
    _say(recorder, "nothing is open")
    recorder.close()
    session = agent_files.session_folder("s1") / EVENTS_FILENAME
    assert [r["payload"].get("text") for r in _written(session)] == ["nothing is open"]


def test_switching_experiments_keeps_one_conversation(microscope, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    recorder = EventRecorder(microscope, experiment_path=a)
    recorder.set_agent_session("s1")
    _say(recorder, "on a")
    recorder.set_experiment(b)
    _say(recorder, "on b")
    recorder.close()

    files = agent_files.record_files("s1", str(b))
    # Neither experiment's file holds any of it.
    assert files == [agent_files.session_folder("s1") / EVENTS_FILENAME]
    assert not (b / EVENTS_FILENAME).exists()
    rows, _ = read_rows([], files)
    assert [row.text for row in rows if row.text in ("on a", "on b")] == [
        "on a",
        "on b",
    ]


def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _message(event_id, t, text, actor=OPERATOR):
    return {
        "id": event_id,
        "seq": int(event_id.split("-")[1]),
        "kind": NOTE,
        "timestamp": t,
        "actor": actor,
        "payload": note_record(text, to=AGENT, id=event_id),
    }


def test_an_old_experiment_reads_its_own_chat(tmp_path):
    experiment = tmp_path / "old"
    _write(experiment / EVENTS_FILENAME, [_message("x-1", 1.0, "from before")])
    files = agent_files.record_files(None, str(experiment))
    assert files == [experiment / EVENTS_FILENAME]
    rows, _ = read_rows([], files)
    assert [row.text for row in rows] == ["from before"]


def test_the_chat_reads_both_files_by_time(tmp_path):
    session = tmp_path / "s" / EVENTS_FILENAME
    experiment = tmp_path / "e" / EVENTS_FILENAME
    _write(session, [_message("x-1", 1.0, "one"), _message("x-3", 3.0, "three")])
    _write(experiment, [_message("x-2", 2.0, "two"), _message("x-3", 3.0, "three")])
    rows, _ = read_rows([], [session, experiment])
    assert [row.text for row in rows] == ["one", "two", "three"]


def test_a_resumed_agent_hears_what_came_while_it_was_away(tmp_path):
    session = tmp_path / "s" / EVENTS_FILENAME
    experiment = tmp_path / "e" / EVENTS_FILENAME
    seen = {"id": "x-1", "kind": "task_started", "timestamp": 1.0, "payload": {}}
    later = {"id": "x-3", "kind": "task_completed", "timestamp": 3.0, "payload": {}}
    _write(experiment, [seen, later])
    _write(
        session,
        [_message("x-0", 0.5, "already read"), _message("x-2", 2.0, "while away")],
    )

    found, resumed = read_events_after_in([session, experiment], "x-1")

    assert resumed
    assert [r["id"] for r in found] == ["x-2", "x-3"]
    assert read_events_after_in([session, experiment], "gone") == ([], False)


def test_with_no_experiment_open_the_session_keeps_everything(microscope, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    recorder = EventRecorder(microscope, experiment_path=plain, plain=True)
    recorder.set_agent_session("s1")
    recorder.buffer.append(NOTE, note_record("[intake] one grid"), actor=AGENT)
    experiment = tmp_path / "exp"
    experiment.mkdir()
    recorder.set_experiment(experiment)
    recorder.buffer.append(NOTE, note_record("[run_state] grid 1"), actor=AGENT)
    recorder.close()

    session = agent_files.session_folder("s1") / EVENTS_FILENAME
    texts = [r["payload"].get("text") for r in _written(session)]
    assert texts == ["[intake] one grid"]
    assert [
        r["payload"].get("text") for r in _written(experiment / EVENTS_FILENAME)
    ] == ["[run_state] grid 1"]
    assert not (plain / EVENTS_FILENAME).exists()


def test_notes_are_read_from_both_records(tmp_path):
    from fibsem.server.agent.notes import latest_notes

    session = tmp_path / "s" / EVENTS_FILENAME
    experiment = tmp_path / "e" / EVENTS_FILENAME

    def note(event_id, t, text):
        return {
            "id": event_id,
            "kind": NOTE,
            "timestamp": t,
            "actor": AGENT,
            "payload": {"text": text},
        }

    _write(
        session, [note("x-1", 1.0, "[intake] early"), note("x-3", 3.0, "[run_state] b")]
    )
    _write(experiment, [note("x-2", 2.0, "[run_state] a")])

    answer = latest_notes([], [session, experiment], ["intake", "run_state"], each=2)

    assert [(n["tag"], n["text"]) for n in answer["notes"]] == [
        ("intake", "[intake] early"),
        ("run_state", "[run_state] b"),
        ("run_state", "[run_state] a"),
    ]


def test_a_reconnected_instrument_keeps_the_running_session(microscope, tmp_path):
    from fibsem.workflows.session import AppSession

    session = AppSession()
    session.set_agent_session("s1")
    session.microscope = microscope
    session.start_recording()
    _say(session.recorder, "after connecting")
    session.stop_recording()
    session.set_agent_session(None)
    file = agent_files.session_folder("s1") / EVENTS_FILENAME
    assert [r["payload"]["text"] for r in _written(file)] == ["after connecting"]
