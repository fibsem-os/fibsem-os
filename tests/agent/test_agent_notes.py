"""The latest notes by tag, from the whole record: past what the live buffer
holds, and across a restart."""

import json

import pytest
from fastapi.testclient import TestClient

from fibsem.server.agent.host import EVENTS_FILENAME, build_agent_host
from fibsem.server.agent.notes import NotesRefused, asked_tags, latest_notes, tag_of
from fibsem.server.auth import AuthConfig

AGENT = "agent-token"
OPERATOR = "operator-token"


def _note(seq, text, **payload):
    return {
        "id": f"s-{seq}",
        "seq": seq,
        "timestamp": float(seq),
        "kind": "note",
        "payload": {"text": text, **payload},
    }


def _progress(seq):
    return {"id": f"s-{seq}", "seq": seq, "kind": "milling_progress", "payload": {}}


def test_a_tag_is_the_notes_first_bracketed_word():
    assert tag_of(_note(1, "[run_state v3] Lamella 01 polished")) == "run_state"
    assert tag_of(_note(1, "  [Intake] brief")) == "intake"
    assert tag_of(_note(1, "no tag here")) is None
    # A message to the operator is not a note.
    assert tag_of(_note(1, "[run_state] x", to="operator")) is None
    assert tag_of({"kind": "action", "payload": {"text": "[intake]"}}) is None
    assert asked_tags(["[intake], run_state", "RUN_STATE"]) == ["intake", "run_state"]
    for bad in ([], [" , "], ["run state"], [",".join(f"t{i}" for i in range(11))]):
        with pytest.raises(NotesRefused):
            asked_tags(bad)


def test_the_newest_of_each_tag_from_the_buffer_then_the_file(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    old = [_note(1, "[intake] the brief"), _note(2, "[run_state v1] start")]
    # A long run's progress: far more than the live buffer keeps.
    old += [_progress(seq) for seq in range(3, 5000)]
    old += [_note(5000, "[run_state v2] two done")]
    path.write_text("".join(json.dumps(r) + "\n" for r in old))
    held = [_note(5000, "[run_state v2] two done"), _note(5001, "[run_state v3] three")]

    answer = latest_notes(held, [path], ["intake", "run_state", "recipe_card"])
    assert [(n["tag"], n["text"]) for n in answer["notes"]] == [
        ("intake", "[intake] the brief"),
        ("run_state", "[run_state v3] three"),
    ]
    assert answer["missing"] == ["recipe_card"]
    # Each counts the buffer's and the file's together, without repeats.
    two = latest_notes(held, [path], ["run_state"], each=3)
    assert [n["id"] for n in two["notes"]] == ["s-5001", "s-5000", "s-2"]
    # With no file (no experiment open), the buffer alone.
    alone = latest_notes(held, [], ["intake"])
    assert alone == {"notes": [], "missing": ["intake"]}
    with pytest.raises(NotesRefused):
        latest_notes(held, [path], ["intake"], each=0)


def test_the_route_reads_a_note_written_before_a_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    auth = AuthConfig(token=AGENT, operator_token=OPERATOR)
    first = build_agent_host(auth, tmp_path / "host")
    agent = TestClient(first, headers={"Authorization": f"Bearer {AGENT}"})
    assert (
        agent.post(
            "/app/experiment/notes", json={"text": "[intake] two lamellae, 200 nm"}
        ).status_code
        == 200
    )
    first.state.host.close()

    again = build_agent_host(auth, tmp_path / "host")
    agent = TestClient(again, headers={"Authorization": f"Bearer {AGENT}"})
    agent.post("/app/experiment/notes", json={"text": "[run_state v1] sites picked"})
    # The live buffer starts empty after the restart: the long-poll has only
    # the new note, the notes read has both.
    held = agent.get("/app/events").json()["events"]
    assert [e["payload"]["text"] for e in held] == ["[run_state v1] sites picked"]
    answer = agent.get("/app/experiment/notes?tags=intake,run_state").json()
    again.state.host.close()
    assert [n["text"] for n in answer["notes"]] == [
        "[intake] two lamellae, 200 nm",
        "[run_state v1] sites picked",
    ]
    assert answer["missing"] == []
    refused = agent.get("/app/experiment/notes?tags=run%20state")
    assert refused.status_code == 422
    assert agent.get("/app/experiment/notes").status_code == 422
