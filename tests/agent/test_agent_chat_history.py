"""The Agent page chat from the whole record: past an app restart, which
empties the live buffer, and back through a long run a page at a time."""

import json

from fastapi.testclient import TestClient

from fibsem.acting import AGENT, OPERATOR
from fibsem.ai.agent.recent import MUTED, EarlierRecords, read_rows
from fibsem.server.agent.host import EVENTS_FILENAME, build_agent_host
from fibsem.server.auth import AuthConfig

AGENT_TOKEN = "agent-token"
OPERATOR_TOKEN = "operator-token"


def _record(stream, seq, kind, payload, actor=AGENT):
    return {
        "id": f"{stream}-{seq}",
        "seq": seq,
        "timestamp": float(seq),
        "t": f"2026-11-23T10:{seq // 60 % 60:02d}:{seq % 60:02d}.000",
        "kind": kind,
        "actor": actor,
        "payload": payload,
    }


def _note(stream, seq, text):
    return _record(stream, seq, "note", {"text": text})


def _write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_after_a_restart_the_chat_comes_back_from_the_file(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    before = [
        _record("a", 1, "note", {"text": "Skip 04.", "to": "agent"}, OPERATOR),
        _note("a", 2, "Skipping 04."),
        _record(
            "a",
            3,
            "command_started",
            {"service": "imaging", "command": "overview", "run_id": "r1"},
        ),
        _record("a", 4, "milling_progress", {"progress": 0.5}),
    ]
    _write(path, before)

    # The app started again: its buffer holds only what came since.
    held = [{**_note("b", 1, "Back after the restart."), "timestamp": 100.0}]
    rows, more = read_rows(held, [path])
    assert more is False
    assert [row.text for row in rows] == [
        "Skip 04.",
        "Skipping 04.",
        "Overview (imaging)",
        "Back after the restart.",
    ]
    # What was running when the app stopped ended with it.
    assert (rows[2].outcome, rows[2].tone) == ("app restarted", MUTED)
    # Nothing new since the restart: the buffer's stream says so.
    rows, _ = read_rows([], [path], stream="b")
    assert rows[-1].outcome == "app restarted"
    assert read_rows([], [path], stream="a")[0][-1].outcome == "running"
    # With no file, the buffer alone, as before.
    assert [row.text for row in read_rows(held, [])[0]] == ["Back after the restart."]


def test_show_earlier_pages_back_through_a_long_run(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    records = []
    seq = 0
    for n in range(120):
        seq += 1
        records.append(_note("a", seq, f"note {n}"))
        # A long run's progress between: far more than the buffer keeps.
        for _ in range(40):
            seq += 1
            records.append(_record("a", seq, "milling_progress", {}))
    _write(path, records)
    # The buffer holds the newest records, also on the file.
    held = records[-200:]
    earlier = EarlierRecords()

    rows, more = read_rows(held, [path], 50, earlier)
    assert [row.text for row in rows] == [f"note {n}" for n in range(70, 120)]
    assert more is True
    rows, more = read_rows(held, [path], 100, earlier)
    assert [row.text for row in rows] == [f"note {n}" for n in range(20, 120)]
    assert more is True
    # All of it: nothing earlier, and nothing twice.
    rows, more = read_rows(held, [path], 150, earlier)
    assert [row.text for row in rows] == [f"note {n}" for n in range(120)]
    assert more is False


def test_the_page_reads_the_chat_across_a_host_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    auth = AuthConfig(token=AGENT_TOKEN, operator_token=OPERATOR_TOKEN)
    headers = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
    first = build_agent_host(auth, tmp_path / "host")
    operator = TestClient(first, headers=headers)
    for text in ("one", "two", "three"):
        assert (
            operator.post("/app/agent/messages", json={"text": text}).status_code == 201
        )
    first.state.host.close()

    again = build_agent_host(auth, tmp_path / "host")
    operator = TestClient(again, headers=headers)
    try:
        answer = operator.get("/app/agent/recent").json()
        assert [row["text"] for row in answer["rows"]] == ["one", "two", "three"]
        assert answer["more"] is False
        page = operator.get("/app/agent/recent?rows=2").json()
        assert [row["text"] for row in page["rows"]] == ["two", "three"]
        assert page["more"] is True
        assert operator.get("/app/agent/recent?rows=0").status_code == 422
    finally:
        again.state.host.close()
