"""The agent's transcript, kept beside the experiment, and what the agent said
it can do: ``/app/agent/transcript`` and the session's facts."""

import base64
import json
import time
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fibsem.ai.agent.levels import AgentLevel
from fibsem.ai.agent.session import session_facts, session_line
from fibsem.server import AuthConfig
from fibsem.server.agent.session import report
from fibsem.server.agent.transcript import build_agent_transcript_router


@pytest.fixture
def server(tmp_path):
    """The route on its own, over an experiment folder that can be closed."""
    experiment = {"path": str(tmp_path / "exp")}
    (tmp_path / "exp").mkdir()
    app = FastAPI()
    app.state.auth = AuthConfig.generate(level=AgentLevel.WATCH)
    app.include_router(build_agent_transcript_router(lambda: experiment["path"]))
    client = TestClient(
        app, headers={"Authorization": f"Bearer {app.state.auth.token}"}
    )
    return client, experiment, tmp_path / "exp"


def _post(client, session_id, *entries):
    return client.post(
        "/app/agent/transcript",
        json={"session_id": session_id, "entries": list(entries)},
    )


def test_the_lines_are_kept_in_their_own_file_and_read_back_in_order(server):
    client, _, folder = server
    said = {"seq": 1, "kind": "said", "text": "Measuring it.", "age_s": 5}
    ran = {"seq": 2, "kind": "step", "verb": "Ran", "text": "python a.py", "age_s": 2}
    assert _post(client, "s1", said, ran).json() == {"kept": 2}
    failed = {"seq": 3, "kind": "failed", "text": "no cv2", "step": 2}
    _post(client, "s1", failed)
    _post(client, "s2", {"kind": "step", "verb": "Wrote", "text": "b.py"})

    kept = (folder / "agent" / "transcript-s1.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in kept] == ["said", "step", "failed"]
    # Not on the record: the experiment's events stay as they were.
    assert not (folder / "events.jsonl").exists()

    entries = client.get("/app/agent/transcript").json()["entries"]
    assert [e["text"] for e in entries] == [
        "Measuring it.",
        "python a.py",
        "no cv2",
        "b.py",
    ]
    assert entries[2]["step"] == 2 and entries[2]["session_id"] == "s1"
    # Stamped by the server, back by its age: the order things happened in.
    times = [e["t"] for e in entries]
    since = client.get("/app/agent/transcript", params={"since": times[1]}).json()
    assert [e["text"] for e in since["entries"]] == ["python a.py", "no cv2", "b.py"]


def test_with_no_experiment_open_it_waits_in_the_sessions_folder(server, tmp_path):
    client, experiment, folder = server
    experiment["path"] = None
    report(client.app.state.auth, {"session_id": "s1"})
    assert _post(client, "s1", {"kind": "said", "text": "before"}).status_code == 200
    assert not (folder / "agent").exists()
    entries = client.get("/app/agent/transcript").json()["entries"]
    assert [e["text"] for e in entries] == ["before"]
    # The first experiment that opens takes it, and what follows.
    experiment["path"] = str(folder)
    _post(client, "s1", {"kind": "said", "text": "after"})
    kept = (folder / "agent" / "transcript-s1.jsonl").read_text().splitlines()
    assert [json.loads(line)["text"] for line in kept] == ["before", "after"]
    entries = client.get("/app/agent/transcript").json()["entries"]
    assert [e["text"] for e in entries] == ["before", "after"]


@pytest.mark.parametrize(
    "session_id, entry",
    [
        ("../up", {"kind": "said", "text": "x"}),
        ("s1", {"kind": "shout", "text": "x"}),
        ("s1", {"kind": "said", "text": "x" * 2001}),
        ("s1", {"kind": "said", "text": "x", "age_s": "soon"}),
        ("s1", {"kind": "step", "text": "x", "verb": 3}),
        ("s1", {"kind": "done", "text": "", "step": 2, "took_s": -1}),
        ("s1", {"kind": "done", "text": "", "step": 2, "took_s": "long"}),
        ("s1", "said x"),
    ],
)
def test_a_bad_post_is_refused_and_keeps_nothing(server, session_id, entry):
    client, _, folder = server
    assert _post(client, session_id, entry).status_code == 422
    assert not (folder / "agent").exists()


def test_the_agent_says_what_it_can_do_and_the_chip_lists_it():
    auth = AuthConfig.generate(level=AgentLevel.WATCH)
    session, _ = report(
        auth,
        {
            "session_id": "abc",
            "model": "claude-opus",
            "harness": "fibsem-agent",
            "effort": "high",
            "code": True,
            "workdir": "/Users/lab/fibsem-work",
            "skills": ["autolamella", "autolamella-lead"],
            "resumed": False,
        },
    )
    assert dict(session_facts(session)) == {
        "Harness": "fibsem-agent",
        "Model": "claude-opus, effort high",
        "Code": "On in /Users/lab/fibsem-work (sandboxed)",
        "Skills": "autolamella, autolamella-lead",
        "Session id": "abc (new)",
    }
    off, _ = report(auth, {"session_id": "def", "code": False, "resumed": True})
    facts = dict(session_facts(off))
    assert facts["Code"] == "Off" and facts["Session id"] == "def (resumed)"
    files, _ = report(auth, {"session_id": "f", "files": True, "workdir": "/w"})
    assert dict(session_facts(files))["Code"] == "Off; reads and writes files in /w"
    # Claude Code says none of it: nothing to list but what it reported.
    assert session_facts({"model": "m"}) == [("Model", "m")]
    named = {"playbook": {"name": "night", "hash": "abc"}}
    assert session_facts(named) == [("Playbook", "night (abc)")]
    assert session_facts(None) == []


def test_the_agent_pages_session_line_leaves_the_goal_and_budget_to_the_panel():
    session = {
        "model": "claude-opus",
        "goal": "Mill 01",
        "code": True,
        "spent_usd": 3.0,
        "budget_usd": 4.0,
        "started": "2026-10-23T23:12:00+00:00",
    }
    line = session_line(session)
    started = datetime.fromisoformat(session["started"]).astimezone()
    assert line == f"claude-opus · code mode · started {started:%H:%M}"
    ended = {"model": "claude-opus", "ended": True, "reason": "its budget is spent"}
    assert session_line(ended) == "claude-opus · ended: its budget is spent"
    assert session_line({"goal": "Mill 01"}) is None
    assert session_line(None) is None


@pytest.mark.parametrize(
    "body",
    [
        {"code": "yes"},
        {"files": "yes"},
        {"skills": "a"},
        {"skills": [1]},
        {"resumed": 1},
    ],
)
def test_a_bad_capability_is_refused(body):
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        report(AuthConfig.generate(level=AgentLevel.WATCH), body)


def test_an_old_entry_is_placed_by_its_age_capped_at_a_day(server):
    client, _, _ = server
    _post(client, "s1", {"kind": "said", "text": "x", "age_s": 10 * 24 * 3600})
    (entry,) = client.get("/app/agent/transcript").json()["entries"]
    from datetime import datetime

    age = time.time() - datetime.fromisoformat(entry["t"]).timestamp()
    assert 24 * 3600 - 5 < age < 24 * 3600 + 5


def test_a_done_line_names_its_step_and_each_post_is_an_event(tmp_path):
    """A done line says how long its step took; each post puts one live event
    on the app's events, so the web page reads the lines at once."""
    (tmp_path / "exp").mkdir()
    emitted = []
    app = FastAPI()
    app.state.auth = AuthConfig.generate(level=AgentLevel.WATCH)
    app.include_router(
        build_agent_transcript_router(
            lambda: str(tmp_path / "exp"),
            lambda kind, data: emitted.append((kind, data)),
        )
    )
    client = TestClient(
        app, headers={"Authorization": f"Bearer {app.state.auth.token}"}
    )
    ran = {"seq": 1, "kind": "step", "verb": "Called", "text": "milling_run"}
    done = {"seq": 2, "kind": "done", "text": "", "step": 1, "took_s": 312.04}
    assert _post(client, "s1", ran, done).json() == {"kept": 2}
    entries = client.get("/app/agent/transcript").json()["entries"]
    assert entries[1]["kind"] == "done" and entries[1]["step"] == 1
    assert entries[1]["took_s"] == 312.0
    assert emitted == [("agent_transcript", {"session_id": "s1", "kept": 2})]
    # A live kind: a run's history leaves it out.
    from fibsem.workflows.events.kinds import LIVE_KINDS

    assert "agent_transcript" in LIVE_KINDS


PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n not really").decode()


def test_a_picture_is_kept_beside_the_transcript_and_served(server):
    client, _, folder = server
    step = {"seq": 4, "kind": "step", "verb": "Called", "text": "notebook read_page"}
    picture = {
        "seq": 5,
        "kind": "picture",
        "text": "",
        "step": 4,
        "image": PNG,
        "media_type": "image/png",
    }
    assert _post(client, "s1", step, picture).json() == {"kept": 2}
    saved = folder / "agent" / "pictures" / "s1-5.png"
    assert saved.read_bytes() == base64.b64decode(PNG)
    # The line keeps the picture's path, not the picture.
    line = json.loads(
        (folder / "agent" / "transcript-s1.jsonl").read_text().splitlines()[-1]
    )
    assert line["path"] == "pictures/s1-5.png" and "image" not in line
    entries = client.get("/app/agent/transcript").json()["entries"]
    assert entries[-1]["path"] == "pictures/s1-5.png"

    served = client.get("/app/agent/transcript/pictures/s1-5.png")
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"
    assert served.content == base64.b64decode(PNG)
    for name in ("s1-6.png", "..%2Ftranscript-s1.jsonl", "s1-5.svg"):
        missing = client.get(f"/app/agent/transcript/pictures/{name}")
        assert missing.status_code == 404, name


@pytest.mark.parametrize(
    "change",
    [
        {"media_type": "image/svg+xml"},
        {"image": "not base64!"},
        {"image": ""},
        {"seq": None},
    ],
)
def test_a_bad_picture_is_refused_and_keeps_nothing(server, change):
    client, _, folder = server
    picture = {"seq": 5, "kind": "picture", "text": "", "image": PNG}
    picture["media_type"] = "image/png"
    picture.update(change)
    picture = {k: v for k, v in picture.items() if v is not None}
    assert _post(client, "s1", picture).status_code == 422
    assert not (folder / "agent" / "pictures").exists()


def test_a_waiting_picture_moves_into_the_experiment(server):
    client, experiment, folder = server
    experiment["path"] = None
    report(client.app.state.auth, {"session_id": "s1"})
    picture = {"seq": 2, "kind": "picture", "text": "", "image": PNG}
    picture["media_type"] = "image/png"
    assert _post(client, "s1", picture).status_code == 200
    assert client.get("/app/agent/transcript/pictures/s1-2.png").status_code == 200
    experiment["path"] = str(folder)
    _post(client, "s1", {"kind": "said", "text": "after"})
    assert (folder / "agent" / "pictures" / "s1-2.png").is_file()
    assert client.get("/app/agent/transcript/pictures/s1-2.png").status_code == 200


def test_a_subagent_s_lines_keep_their_run_and_its_call_names_the_subagent(server):
    client, _, _ = server
    call = {
        "seq": 1,
        "kind": "step",
        "verb": "Asked",
        "text": "Check 02",
        "subagent": "autolamella-verifier",
    }
    said = {"seq": 2, "kind": "said", "text": "Reading.", "run": 1}
    assert _post(client, "s1", call, said).json() == {"kept": 2}
    first, second = client.get("/app/agent/transcript").json()["entries"]
    assert first["subagent"] == "autolamella-verifier" and "run" not in first
    assert second["run"] == 1
    done = {"kind": "done", "text": "left edge", "step": 1, "verdict": "concern"}
    _post(client, "s1", done)
    assert (
        client.get("/app/agent/transcript").json()["entries"][-1]["verdict"]
        == "concern"
    )
    assert _post(client, "s1", {**done, "verdict": "maybe"}).status_code == 422
    assert _post(client, "s1", {**done, "attention": "yes"}).status_code == 422
    assert (
        _post(client, "s1", {"kind": "said", "text": "x", "run": "1"}).status_code
        == 422
    )
    assert (
        _post(
            client, "s1", {"kind": "step", "text": "x", "subagent": "a" * 61}
        ).status_code
        == 422
    )


def test_a_resumed_session_s_lines_follow_on_and_a_cut_step_has_ended(server):
    """A resumed session numbers its lines from 1 again: read back, they follow
    on from the run before, and that run's unanswered steps have ended (the page
    would otherwise show an Auditor run from last week as running)."""
    client, _, _ = server
    call = {"seq": 1, "kind": "step", "text": "Audit", "subagent": "auditor"}
    said = {"seq": 2, "kind": "said", "text": "Reading.", "run": 1}
    read = {"seq": 3, "kind": "step", "text": "Read log", "run": 1}
    _post(client, "s1", call, said, read)
    # Resumed: the new run's own step 1, answered, and its step 2 still running.
    again = {"seq": 1, "kind": "step", "text": "Read plan"}
    done = {"seq": 2, "kind": "done", "text": "", "step": 1}
    now = {"seq": 3, "kind": "step", "text": "Take image"}
    _post(client, "s1", again, done, now)
    lines = client.get("/app/agent/transcript").json()["entries"]
    assert [line["seq"] for line in lines] == [1, 2, 3, 4, 5, 6]
    assert lines[1]["run"] == 1 and lines[2]["run"] == 1
    assert lines[4]["step"] == 4
    assert [bool(line.get("ended")) for line in lines] == [
        True,
        False,
        True,
        False,
        False,
        False,
    ]


def test_every_subagent_run_is_listed_and_one_run_s_lines_read(server):
    """The Subagents tab lists every run however old: each Agent call's line
    and its answer; a run's own lines come on their own."""
    client, _, _ = server
    old = {"seq": 1, "kind": "step", "text": "Find sites", "subagent": "site-finder"}
    said = {"seq": 2, "kind": "said", "text": "Looking.", "run": 1}
    looked = {"seq": 3, "kind": "step", "text": "Read grid", "run": 1}
    looked_done = {"seq": 4, "kind": "done", "text": "", "step": 3, "run": 1}
    found = {"seq": 5, "kind": "done", "text": "Six sites.", "step": 1}
    mine = {"seq": 6, "kind": "step", "text": "Move stage"}
    _post(client, "s1", {**old, "age_s": 3600}, said, looked, looked_done, found, mine)
    check = {"seq": 1, "kind": "step", "text": "Check", "subagent": "verifier"}
    _post(client, "s2", check)
    runs = client.get("/app/agent/transcript/runs").json()["entries"]
    assert [(r["session_id"], r["seq"], r["kind"]) for r in runs] == [
        ("s1", 1, "step"),
        ("s1", 5, "done"),
        ("s2", 1, "step"),
    ]
    lines = client.get("/app/agent/transcript/runs/s1/1").json()["entries"]
    assert [line["seq"] for line in lines] == [2, 3, 4]
    assert client.get("/app/agent/transcript/runs/s9/1").json() == {"entries": []}
    assert client.get("/app/agent/transcript/runs/a.b/1").status_code == 422


def test_a_transcript_file_is_read_once_then_only_what_was_added(tmp_path):
    from fibsem.server.agent import transcript

    path = tmp_path / "transcript-s1.jsonl"
    line = {"t": "2026-11-23T10:00:00+11:00", "kind": "said", "text": "a", "seq": 1}
    path.write_text(json.dumps(line) + "\n" + '{"t": "2026-11-23T10:0', "utf-8")
    assert [e["seq"] for e in transcript.read(tmp_path)] == [1]
    with open(path, "a", encoding="utf-8") as f:
        f.write('1:00+11:00", "kind": "said", "text": "b", "seq": 2}\n')
    assert [e["seq"] for e in transcript.read(tmp_path)] == [1, 2]
    # Read back, the renumbering of one read never leaks into the next.
    assert [e["seq"] for e in transcript.read(tmp_path)] == [1, 2]
    path.write_text(json.dumps({**line, "seq": 7}) + "\n", "utf-8")
    assert [e["seq"] for e in transcript.read(tmp_path)] == [7]
