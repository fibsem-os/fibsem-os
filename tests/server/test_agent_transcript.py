"""The agent's transcript, kept beside the experiment, and what the agent said
it can do: ``/app/agent/transcript`` and the session's facts."""

import json
import time

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.ai.agent.session import session_facts  # noqa: E402
from fibsem.server import AuthConfig  # noqa: E402
from fibsem.server.agent_session import report  # noqa: E402
from fibsem.server.agent_transcript import build_agent_transcript_router  # noqa: E402


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


def test_with_no_experiment_open_nothing_is_kept(server):
    client, experiment, _ = server
    experiment["path"] = None
    answer = _post(client, "s1", {"kind": "said", "text": "x"})
    assert answer.status_code == 409
    assert answer.json()["detail"]["error_type"] == "no_experiment"
    assert client.get("/app/agent/transcript").json() == {"entries": []}


@pytest.mark.parametrize(
    "session_id, entry",
    [
        ("../up", {"kind": "said", "text": "x"}),
        ("s1", {"kind": "shout", "text": "x"}),
        ("s1", {"kind": "said", "text": "x" * 2001}),
        ("s1", {"kind": "said", "text": "x", "age_s": "soon"}),
        ("s1", {"kind": "step", "text": "x", "verb": 3}),
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
            "skills": ["autolamella-agent", "supervise-autolamella"],
            "resumed": False,
        },
    )
    assert dict(session_facts(session)) == {
        "Harness": "fibsem-agent",
        "Model": "claude-opus, effort high",
        "Code": "On in /Users/lab/fibsem-work (sandboxed)",
        "Skills": "autolamella-agent, supervise-autolamella",
        "Session id": "abc (new)",
    }
    off, _ = report(auth, {"session_id": "def", "code": False, "resumed": True})
    facts = dict(session_facts(off))
    assert facts["Code"] == "Off" and facts["Session id"] == "def (resumed)"
    # Claude Code says none of it: nothing to list but what it reported.
    assert session_facts({"model": "m"}) == [("Model", "m")]
    assert session_facts(None) == []


@pytest.mark.parametrize(
    "body", [{"code": "yes"}, {"skills": "a"}, {"skills": [1]}, {"resumed": 1}]
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
