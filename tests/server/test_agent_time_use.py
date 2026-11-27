"""Where a session's time went, from its transcript: ``/app/agent/time`` and
`fibsem.server.agent.time_use`."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fibsem.ai.agent.levels import AgentLevel
from fibsem.server import AuthConfig
from fibsem.server.agent import time_use
from fibsem.server.agent.transcript import build_agent_transcript_router

T0 = datetime(2026, 11, 27, 21, 0, tzinfo=timezone.utc)


def _at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def _step(seq, at, tool, **extra):
    return {
        "seq": seq,
        "kind": "step",
        "verb": "Called",
        "text": tool,
        "tool": tool,
        "_at": _at(at),
        **extra,
    }


def _done(seq, at, step, took, **extra):
    return {
        "seq": seq,
        "kind": "done",
        "text": "",
        "step": step,
        "took_s": took,
        "_at": _at(at),
        **extra,
    }


def _turn(seq, at, took):
    return {"seq": seq, "kind": "turn", "text": "", "took_s": took, "_at": _at(at)}


def _session():
    """A turn of 100 s: an image (20 s), a crop (5 s) and a worker (50 s,
    which moved the stage for 10 s and read an item for 2 s); 300 s asleep;
    then a turn of 40 s with a 30 s mill."""
    return [
        _step(1, 0, "beam_acquire_image"),
        _done(2, 20, 1, 20.0),
        _step(3, 22, "analysis_crop_image"),
        _done(4, 27, 3, 5.0),
        _step(5, 30, "Agent", subagent="fibsem-worker"),
        _step(6, 32, "stage_stable_move", run=5),
        _done(7, 42, 6, 10.0, run=5),
        _step(8, 45, "item_get", run=5),
        _done(9, 47, 8, 2.0, run=5),
        _done(10, 80, 5, 50.0),
        _turn(11, 100, 100.0),
        _step(12, 405, "milling_run_stage"),
        _done(13, 435, 12, 30.0),
        _turn(14, 440, 40.0),
    ]


def test_a_session_splits_into_model_microscope_reads_subagents_and_between_turns():
    found = time_use.summarise(_session(), since=_at(0), until=_at(440))
    assert found["seconds"] == 440 and found["turns"] == 2
    assert found["in_turns_s"] == 140 and found["between_turns_s"] == 300
    lead = found["lead"]
    assert lead["microscope_s"] == 50  # the image and the mill
    assert lead["reads_s"] == 5
    assert lead["subagents_s"] == 50
    assert lead["tools_s"] == 105
    assert lead["model_s"] == 35  # 140 s in turns, 105 s of it in tools
    (worker,) = found["subagents"]
    assert worker == {
        "name": "fibsem-worker",
        "runs": 1,
        "wall_s": 50,
        "model_s": 38,
        "microscope_s": 10,
        "reads_s": 2,
    }
    (run,) = found["subagent_runs"]
    assert run["name"] == "fibsem-worker" and run["microscope_s"] == 10
    assert run["microscope_tools"] == ["stage_stable_move"]
    assert found["slowest_tools"][0] == {
        "tool": "milling_run_stage",
        "calls": 1,
        "total_s": 30,
        "mean_s": 30.0,
    }
    assert found["notes"] == []


def test_a_window_counts_only_what_falls_in_it():
    # The second turn alone: the mill, and the model's 10 s around it.
    found = time_use.summarise(_session(), since=_at(400), until=_at(440))
    assert found["turns"] == 1 and found["in_turns_s"] == 40
    assert found["lead"]["microscope_s"] == 30 and found["lead"]["model_s"] == 10
    assert found["subagents"] == []


@pytest.mark.parametrize(
    "tool, kind",
    [
        ("beam_acquire_image", "microscope"),
        ("stage_vertical_move", "microscope"),
        ("stage_get_position", "reads"),
        ("milling_preview_patterns", "reads"),
        ("service_wait_for_run", "microscope"),
        ("analysis_measure_coincidence_offset", "reads"),
        ("Skill", "reads"),
        ("Agent", "subagents"),
    ],
)
def test_each_tool_has_its_category(tool, kind):
    assert time_use.category(tool) == kind


def test_a_line_from_before_tools_were_named_is_read_from_its_words():
    old = {
        "seq": 1,
        "kind": "step",
        "verb": "Called",
        "text": "beam_auto_focus {}",
        "_at": _at(0),
    }
    found = time_use.summarise([old, _done(2, 8, 1, 8.0)], until=_at(10))
    assert found["lead"]["microscope_s"] == 8
    # No turn lines: the model's time is not known, and it says so.
    assert "model_s" not in found["lead"] and found["notes"]


@pytest.fixture
def client(tmp_path):
    (tmp_path / "exp").mkdir()
    app = FastAPI()
    app.state.auth = AuthConfig.generate(level=AgentLevel.WATCH)
    app.include_router(build_agent_transcript_router(lambda: str(tmp_path / "exp")))
    return TestClient(app, headers={"Authorization": f"Bearer {app.state.auth.token}"})


def test_the_route_answers_a_sessions_time_and_the_page_never_sees_turn_lines(client):
    entries = [
        {
            "seq": 1,
            "kind": "step",
            "verb": "Called",
            "text": "beam_acquire_image",
            "tool": "beam_acquire_image",
            "age_s": 60,
        },
        {"seq": 2, "kind": "done", "text": "", "step": 1, "took_s": 20, "age_s": 40},
        {"seq": 3, "kind": "turn", "text": "", "took_s": 60, "model_s": 35, "age_s": 0},
    ]
    posted = client.post(
        "/app/agent/transcript", json={"session_id": "s1", "entries": entries}
    )
    assert posted.json() == {"kept": 3}
    shown = client.get("/app/agent/transcript").json()["entries"]
    assert [e["kind"] for e in shown] == ["step", "done"]
    assert shown[0]["tool"] == "beam_acquire_image"

    found = client.get("/app/agent/time", params={"session_id": "s1"}).json()
    assert found["session_id"] == "s1" and found["turns"] == 1
    assert found["lead"]["microscope_s"] == 20
    assert 39 <= found["lead"]["model_s"] <= 41
    bad = client.get("/app/agent/time", params={"session_id": "s1", "since": "monday"})
    assert bad.status_code == 422
