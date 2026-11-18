"""The replay page of a live run, from its own record."""

import json

import numpy as np
import tifffile

from fibsem.ai.evals.core import replay


def _event(t, kind, payload, actor="agent"):
    return {"timestamp": t, "kind": kind, "actor": actor, "payload": payload}


def _run(tmp_path):
    repeat = tmp_path / "out" / "case-1" / "r1"
    experiment = repeat / "home" / "experiments" / "exp"
    experiment.mkdir(parents=True)
    image = experiment / "P00_ib.tif"
    tifffile.imwrite(
        image, (np.random.default_rng(0).random((64, 96)) * 255).astype(np.uint8)
    )
    stage = {"name": "Polishing Mill 01", "milling": {"milling_current": 6e-11}}
    events = [
        _event(0.0, "grid_loaded", {"grid": "Grid-01"}, "operator"),
        _event(
            1.0,
            "milling_stage_started",
            {"milling_task_name": "Polishing", "stage": stage},
        ),
        _event(2.0, "milling_progress", {"status": "running"}),
        _event(
            3.0, "milling", {"milling_task_name": "Polishing", "status": "completed"}
        ),
        _event(
            4.0,
            "image_acquired",
            {"path": str(image), "beam_type": "ION", "filename": "P00", "hfw": 3e-5},
        ),
    ]
    (experiment / replay.EVENTS).write_text("\n".join(json.dumps(e) for e in events))
    (repeat / replay.SIM_RECORD).write_text(
        json.dumps({"passes": [{"pass": 0, "gap": 9e-7, "gap_mean": 9e-7}]})
    )
    (repeat / "result.json").write_text(
        json.dumps({"case": "case-1", "repeat": 1, "scores": [{"result": "hit"}]})
    )
    (repeat / "work").mkdir()
    (repeat / "work" / "answer.json").write_text(json.dumps({"thickness_nm": 900}))
    return repeat


def test_steps_name_each_event_and_count_the_passes(tmp_path):
    repeat = _run(tmp_path)
    steps = replay.steps_of(replay._events(repeat / "home"), [{"pass": 0}])
    assert [s["kind"] for s in steps] == [
        "grid_loaded",
        "milling_stage_started",
        "milling",
        "image_acquired",
    ]
    assert steps[1]["text"] == "milling Polishing: Polishing Mill 01 at 60 pA"
    assert [s["passes_done"] for s in steps] == [0, 0, 1, 1]


def test_the_page_holds_the_steps_and_the_image(tmp_path):
    repeat = _run(tmp_path)
    written = replay.write_replays(tmp_path / "out")
    assert written == [repeat / replay.REPLAY]
    page = written[0].read_text()
    assert "case-1" in page and "Polishing Mill 01" in page and "iVBOR" in page
