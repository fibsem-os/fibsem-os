import json
from datetime import datetime, timezone

import pytest
import yaml

from fibsem.ai.evals.__main__ import main
from fibsem.ai.evals.core import cut, episode, segments
from fibsem.ai.evals.core.cases import check_case, load_case

BASE = datetime(2026, 11, 20, 3, tzinfo=timezone.utc).timestamp()


def _episode(tmp_path):
    exp = tmp_path / "grid-01"
    (exp / "Lam-01").mkdir(parents=True)
    for name in ("setup_ib.tif", "p1_eb.tif", "p1_ib.tif", "p2_eb.tif", "p2_ib.tif"):
        (exp / "Lam-01" / name).write_bytes(name.encode())
    (exp / "Lam-01" / "final_ib.tif").write_bytes(b"after")

    def image(i, t, name):
        return {
            "id": f"s-{i}",
            "timestamp": BASE + t,
            "t": f"2026-11-20T03:00:{t:02d}+00:00",
            "kind": "image_acquired",
            "actor": "agent",
            "item": "Lam-01",
            "payload": {"path": f"D:\\data\\grid-01\\Lam-01\\{name}"},
        }

    rows = [
        {
            "id": "s-0",
            "timestamp": BASE + 1,
            "kind": "agent_session_started",
            "actor": "agent",
            "payload": {"session_id": "abc"},
        },
        image(1, 2, "setup_ib.tif"),
        {
            "id": "s-2",
            "timestamp": BASE + 3,
            "kind": "action",
            "actor": "agent",
            "item": "Lam-01",
            "payload": {},
        },
        image(3, 4, "p1_eb.tif"),
        image(4, 5, "p1_ib.tif"),
        image(5, 6, "p2_eb.tif"),
        image(6, 7, "p2_ib.tif"),
        {
            "id": "s-7",
            "timestamp": BASE + 8,
            "kind": "action",
            "actor": "agent",
            "item": "Lam-01",
            "payload": {},
        },
        image(8, 9, "final_ib.tif"),
    ]
    (exp / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    store = tmp_path / "sessions" / "2026-11-20-abc"
    store.mkdir(parents=True)
    (store / "session.json").write_text(json.dumps({"session_id": "abc", "model": "m"}))
    (store / "transcript.jsonl").write_text(
        json.dumps({"timestamp": "2026-11-20T03:00:02Z", "said": "before"})
        + "\n"
        + json.dumps({"timestamp": "2026-11-20T03:00:30Z", "said": "after"})
        + "\n"
    )
    ep = episode.pack(exp, tmp_path / "ep", [tmp_path / "sessions"])
    segments.save(
        ep,
        {
            "format": 0,
            "segments": [
                {
                    "id": "seg-01",
                    "task": "setup_placement",
                    "item": "Lam-01",
                    "start_event": "s-2",
                    "end_event": "s-2",
                    "inputs": {"fib_image": "Lam-01/setup_ib.tif"},
                    "facts": {"milling_angle_deg": 15},
                    "decision": "centred on the target",
                    "answer": {"x_px": 10, "y_px": 20},
                    "override": {"what": "moved 1 um down", "said": "lower"},
                    "status": "kept",
                },
                {
                    "id": "seg-02",
                    "task": "polishing_stop",
                    "item": "Lam-01",
                    "start_event": "s-7",
                    "end_event": "s-7",
                    "inputs": {
                        "passes": [
                            {
                                "name": "P1",
                                "sem": "Lam-01/p1_eb.tif",
                                "fib": "Lam-01/p1_ib.tif",
                                "thickness_nm": 300,
                            },
                            {
                                "name": "P2",
                                "sem": "Lam-01/p2_eb.tif",
                                "fib": "Lam-01/p2_ib.tif",
                            },
                        ]
                    },
                    "facts": {"target_nm": 200},
                    "decision": "stopped after P2",
                    "status": "kept",
                },
                {
                    "id": "seg-03",
                    "task": "fm_target",
                    "item": "Lam-01",
                    "start_event": "s-2",
                    "end_event": "s-2",
                    "inputs": {"fm_stack": "Lam-01/setup_ib.tif"},
                    "decision": "target found",
                    "status": "kept",
                },
                {
                    "id": "seg-04",
                    "task": "setup_placement",
                    "item": "Lam-01",
                    "start_event": "s-2",
                    "end_event": "s-2",
                    "inputs": {"fib_image": "Lam-01/setup_ib.tif"},
                    "decision": "dropped one",
                    "status": "dropped",
                },
            ],
        },
    )
    return ep


def test_cut_cases_from_kept_segments(tmp_path):
    ep = _episode(tmp_path)
    made = cut.cut_episode(ep, tmp_path / "cases")
    assert len(made["made"]) == 2
    assert any("no cutter yet for fm_target" in s for s in made["skipped"])

    placement = load_case(tmp_path / "cases" / "setup-placement" / "ep-seg-01")
    assert check_case(placement).ok
    assert placement.status == "unkeyed"
    assert placement.tier == "real"
    assert placement.data["inputs"]["fib_image"] == "experiment/Lam-01/setup_ib.tif"
    assert placement.data["inputs"]["milling_angle_deg"] == 15
    assert placement.data["live"]["override"]["said"] == "lower"
    assert placement.data["source"]["start_event"] == "s-2"
    # nothing after the cut in experiment/, what came after in reference/
    seen = {p.name for p in (placement.path / "experiment").rglob("*.tif")}
    assert seen == {"setup_ib.tif"}
    later = {p.name for p in (placement.path / "reference" / "later").rglob("*")}
    assert "final_ib.tif" in later and "p1_ib.tif" in later
    transcript = (placement.path / "reference" / cut.TRANSCRIPT).read_text()
    assert "before" in transcript and "after" not in transcript

    polishing = load_case(tmp_path / "cases" / "polishing-stop" / "ep-seg-02")
    assert check_case(polishing).ok
    assert [p["name"] for p in polishing.data["passes"]] == ["P1", "P2"]
    assert polishing.data["passes"][0]["measured"] == {"median_nm": 300}
    assert "200 nm" in polishing.data["brief"]
    context = yaml.safe_load((polishing.path / cut.CONTEXT).read_text())
    assert context["earlier_decisions"][0]["override"]["what"] == "moved 1 um down"


def test_cut_refuses_segments_with_problems(tmp_path):
    ep = _episode(tmp_path)
    data = segments.load(ep)
    data["segments"][0]["start_event"] = "nope"
    segments.save(ep, data)
    with pytest.raises(cut.CutError):
        cut.cut_episode(ep, tmp_path / "cases")


def test_cli_from_run(tmp_path, capsys):
    ep = _episode(tmp_path)
    assert main(["from-run", str(ep.path), "--out", str(tmp_path / "cases")]) == 0
    out = capsys.readouterr().out
    assert "2 cases, 1 skipped" in out
    assert main(["check", str(tmp_path / "cases")]) == 0
