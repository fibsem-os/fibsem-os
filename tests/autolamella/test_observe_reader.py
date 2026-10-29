"""The Observer's offline reader (FIB-1282): an experiment folder read with no
app running, never written to, and never through ``Experiment.load()``."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

READER = (
    Path(__file__).resolve().parents[2]
    / ".claude"
    / "skills"
    / "autolamella-agent"
    / "read_record.py"
)

FIDUCIAL_ERROR = "Alignment failed: fiducial not found in the FIB image"


@pytest.fixture(scope="module")
def reader():
    spec = importlib.util.spec_from_file_location("read_record", READER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _history(name, status, message, start, end):
    return {
        "name": name,
        "status": status,
        "status_message": message,
        "start_timestamp": f"2026-10-12T01:{start}+00:00",
        "end_timestamp": f"2026-10-12T01:{end}+00:00",
    }


def _event(seq, t, kind, item, task, payload=None, actor="task"):
    return {
        "session": "s",
        "actor": actor,
        "experiment": {"id": "e", "name": "observed"},
        "item": {"id": item, "name": item} if item else None,
        "task": {"id": f"{item}-{task}", "name": task} if task else None,
        "seq": seq,
        "timestamp": 0,
        "t": f"2026-10-12T01:{t}+00:00",
        "kind": kind,
        "payload": payload or {},
    }


@pytest.fixture
def folder(tmp_path) -> Path:
    """Two lamellae; Mill Fiducial fails on the second, and the agent left a note."""
    root = tmp_path / "observed"
    for name in ("01-tough-mako", "02-loved-liger"):
        (root / name).mkdir(parents=True)
    (root / "01-tough-mako" / "ref_Mill Fiducial_final_ib.tif").write_bytes(b"")

    positions = [
        {
            "petname": "01-tough-mako",
            "number": 1,
            "task_state": {"name": "Mill Fiducial", "status": "Completed"},
            "task_history": [
                _history("Mill Fiducial", "Completed", "Finished", "00:00", "00:30")
            ],
            "defect": {"verdict": "UNASSESSED"},
        },
        {
            "petname": "02-loved-liger",
            "number": 2,
            "task_state": {"name": "Mill Fiducial", "status": "Failed"},
            "task_history": [
                _history("Mill Fiducial", "Failed", FIDUCIAL_ERROR, "01:00", "01:10")
            ],
            "defect": {"verdict": "UNASSESSED"},
        },
    ]
    (root / "experiment.yaml").write_text(
        yaml.safe_dump({"name": "observed", "positions": positions})
    )
    (root / "protocol.yaml").write_text(
        yaml.safe_dump(
            {"workflow": {"tasks": [{"name": "Mill Fiducial", "required": True}]}}
        )
    )
    outputs = {"final_fib": ["ref_Mill Fiducial_final_ib.tif"]}
    events = [
        _event(1, "00:00", "task_started", "01-tough-mako", "Mill Fiducial"),
        _event(
            2,
            "00:30",
            "task_completed",
            "01-tough-mako",
            "Mill Fiducial",
            {"task_state": {"outputs": outputs}},
        ),
        _event(3, "01:00", "task_started", "02-loved-liger", "Mill Fiducial"),
        _event(
            4,
            "01:10",
            "task_failed",
            "02-loved-liger",
            "Mill Fiducial",
            {"error": FIDUCIAL_ERROR, "task_state": {"step": "STARTED"}},
        ),
        _event(5, "01:30", "task_started", "02-loved-liger", "Rough Milling"),
        _event(
            6,
            "01:31",
            "alignment",
            "02-loved-liger",
            "Rough Milling",
            {
                "status": "skipped",
                "reason": "reference image ref_alignment_ib.tif does not exist",
            },
        ),
        _event(7, "01:40", "task_completed", "02-loved-liger", "Rough Milling"),
        _event(
            8,
            "02:00",
            "agent_note",
            None,
            None,
            {"text": "fiducial out of frame", "item_name": "02-loved-liger"},
            actor="agent",
        ),
    ]
    (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (root / "logfile.log").write_text(
        "2026-10-12 01:00:20,000 — root — ERROR — run:12 — long before the run\n"
        "2026-10-12 01:01:09,500 — root — ERROR — _run:40 — cross-correlation 0.12\n"
        "Traceback (most recent call last):\n"
        "RuntimeError: fiducial not found\n"
        "2026-10-12 01:01:09,600 — root — INFO — _run:41 — not a warning\n",
        encoding="utf-8",
    )
    return root


def _files(folder: Path):
    return sorted((p, p.stat().st_mtime_ns) for p in folder.rglob("*"))


def test_the_overview_counts_yield_and_names_the_failure(reader, folder, capsys):
    before = _files(folder)
    reader.main([str(folder)])
    out = capsys.readouterr().out

    assert "Yield: 1 of 2 lamellae" in out
    assert (
        f"02-loved-liger / Mill Fiducial at 2026-10-12 01:01:10: {FIDUCIAL_ERROR}"
        in out
    )
    assert "Mill Fiducial: 1 completed, 1 failed" in out
    assert _files(folder) == before, "the reader wrote into the experiment"


def test_an_item_is_found_by_its_number_with_its_runs_images_and_notes(
    reader, folder, capsys
):
    reader.main([str(folder), "--item", "1"])
    first = capsys.readouterr().out
    assert first.startswith("Lamella 01-tough-mako")
    image = folder / "01-tough-mako" / "ref_Mill Fiducial_final_ib.tif"
    assert f"final_fib: {image}" in first

    reader.main([str(folder), "--item", "2"])
    second = capsys.readouterr().out
    assert f"Mill Fiducial: failed, 10s (waiting 0s) -- {FIDUCIAL_ERROR}" in second
    assert "stopped at step: STARTED" in second
    # the later run that went without what the failed one should have left
    assert (
        "alignment skipped: reference image ref_alignment_ib.tif does not exist"
        in second
    )
    assert "agent: fiducial out of frame" in second


def test_failures_show_the_log_around_the_failed_run_only(reader, folder, capsys):
    reader.main([str(folder), "--failures"])
    out = capsys.readouterr().out

    assert "cross-correlation 0.12" in out
    assert "RuntimeError: fiducial not found" in out
    assert "stopped at step: STARTED" in out
    assert "long before the run" not in out
    assert "not a warning" not in out


def test_without_events_the_failure_comes_from_the_task_history(reader, folder, capsys):
    (folder / "events.jsonl").unlink()
    reader.main([str(folder)])
    out = capsys.readouterr().out

    assert "No events.jsonl" in out
    assert FIDUCIAL_ERROR in out


def test_a_preview_is_never_written_into_the_experiment(reader, folder):
    tif = folder / "01-tough-mako" / "ref_Mill Fiducial_final_ib.tif"
    with pytest.raises(SystemExit, match="outside the experiment folder"):
        reader.main([str(folder), "--preview", str(tif), "--out", str(folder / "x")])


def test_a_lamella_left_unassessed_with_a_reason_is_not_yield(reader):
    """A lead's stop for the morning (UNASSESSED, with why) is a defect line;
    UNASSESSED with no reason is a lamella nobody has judged."""
    left = {"defect": {"verdict": "UNASSESSED", "reason": "no SEM of the face"}}
    assert reader._defect(left) == "UNASSESSED (no SEM of the face)"
    assert reader._defect({"defect": {"verdict": "UNASSESSED"}}) == ""
    assert reader._defect({}) == ""


def test_the_overview_counts_lamellae_left_for_review_apart(reader, folder, capsys):
    path = folder / "experiment.yaml"
    experiment = yaml.safe_load(path.read_text())
    for item in experiment["positions"]:
        item["defect"] = {"verdict": "UNASSESSED", "reason": "stopped one pass short"}
    path.write_text(yaml.safe_dump(experiment))
    reader.main([str(folder)])
    out = capsys.readouterr().out
    assert "Yield: 0 of 2 lamellae" in out
    assert "defect: UNASSESSED (stopped one pass short)" in out
    assert "1 more completed them but were left UNASSESSED" in out
