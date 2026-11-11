"""The HTML report: every task gets its figure from made-up images and answers."""

import json

import numpy as np
import pytest
import tifffile
import yaml

from fibsem.ai.evals import visual
from fibsem.ai.evals.__main__ import main
from fibsem.ai.evals.cases import load_case
from fibsem.ai.evals.scoring import score_case, score_live

rng = np.random.default_rng(0)


def image(path, shape=(400, 600)):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, rng.integers(0, 255, shape, dtype=np.uint8))


PASSES = [
    {
        "name": n,
        "fib": f"experiment/{n}.tif",
        "measured": {"median_nm": m, "min_nm": m - 50, "max_nm": m + 50},
    }
    for n, m in (("p1", 400), ("p2", 300), ("p3", 290))
]
SQUARES = [{"square": n, "x_px": 100 * n, "y_px": 200} for n in (1, 2, 3)]

CASES = {
    "site": (
        {
            "task": "site_selection",
            "inputs": {"overview": "experiment/overview.tif"},
            "key": {"good": [{"square": 1, "x_px": 100, "y_px": 200}]},
        },
        {"picks": [{"x_px": 105, "y_px": 198, "reason": "clear"}]},
    ),
    "find": (
        {
            "task": "square_finding",
            "inputs": {"overview": "experiment/overview.tif"},
            "key": {"squares": SQUARES, "pitch_px": 100, "angle_deg": 0},
        },
        {"squares": [{"x_px": 100, "y_px": 200}], "pitch_px": 100, "angle_deg": 0},
    ),
    "quality": (
        {
            "task": "square_quality",
            "inputs": {"overview": "experiment/overview.tif"},
            "squares": SQUARES,
            "key": {"labels": {"1": "good", "2": "damaged", "3": "thick"}},
        },
        {"labels": {"1": "good", "2": {"label": "good"}}},
    ),
    "fm": (
        {
            "task": "fm_target",
            "stack": {
                "file": "experiment/stack.ome.tiff",
                "shape": [2, 3, 300, 300],
                "channels": ["reflection", "gfp"],
                "target_channel": "gfp",
                "pixel_size_um": 0.1,
            },
            "key": {
                "usable": "yes",
                "target": {"x_px": 150, "y_px": 140, "z_slice": 1},
            },
        },
        {"usable": "yes", "x_px": 152, "y_px": 141, "z_slice": 1},
    ),
    "corr": (
        {
            "task": "correlation",
            "inputs": {
                "fib_image": "experiment/fib.tif",
                "fib_pixel_size_um": 0.05,
                "fm_stack": "experiment/stack.ome.tiff",
            },
            "key": {
                "target_fib_px": {"x": 300, "y": 200},
                "target_fm": {"x_px": 150, "y_px": 140},
                "fiducials_fm_px": [[40, 50, 1]],
                "fiducials_fib_px": [[52, 61]],
            },
        },
        {
            "x_px": 301,
            "y_px": 230,
            "fiducials": [{"fm_px": [1, 2, 3], "fib_px": [50, 60]}],
        },
    ),
    "polish": (
        {
            "task": "polishing_stop",
            "passes": PASSES,
            "key": {"stop_after": "p2"},
            "live": {
                "source": "the real run",
                "answer": {"passes": [{"name": "p3", "decision": "stop"}]},
            },
        },
        {
            "passes": [
                {"name": "p1", "decision": "carry_on"},
                {"name": "p3", "decision": "stop"},
            ]
        },
    ),
    "parts": (
        {
            "task": "polishing_stop",
            "passes": PASSES,
            "key": {"stop_after": "p2"},
            "parts": {
                "a": {
                    "task": "target_present",
                    "key": {"p1": "yes", "p2": "yes", "p3": "no"},
                },
                "b": {"task": "polishing_stop", "headline": False},
            },
        },
        {
            "parts": {
                "a": {"present": {"p1": "yes", "p2": "yes", "p3": "yes"}},
                "b": {"passes": [{"name": "p2", "decision": "stop"}]},
            }
        },
    ),
}


@pytest.fixture
def run(tmp_path):
    cases_root, out = tmp_path / "cases", tmp_path / "out"
    for name, (data, answer) in CASES.items():
        folder = cases_root / name
        folder.mkdir(parents=True)
        data = {"id": name, "format": 0, "brief": f"{name} brief", **data}
        (folder / "case.yaml").write_text(yaml.safe_dump(data))
        image(folder / "experiment" / "overview.tif", (400, 600))
        image(folder / "experiment" / "fib.tif", (400, 600))
        for entry in PASSES:
            image(folder / entry["fib"], (64, 96))
        tifffile.imwrite(
            folder / "experiment" / "stack.ome.tiff",
            rng.integers(0, 4000, (2, 3, 300, 300), dtype=np.uint16),
        )
        case = load_case(folder)
        for repeat in (1, 2):
            run_folder = out / name / f"r{repeat}"
            (run_folder / "work").mkdir(parents=True)
            (run_folder / "work" / "answer.json").write_text(json.dumps(answer))
            (run_folder / "agent.log").write_text("log")
            result = {
                "case": name,
                "repeat": repeat,
                "task": case.task,
                "status": "answered",
                "spent_usd": 0.5,
                "seconds": 90,
                "scores": [s.to_dict() for s in score_case(case, answer)],
            }
            if name != "site":
                result["case_path"] = str(folder)
            (run_folder / "result.json").write_text(json.dumps(result))
    return cases_root, out


def test_every_task_gets_a_figure(run):
    cases_root, out = run
    page = visual.write_html(out, cases_root).read_text()
    # One figure per case, and one per part of the case with parts.
    assert page.count("data:image/png;base64,") == len(CASES) + 1
    assert "no figure" not in page
    assert "r2" in page and "agent</a>" in page and "unsafe" in page


def test_cases_are_found_by_their_path_without_a_cases_folder(run):
    _, out = run
    page = visual.write_html(out).read_text()
    # Only the site case's results lack a case_path.
    assert page.count("Case not found") == 1


def test_a_figure_that_fails_leaves_the_rest(run, monkeypatch):
    cases_root, out = run

    def broken(case, runs, **kw):
        raise OSError("no such file")

    monkeypatch.setitem(visual.DRAW, "correlation", broken)
    page = visual.write_html(out, cases_root).read_text()
    assert "correlation: no figure (no such file)" in page
    assert page.count("data:image/png;base64,") == len(CASES)


def test_the_correlation_figure_shows_both_images(run):
    cases_root, out = run
    case = load_case(cases_root / "corr")
    runs = [run for _, run in visual.load_runs(out)["corr"]]
    fig = visual.draw_correlation(case, runs)
    assert len(fig.axes) == 2
    # Without the stack, the FIB image alone.
    case.data["inputs"]["fm_stack"] = "experiment/missing.ome.tiff"
    assert len(visual.draw_correlation(case, runs).axes) == 1


def test_report_html_from_the_command_line(run, capsys):
    cases_root, out = run
    assert main(["report", str(out), "--html", "--cases", str(cases_root)]) == 0
    assert (out / "report.html").is_file()
    assert "Wrote" in capsys.readouterr().out


def test_the_real_run_is_scored_against_the_key(run):
    cases_root, out = run
    case = load_case(cases_root / "polish")
    (live,) = score_live(case)
    assert live.result == "unsafe"
    runs = [run for _, run in visual.load_runs(out)["polish"]]
    (row,) = visual.live_rows(case, runs)
    # Both repeats stopped at p3 as well.
    assert row["live"] == "unsafe" and row["same"] == 2 and len(row["eval"]) == 2
    assert score_live(load_case(cases_root / "site")) is None
    page = visual.write_html(out, cases_root).read_text()
    assert "Eval against the real run" in page and "2 of 2" in page
