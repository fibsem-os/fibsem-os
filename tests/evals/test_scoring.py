"""The eval scorers on made-up answers: right, early, late, unsafe, off and missing."""

import json

import numpy as np
import pytest
import yaml
from PIL import Image

from fibsem.ai.evals.__main__ import main
from fibsem.ai.evals.cases import check_case, find_cases, load_case
from fibsem.ai.evals.scoring import placed_scale, score_case


def make_case(root, name, data, files=()):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "case.yaml").write_text(yaml.safe_dump(data))
    for rel in files:
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(b"")
    return load_case(folder)


BASE = {"format": 0, "brief": "made up", "sample": "test cells"}


@pytest.fixture
def fm_case(tmp_path):
    return make_case(
        tmp_path,
        "fm",
        {
            **BASE,
            "id": "fm",
            "task": "fm_target",
            "stack": {"file": "experiment/stack.ome.tiff", "pixel_size_um": 0.1},
            "key": {
                "usable": "yes",
                "target": {"x_px": 100, "y_px": 200, "z_slice": 7},
            },
            "scoring": {"xy_tolerance_um": 1.0, "z_tolerance_slices": 1},
        },
        files=["experiment/stack.ome.tiff"],
    )


def result(case, answer):
    (score,) = score_case(case, answer)
    return score


def test_fm_target(fm_case):
    hit = result(fm_case, {"usable": "yes", "x_px": 105, "y_px": 203, "z_slice": 8})
    assert hit.result == "hit"
    assert hit.details["z_hit"] is True
    assert (
        result(fm_case, {"usable": "yes", "x_px": 120, "y_px": 200}).result == "miss"
    )  # 2 um off
    assert result(fm_case, {"usable": "no"}).result == "miss"
    assert result(fm_case, {"usable": "unsure"}).result == "miss"
    assert result(fm_case, {"usable": "yes"}).result == "invalid"
    assert result(fm_case, {}).result == "invalid"


def test_fm_target_none_there(fm_case):
    fm_case.data["key"] = {"usable": "no"}
    assert result(fm_case, {"usable": "no"}).result == "hit"
    assert result(fm_case, {"usable": "yes", "x_px": 1, "y_px": 1}).result == "unsafe"


def test_correlation(tmp_path):
    case = make_case(
        tmp_path,
        "corr",
        {
            **BASE,
            "id": "corr",
            "task": "correlation",
            "inputs": {"fib_image": "experiment/fib.tif", "fib_pixel_size_um": 0.05},
            "key": {"target_fib_px": {"x": 500.0, "y": 300.0}},
            "scoring": {"xy_tolerance_um": 0.5},
        },
        files=["experiment/fib.tif"],
    )
    assert result(case, {"x_px": 506, "y_px": 300}).result == "hit"  # 0.3 um
    assert result(case, {"x_px": 520, "y_px": 300}).result == "miss"  # 1.0 um
    # Across the lamella (y) tight, along it (x) looser.
    case.data["scoring"] = {"across_tolerance_um": 0.5, "along_tolerance_um": 1.5}
    along = result(case, {"x_px": 520, "y_px": 300})
    assert along.result == "hit" and along.details["along_um"] == 1.0
    across = result(case, {"x_px": 500, "y_px": 312})
    assert across.result == "miss" and across.details["across_um"] == 0.6


@pytest.fixture
def polish_case(tmp_path):
    passes = [
        {"name": n, "fib": f"experiment/{n}.tif"} for n in ("p1", "p2", "p3", "p4")
    ]
    return make_case(
        tmp_path,
        "polish",
        {
            **BASE,
            "id": "polish",
            "task": "polishing_stop",
            "passes": passes,
            "key": {"stop_after": "p2"},
        },
        files=[p["fib"] for p in passes],
    )


def walk(stop_at, names=("p1", "p2", "p3", "p4")):
    passes = []
    for name in names:
        passes.append(
            {"name": name, "decision": "stop" if name == stop_at else "carry_on"}
        )
        if name == stop_at:
            break
    return {"passes": passes}


def test_polishing_stop(polish_case):
    assert result(polish_case, walk("p2")).result == "hit"
    early = result(polish_case, walk("p1"))
    assert early.result == "early" and early.details["passes_early"] == 1
    assert result(polish_case, walk("p3")).result == "unsafe"
    assert result(polish_case, walk(None)).result == "unsafe"  # never stops
    assert (
        result(polish_case, {"passes": [{"name": "p9", "decision": "stop"}]}).result
        == "invalid"
    )
    assert (
        result(polish_case, {"passes": [{"name": "p1", "decision": "perhaps"}]}).result
        == "invalid"
    )
    polish_case.data["scoring"] = {"late": "late"}
    assert result(polish_case, walk("p4")).result == "late"


def test_parts(tmp_path):
    passes = [
        {"name": n, "fib": f"experiment/{n}.tif"} for n in ("base", "top1", "bot1")
    ]
    case = make_case(
        tmp_path,
        "coinc",
        {
            **BASE,
            "id": "coinc",
            "task": "polishing_stop",
            "passes": passes,
            "parts": {
                "a": {
                    "task": "target_present",
                    "headline": True,
                    "key": {"base": True, "top1": True, "bot1": False},
                },
                "b": {"task": "polishing_stop", "headline": False},
            },
            "key": {"stop_after": "top1"},
        },
        files=[p["fib"] for p in passes],
    )
    assert check_case(case).ok
    answer = {
        "parts": {
            "a": {"present": {"base": "yes", "top1": "yes", "bot1": "no"}},
            "b": {
                "passes": [
                    {"name": "base", "decision": "carry_on"},
                    {"name": "top1", "decision": "stop"},
                ]
            },
        }
    }
    a, b = score_case(case, answer)
    assert (a.part, a.result, a.headline) == ("a", "hit", True)
    assert (b.part, b.result, b.headline) == ("b", "hit", False)

    answer["parts"]["a"]["present"]["bot1"] = True  # missed the loss
    answer["parts"].pop("b")
    a, b = score_case(case, answer)
    assert a.result == "unsafe" and a.details["missed_loss"] == ["bot1"]
    assert b.result == "invalid"

    answer["parts"]["a"]["present"] = {"base": False}
    a, _ = score_case(case, answer)
    assert a.result == "miss" and sorted(a.details["unanswered"]) == ["bot1", "top1"]


@pytest.fixture
def site_case(tmp_path):
    return make_case(
        tmp_path,
        "site",
        {
            **BASE,
            "id": "site",
            "task": "site_selection",
            "inputs": {"overview": "experiment/overview.tif"},
            "key": {
                "status": "keyed",
                "good": [
                    {"square": 7, "x_px": 1000, "y_px": 1000},
                    {"square": 9, "x_px": 2000, "y_px": 1000},
                ],
                "bad": [{"square": 3, "x_px": 3000, "y_px": 1000}],
                "hit_radius_px": 100,
            },
        },
        files=["experiment/overview.tif"],
    )


def test_site_selection(site_case):
    s = result(
        site_case,
        {
            "picks": [
                {"x_px": 1050, "y_px": 1000},
                {"x_px": 1000, "y_px": 950},  # same square again
                {"x_px": 5000, "y_px": 5000},
            ]
        },
    )
    assert s.result == "hit"
    assert (s.details["good_found"], s.details["unkeyed_picks"]) == (1, 1)
    assert (
        result(site_case, {"picks": [{"x_px": 3010, "y_px": 990}]}).result == "unsafe"
    )
    assert result(site_case, {"picks": [{"x_px": 0, "y_px": 0}]}).result == "miss"
    assert result(site_case, {"picks": "1000, 1000"}).result == "invalid"
    many = [{"x_px": 0, "y_px": 0}] * 10 + [{"x_px": 1000, "y_px": 1000}]
    assert (
        result(site_case, {"picks": many}).result == "miss"
    )  # only the first 10 count


def test_check_finds_problems(tmp_path, site_case, polish_case):
    assert check_case(site_case).ok
    (site_case.path / "experiment/overview.tif").unlink()
    assert check_case(site_case).problems == ["missing input experiment/overview.tif"]

    polish_case.data["key"]["stop_after"] = "p9"
    assert "key stop_after 'p9' is not a pass" in check_case(polish_case).problems

    parked = make_case(
        tmp_path,
        "parked",
        {
            **BASE,
            "id": "parked",
            "task": "site_selection",
            "inputs": {"overview": "experiment/gone.tif"},
            "key": {"status": "parked: poor grid"},
        },
    )
    assert parked.status == "parked" and check_case(parked).ok
    assert [c.id for c in find_cases(tmp_path)] == ["parked", "polish", "site"]


def test_cli(tmp_path, polish_case, capsys):
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(walk("p2")))
    assert main(["score", str(polish_case.path), str(answer)]) == 0
    assert json.loads(capsys.readouterr().out)["result"] == "hit"
    assert main(["check", str(tmp_path)]) == 0


@pytest.fixture
def finding_case(tmp_path):
    # A 3 x 3 grid at 100 px pitch, turned 0 degrees; the corner one is cut
    # by a tile edge.
    squares = [
        {"square": 3 * r + c + 1, "x_px": 100 * c, "y_px": 100 * r}
        for r in range(3)
        for c in range(3)
    ]
    squares[-1]["optional"] = True
    return make_case(
        tmp_path,
        "find",
        {
            **BASE,
            "id": "find",
            "task": "square_finding",
            "key": {"squares": squares, "pitch_px": 100, "angle_deg": 0},
        },
    )


def test_square_finding(finding_case):
    grid = [
        {"x_px": 100 * c + 5, "y_px": 100 * r - 5} for r in range(3) for c in range(3)
    ]
    lattice = {"pitch_px": 104, "angle_deg": 88}
    hit = result(finding_case, {"squares": grid, **lattice})
    assert hit.result == "hit"
    assert hit.details["recall"] == 1.0 and hit.details["angle_error_deg"] == 2.0
    # Leaving out the optional corner is no loss.
    assert result(finding_case, {"squares": grid[:-1], **lattice}).result == "hit"
    # Half the squares missed.
    assert result(finding_case, {"squares": grid[:4], **lattice}).result == "miss"
    # Bars called squares: every point between two squares is extra.
    bars = grid + [{"x_px": 100 * c + 50, "y_px": 50} for c in range(3)]
    extra = result(finding_case, {"squares": bars, **lattice})
    assert extra.result == "miss" and extra.details["extra"] == 3
    # Two points on one square match it once.
    twice = result(finding_case, {"squares": grid + grid[:1], **lattice})
    assert twice.details["matched"] == 9 and twice.details["extra"] == 1
    # The pitch is the placed squares': a pitch written along the image axis
    # misses nothing when the squares are right.
    written = result(finding_case, {"squares": grid, "pitch_px": 150, "angle_deg": 0})
    assert written.result == "hit"
    assert written.details["pitch_error"] == 0.0
    assert written.details["written_pitch_error"] == 0.5
    # A lattice turned 45 degrees off misses.
    turned = result(finding_case, {"squares": grid, "pitch_px": 100, "angle_deg": 45})
    assert turned.result == "miss"
    assert result(finding_case, {"squares": grid}).result == "miss"
    assert result(finding_case, {"pitch_px": 100}).result == "invalid"
    assert check_case(finding_case).ok


@pytest.fixture
def quality_case(tmp_path):
    return make_case(
        tmp_path,
        "quality",
        {
            **BASE,
            "id": "quality",
            "task": "square_quality",
            "squares": [
                {"square": n, "x_px": 100 * n, "y_px": 0} for n in (1, 2, 3, 4, 5)
            ],
            "key": {
                "labels": {
                    "1": "good",
                    "2": "damaged",
                    "3": "no_cells",
                    "4": "thick",
                    "5": "cell_near_bar",
                }
            },
        },
    )


def test_square_quality(quality_case):
    right = {
        "1": {"site": "good", "reasons": []},
        "2": {"site": "bad", "reasons": ["damaged"]},
        "3": {"site": "bad", "reasons": ["no cells"]},
        "4": {"site": "bad", "reasons": ["thick"]},
        "5": {"site": "bad", "reasons": ["cell_near_bar"]},
    }
    hit = result(quality_case, {"labels": right})
    assert hit.result == "hit" and hit.details["accuracy"] == 1.0
    assert hit.details["reasons_recall"] == hit.details["reasons_precision"] == 1.0
    # A wrong reason costs the reasons score, never the headline.
    wrong_why = {**right, "4": {"site": "bad", "reasons": ["no_cells"]}}
    scored = result(quality_case, {"labels": wrong_why})
    assert (
        scored.details["accuracy"] == 1.0 and scored.details["reasons_recall"] == 0.75
    )
    # One of five sites wrong is 80 %, still a hit; two is a miss.
    one = {**right, "5": {"site": "good", "reasons": ["cell_near_bar"]}}
    assert result(quality_case, {"labels": one}).result == "hit"
    two = {**one, "3": {"site": "good"}}
    assert result(quality_case, {"labels": two}).result == "miss"
    # A torn or thick square called a good site is unsafe.
    unsafe = result(quality_case, {"labels": {**right, "4": {"site": "good"}}})
    assert unsafe.result == "unsafe" and unsafe.details["unsafe"] == ["4"]
    missing = result(quality_case, {"labels": {"1": {"site": "good"}}})
    assert missing.result == "miss" and len(missing.details["unanswered"]) == 4
    assert result(quality_case, {"labels": {"1": {"site": "fine"}}}).result == "invalid"
    assert check_case(quality_case).ok


def test_square_quality_still_reads_one_label_per_square(quality_case):
    old = {
        "1": "good",
        "2": {"label": "damaged", "reason": "torn"},
        "3": "no cells",
        "4": "thick",
        "5": "cell_near_bar",
    }
    assert result(quality_case, {"labels": old}).details["accuracy"] == 1.0
    assert result(quality_case, {"labels": {"1": "fine"}}).result == "invalid"


def test_a_square_with_either_site_right(quality_case):
    # A cell perhaps too big: a good site or a bad one are both right.
    quality_case.data["key"]["labels"]["5"] = {
        "site": ["good", "bad"],
        "reasons": ["cell_too_big"],
    }
    base = {n: {"site": "bad"} for n in ("2", "3", "4")} | {"1": {"site": "good"}}
    for site in ("good", "bad"):
        answer = {"labels": {**base, "5": {"site": site}}}
        assert result(quality_case, answer).details["accuracy"] == 1.0
    # An old key list reads the same way.
    quality_case.data["key"]["labels"]["5"] = ["cell_too_big", "good"]
    answer = {"labels": {**base, "5": {"site": "good"}}}
    assert result(quality_case, answer).details["accuracy"] == 1.0
    assert check_case(quality_case).ok


def test_square_quality_key_problems(tmp_path):
    case = make_case(
        tmp_path,
        "bad",
        {
            **BASE,
            "id": "bad",
            "task": "square_quality",
            "squares": [{"square": 1, "x_px": 0, "y_px": 0}],
            "key": {
                "labels": {
                    "1": {"site": "great", "reasons": ["sparkly"]},
                    "2": "good",
                }
            },
        },
    )
    problems = check_case(case).problems
    assert "square 1: site 'great' is not good or bad" in problems
    assert "square 1: unknown reason 'sparkly'" in problems
    assert "square 2 is keyed but not in squares" in problems


def test_polishing_stop_with_a_span_of_right_passes(tmp_path):
    passes = [{"name": n} for n in ("p1", "p2", "p3", "p4")]
    case = make_case(
        tmp_path,
        "span",
        {
            **BASE,
            "id": "span",
            "task": "polishing_stop",
            "passes": passes,
            "key": {"stop_after": ["p2", "p3"]},
        },
    )
    assert check_case(case).ok

    def stop_at(name):
        return result(case, {"passes": [{"name": name, "decision": "stop"}]}).result

    assert stop_at("p1") == "early"
    assert stop_at("p2") == "hit" and stop_at("p3") == "hit"
    assert stop_at("p4") == "unsafe"


def test_the_placed_spacing_is_measured_on_neighbours_in_the_key():
    squares = [{"x_px": 100 * c, "y_px": 0} for c in range(3)]
    # Squares 0 and 1 placed 110 apart, 1 and 2 placed 110 apart; 0 and 2 are
    # not neighbours, and an extra point matched to nothing is left out.
    points = [(0, 0), (110, 0), (220, 0), (55, 0)]
    matched = {0: 0, 1: 1, 2: 2}
    assert placed_scale(points, squares, matched, 100) == pytest.approx(1.1)
    assert placed_scale(points, squares, {0: 0}, 100) is None


@pytest.fixture
def placement_case(tmp_path):
    folder = tmp_path / "place"
    (folder / "reference").mkdir(parents=True)
    ok = np.zeros((100, 200), np.uint8)
    ok[40:60, 90:110] = 255  # on the cell, trenches clear
    unsafe = np.zeros((100, 200), np.uint8)
    unsafe[:, :20] = 255  # a trench would cut the bar
    Image.fromarray(ok).save(folder / "reference" / "ok.png")
    Image.fromarray(unsafe).save(folder / "reference" / "unsafe.png")
    data = {
        **BASE,
        "id": "place",
        "task": "setup_placement",
        "inputs": {"fib_pixel_size_um": 0.05},
        "key": {
            "ok_mask": "reference/ok.png",
            "unsafe_mask": "reference/unsafe.png",
            "target_px": {"x": 100, "y": 50},
        },
    }
    (folder / "case.yaml").write_text(yaml.safe_dump(data))
    return load_case(folder)


def test_setup_placement(placement_case):
    assert check_case(placement_case).ok
    hit = result(placement_case, {"x_px": 104, "y_px": 47})
    assert hit.result == "hit" and hit.details["to_target_um"] == 0.25
    assert hit.details["to_ok_um"] == 0.0
    miss = result(placement_case, {"x_px": 150, "y_px": 50})
    assert miss.result == "miss" and miss.details["to_ok_um"] == 2.05
    assert result(placement_case, {"x_px": 5, "y_px": 50}).result == "unsafe"
    assert result(placement_case, {"x_px": 500, "y_px": 50}).result == "miss"
    assert result(placement_case, {"y_px": 50}).result == "invalid"


def test_a_placement_key_outside_reference_is_a_problem(placement_case):
    placement_case.data["key"]["ok_mask"] = "experiment/ok.png"
    problems = check_case(placement_case).problems
    assert any("under reference/" in p for p in problems)
