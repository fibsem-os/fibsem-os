"""Grid cases made from the simulator: the scene's options, the labelling
rules, the key against the rendered overview, and sim scores kept apart."""

import json

import numpy as np
import pytest
import tifffile

from fibsem.ai.evals import report, sim_cases
from fibsem.ai.evals.__main__ import main
from fibsem.ai.evals.cases import check_case, load_case

SMALL = {"resolution": (768, 512), "hfw": 520e-6}
# a bare grid: film, bars and nothing else, so a pixel says which it is
BARE = {
    "cell_type": "none",
    "ice_density": 0.0,
    "frost_density": 0.0,
    "sphere_density": 0.0,
    "crack_density": 0.0,
    "contamination_density": 0.0,
    "rip_fraction": 0.0,
    "noise_sigma": 0.0,
    "noise_fraction": 0.0,
    "centre_mark": False,
}


def test_a_seed_always_draws_the_same_scene_and_overrides_win():
    assert sim_cases.scene_options(5) == sim_cases.scene_options(5)
    assert sim_cases.scene_options(5) != sim_cases.scene_options(6)
    options = sim_cases.scene_options(5, {"crack_density": 0.0, "seed": 99})
    assert options["crack_density"] == 0.0
    assert options["seed"] == 5


def _measures(rip=0.0, ice=0.0, cells=1, clear=1, **ends):
    end = {"ice_fraction": ice, "cells": cells, "clear_cells": clear}
    return {
        "rip_fraction": rip,
        "crack_length": 0.0,
        "beyond": False,
        "low": {**end, **ends.get("low", {})},
        "mid": dict(end),
        "high": {**end, **ends.get("high", {})},
    }


@pytest.mark.parametrize(
    "measures, label",
    [
        (_measures(rip=0.2, ice=0.5), "damaged"),
        (_measures(ice=0.2), "thick"),
        (_measures(cells=0, clear=0), "no_cells"),
        (_measures(cells=2, clear=0), "cell_near_bar"),
        (_measures(), "good"),
    ],
)
def test_labels_go_by_the_rules_in_order(measures, label):
    assert sim_cases.label_square(measures) == (label, True)


def test_a_square_close_to_a_threshold_is_not_certain():
    # ice just over the line, and a cell that only clears the bars when small
    assert sim_cases.label_square(_measures(ice=0.035)) == ("thick", False)
    near = _measures(high={"clear_cells": 0})
    assert sim_cases.label_square(near) == ("good", False)


@pytest.fixture(scope="module")
def bare(tmp_path_factory):
    out = tmp_path_factory.mktemp("cases")
    folders = sim_cases.write_cases(3, out, BARE, **SMALL)
    return out, folders


def test_three_checked_cases_in_the_sim_tier(bare):
    out, folders = bare
    tasks = set()
    for folder in folders:
        case = load_case(folder)
        assert check_case(case).ok, check_case(case).problems
        assert case.tier == "sim"
        tasks.add(case.task)
        # a bare grid has no good square to pick, so no site to key
        expected = "unkeyed" if case.task == "site_selection" else "keyed"
        assert case.status == expected
    assert tasks == {"site_selection", "square_finding", "square_quality"}
    reference = out / "square-finding" / "sim-0003" / "reference"
    assert (reference / "preview-key.jpg").exists()
    assert json.loads((reference / "measures.json").read_text())


def test_the_key_puts_each_square_on_film_and_its_pitch_on_the_bars(bare):
    out, _ = bare
    case = load_case(out / "square-finding" / "sim-0003")
    image = tifffile.imread(
        out / "site-selection" / "sim-0003" / "experiment" / "overview.tif"
    )
    key = case.key
    h, w = image.shape
    film = np.median(image)
    angle = np.radians(key["angle_deg"])
    # half a pitch along the bars' direction is the middle of a bar
    step = key["pitch_px"] / 2 * np.array([np.cos(angle), np.sin(angle)])
    checked = 0
    for square in key["squares"]:
        x, y = square["x_px"], square["y_px"]
        bx, by = x + step[0], y + step[1]
        if square.get("optional") or not (5 < bx < w - 5 and 5 < by < h - 5):
            continue
        centre = image[int(y) - 3 : int(y) + 4, int(x) - 3 : int(x) + 4].mean()
        bar = image[int(by) - 3 : int(by) + 4, int(bx) - 3 : int(bx) + 4].mean()
        assert abs(centre - film) < 5
        assert bar > film + 10
        checked += 1
    assert checked >= 3


def test_a_bare_grid_has_no_cells_anywhere(bare):
    out, _ = bare
    case = load_case(out / "square-quality" / "sim-0003")
    assert set(case.key["labels"].values()) == {"no_cells"}
    assert not load_case(out / "site-selection" / "sim-0003").key["good"]


def _result(tier, result="hit"):
    return {
        "case": f"{tier}-case",
        "tier": tier,
        "spent_usd": 0.1,
        "seconds": 10,
        "scores": [{"task": "square_finding", "result": result, "headline": True}],
    }


def test_sim_scores_get_their_own_rows_after_the_real_ones():
    rows = report.summarise([_result("sim"), _result("real", "miss"), _result("sim")])
    assert [(r["tier"], r["runs"], r["hit"]) for r in rows] == [
        ("real", 1, 0),
        ("sim", 2, 2),
    ]
    table = report.table(rows)
    assert "| real | square_finding" in table and "| sim | square_finding" in table


def test_results_from_before_tiers_count_as_real():
    old = _result("real")
    del old["tier"]
    (row,) = report.summarise([old])
    assert row["tier"] == "real"


def test_an_unknown_tier_is_a_problem(tmp_path):
    (tmp_path / "case.yaml").write_text(
        "id: x\ntask: square_finding\nformat: 0\nbrief: b\ntier: fake\n"
    )
    assert "unknown tier 'fake'" in check_case(load_case(tmp_path)).problems


def test_run_only_the_sim_tier(bare, tmp_path, capsys, monkeypatch):
    out, _ = bare
    seen = []

    def run_cases(cases, config, on_result):
        config.out.mkdir(parents=True)
        seen.extend(c.tier for c in cases)

    monkeypatch.setattr("fibsem.ai.evals.__main__.run_cases", run_cases)
    assert (
        main(["run", str(out), "--out", str(tmp_path / "runs"), "--tier", "real"]) == 0
    )
    assert seen == []
    assert (
        main(["run", str(out), "--out", str(tmp_path / "runs2"), "--tier", "sim"]) == 0
    )
    assert seen == ["sim"] * 2
