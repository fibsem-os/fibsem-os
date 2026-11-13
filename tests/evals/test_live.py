"""Live sim cases: scoring where the sites really are, the scripted agent's
image rules, the runner's part, and one scripted run end to end (slow)."""

import json
import math

import numpy as np
import pytest
import yaml

from fibsem.ai.evals.cases import load_case
from fibsem.ai.evals.live.scripted import assess_site, find_cells
from fibsem.ai.evals.runner import live_server_flags, task_text
from fibsem.ai.evals.scoring import InvalidAnswer, score_case
from fibsem.ai.evals.tasks.live_site_selection import (
    _positions,
    score_live_site_selection,
    site_truth,
)

PITCH, BAR = 125e-6, 35e-6

# Stage axes along the grid's, y flipped as the Arctis's SEM shows it.
KEY = {
    "status": "keyed",
    "frame": {
        "origin": [0.0, 0.0],
        "u": [1.0, 0.0],
        "v": [0.0, -1.0],
        "pitch": PITCH,
        "bar_width": BAR,
    },
    "squares": [
        {"index": [0, 0], "x": 0.0, "y": 0.0, "label": "good", "certain": True},
        {"index": [1, 0], "x": PITCH, "y": 0.0, "label": "thick", "certain": True},
        {"index": [2, 0], "x": 2 * PITCH, "y": 0.0, "label": "good", "certain": False},
        {
            "index": [0, 1],
            "x": 0.0,
            "y": -PITCH,
            "label": "cell_near_bar",
            "certain": True,
        },
    ],
    "cells": [
        {
            "x": 0.0,
            "y": 0.0,
            "radius_um": 10.0,
            "square": [0, 0],
            "on_film": True,
            "clear": True,
        },
        {
            "x": PITCH,
            "y": 0.0,
            "radius_um": 10.0,
            "square": [1, 0],
            "on_film": True,
            "clear": True,
        },
        {
            "x": 2 * PITCH,
            "y": 0.0,
            "radius_um": 10.0,
            "square": [2, 0],
            "on_film": True,
            "clear": True,
        },
        {
            "x": 30e-6,
            "y": -PITCH,
            "radius_um": 12.0,
            "square": [0, 1],
            "on_film": True,
            "clear": False,
        },
    ],
}


def _case(tmp_path, **scoring):
    folder = tmp_path / "live"
    folder.mkdir(parents=True)
    data = {
        "id": "live-1",
        "task": "live_site_selection",
        "tier": "sim",
        "brief": "find sites",
        "sim_run": {"microscope": "microscope.yaml", "level": "drive"},
        "scoring": scoring,
        "key": KEY,
    }
    (folder / "case.yaml").write_text(yaml.safe_dump(data))
    return load_case(folder)


def _item(name, x, y, t=0.0):
    pose = {"x": x, "y": y, "z": 0.0, "r": 0.0, "t": t}
    return {"name": name, "poses": {"MILLING": pose}}


def test_a_site_is_good_only_on_a_clear_cell_in_a_clean_square():
    assert site_truth(KEY, 2e-6, 1e-6)["truth"] == "good"
    assert site_truth(KEY, 40e-6, 0.0)["why"] == "no cell there"
    assert site_truth(KEY, PITCH / 2, 0.0)["why"] == "on a grid bar"
    assert site_truth(KEY, PITCH, 0.0)["why"] == "the square is thick"
    assert site_truth(KEY, 30e-6, -PITCH)["why"] == "the cell is too near a bar"
    unsure = site_truth(KEY, 2 * PITCH, 0.0)
    assert unsure["truth"] == "unsure" and unsure["was"] == "good"
    assert site_truth(KEY, 5 * PITCH, 0.0)["truth"] == "unkeyed"


def test_sites_are_scored_where_the_run_saved_them(tmp_path):
    case = _case(tmp_path)
    answer = {
        # the agent's word for where Site-02 is counts for nothing
        "sites": [
            {"item": "Site-01", "verdict": "good"},
            {"item": "Site-02", "verdict": "bad", "stage_position": {"x": 0, "y": 0}},
        ],
        "record": {
            "items": [_item("Site-01", 1e-6, 0.0), _item("Site-02", 40e-6, 0.0)]
        },
    }
    result, details = score_live_site_selection(case, answer, KEY)
    assert result == "hit"
    assert details["good_sites"] == 1 and details["verdict_accuracy"] == 1.0
    assert [s["truth"] for s in details["per_site"]] == ["good", "bad"]


def test_a_bad_site_called_good_is_costly_and_a_miss(tmp_path):
    case = _case(tmp_path)
    answer = {
        "sites": [
            {"item": "Site-01", "verdict": "good"},
            {"item": "Site-02", "verdict": "good"},
        ],
        "record": {"items": [_item("Site-01", 0.0, 0.0), _item("Site-02", PITCH, 0.0)]},
    }
    result, details = score_live_site_selection(case, answer, KEY)
    assert result == "miss" and details["costly"] == 1
    assert details["per_site"][1]["costly"] is True


def test_one_cell_saved_twice_is_one_good_site(tmp_path):
    case = _case(tmp_path)
    answer = {
        "sites": [{"item": "A", "verdict": "good"}, {"item": "B", "verdict": "good"}],
        "record": {"items": [_item("A", 0.0, 0.0), _item("B", 3e-6, 0.0)]},
    }
    _, details = score_live_site_selection(case, answer, KEY)
    assert details["good_sites"] == 1 and details["good_fraction"] == 0.5


def test_a_site_not_saved_or_saved_tilted_is_not_scored_there(tmp_path):
    case = _case(tmp_path)
    answer = {
        "sites": [{"item": "A", "verdict": "good"}, {"item": "B", "verdict": "good"}],
        "record": {"items": [_item("A", 0.0, 0.0, t=math.radians(-20))]},
    }
    _, details = score_live_site_selection(case, answer, KEY)
    assert details["not_saved"] == 2 and details["good_sites"] == 0
    assert _positions({"record": {"items": [_item("A", 1.0, 2.0)]}})["A"]["x"] == 1.0


OPTICS = {
    "sem_angle": 4e-3,
    "sem_focus_wd": 7e-3,
    "sem_focus_per_z": 1.0,
    "coincident_z": -10e-6,
    "coincidence_per_z": 1.3,
}


def _image(x, y, z, wd, path="sites/A_eb.tif", hfw=80e-6, time=1.0):
    return {
        "path": path,
        "beam": "ELECTRON",
        "hfw": hfw,
        "pixel_size": hfw / 1536,
        "working_distance": wd,
        "stage_position": {"x": x, "y": y, "z": z, "r": 0.0, "t": 0.0},
        "time": time,
    }


def test_a_site_is_scored_on_coincidence_and_the_focus_of_its_image(tmp_path):
    case = _case(tmp_path)
    key = {**KEY, "optics": OPTICS}
    ready = _item("A", 0.0, 0.0)
    ready["poses"]["MILLING"]["z"] = -10.2e-6
    off = _item("B", 8e-6, 0.0)  # the same cell, at the boot height
    answer = {
        "sites": [{"item": "A", "verdict": "good"}, {"item": "B", "verdict": "good"}],
        "record": {
            "overviews": ["overview-07-00-00.tif"],
            "items": [ready, off],
            "images": [
                # focused for its z; then one out of focus, older, ignored
                _image(0.0, 0.0, -10.2e-6, 7e-3 - 10.2e-6, time=2.0),
                _image(0.0, 0.0, -10.2e-6, 7e-3 + 40e-6, time=1.0),
                _image(8e-6, 0.0, 0.0, 7e-3 + 60e-6, path="sites/B_eb.tif"),
            ],
        },
    }
    result, details = score_live_site_selection(case, answer, key)
    a, b = details["per_site"]
    assert a["coincident"] and a["sharp"] and a["blur_px"] < 0.5
    assert not b["coincident"] and b["coincidence_um"] == pytest.approx(13.0)
    assert not b["sharp"]
    assert details["steps"] == {
        "overview": True,
        "sites_saved": 2,
        "sites_imaged": 2,
        "verdicts": 2,
    }
    # B was called good but left off coincidence: not ready to mill
    assert (details["ready"], details["said_good"]) == (1, 2)
    assert result == "miss"
    # coincident but blurred passes unless sharpness is asked for
    b_ready = _item("B", 8e-6, 0.0)
    b_ready["poses"]["MILLING"]["z"] = -10e-6
    answer["record"]["items"][1] = b_ready
    assert score_live_site_selection(case, answer, key)[0] == "hit"
    strict = _case(tmp_path / "strict", require_sharp=True)
    assert score_live_site_selection(strict, answer, key)[0] == "miss"


def test_an_answer_without_sites_or_with_a_wrong_verdict_is_invalid(tmp_path):
    case = _case(tmp_path)
    with pytest.raises(InvalidAnswer):
        score_live_site_selection(case, {"sites": []}, KEY)
    with pytest.raises(InvalidAnswer):
        score_live_site_selection(
            case, {"sites": [{"item": "A", "verdict": "maybe"}]}, KEY
        )
    assert score_case(case, {"sites": []})[0].result == "invalid"


def test_a_live_case_serves_its_own_microscope_and_says_to_work_it(tmp_path):
    case = _case(tmp_path)
    flags = live_server_flags(case)
    assert flags[0] == "--config" and flags[1].endswith("microscope.yaml")
    assert "--serve-microscope" in flags
    text = task_text(case, [])
    assert "simulator" in text and "Look and read only" not in text


def _grid_image(px=0.5e-6, cell_at=(200, 200)):
    """Film squares (dark) between bars (mid grey), one bright cell."""
    image = np.full((400, 400), 145.0)
    image[40:360, 40:360] = 113.0
    yy, xx = np.mgrid[:400, :400]
    core = np.hypot(yy - cell_at[1], xx - cell_at[0]) < 7e-6 / px
    image[core] = 175.0
    rng = np.random.default_rng(0)
    return np.clip(image + rng.normal(0, 3, image.shape), 0, 255).astype(np.uint8)


def test_the_scripted_agent_finds_a_cell_and_whether_film_is_all_round_it():
    clear = find_cells(_grid_image(), 0.5e-6)
    assert len(clear) == 1 and clear[0]["film_share"] > 0.95
    near_bar = find_cells(_grid_image(cell_at=(60, 200)), 0.5e-6)
    assert near_bar and near_bar[0]["film_share"] < 0.9


def test_the_scripted_agent_calls_a_site_good_only_with_a_clear_cell_at_centre():
    image = _grid_image()[40:360, 40:360]  # a site image: film only
    assert assess_site(image, 0.5e-6, 32.0)[0] == "good"
    empty = np.full((320, 320), 113, dtype=np.uint8)
    verdict, reason = assess_site(empty, 0.5e-6, 32.0)
    assert verdict == "bad" and "no cell" in reason


@pytest.mark.slow
def test_a_scripted_run_of_a_live_case_end_to_end(tmp_path, monkeypatch):
    """The case's microscope on a real App Server, the scripted agent at it,
    and the run scored from what it saved."""
    from fibsem.ai.evals.live import runs as live_runs
    from fibsem.ai.evals.live import sim_sites
    from fibsem.ai.evals.runner import RunConfig, run_case

    folder = sim_sites.write_case(1, tmp_path / "cases")
    case = load_case(folder)
    config = RunConfig(
        out=tmp_path / "runs",
        repeats=1,
        timeout_s=600,
        knowledge=tmp_path / "no-knowledge",
        agent_command=live_runs.scripted_command(),
    )
    result = run_case(case, 1, config)
    assert result["status"] == "answered", (tmp_path / "runs").as_posix()
    run = tmp_path / "runs" / case.id / "r1"
    record = json.loads((run / live_runs.RECORD).read_text())
    assert record["items"]
    score = result["scores"][0]
    assert score["result"] in ("hit", "miss")
    details = score["details"]
    assert details["not_saved"] == 0 and details["good_sites"] >= 1
    assert details["steps"]["overview"] and details["steps"]["sites_imaged"] >= 1
    assert details["coincident"] >= 1
