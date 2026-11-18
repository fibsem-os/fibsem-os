"""Live correlation: the lamella's point of interest against the target's
true place in the FIB's view from the set-up's pose."""

import pytest
import yaml

from fibsem.ai.evals.core.cases import load_case
from fibsem.ai.evals.core.scoring import InvalidAnswer
from fibsem.ai.evals.tasks.correlation.task_live import score_live_correlation

POSE = {"x": 1e-5, "y": 2e-5, "z": -1e-5, "r": 0.0, "t": -0.4}
KEY = {
    "status": "keyed",
    "item": "Lam-01",
    "target_kind": "cilium",
    "target_poi_m": [4e-6, -3e-6],
    "milling_pose": POSE,
    "burns": [[0.2, 0.3], [0.7, 0.6]],
}


def _case(tmp_path):
    folder = tmp_path / "corr"
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "id": "live-corr",
        "task": "live_correlation",
        "tier": "sim",
        "brief": "correlate and place the lamella",
        "sim_run": {"microscope": "microscope.yaml", "level": "drive"},
        "scoring": {
            "across_tolerance_um": 0.75,
            "along_tolerance_um": 1.0,
            "same_pose_um": 0.2,
        },
        "key": KEY,
    }
    (folder / "case.yaml").write_text(yaml.safe_dump(data))
    return load_case(folder)


def _answer(poi=None, target_m=None, pose=POSE):
    answer = {
        "item": "Lam-01",
        "record": {
            "items": [
                {
                    "name": "Lam-01",
                    "id": "a",
                    "poses": {"MILLING": dict(pose)},
                    "poi": poi,
                }
            ]
        },
    }
    if target_m is not None:
        answer["target_m"] = target_m
    return answer


def test_a_point_of_interest_on_the_target_is_a_hit(tmp_path):
    result, details = score_live_correlation(
        _case(tmp_path), _answer({"x": 4.5e-6, "y": -2.5e-6}), KEY
    )
    assert result == "hit"
    assert details["scored"] == "point of interest"
    assert details["across_um"] == pytest.approx(0.5)
    assert details["along_um"] == pytest.approx(0.5)


def test_off_across_the_lamella_misses_sooner_than_along_it(tmp_path):
    case = _case(tmp_path)
    across, _ = score_live_correlation(case, _answer({"x": 4e-6, "y": -2e-6}), KEY)
    along, _ = score_live_correlation(case, _answer({"x": 4.9e-6, "y": -3e-6}), KEY)
    assert (across, along) == ("miss", "hit")


def test_the_centre_misses(tmp_path):
    result, details = score_live_correlation(
        _case(tmp_path), _answer({"x": 0.0, "y": 0.0}, target_m=[0.0, 0.0]), KEY
    )
    assert result == "miss" and details["scored"] == "answer"


def test_moving_the_stage_is_a_miss(tmp_path):
    moved = dict(POSE, y=POSE["y"] + 2e-6)
    result, details = score_live_correlation(
        _case(tmp_path), _answer({"x": 4e-6, "y": -3e-6}, pose=moved), KEY
    )
    assert result == "miss" and "moved" in details["verdict"]


def test_no_point_at_all_is_invalid(tmp_path):
    with pytest.raises(InvalidAnswer):
        score_live_correlation(_case(tmp_path), _answer(None), KEY)
