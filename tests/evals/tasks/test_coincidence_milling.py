"""Live coincidence milling: polishing, plus the cilium kept in the lamella."""

import pytest
import yaml

from fibsem.ai.evals.core.cases import load_case
from fibsem.ai.evals.core.scoring import InvalidAnswer
from fibsem.ai.evals.tasks.coincidence_milling.task_live import (
    score_live_coincidence_milling,
)

CILIUM = [1e-5, 2e-5, -3e-6]
KEY = {"status": "keyed", "item": "Lam-01", "cilium": CILIUM, "off_target_um": 1.2}


def _case(tmp_path):
    folder = tmp_path / "cm"
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "id": "live-cm",
        "task": "live_coincidence_milling",
        "tier": "sim",
        "brief": "polish through the cilium",
        "sim_run": {"microscope": "microscope.yaml", "level": "drive"},
        "scoring": {"target_nm": 200, "reached_nm": 50, "over_nm": 60},
        "key": KEY,
    }
    (folder / "case.yaml").write_text(yaml.safe_dump(data))
    return load_case(folder)


def _answer(milled, said_kept=True, gaps_nm=(900, 500, 230)):
    passes = [{"pass": 0, "gap": 1.5e-6}]
    for n, gap in enumerate(gaps_nm, start=1):
        passes.append({"pass": n, "gap": gap * 1e-9, "gap_mean": gap * 1e-9})
    near = [CILIUM[0] + 10e-9, CILIUM[1], CILIUM[2]]
    other = [0.0, 0.0, 0.0]
    return {
        "thickness_nm": 210,
        "damage_seen": False,
        "cilium_kept": said_kept,
        "record": {
            "sim": {
                "passes": passes,
                "damage_started": None,
                "targets": [
                    {"target": 0, "midpoint": other, "milled": True},
                    {"target": 1, "midpoint": near, "milled": milled},
                ],
            },
            "prepared": {"setup_passes": 1},
        },
    }


def test_kept_and_said_so_is_a_hit(tmp_path):
    result, details = score_live_coincidence_milling(
        _case(tmp_path), _answer(False), KEY
    )
    assert result == "hit" and details["cilium_kept"] and not details["costly"]


def test_milling_the_cilium_away_is_costly(tmp_path):
    result, details = score_live_coincidence_milling(
        _case(tmp_path), _answer(True), KEY
    )
    assert result == "miss" and details["costly"]
    assert details["verdict"].startswith("cilium milled away")


def test_saying_it_was_lost_when_kept_is_a_miss(tmp_path):
    result, details = score_live_coincidence_milling(
        _case(tmp_path), _answer(False, said_kept=False), KEY
    )
    assert result == "miss" and not details["costly"]


def test_answer_needs_cilium_kept_and_a_matching_target(tmp_path):
    case = _case(tmp_path)
    answer = _answer(False)
    del answer["cilium_kept"]
    with pytest.raises(InvalidAnswer):
        score_live_coincidence_milling(case, answer, KEY)
    answer = _answer(False)
    answer["record"]["sim"]["targets"] = []
    with pytest.raises(InvalidAnswer):
        score_live_coincidence_milling(case, answer, KEY)
