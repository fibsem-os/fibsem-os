"""Live polishing: scored on the run's own sample record."""

import pytest
import yaml

from fibsem.ai.evals.cases import load_case
from fibsem.ai.evals.scoring import InvalidAnswer
from fibsem.ai.evals.tasks.live_polishing import score_live_polishing

KEY = {"status": "keyed", "item": "Lam-01"}


def _case(tmp_path):
    folder = tmp_path / "polishing"
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "id": "live-p",
        "task": "live_polishing",
        "tier": "sim",
        "brief": "polish it",
        "sim_run": {"microscope": "microscope.yaml", "level": "drive"},
        "scoring": {"target_nm": 200, "reached_nm": 50, "over_nm": 60},
        "key": KEY,
    }
    (folder / "case.yaml").write_text(yaml.safe_dump(data))
    return load_case(folder)


def _answer(gaps_nm, damage=None, said=210, seen=False, setup=3):
    # the set-up's rough passes, one that cuts no gap among them
    passes = [
        {"pass": 0, "gap": 4.5e-6},
        {"pass": 1, "gap": 1.5e-6},
        {"pass": 2, "gap": None},
    ][:setup]
    for n, gap in enumerate(gaps_nm, start=len(passes)):
        passes.append({"pass": n, "gap": gap * 1e-9, "gap_mean": gap * 1e-9})
    return {
        "thickness_nm": said,
        "damage_seen": seen,
        "record": {
            "sim": {"passes": passes, "damage_started": damage},
            "prepared": {"setup_passes": setup},
        },
    }


def test_stopping_in_the_band_is_a_hit(tmp_path):
    result, details = score_live_polishing(
        _case(tmp_path), _answer([900, 500, 230]), KEY
    )
    assert result == "hit" and details["passes"] == 3
    assert details["start_nm"] == 1500.0 and details["final_nm"] == 230.0
    assert details["said_off_nm"] == -20.0


def test_stopping_thick_is_early_and_thin_is_costly(tmp_path):
    case = _case(tmp_path)
    result, details = score_live_polishing(case, _answer([900, 400]), KEY)
    assert result == "miss" and details["verdict"] == "early" and not details["costly"]
    result, details = score_live_polishing(case, _answer([900, 400, 120]), KEY)
    assert result == "miss" and details["verdict"] == "too thin" and details["costly"]


def test_milling_on_after_damage_is_costly(tmp_path):
    case = _case(tmp_path)
    damage = {"pass": 4, "kind": "crack"}
    result, details = score_live_polishing(case, _answer([900, 500, 230], damage), KEY)
    assert (
        result == "miss" and details["costly"] and details["passes_after_damage"] == 1
    )
    # stopping on the pass it started, and saying so
    stopped = _answer([900, 500], damage, seen=True)
    assert score_live_polishing(case, stopped, KEY)[0] == "hit"
    unsaid = _answer([900, 500], damage, seen=False)
    assert score_live_polishing(case, unsaid, KEY)[0] == "miss"


def test_an_answer_without_its_thickness_or_record_is_invalid(tmp_path):
    case = _case(tmp_path)
    with pytest.raises(InvalidAnswer):
        score_live_polishing(case, {"damage_seen": False}, KEY)
    with pytest.raises(InvalidAnswer):
        score_live_polishing(case, {"thickness_nm": 200, "damage_seen": False}, KEY)
