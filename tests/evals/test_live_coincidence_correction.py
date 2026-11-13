"""Live coincidence correction: scored at the height the site was saved."""

import pytest
import yaml

from fibsem.ai.evals.cases import load_case
from fibsem.ai.evals.scoring import InvalidAnswer, score_case
from fibsem.ai.evals.tasks.live_coincidence_correction import (
    score_live_coincidence_correction,
)

OPTICS = {
    "sem_angle": 4e-3,
    "sem_focus_wd": 7e-3,
    "sem_focus_per_z": 1.0,
    "coincident_z": -10e-6,
    "coincidence_per_z": 1.3,
}
KEY = {
    "status": "keyed",
    "site": "Site-01",
    "start": {"x": 1e-4, "y": 2e-5, "z": 0.0, "r": 0.0, "t": 0.0},
    "start_off_um": 13.0,
    "cell": {"x": 1e-4, "y": 2e-5, "radius_um": 10.0},
    "optics": OPTICS,
}


def _case(tmp_path):
    folder = tmp_path / "coincidence"
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "id": "live-c",
        "task": "live_coincidence_correction",
        "tier": "sim",
        "brief": "make it coincident",
        "sim_run": {"microscope": "microscope.yaml", "level": "drive"},
        "key": KEY,
    }
    (folder / "case.yaml").write_text(yaml.safe_dump(data))
    return load_case(folder)


def _answer(z, said=True, x=1e-4, name="Site-01"):
    pose = {"x": x, "y": 2e-5, "z": z, "r": 0.0, "t": 0.0}
    return {
        "site": "Site-01",
        "coincident": said,
        "record": {"items": [{"name": name, "poses": {"MILLING": pose}}]},
    }


def test_a_site_saved_at_the_coincident_height_is_a_hit(tmp_path):
    result, details = score_live_coincidence_correction(
        _case(tmp_path), _answer(-10.3e-6), KEY
    )
    assert result == "hit"
    assert details["final_off_um"] == pytest.approx(0.39)
    assert details["start_off_um"] == 13.0 and not details["costly"]


def test_saying_coincident_when_far_off_is_costly(tmp_path):
    result, details = score_live_coincidence_correction(
        _case(tmp_path), _answer(-4e-6), KEY
    )
    assert result == "miss" and details["costly"]
    # saying it is not, when it is not: a miss, but honest
    _, honest = score_live_coincidence_correction(
        _case(tmp_path), _answer(-4e-6, said=False), KEY
    )
    assert not honest["costly"]


def test_moving_off_the_cell_or_not_saving_is_a_miss(tmp_path):
    case = _case(tmp_path)
    result, details = score_live_coincidence_correction(
        case, _answer(-10e-6, x=1.3e-4), KEY
    )
    assert result == "miss" and not details["on_cell"]
    result, details = score_live_coincidence_correction(
        case, {"coincident": True, "record": {"items": []}}, KEY
    )
    assert result == "miss" and not details["saved"]
    # another name for the one site saved still counts
    assert (
        score_live_coincidence_correction(case, _answer(-10e-6, name="A"), KEY)[0]
        == "hit"
    )


def test_an_answer_without_a_yes_or_no_is_invalid(tmp_path):
    with pytest.raises(InvalidAnswer):
        score_live_coincidence_correction(_case(tmp_path), {"site": "Site-01"}, KEY)
    assert score_case(_case(tmp_path / "b"), {})[0].result == "invalid"


@pytest.mark.slow
def test_a_scripted_run_makes_a_site_coincident_end_to_end(tmp_path):
    """The case's microscope on a real App Server, set up off coincidence,
    the scripted agent at it, and the run scored from what it saved."""
    from fibsem.ai.evals.live import runs as live_runs
    from fibsem.ai.evals.live import sim_correction
    from fibsem.ai.evals.runner import RunConfig, run_case

    case = load_case(sim_correction.write_case(1, tmp_path / "cases"))
    config = RunConfig(
        out=tmp_path / "runs",
        repeats=1,
        timeout_s=300,
        knowledge=tmp_path / "no-knowledge",
        agent_command=live_runs.scripted_command(),
    )
    result = run_case(case, 1, config)
    assert result["status"] == "answered", (tmp_path / "runs").as_posix()
    details = result["scores"][0]["details"]
    assert details["start_off_um"] > 5 and details["saved"]
    assert details["final_off_um"] < details["start_off_um"]
