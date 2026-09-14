"""The fit verdict (FIB-956), calibrated on ten saved runs: eight METEOR runs
from one session that are the good population, and two Arctis runs known to
be bad. The fixture carries the pairs, the POI, the pixel sizes and a prior
taken from another run's isotropic fit; names are system and lamella number
only."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from fibsem.correlation.geometry import NominalTransform
from fibsem.correlation.verdict import (
    LOO_BAD_UM,
    MIRROR_RATIO_AMBIGUOUS,
    FitDiagnostics,
    diagnose,
    reject_outliers,
    verdict,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fit_diagnostics.json"
RUNS = {r["name"]: r for r in json.loads(FIXTURE.read_text())["runs"]}

FORBIDDEN_WORDS = ("mirror", "leave-one-out", "jackknife", "hull", "pairs", "RMS")


def _prior(r: dict) -> NominalTransform:
    R = np.asarray(r["prior"]["rotation"], dtype=float)
    s = float(r["prior"]["scale"])
    return NominalTransform(
        projection=s * R[:2, :],
        translation=np.zeros(2),
        scale=s,
        fm_pixel_size=r["fm_pixel_size"],
        fm_pixel_size_z=r["fm_pixel_size_z"],
        fib_pixel_size=r["fib_pixel_size"],
    )


def _sizes(r: dict) -> dict:
    return dict(
        fib_pixel_size=r["fib_pixel_size"],
        fm_pixel_size=r["fm_pixel_size"],
        fm_pixel_size_z=r["fm_pixel_size_z"],
    )


def _diagnose(r: dict, **kw) -> FitDiagnostics:
    return diagnose(r["fib"], r["fm"], _prior(r), poi=r["poi"], **_sizes(r), **kw)


def _verdict(r: dict):
    d = _diagnose(r)
    return d, verdict(d, reject_outliers(r["fib"], r["fm"], _prior(r), **_sizes(r)))


# ── the calibration holds ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name, tier",
    [
        ("meteor-01-1", "good"),
        ("meteor-02-2", "good"),
        ("meteor-02-3", "good"),
        ("meteor-07-4", "check"),  # target 16 µm outside the fiducials
        ("meteor-08-5", "good"),
        ("meteor-10-6", "check"),  # one pair 1.5 µm off
        ("meteor-12-7", "good"),
        ("meteor-13-8", "check"),  # one pair 1.1 µm off, in depth
        ("arctis-01-1", "bad"),
        ("arctis-01-2", "bad"),
    ],
)
def test_each_saved_run_gets_the_tier_it_was_calibrated_to(name, tier):
    d, v = _verdict(RUNS[name])
    assert v.tier == tier, v.text


def test_the_good_population_is_where_the_thresholds_were_set():
    for name, r in RUNS.items():
        if not name.startswith("meteor"):
            continue
        d = _diagnose(r)
        assert d.rms_um < 0.5
        assert d.loo_median_um < 0.5
        assert d.mirror_ratio > MIRROR_RATIO_AMBIGUOUS
        assert abs(d.scale_ratio - 1.0) < 0.02


def test_leave_one_out_names_the_pair_rms_hides_and_suggests_its_depth():
    d, v = _verdict(RUNS["meteor-13-8"])
    assert d.rms_um < 0.35  # nothing in the RMS
    assert d.worst == 7  # FM 8
    assert d.pairs[7].loo_error_um == pytest.approx(1.09, abs=0.1)
    assert d.suggested_z == pytest.approx(36.0)  # the pick is at 32
    assert v.headline == "Check FM 8."
    assert v.reasons[0].pair == 7
    assert "slice 36" in v.reasons[0].text
    # a Check-tier pair is flagged, never removed automatically
    assert (
        reject_outliers(
            RUNS["meteor-13-8"]["fib"],
            RUNS["meteor-13-8"]["fm"],
            _prior(RUNS["meteor-13-8"]),
            **_sizes(RUNS["meteor-13-8"]),
        )
        == []
    )


def test_a_target_outside_the_fiducials_is_said_in_microns():
    d, v = _verdict(RUNS["meteor-07-4"])
    assert d.poi_hull_distance_um > 5
    assert v.headline == "Check the target."
    assert "16 µm from the nearest fiducial" in v.reasons[0].text
    # inside the hull reads as zero
    assert _diagnose(RUNS["meteor-12-7"]).poi_hull_distance_um == 0.0


def test_the_bad_runs_are_ambiguous_in_depth_and_lose_their_worst_pairs():
    r = RUNS["arctis-01-1"]
    d, v = _verdict(r)
    assert d.mirror_ratio < MIRROR_RATIO_AMBIGUOUS
    assert d.loo_max_um > LOO_BAD_UM
    rejected = reject_outliers(r["fib"], r["fm"], _prior(r), **_sizes(r))
    assert 5 in rejected  # FM 6, 15 µm off
    assert len(r["fib"]) - len(rejected) >= 4
    assert v.headline == "Poor fit. Do not continue."
    assert "cannot say which way is deeper" in v.reasons[0].text
    assert "depths do not agree" in v.reasons[0].text  # scattered, not coplanar
    assert v.reasons[1].pair == 5 and "was removed" in v.reasons[1].text


def test_the_second_bad_run_also_fails_the_scale_check():
    d = _diagnose(RUNS["arctis-01-2"])
    assert abs(d.scale_ratio - 1.0) > 0.03


# ── the rules on synthetic damage ────────────────────────────────────────


def test_moving_one_good_pair_far_is_caught_named_and_removed():
    r = copy.deepcopy(RUNS["meteor-12-7"])
    r["fm"][2][0] += 60.0  # ~4 µm sideways on FM 3
    d, v = _verdict(r)
    assert d.worst == 2
    assert d.pairs[2].loo_error_um > LOO_BAD_UM
    assert reject_outliers(r["fib"], r["fm"], _prior(r), **_sizes(r)) == [2]
    assert v.tier == "bad"
    assert any(x.pair == 2 and "was removed" in x.text for x in v.reasons)


def test_accepted_predictions_are_counted_and_reported():
    r = RUNS["meteor-12-7"]
    n = len(r["fib"])
    d = _diagnose(r, accepted=[True] * (n - 3) + [False] * 3)
    assert d.n_accepted == n - 3
    v = verdict(d)
    assert v.tier == "check"
    assert "Only 3 fiducials were placed by you" in v.text


def test_coplanar_fiducials_read_as_depth_from_the_prior():
    r = copy.deepcopy(RUNS["meteor-12-7"])
    z = float(np.mean([p[2] for p in r["fm"]]))
    for p in r["fm"]:
        p[2] = z
    d = _diagnose(r)
    assert d.depth_span_um == 0.0
    v = verdict(d)
    assert any("same depth" in x.text for x in v.reasons)


# ── wording and serialisation ────────────────────────────────────────────


def test_the_verdict_never_uses_fit_vocabulary():
    for r in RUNS.values():
        _, v = _verdict(r)
        for word in FORBIDDEN_WORDS:
            assert word not in v.text, (r["name"], word, v.text)
        assert "fiducial" in v.text or v.tier == "good"


def test_diagnostics_serialise_without_the_rotation():
    d = _diagnose(RUNS["meteor-01-1"])
    payload = json.loads(json.dumps(d.to_dict()))
    assert payload["n_pairs"] == 9
    assert len(payload["pairs"]) == 9
    assert "rotation" not in payload


def test_too_few_pairs_is_refused():
    r = RUNS["meteor-01-1"]
    with pytest.raises(ValueError):
        diagnose(r["fib"][:2], r["fm"][:2], _prior(r), **_sizes(r))
