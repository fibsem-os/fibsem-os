"""The transform prior from a previous run (FIB-956): the fitted rotation and
scale of an earlier correlation, this lamella's first, the geometry only as
the fallback; the translation is never taken from it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fibsem.correlation.geometry import NominalTransform
from fibsem.correlation.history import CorrelationRun
from fibsem.correlation.prior import (
    experiment_runs,
    prior_from_runs,
    transform_from_result,
)
from fibsem.correlation.structures import (
    CorrelationInputData,
    CorrelationResult,
    CorrelationState,
)

FIXTURE = Path(__file__).parent / "fixtures" / "nominal_transform.json"
ENTRIES = json.loads(FIXTURE.read_text())["entries"]
ARCTIS = next(e for e in ENTRIES if e["system"] == "arctis")
METEOR = [e for e in ENTRIES if e["system"] == "meteor"]


def _result(entry: dict, *, fm_z_scale: float, fib_px: float) -> CorrelationResult:
    fit = entry["fit"]
    return CorrelationResult(
        scale=float(fit["scale"]),
        rotation_quaternion=fit["rotation"],
        translation=fit["translation"],
        fm_z_scale=fm_z_scale,
        input_data=CorrelationInputData(stored_fib_image_pixel_size=fib_px),
    )


def _run(name: str, result) -> CorrelationRun:
    return CorrelationRun(
        path=f"/x/{name}", name=name, state=CorrelationState(result=result)
    )


def test_transform_from_result_keeps_rotation_and_scale_not_translation():
    e = ARCTIS
    zan = e["fm"]["pixel_size_z"] / e["fm"]["pixel_size_x"]
    res = _result(e, fm_z_scale=zan, fib_px=e["fib"]["pixel_size"])
    t = transform_from_result(
        res,
        fm_pixel_size=e["fm"]["pixel_size_x"],
        fm_pixel_size_z=e["fm"]["pixel_size_z"],
        fib_pixel_size=e["fib"]["pixel_size"],
    )
    R = np.asarray(e["fit"]["rotation"])
    assert np.allclose(t.projection, e["fit"]["scale"] * R[:2, :])
    assert t.angle_to(R) < 1e-6
    assert np.allclose(t.translation, 0.0)
    assert t.z_anisotropy == pytest.approx(zan)


def test_scale_follows_the_current_fib_pixel_size():
    e = ARCTIS
    res = _result(e, fm_z_scale=2.0, fib_px=e["fib"]["pixel_size"])
    half = transform_from_result(
        res,
        fm_pixel_size=e["fm"]["pixel_size_x"],
        fm_pixel_size_z=e["fm"]["pixel_size_z"],
        fib_pixel_size=e["fib"]["pixel_size"] / 2,
    )
    # half the FIB pixel size: twice the FIB pixels per FM pixel
    assert half.scale == pytest.approx(2 * e["fit"]["scale"])


def test_a_raw_slice_fit_is_brought_to_isotropic_units():
    e = ARCTIS
    zan = e["fm"]["pixel_size_z"] / e["fm"]["pixel_size_x"]
    iso = transform_from_result(
        _result(e, fm_z_scale=zan, fib_px=None),
        fm_pixel_size=e["fm"]["pixel_size_x"],
        fm_pixel_size_z=e["fm"]["pixel_size_z"],
        fib_pixel_size=e["fib"]["pixel_size"],
    )
    raw = transform_from_result(
        _result(e, fm_z_scale=1.0, fib_px=None),
        fm_pixel_size=e["fm"]["pixel_size_x"],
        fm_pixel_size_z=e["fm"]["pixel_size_z"],
        fib_pixel_size=e["fib"]["pixel_size"],
    )
    # the depth column was divided by the anisotropy; the in-plane map is untouched
    assert np.allclose(raw.projection[:, 2] * zan, iso.projection[:, 2])
    assert np.allclose(raw.projection[:, :2], iso.projection[:, :2])


def test_unusable_results_are_skipped_and_the_order_is_honoured():
    e = ARCTIS
    good = _result(e, fm_z_scale=2.0, fib_px=None)
    runs = [
        ("this lamella, run b", _run("b", None)),
        ("this lamella, run a", _run("a", CorrelationResult(scale=0.0))),
        ("other, run z", _run("z", good)),
    ]
    prior = prior_from_runs(
        runs,
        fm_pixel_size=e["fm"]["pixel_size_x"],
        fm_pixel_size_z=e["fm"]["pixel_size_z"],
        fib_pixel_size=e["fib"]["pixel_size"],
    )
    assert prior is not None and prior.source == "other, run z"
    assert (
        prior_from_runs([], fm_pixel_size=1, fm_pixel_size_z=1, fib_pixel_size=1)
        is None
    )


def test_meteor_fits_agree_with_each_other_far_better_than_with_geometry():
    """The reason a previous run is a better prior than the geometry."""
    fits = [np.asarray(e["fit"]["rotation"]) for e in METEOR]
    t = NominalTransform(
        projection=fits[0][:2, :],
        translation=np.zeros(2),
        scale=1.0,
        fm_pixel_size=1.0,
        fm_pixel_size_z=1.0,
        fib_pixel_size=1.0,
    )
    # within two degrees of each other (the geometry sits five to six away)
    assert max(t.angle_to(R) for R in fits[1:]) < 2.0


def test_experiment_runs_puts_this_lamella_first_newest_first(tmp_path):
    e = ARCTIS
    exp = tmp_path
    for lamella, names in (("01-a", ["r1", "r2"]), ("02-b", ["r3"])):
        for name in names:
            folder = exp / lamella / "Correlation" / name
            folder.mkdir(parents=True)
            CorrelationState(result=_result(e, fm_z_scale=2.0, fib_px=None)).save(
                str(folder / "correlation.json")
            )
    (exp / "experiment.yaml").write_text("positions: []\n")
    ordered = [label for label, _ in experiment_runs(str(exp), str(exp / "02-b"))]
    assert ordered == ["this lamella, run r3", "01-a, run r2", "01-a, run r1"]
    ordered = [label for label, _ in experiment_runs(str(exp))]
    assert ordered == ["01-a, run r2", "01-a, run r1", "02-b, run r3"]


# ── the placement offset (FIB-979) ──────────────────────────────────────


def test_placement_offset_round_trips_through_the_result_file():
    result = CorrelationResult(placement_offset=[-0.4, -8.8])
    assert CorrelationResult.from_dict(result.to_dict()).placement_offset == [
        -0.4,
        -8.8,
    ]
    assert CorrelationResult.from_dict({"poi": []}).placement_offset is None


def test_placement_offset_comes_from_the_first_run_that_measured_it_well():
    from fibsem.correlation.prior import placement_offset_from_runs

    px = ARCTIS["fib"]["pixel_size"]

    def result(offset, rms_um, updated_at):
        return CorrelationResult(
            placement_offset=offset,
            rms_error=rms_um * 1e-6 / px,
            updated_at=updated_at,
            input_data=CorrelationInputData(stored_fib_image_pixel_size=px),
        )

    now = 1_000_000.0
    runs = [
        ("this lamella, run a", _run("a", result(None, 0.3, now))),  # pre-FIB-979
        ("this lamella, run b", _run("b", result([-18.3, -11.4], 5.0, now))),  # poor
        ("02-other, run c", _run("c", result([-0.4, -8.8], 0.5, now - 2 * 86400))),
        ("03-other, run d", _run("d", result([9.0, 9.0], 0.2, now))),
    ]
    got = placement_offset_from_runs(runs, now=now)
    assert got is not None
    assert got.source == "02-other, run c"
    assert np.allclose(got.offset_um, [-0.4, -8.8])
    assert got.age_days == pytest.approx(2.0)
    assert placement_offset_from_runs(runs[:2], now=now) is None
