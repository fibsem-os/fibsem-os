"""Predicted fiducials (FIB-956): the map is the geometry's until the user's
pairs correct it, predictions never count as pairs, and a confirmed point is
never moved by a projection.

Built over the nominal-transform fixture so the map is a real instrument's.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fibsem.correlation.geometry import NominalTransform
from fibsem.correlation.prediction import (
    MIN_PAIRS_FOR_ROTATION_REFIT,
    build_projection,
    independent_pairs,
    place_predictions,
    predictions_for,
    suggest_indices,
    usable_pairs,
)
from fibsem.correlation.structures import (
    Coordinate,
    PointProvenance,
    PointStatus,
    PointType,
    PointXYZ,
)

FIXTURE = Path(__file__).parent / "fixtures" / "nominal_transform.json"
ENTRIES = json.loads(FIXTURE.read_text())["entries"]
ARCTIS = next(e for e in ENTRIES if e["system"] == "arctis")


@pytest.fixture
def nominal() -> NominalTransform:
    fit = ARCTIS["fit"]
    R = np.asarray(fit["rotation"], dtype=float)
    s = float(fit["scale"])
    return NominalTransform(
        projection=s * R[:2, :],
        translation=np.asarray(fit.get("translation", [0.0, 0.0]), dtype=float)[:2],
        scale=s,
        fm_pixel_size=ARCTIS["fm"]["pixel_size_x"],
        fm_pixel_size_z=ARCTIS["fm"]["pixel_size_z"],
        fib_pixel_size=ARCTIS["fib"]["pixel_size"],
    )


def _fib(entry: dict) -> list:
    return [
        Coordinate(PointXYZ(x, y, 0.0), PointType.FIB)
        for x, y in entry["coords"]["fib"]
    ]


# ── pairs ────────────────────────────────────────────────────────────────


def test_predictions_are_not_pairs_and_accepted_predictions_are_not_evidence():
    fib = _fib(ARCTIS)
    fm = predictions_for(fib, [], z_slice=5.0)
    assert len(fm) == len(fib)
    assert all(c.status in PointStatus.TENTATIVE for c in fm)
    assert all(c.provenance == PointProvenance.PROJECTED for c in fm)
    assert usable_pairs(fib, fm) == []
    # three of them are the ones to drag first
    assert sum(1 for c in fm if c.status == PointStatus.SUGGESTED) == 3
    # accepted where the map put them: usable, but not evidence
    fm[0].status = PointStatus.CONFIRMED
    assert len(usable_pairs(fib, fm)) == 1
    assert independent_pairs(fib, fm) == []
    # moved by the user: evidence
    fm[1].status = PointStatus.ADJUSTED
    fm[1].provenance = PointProvenance.USER
    assert len(independent_pairs(fib, fm)) == 1
    # rejected: neither
    fm[1].status = PointStatus.REJECTED
    assert independent_pairs(fib, fm) == []


def test_predictions_only_fill_the_unpaired_tail_and_do_not_suggest_once_paired():
    fib = _fib(ARCTIS)
    fm = [
        Coordinate(
            PointXYZ(1.0, 2.0, 3.0), PointType.FM, provenance=PointProvenance.USER
        )
    ]
    new = predictions_for(fib, fm, z_slice=5.0)
    assert len(new) == len(fib) - 1
    assert all(c.status == PointStatus.PREDICTED for c in new)


def test_suggest_indices_spans_the_set():
    pts = [(0, 0), (1, 0), (0, 1), (10, 10), (10, 0), (0, 10), (5, 5)]
    chosen = suggest_indices(pts, k=3)
    assert len(chosen) == 3 and len(set(chosen)) == 3
    # farthest-point sampling reaches the corners, not the cluster at the origin
    assert {3, 4, 5} & set(chosen)
    assert suggest_indices(pts[:2], k=3) == [0, 1]


# ── the map ──────────────────────────────────────────────────────────────


def test_with_no_pairs_the_map_is_the_geometry_and_predictions_are_rigid(nominal):
    fib = _fib(ARCTIS)
    fm = predictions_for(fib, [], z_slice=5.0)
    proj = build_projection(nominal, fib, fm, z_slice=5.0)
    assert proj.n_pairs == 0 and not proj.rotation_refit
    assert np.allclose(proj.matrix, nominal.projection)
    moved = place_predictions(proj, fib, fm)
    assert len(moved) == len(fm)
    for a, b in zip(fib, fm):
        assert b.point.z == pytest.approx(5.0)
        assert proj.fib_from_fm(b.point.x, b.point.y, b.point.z) == pytest.approx(
            (a.point.x, a.point.y), abs=1e-6
        )
    # suggestions survive a projection without pairs
    assert sum(1 for c in fm if c.status == PointStatus.SUGGESTED) == 3


def test_a_stage_translation_off_the_image_is_replaced_by_centring(nominal):
    fib = _fib(ARCTIS)
    fm = predictions_for(fib, [], z_slice=5.0)
    far = NominalTransform(
        projection=nominal.projection,
        translation=np.array([1e6, 1e6]),
        scale=nominal.scale,
        fm_pixel_size=nominal.fm_pixel_size,
        fm_pixel_size_z=nominal.fm_pixel_size_z,
        fib_pixel_size=nominal.fib_pixel_size,
    )
    h, w = ARCTIS["fm"]["shape"]
    proj = build_projection(far, fib, fm, z_slice=5.0, fm_shape=(h, w))
    assert proj.centred and "centred" in proj.note
    place_predictions(proj, fib, fm)
    centre = np.mean([[c.point.x, c.point.y] for c in fm], axis=0)
    assert centre == pytest.approx((w / 2, h / 2), abs=1e-6)
    # and without the fallback the same map lands nowhere useful
    proj = build_projection(far, fib, fm, z_slice=5.0)
    assert not proj.centred


def test_one_confirmed_pair_moves_the_rest_by_its_offset_and_leaves_it_alone(nominal):
    fib = _fib(ARCTIS)
    fm = predictions_for(fib, [], z_slice=5.0)
    place_predictions(build_projection(nominal, fib, fm, z_slice=5.0), fib, fm)
    before = np.array([[c.point.x, c.point.y] for c in fm])
    i = next(k for k, c in enumerate(fm) if c.status == PointStatus.SUGGESTED)
    fm[i].point.x += 40.0
    fm[i].point.y -= 25.0
    fm[i].status = PointStatus.ADJUSTED
    fm[i].provenance = PointProvenance.USER
    proj = build_projection(nominal, fib, fm, z_slice=5.0)
    assert proj.n_pairs == 1 and not proj.rotation_refit
    assert "translation from 1 pair" == proj.note
    moved = place_predictions(proj, fib, fm)
    assert fm[i] not in moved
    after = np.array([[c.point.x, c.point.y] for c in fm])
    delta = after - before
    others = [k for k in range(len(fm)) if k != i]
    assert np.allclose(delta[others], delta[i], atol=1e-6)
    # the rest stay in the displayed slice (a FIB point fixes a line through the
    # volume, and the prediction is where that line crosses the slice on
    # screen), and are no longer "suggested"
    assert all(fm[k].point.z == pytest.approx(5.0) for k in others)
    assert all(fm[k].status == PointStatus.PREDICTED for k in others)


def test_predictions_follow_the_displayed_slice(nominal):
    fib = _fib(ARCTIS)
    fm = predictions_for(fib, [], z_slice=5.0)
    place_predictions(build_projection(nominal, fib, fm, z_slice=5.0), fib, fm)
    at5 = np.array([[c.point.x, c.point.y] for c in fm])
    place_predictions(build_projection(nominal, fib, fm, z_slice=9.0), fib, fm)
    at9 = np.array([[c.point.x, c.point.y] for c in fm])
    assert all(c.point.z == pytest.approx(9.0) for c in fm)
    # every prediction slid by the same in-plane step: four slices along the
    # depth line, which the transform maps back into the FIB image exactly
    step = at9 - at5
    assert np.allclose(step, step[0], atol=1e-6)
    assert np.linalg.norm(step[0]) > 0
    for a, b in zip(fib, fm):
        proj = build_projection(nominal, fib, fm, z_slice=9.0)
        assert proj.fib_from_fm(b.point.x, b.point.y, 9.0) == pytest.approx(
            (a.point.x, a.point.y), abs=1e-6
        )


def test_accepted_predictions_do_not_refine_the_map(nominal):
    fib = _fib(ARCTIS)
    fm = predictions_for(fib, [], z_slice=5.0)
    place_predictions(build_projection(nominal, fib, fm, z_slice=5.0), fib, fm)
    for c in fm:
        c.status = PointStatus.CONFIRMED  # accepted where the map put them
    proj = build_projection(nominal, fib, fm, z_slice=5.0)
    assert proj.n_pairs == 0
    assert np.allclose(proj.translation, nominal.translation)


def test_enough_user_pairs_refit_the_rotation_from_the_seed(nominal):
    fib = _fib(ARCTIS)
    fm = predictions_for(fib, [], z_slice=5.0)
    place_predictions(build_projection(nominal, fib, fm, z_slice=5.0), fib, fm)
    # the user confirms the predictions after a small rigid shift
    n = MIN_PAIRS_FOR_ROTATION_REFIT
    for c in fm[:n]:
        c.point.x += 3.0
        c.point.y -= 2.0
        c.status = PointStatus.ADJUSTED
        c.provenance = PointProvenance.USER
    proj = build_projection(nominal, fib, fm, z_slice=5.0)
    assert proj.n_pairs == n and proj.rotation_refit
    assert "rotation refitted" in proj.note
    # a rigid shift of the inputs is a translation: the rotation the refit
    # finds is the geometry's, within the solver's tolerance
    assert nominal.angle_to(_complete(proj.matrix / nominal.scale)) < 1.0


def _complete(rows):
    from fibsem.correlation.geometry import _complete_rotation

    return _complete_rotation(rows)
