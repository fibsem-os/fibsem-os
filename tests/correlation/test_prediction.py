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
    assert sum(1 for c in fm if c.suggested) == 3
    # accepted where the map put them: usable, but not evidence
    fm[0].status = PointStatus.ACCEPTED
    assert len(usable_pairs(fib, fm)) == 1
    assert independent_pairs(fib, fm) == []
    # moved by the user: evidence
    fm[1].status = PointStatus.PLACED
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
    assert proj.n_pairs == 0
    assert np.allclose(proj.matrix, nominal.projection)
    moved = place_predictions(proj, fib, fm)
    assert len(moved) == len(fm)
    for a, b in zip(fib, fm):
        assert b.point.z == pytest.approx(5.0)
        assert proj.fib_from_fm(b.point.x, b.point.y, b.point.z) == pytest.approx(
            (a.point.x, a.point.y), abs=1e-6
        )
    # suggestions survive a projection without pairs
    assert sum(1 for c in fm if c.suggested) == 3


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
    i = next(k for k, c in enumerate(fm) if c.suggested)
    fm[i].point.x += 40.0
    fm[i].point.y -= 25.0
    fm[i].point.z = 7.0
    fm[i].status = PointStatus.PLACED
    fm[i].provenance = PointProvenance.USER
    proj = build_projection(nominal, fib, fm, z_slice=5.0)
    assert proj.n_pairs == 1
    assert "translation from 1 pair" == proj.note
    moved = place_predictions(proj, fib, fm)
    assert fm[i] not in moved
    others = [k for k in range(len(fm)) if k != i]
    # every re-placed prediction maps back onto its FIB point through the
    # translation the placed pair fixed
    for k in others:
        assert proj.fib_from_fm(fm[k].point.x, fm[k].point.y, fm[k].point.z) == (
            pytest.approx((fib[k].point.x, fib[k].point.y), abs=1e-6)
        )
    assert proj.fib_from_fm(fm[i].point.x, fm[i].point.y, fm[i].point.z) == (
        pytest.approx((fib[i].point.x, fib[i].point.y), abs=1e-6)
    )
    # the rest move to the placed pair's slice (the burns share a surface) and
    # are no longer "suggested"
    assert all(fm[k].point.z == pytest.approx(7.0) for k in others)
    assert all(
        fm[k].status == PointStatus.PREDICTED and not fm[k].suggested for k in others
    )


def test_accepted_predictions_do_not_refine_the_map(nominal):
    fib = _fib(ARCTIS)
    fm = predictions_for(fib, [], z_slice=5.0)
    place_predictions(build_projection(nominal, fib, fm, z_slice=5.0), fib, fm)
    for c in fm:
        c.status = PointStatus.ACCEPTED  # taken where the map put them
    proj = build_projection(nominal, fib, fm, z_slice=5.0)
    assert proj.n_pairs == 0
    assert np.allclose(proj.translation, nominal.translation)


def test_many_pairs_still_move_only_the_translation(nominal):
    """Even with more pairs than a rotation has degrees of freedom the
    re-projection keeps the prior's rotation: five burns on one surface
    refitted it 18 degrees off on a METEOR lamella and tripled the ring
    error on an Arctis one. The final fit owns the rotation."""
    fib = _fib(ARCTIS)
    fm = predictions_for(fib, [], z_slice=5.0)
    place_predictions(build_projection(nominal, fib, fm, z_slice=5.0), fib, fm)
    for c in fm[:6]:
        c.point.x += 3.0
        c.point.y -= 2.0
        c.status = PointStatus.PLACED
        c.provenance = PointProvenance.USER
    proj = build_projection(nominal, fib, fm, z_slice=5.0)
    assert proj.n_pairs == 6
    assert np.allclose(proj.matrix, nominal.projection)
    assert proj.note == "translation from 6 pairs"
    # the FM points moved, so the FIB-side translation moves by minus the
    # shift put through the prior's in-plane map
    shift = nominal.projection[:, :2] @ [3.0, -2.0]
    assert np.allclose(proj.translation, nominal.translation - shift)


def test_legacy_status_strings_read_back_into_the_vocabulary():
    for old, new in PointStatus.LEGACY.items():
        c = Coordinate.from_dict(
            {"point": {"x": 1, "y": 2, "z": 3}, "point_type": "FM", "status": old}
        )
        assert c.status == new
        assert not c.suggested
    assert (
        Coordinate.from_dict(
            {"point": {"x": 1, "y": 2, "z": 3}, "point_type": "FM"}
        ).status
        == ""
    )
