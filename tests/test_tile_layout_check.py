"""The overview tile layout check (`fibsem.imaging.tiling.layout_check`).

Tiles are cut from one synthetic scene at known positions, so the truth is exact: the
check must recover how far apart they really are using only the pixels and the plan.
"""

import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from fibsem.imaging.tiling.layout_check import (
    TileLayoutCheck,
    measure_pair_offset,
)

H, W = 200, 300
OVERLAP = 0.15
STEP_Y, STEP_X = int(round(H * (1 - OVERLAP))), int(round(W * (1 - OVERLAP)))


def _scene(shape=(1100, 1400), seed=0) -> np.ndarray:
    """Blobby texture with structure at several scales, as uint8."""
    rng = np.random.default_rng(seed)
    scene = sum(
        gaussian_filter(rng.standard_normal(shape), sigma) * sigma
        for sigma in (2, 6, 15)
    )
    scene = (scene - scene.min()) / np.ptp(scene)
    return (scene * 255).astype(np.uint8)


def _tileset(nrows, ncols, scale=(1.0, 1.0), origin=(50, 50)):
    """Tiles whose true step is `scale` times the planned one, per (y, x)."""
    scene = _scene()
    tiles, planned = {}, {}
    for r in range(nrows):
        for c in range(ncols):
            y = int(round(r * STEP_Y * scale[0])) + origin[0]
            x = int(round(c * STEP_X * scale[1])) + origin[1]
            tiles[(r, c)] = scene[y : y + H, x : x + W]
            planned[(r, c)] = (float(r * STEP_Y), float(c * STEP_X))
    return tiles, planned


def _run(tiles, planned, order=None):
    check = TileLayoutCheck(planned)
    for key in order or sorted(tiles):
        check.add(key, tiles[key])
    return check


def test_measure_pair_offset_recovers_a_known_shift():
    scene = _scene()
    a = scene[100 : 100 + H, 100 : 100 + W]
    b = scene[107 : 107 + H, 100 + STEP_X - 12 : 100 + STEP_X - 12 + W]

    (dy, dx), score = measure_pair_offset(a, b, planned=(0, STEP_X))

    assert dy == pytest.approx(7, abs=0.6)
    assert dx == pytest.approx(STEP_X - 12, abs=0.6)
    assert score > 0.9


def test_measure_pair_offset_refuses_flat_tiles():
    flat = np.full((H, W), 128, dtype=np.uint8)
    assert measure_pair_offset(flat, flat, planned=(0, STEP_X)) is None


def test_a_layout_that_matches_the_plan_raises_no_warning():
    summary = _run(*_tileset(3, 3)).summary()

    assert summary.x.step_fraction == pytest.approx(1.0, abs=0.01)
    assert summary.y.step_fraction == pytest.approx(1.0, abs=0.01)
    assert summary.pairs_measured == summary.pairs_total == 12
    assert summary.warning is None


def test_a_scale_error_is_measured_and_warned_about():
    """The picture moves 95% of each planned step in x: a pixel size or stage scale
    5% out, as on a real tilted SEM overview."""
    summary = _run(*_tileset(2, 3, scale=(1.0, 0.95))).summary()

    assert summary.x.step_fraction == pytest.approx(0.95, abs=0.01)
    assert summary.y.step_fraction == pytest.approx(1.0, abs=0.01)
    assert summary.warning is not None
    assert "in x" in summary.warning and "in y" not in summary.warning


def test_featureless_tiles_give_no_summary_and_no_warning():
    tiles, planned = _tileset(2, 2)
    tiles = {k: np.full((H, W), 90, dtype=np.uint8) for k in tiles}

    summary = _run(tiles, planned).summary()

    assert summary.x is None and summary.y is None
    assert summary.warning is None


def test_tiles_arriving_out_of_order_are_all_paired_and_released():
    """A spiral starts in the middle; every neighbour pair is still measured once, and
    no tile is held once its neighbours are in."""
    tiles, planned = _tileset(3, 3)
    spiral = [(1, 1), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2)]

    check = _run(tiles, planned, order=spiral)

    assert len(check.pairs) == 12
    assert {(p.a, p.b) for p in check.pairs} == {
        (a, b)
        for a in planned
        for b in ((a[0], a[1] + 1), (a[0] + 1, a[1]))
        if b in planned
    }
    assert check._held == {}


def test_a_sparse_run_only_pairs_the_tiles_it_acquires():
    tiles, planned = _tileset(2, 2)
    del planned[(1, 1)]
    del tiles[(1, 1)]

    check = _run(tiles, planned)

    assert {(p.a, p.b) for p in check.pairs} == {((0, 0), (0, 1)), ((0, 0), (1, 0))}
    assert check._held == {}


def test_save_writes_the_summary_and_every_pair(tmp_path):
    check = _run(*_tileset(2, 3, scale=(1.0, 0.95)))
    path = tmp_path / "tile-layout.json"

    check.save(str(path), pixel_size=3.3e-7)

    record = json.loads(path.read_text())
    assert record["warning"] == check.summary().warning
    assert record["pixel_size"] == 3.3e-7
    assert record["summary"]["x"]["step_fraction"] == pytest.approx(0.95, abs=0.01)
    assert len(record["pairs"]) == 7
    assert {"a", "b", "planned", "measured", "score", "axis"} <= set(record["pairs"][0])
