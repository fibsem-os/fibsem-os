"""Experimental tile registration (`fibsem.imaging.tiling.registration`).

Tiles are cut from one synthetic scene at known, deliberately jittered positions, so
the truth is exact: registration must recover where each tile came from, using only
the layout check's measurements of the pixels.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from fibsem.imaging.tiling.layout_check import TileLayoutCheck
from fibsem.imaging.tiling.registration import compose_mosaic, registered_positions
from fibsem.structures import OverviewAcquisitionSettings

H, W = 200, 300
OVERLAP = 0.15
STEP_Y, STEP_X = int(round(H * (1 - OVERLAP))), int(round(W * (1 - OVERLAP)))


def _scene(shape=(1100, 1400), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scene = sum(
        gaussian_filter(rng.standard_normal(shape), sigma) * sigma
        for sigma in (2, 6, 15)
    )
    scene = (scene - scene.min()) / np.ptp(scene)
    return (scene * 255).astype(np.uint8)


def _cut(nrows, ncols, errors, scale=(1.0, 1.0), origin=(100, 100)):
    """Tiles cut at the planned grid (times `scale`) plus `errors[(r, c)]`."""
    scene = _scene()
    tiles, planned, truth = {}, {}, {}
    for r in range(nrows):
        for c in range(ncols):
            ey, ex = errors.get((r, c), (0, 0))
            y = int(round(r * STEP_Y * scale[0])) + ey
            x = int(round(c * STEP_X * scale[1])) + ex
            tiles[(r, c)] = scene[
                origin[0] + y : origin[0] + y + H, origin[1] + x : origin[1] + x + W
            ]
            planned[(r, c)] = (float(r * STEP_Y), float(c * STEP_X))
            truth[(r, c)] = (float(y), float(x))
    return tiles, planned, truth


def _register(tiles, planned):
    check = TileLayoutCheck(planned)
    for key in sorted(tiles):
        check.add(key, tiles[key])
    return registered_positions(planned, check.pairs, check.summary(), tile_size=W)


def _relative(positions, key=(0, 0)):
    oy, ox = positions[key]
    return {k: (y - oy, x - ox) for k, (y, x) in positions.items()}


ERRORS = {(0, 1): (5, -9), (1, 0): (-6, 4), (1, 1): (8, 11), (0, 2): (-3, 7)}


def test_registration_recovers_where_each_tile_was_cut():
    tiles, planned, truth = _cut(2, 3, ERRORS)

    positions, note = _register(tiles, planned)

    assert positions is not None, note
    got, want = _relative(positions), _relative(truth)
    for key in tiles:
        assert got[key] == pytest.approx(want[key], abs=1.0), key


def test_registration_keeps_the_mosaic_centred():
    """The mean correction is zero, so the mosaic's centre -- the grid centre its
    metadata records -- does not move."""
    tiles, planned, _ = _cut(2, 3, ERRORS)

    positions, _ = _register(tiles, planned)

    shifts = np.array([np.subtract(positions[k], planned[k]) for k in planned])
    np.testing.assert_allclose(shifts.mean(axis=0), 0.0, atol=1e-6)


def test_registration_is_refused_when_the_tiles_do_not_match_the_stage():
    """A 6% scale error: registering would put positions out by 6% of their distance
    from the centre, so the plan is kept and the note says why."""
    tiles, planned, _ = _cut(2, 3, {}, scale=(1.0, 0.94))

    positions, note = _register(tiles, planned)

    assert positions is None
    assert "left as planned" in note


def test_registration_needs_enough_matched_pairs():
    """A 1x2 has one pair: nothing could outvote a false match."""
    tiles, planned, _ = _cut(1, 2, {(0, 1): (4, -6)})

    positions, note = _register(tiles, planned)

    assert positions is None
    assert "needs 3" in note


def test_featureless_tiles_are_left_as_planned():
    tiles, planned, _ = _cut(2, 2, {})
    tiles = {k: np.full((H, W), 90, dtype=np.uint8) for k in tiles}

    positions, _ = _register(tiles, planned)

    assert positions is None


def test_compose_at_the_true_positions_rebuilds_the_scene():
    scene = _scene()
    tiles, _, truth = _cut(2, 2, {(0, 1): (5, -9), (1, 0): (-6, 4), (1, 1): (8, 11)})
    shape = (STEP_Y + H, STEP_X + W)

    mosaic = compose_mosaic(tiles, truth, shape)

    assert mosaic.shape == shape and mosaic.dtype == np.uint8
    inner = (slice(20, STEP_Y + H - 20), slice(30, STEP_X + W - 30))
    expected = scene[100 : 100 + shape[0], 100 : 100 + shape[1]]
    np.testing.assert_array_equal(mosaic[inner], expected[inner])


def test_compose_clips_a_tile_moved_off_the_canvas():
    tiles, planned, _ = _cut(2, 2, {})
    positions = dict(planned)
    positions[(0, 0)] = (-10.0, -20.0)

    mosaic = compose_mosaic(tiles, positions, (STEP_Y + H, STEP_X + W))

    assert mosaic.shape == (STEP_Y + H, STEP_X + W)


def test_the_setting_is_off_by_default_and_round_trips():
    assert OverviewAcquisitionSettings().register_tiles is False
    assert OverviewAcquisitionSettings.from_dict({}).register_tiles is False
    on = OverviewAcquisitionSettings(register_tiles=True)
    assert OverviewAcquisitionSettings.from_dict(on.to_dict()).register_tiles is True
