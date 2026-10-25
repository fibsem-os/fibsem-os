"""Experimental tile registration (`fibsem.imaging.tiling.registration`).

Tiles are cut from one synthetic scene at known, deliberately jittered positions, so
the truth is exact: registration must recover where each tile came from, using only
the layout check's measurements of the pixels.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from fibsem.imaging.tiling.layout_check import TileLayoutCheck
from fibsem.imaging.tiling.registration import (
    MAX_SHIFT,
    compose_mosaic,
    measure_registration,
)
from fibsem.structures.settings import OverviewAcquisitionSettings

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


# Small enough that the test's shifts are well inside `MAX_SHIFT`.
PIXEL_SIZE = 50e-9


def _register(tiles, planned, pixel_size=PIXEL_SIZE, requested=True):
    check = TileLayoutCheck(planned)
    for key in sorted(tiles):
        check.add(key, tiles[key])
    return measure_registration(
        check, planned, tile_size=W, pixel_size=pixel_size, requested=requested
    )


def _relative(positions, key=(0, 0)):
    oy, ox = positions[key]
    return {k: (y - oy, x - ox) for k, (y, x) in positions.items()}


ERRORS = {(0, 1): (5, -9), (1, 0): (-6, 4), (1, 1): (8, 11), (0, 2): (-3, 7)}


def test_registration_recovers_where_each_tile_was_cut():
    tiles, planned, truth = _cut(2, 3, ERRORS)

    registration = _register(tiles, planned)

    assert registration.applied, registration.note
    got, want = _relative(registration.positions), _relative(truth)
    for key in tiles:
        assert got[key] == pytest.approx(want[key], abs=1.0), key


def test_registration_keeps_the_mosaic_centred():
    """The mean correction is zero, so the mosaic's centre -- the grid centre its
    metadata records -- does not move."""
    tiles, planned, _ = _cut(2, 3, ERRORS)

    positions = _register(tiles, planned).positions

    shifts = np.array([np.subtract(positions[k], planned[k]) for k in planned])
    np.testing.assert_allclose(shifts.mean(axis=0), 0.0, atol=1e-6)


def test_a_scale_error_within_the_limit_is_aligned_and_recorded():
    """A 6% step error, as the Arctis's 104/107%: aligned while no tile moves more
    than `MAX_SHIFT`, and every tile's move recorded in metres."""
    tiles, planned, _ = _cut(2, 3, {}, scale=(1.0, 0.94))

    registration = _register(tiles, planned)

    assert registration.applied, registration.note
    assert 0 < registration.largest_shift <= MAX_SHIFT
    assert set(registration.shifts) == {f"{r},{c}" for r, c in planned}
    largest = max(np.hypot(*shift) for shift in registration.shifts.values())
    assert largest == pytest.approx(registration.largest_shift)


def test_a_move_past_the_limit_keeps_the_plan_and_says_why():
    """The same 6% at a pixel size where the corners move tens of µm: the plan is
    kept, the note names the move and the limit, and the moves are still recorded."""
    tiles, planned, _ = _cut(2, 3, {}, scale=(1.0, 0.94))

    registration = _register(tiles, planned, pixel_size=3e-6)

    assert not registration.applied and registration.positions is None
    assert registration.largest_shift > MAX_SHIFT
    assert "left as planned" in registration.note and "15 µm" in registration.note
    assert registration.shifts


def test_tiles_not_asked_to_align_are_measured_and_left():
    tiles, planned, _ = _cut(2, 3, ERRORS)

    registration = _register(tiles, planned, requested=False)

    assert not registration.applied and registration.largest_shift > 0
    assert registration.to_dict()["requested"] is False
    assert "as asked" in registration.note


def test_registration_needs_enough_matched_pairs():
    """A 1x2 has one pair: nothing could outvote a false match."""
    tiles, planned, _ = _cut(1, 2, {(0, 1): (4, -6)})

    registration = _register(tiles, planned)

    assert not registration.applied
    assert "needs 3" in registration.note


def test_featureless_tiles_are_left_as_planned():
    tiles, planned, _ = _cut(2, 2, {})
    tiles = {k: np.full((H, W), 90, dtype=np.uint8) for k in tiles}

    registration = _register(tiles, planned)

    assert not registration.applied and registration.positions is None


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
