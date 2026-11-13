"""The simulated FM on a tilted sample: a point shows where its height puts
it along the line of sight, a cut dims only what lies along that line, and
the plane is in focus along one line across the view."""

import numpy as np
import pytest

from fibsem.drivers.demo import sim_solid
from fibsem.drivers.demo.sim_scene import SampleScene

TILT = np.deg2rad(23.0)


def _tilted(pixel=1e-7) -> SampleScene:
    scene = SampleScene(seed=2, cell_type="none")
    scene._fm_tilt = (float(np.cos(TILT)), float(np.sin(TILT)), pixel)
    return scene


def test_looking_straight_at_the_plane_changes_nothing():
    scene = SampleScene(seed=2, cell_type="none")
    assert scene._fm_seen(3e-6, 1e-6) == (3e-6, 1e-6)
    canvas = np.random.default_rng(0).normal(0, 1, (64, 64)).astype(np.float32)
    assert scene._fm_focus(canvas, 0.0) is canvas
    assert scene._fm_lean(2) == 0


def test_a_height_shifts_a_point_down_a_tilted_view():
    """The view is the plane turned by its tilt: a point keeps its distance
    from the view's centre, and one over the plane shows further along."""
    scene = _tilted()
    fs = np.cos(TILT)
    on_plane, raised = scene._fm_seen(4e-6 * fs, 0.0), scene._fm_seen(4e-6 * fs, 2e-6)
    assert on_plane[0] == pytest.approx(4e-6 * fs)
    assert raised[0] - on_plane[0] == pytest.approx(2e-6 * np.sin(TILT))
    assert np.hypot(*raised) == pytest.approx(np.hypot(4e-6, 2e-6))


def test_a_tilted_plane_is_in_focus_along_one_line_only():
    scene = _tilted()
    canvas = np.random.default_rng(0).normal(0, 1, (400, 300)).astype(np.float32)
    out = scene._fm_focus(canvas, 0.0)
    detail = np.abs(np.diff(out, axis=1)).mean(axis=1)
    middle, top, bottom = (
        detail[190:210].mean(),
        detail[:20].mean(),
        detail[-20:].mean(),
    )
    assert middle > 3 * top and middle > 3 * bottom
    # moving the objective moves the line
    out = scene._fm_focus(canvas, 5e-6)
    detail = np.abs(np.diff(out, axis=1)).mean(axis=1)
    assert abs(int(np.argmax(detail)) - 200) > 60


def test_a_cut_is_seen_through_along_the_line_of_sight():
    """What a tilted view sees at a height is what lies that far along its
    line: a layer's rows move with its height."""
    layer = np.zeros((20, 4), dtype=np.float32)
    layer[10] = 1
    assert np.argmax(sim_solid._rows(layer, 3.4)[:, 0]) == 13
    assert np.argmax(sim_solid._rows(layer, -3.4)[:, 0]) == 7
    assert sim_solid._rows(layer, 0.2) is layer
    assert not sim_solid._rows(layer, 40).any()


def test_the_reflection_sees_a_surface_along_the_same_line():
    """Reflected light takes the line of sight the fluorescence does: a top
    over the plane shows further along by its height, a trench shows the
    wall the line comes in under, and looking straight down nothing moves."""
    z = np.zeros((40, 4), dtype=np.float32)
    values = np.full(z.shape, 0.3, dtype=np.float32)
    solid = np.ones(z.shape, dtype=bool)
    z[10:14], values[10:14] = 4.0, 1.0  # a raised block
    z[24:30], solid[24:30] = -2.0, False  # a trench milled through
    lean = 1.0  # a row per um
    out = sim_solid._along(values, z, solid, lean, floor=-2.0)[:, 0]
    # the block's top, 4 um up, shows 4 rows on
    assert np.allclose(out[14:18], 1.0)
    # in front of it the line comes in under the top: its wall, dim
    assert np.allclose(out[10:14], 0.12, atol=0.01)
    # the trench: nothing where the line passes through it, then the far
    # wall under the ice top (2 um of it: two rows), then the top again
    assert not out[24:28].any()
    assert np.allclose(out[28:30], 0.12 * 0.3, atol=0.01)
    assert out[30] == pytest.approx(0.3)
    # the plane itself, at height 0, stays where it is
    assert out[5] == pytest.approx(0.3) and out[20] == pytest.approx(0.3)
