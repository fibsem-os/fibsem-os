"""Milling patterns must land on the same side of the image in both renderers.

FIB-692: `convert_pattern_to_napari_polygon` added the image centre to the
vertex y instead of subtracting it, so polygons drew mirrored about the centre —
a vertex 100 px above centre rendered 100 px below. Microscope +y is up while
image rows go down, so y has to be mirrored; every sibling converter and
`plotting.py`'s own polygon converter do exactly that.

The error is 2 x the distance from the centre, so it vanishes for a pattern on
the centre line. Symmetric, centred fixtures cannot see it — these are
deliberately off-centre and asymmetric.

NOTE these skip on CI, which installs no napari. `fibsem/ui/napari/patterns.py`
imports it at module level, so the converters cannot be reached without it.
"""
import numpy as np
import pytest

pytest.importorskip("napari")

from fibsem.milling.patterning.plotting import _polygon_pattern_to_image_pixels
from fibsem.structures import FibsemPolygonSettings, FibsemRectangleSettings
from fibsem.ui.napari.patterns import (
    convert_pattern_to_napari_polygon,
    convert_pattern_to_napari_rect,
)

SHAPE = (512, 1024)   # (height, width) -> centre (256, 512)
PS = 1e-8             # 10 nm/px, so 1e-6 m = 100 px
ABOVE = 1e-6          # +y in microscope coordinates is UP


def test_a_vertex_above_the_centre_is_drawn_above_the_centre():
    """The defect, stated as plainly as it can be."""
    pattern = FibsemPolygonSettings(vertices=[(0.0, ABOVE)], depth=1e-6)

    (row, col), = convert_pattern_to_napari_polygon(pattern, SHAPE, PS)[0]

    assert row == pytest.approx(156.0), "100 px above a centre of 256"
    assert row < 256, "drawn below the centre — mirrored (FIB-692)"
    assert col == pytest.approx(512.0), "x was never wrong"


def test_the_two_renderers_agree_on_every_vertex():
    """The strongest form: napari and matplotlib draw the same polygon.

    They disagreed by 2x the offset before, and nothing compared them."""
    pattern = FibsemPolygonSettings(
        vertices=[(2e-6, ABOVE), (-3e-6, -2e-6), (1e-6, 0.0)], depth=1e-6
    )

    napari_yx = convert_pattern_to_napari_polygon(pattern, SHAPE, PS)[0]
    mpl_xy = _polygon_pattern_to_image_pixels(pattern, PS, SHAPE)

    assert np.allclose(napari_yx[:, 0], mpl_xy[:, 1]), "y disagrees"
    assert np.allclose(napari_yx[:, 1], mpl_xy[:, 0]), "x disagrees"


def test_a_polygon_agrees_with_a_rectangle_about_which_way_is_up():
    """Cross-check inside the napari module itself: the same offset, two pattern
    types, one answer. The rectangle converter was always right."""
    poly = FibsemPolygonSettings(vertices=[(0.0, ABOVE)], depth=1e-6)
    rect = FibsemRectangleSettings(
        width=1e-6, height=1e-6, depth=1e-6, centre_x=0.0, centre_y=ABOVE
    )

    (poly_row, _), = convert_pattern_to_napari_polygon(poly, SHAPE, PS)[0]
    rect_rows = [row for row, _ in convert_pattern_to_napari_rect(rect, SHAPE, PS)[0]]

    assert poly_row == pytest.approx(sum(rect_rows) / len(rect_rows))


def test_a_vertex_on_the_centre_line_never_showed_the_bug():
    """Why it survived: with y = 0 the mirror is the identity."""
    pattern = FibsemPolygonSettings(vertices=[(4e-6, 0.0)], depth=1e-6)

    (row, col), = convert_pattern_to_napari_polygon(pattern, SHAPE, PS)[0]

    assert (row, col) == pytest.approx((256.0, 912.0))


def test_the_result_is_row_column_for_napari():
    """The swap is real and has to happen after the conversion, not before."""
    pattern = FibsemPolygonSettings(vertices=[(4e-6, ABOVE)], depth=1e-6)

    (row, col), = convert_pattern_to_napari_polygon(pattern, SHAPE, PS)[0]

    assert (row, col) == pytest.approx((156.0, 912.0))


def test_a_polygon_with_no_vertices_is_empty_not_an_error():
    """Degenerate, but it used to raise IndexError from the (x, y) swap."""
    pattern = FibsemPolygonSettings(vertices=np.zeros((0, 2)), depth=1e-6)

    assert convert_pattern_to_napari_polygon(pattern, SHAPE, PS)[0].shape == (0, 2)
