"""The overview plot's labels clear their marker and stay on the image.

Two defects, both measured on a real experiment before the fix:

- labels were offset a fixed ten image pixels from the marker, which at an overview's
  scale is 1.8 pt -- inside a crosshair whose arm reaches 10 pt;
- a label near the right edge was clipped (``clip_on=True``) with no sign it had been,
  so ``02-civil-cub`` read as ``02-civil-c``. Its description, not clipped, ran off the
  image onto the page margin instead.
"""

import os

import numpy as np
import pytest

from fibsem import utils
from fibsem.imaging.tiling.plotting import (
    _label_placement,
    plot_minimap,
    plot_stage_positions_on_image,
)
from fibsem.imaging.tiling.reprojection import reproject_stage_positions_onto_image2
from fibsem.structures import BeamType, FibsemImage, FibsemStagePosition, ImageSettings

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_WIDE = (256, 768)  # (height, width)


@pytest.fixture(scope="module")
def overview() -> FibsemImage:
    import fibsem.config as fibsem_config

    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "microscope-configuration.yaml",
    )
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    image = FibsemImage.generate_blank_image(
        resolution=(_WIDE[1], _WIDE[0]), hfw=900e-6
    )
    image.data = np.zeros(_WIDE, dtype=np.uint8)
    image.metadata.image_settings = ImageSettings(
        hfw=900e-6, beam_type=BeamType.ELECTRON
    )
    image.metadata.microscope_state = microscope.get_microscope_state(
        beam_type=BeamType.ELECTRON
    )
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    return image


def _position(image: FibsemImage, name: str, near_right_edge: bool):
    """A position at the image centre, or near its right-hand edge.

    The direction a stage move appears in on the image follows the recorded scan
    rotation, so both signs are tried rather than assuming one.
    """
    base = image.metadata.microscope_state.stage_position
    offsets = (380e-6, -380e-6) if near_right_edge else (0.0,)
    for dx in offsets:
        position = FibsemStagePosition(
            x=base.x + dx, y=base.y, z=base.z, r=base.r, t=base.t
        )
        position.name = name
        (pt,) = reproject_stage_positions_onto_image2(image, [position])
        if not near_right_edge or pt.x > image.data.shape[1] * 0.85:
            return position, pt
    raise AssertionError("no offset put the position near the right-hand edge")


def _label(fig, name):
    labels = [t for t in fig.axes[0].texts if t.get_text() == name]
    assert labels, f"{name} was not labelled"
    return labels[0]


class TestPlacement:
    def test_the_offset_clears_the_marker_whatever_its_size(self):
        """In points, from the end of the marker's arm -- not a count of image pixels."""
        for half in (5.0, 10.0, 40.0):
            (dx, dy), _, _ = _label_placement(384, 128, _WIDE, marker_half_points=half)
            assert dx > half and dy > half

    def test_a_marker_near_the_right_edge_is_labelled_to_its_left(self):
        (dx, _), ha, _ = _label_placement(760, 128, _WIDE, marker_half_points=10)
        assert dx < 0 and ha == "right"

    def test_a_marker_near_the_top_is_labelled_below_it(self):
        """Offsets are in display space, so a negative dy is down the page."""
        (_, dy), _, va = _label_placement(384, 2, _WIDE, marker_half_points=10)
        assert dy < 0 and va == "top"

    def test_a_marker_in_open_ground_is_labelled_above_and_to_the_right(self):
        (dx, dy), ha, va = _label_placement(384, 128, _WIDE, marker_half_points=10)
        assert dx > 0 and dy > 0
        assert (ha, va) == ("left", "bottom")


@pytest.mark.parametrize("render", [plot_minimap, plot_stage_positions_on_image])
class TestDrawn:
    def test_the_label_starts_clear_of_the_crosshair(self, overview, render):
        position, pt = _position(overview, "01-lamella", near_right_edge=False)
        fig = render(overview, [position], show_names=True)
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax = fig.axes[0]
            box = _label(fig, "01-lamella").get_window_extent(renderer)
            marker_x, _ = ax.transData.transform((pt.x, pt.y))
            gap_points = (box.x0 - marker_x) * 72.0 / fig.dpi
            assert gap_points > 10, f"label starts {gap_points:.1f} pt from the centre"
        finally:
            plt.close(fig)

    def test_a_label_at_the_right_edge_stays_on_the_image(self, overview, render):
        position, _ = _position(
            overview, "02-a-long-lamella-name", near_right_edge=True
        )
        fig = render(overview, [position], show_names=True)
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            axes_box = fig.axes[0].get_window_extent(renderer)
            box = _label(fig, "02-a-long-lamella-name").get_window_extent(renderer)
            assert box.x1 <= axes_box.x1 + 1, "the label ran off the right of the image"
            assert box.x0 >= axes_box.x0 - 1, "the label ran off the left of the image"
        finally:
            plt.close(fig)


def test_a_description_stays_with_its_name_at_the_edge(overview):
    position, _ = _position(overview, "02-lamella", near_right_edge=True)
    fig = plot_minimap(
        overview,
        [position],
        show_names=True,
        show_descriptions=True,
        descriptions={"02-lamella": "slight curtaining, still usable"},
    )
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_box = fig.axes[0].get_window_extent(renderer)
        box = _label(fig, "slight curtaining, still usable").get_window_extent(renderer)
        assert box.x1 <= axes_box.x1 + 1, "the description ran off the image"
    finally:
        plt.close(fig)


def test_the_name_is_the_line_further_from_the_marker(overview):
    """Above the marker, the description sits between it and the name."""
    position, _ = _position(overview, "01-lamella", near_right_edge=False)
    fig = plot_minimap(
        overview,
        [position],
        show_names=True,
        show_descriptions=True,
        descriptions={"01-lamella": "good ice"},
    )
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        name = _label(fig, "01-lamella").get_window_extent(renderer)
        sub = _label(fig, "good ice").get_window_extent(renderer)
        assert name.y0 >= sub.y1 - 1, "the description is drawn over or above the name"
    finally:
        plt.close(fig)
