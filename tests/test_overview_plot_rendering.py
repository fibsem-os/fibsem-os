"""What the exported overview plot must look like, as opposed to what it contains.

The reprojection these plots draw is covered elsewhere (`test_reprojection.py`,
`test_beam_stage_projection.py`); this file covers the *rendering*, which had nothing.
Three defects motivate it, all reproduced against a real experiment before being fixed:

- a wide overview drawn into a square figure spent about two thirds of the exported page
  on blank paper, and `bbox_inches="tight"` could not reclaim any of it;
- labels were offset a fixed number of *image pixels* from their marker, which at any
  real overview scale is inside the crosshair, and were clipped rather than moved at the
  image border, so a name near the edge silently lost its tail;
- `TitledPanel` grew its header row to fill whatever vertical space the panel was given.

Each test below fails on the behaviour it replaces.
"""

import os

import numpy as np
import pytest

from fibsem import utils
from fibsem.imaging.tiling.plotting import (
    _label_placement,
    figsize_for_image,
    plot_minimap,
)
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)

# Deliberately wide. A square test image would pass the figure-shape assertions no
# matter what the code did, which is how this went unnoticed.
_WIDE_SHAPE = (256, 768)  # (height, width) -> 3:1


@pytest.fixture(scope="module")
def microscope():
    import fibsem.config as fibsem_config

    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "microscope-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    return scope


@pytest.fixture(scope="module")
def wide_overview(microscope) -> FibsemImage:
    """A 3:1 image carrying enough metadata for the reprojection to run."""
    image = FibsemImage.generate_blank_image(
        resolution=(_WIDE_SHAPE[1], _WIDE_SHAPE[0]), hfw=900e-6
    )
    image.data = np.zeros(_WIDE_SHAPE, dtype=np.uint8)
    state = microscope.get_microscope_state(beam_type=BeamType.ELECTRON)
    image.metadata.image_settings = ImageSettings(
        hfw=900e-6, beam_type=BeamType.ELECTRON
    )
    image.metadata.microscope_state = state
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    return image


def _page_used_by_axes(ax) -> float:
    """The fraction of the figure's height the axes occupies.

    Deliberately *not* the axes box's aspect ratio, which was the first thing tried and
    is useless here: `imshow` fixes the axes aspect to the image's, and `tight_layout`
    shrinks the box to suit, so the box comes out at the image's aspect whatever shape
    the figure is. The waste is the page *around* that box -- which is exactly what
    `bbox_inches="tight"` also cannot see, since a bounding box is a rectangle and the
    gap between the title and the image lies inside it.
    """
    return ax.get_position().bounds[3]


class TestFigureShape:
    """The figure follows the image, so the page is not mostly margin."""

    def test_matches_the_image_aspect(self):
        width, height = figsize_for_image(_WIDE_SHAPE)
        assert width / height == pytest.approx(3.0)

    @pytest.mark.parametrize(
        "shape",
        [
            (0, 0),  # a degenerate array
            (10,),  # a shape too short to have two axes
            (),  # no shape at all
        ],
    )
    def test_degenerate_shapes_do_not_raise(self, shape):
        """Rendering must not be the thing that reports a malformed image."""
        width, height = figsize_for_image(shape)
        assert width > 0 and height > 0

    @pytest.mark.parametrize("shape", [(1, 10_000), (10_000, 1)])
    def test_extreme_aspects_are_clamped(self, shape):
        """A ribbon of a mosaic still has to leave room for a readable label."""
        width, height = figsize_for_image(shape)
        assert 0.2 <= height / width <= 5.0

    def test_the_axes_fills_the_figure_it_is_given(self, wide_overview):
        """The assertion the old square default failed: 0.355 of the figure height.

        End to end rather than on `figsize_for_image` alone, so that wiring the helper
        up but not calling it -- which is the likelier regression -- is caught too.
        """
        fig = plot_minimap(wide_overview, positions=[], figsize=None)
        try:
            fig.tight_layout()
            used = _page_used_by_axes(fig.axes[0])
            assert used > 0.6, (
                f"the image occupies {used:.0%} of the exported page height; "
                "the rest is blank paper that no bbox can trim"
            )
        finally:
            plt.close(fig)

    def test_an_explicit_figsize_still_wins(self, wide_overview):
        """Callers that know what page they are filling keep control."""
        fig = plot_minimap(wide_overview, positions=[], figsize=(6, 6))
        try:
            assert tuple(fig.get_size_inches()) == (6, 6)
        finally:
            plt.close(fig)


class TestLabelPlacement:
    """Labels clear their marker, and stay on the page."""

    def test_offset_clears_the_marker(self):
        """In points, and scaled to the marker -- not a fixed count of image pixels."""
        (dx_small, _), _, _ = _label_placement(100, 100, _WIDE_SHAPE, 5.0)
        (dx_large, _), _, _ = _label_placement(100, 100, _WIDE_SHAPE, 50.0)
        assert abs(dx_large) > abs(dx_small)
        assert abs(dx_small) > 5.0
        assert abs(dx_large) > 50.0

    def test_a_marker_near_the_right_edge_is_labelled_to_its_left(self):
        (dx, _), ha, _ = _label_placement(
            x=_WIDE_SHAPE[1] - 5, y=128, image_shape=_WIDE_SHAPE, marker_half_points=10
        )
        assert dx < 0 and ha == "right"

    def test_a_marker_near_the_top_is_labelled_below_it(self):
        """Offsets are display-space, so a negative dy is down the page."""
        (_, dy), _, va = _label_placement(
            x=384, y=2, image_shape=_WIDE_SHAPE, marker_half_points=10
        )
        assert dy < 0 and va == "top"

    def test_a_marker_in_open_ground_is_labelled_above_and_right(self):
        (dx, dy), ha, va = _label_placement(
            x=384, y=128, image_shape=_WIDE_SHAPE, marker_half_points=10
        )
        assert dx > 0 and dy > 0
        assert (ha, va) == ("left", "bottom")

    def test_edge_labels_are_drawn_inside_the_image(self, wide_overview):
        """End to end: the name of a position at the right-hand edge lands on the image.

        The behaviour this replaces offset every label the same way and set
        `clip_on=True`, so this label was drawn past the edge and then silently cut in
        half -- which reads as a shorter name, not as a bug.
        """
        state = wide_overview.metadata.microscope_state
        assert state.stage_position is not None
        # A position a little to the right of the image centre, so it reprojects near
        # the right-hand edge of a 900 um field.
        edge = FibsemStagePosition(
            x=state.stage_position.x + 380e-6,
            y=state.stage_position.y,
            z=state.stage_position.z,
            r=state.stage_position.r,
            t=state.stage_position.t,
            coordinate_system="RAW",
        )
        edge.name = "a-long-lamella-name"

        fig = plot_minimap(
            wide_overview, positions=[edge], show_names=True, figsize=None
        )
        try:
            ax = fig.axes[0]
            fig.canvas.draw()
            labels = [a for a in ax.texts if a.get_text() == "a-long-lamella-name"]
            assert labels, "the position was not labelled at all"
            box = labels[0].get_window_extent(fig.canvas.get_renderer())
            axes_box = ax.get_window_extent()
            assert box.x1 <= axes_box.x1 + 1, "the label ran off the right of the image"
            assert box.x0 >= axes_box.x0 - 1, "the label ran off the left of the image"
        finally:
            plt.close(fig)


class TestTitledPanelHeader:
    """The header is a row, not a claim on the whole panel."""

    def test_header_does_not_absorb_vertical_slack(self, qapp):
        from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget

        from fibsem.ui.widgets.custom_widgets import TitledPanel

        content = QWidget()
        QHBoxLayout(content).addWidget(QLabel("a row of controls"))
        panel = TitledPanel("Options", content=content, collapsible=False)

        # A tall neighbour, which is what the overview export dialog is: a panel beside
        # a canvas. Before the fix the header took half of the 800 px.
        tall = QLabel()
        tall.setMinimumHeight(800)

        host = QWidget()
        row = QHBoxLayout(host)
        row.addWidget(tall)
        row.addWidget(panel)
        host.resize(1000, 800)
        host.show()
        qapp.processEvents()

        header_height = panel._header.height()
        assert header_height < 60, (
            f"the header grew to {header_height} px; it should stay a single row"
        )

        host.close()
