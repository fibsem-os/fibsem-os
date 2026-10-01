"""The overview plot's figure takes the image's shape, not a fixed square.

Both renderers defaulted to ``figsize=(15, 15)``. A stitched overview is usually much
wider than it is tall, so most of a square figure is blank paper -- and
``bbox_inches="tight"`` cannot trim it, because the gap sits inside the bounding box,
between the title and the image.
"""

import os

import numpy as np
import pytest

from fibsem import utils
from fibsem.imaging.tiling.plotting import (
    figsize_for_image,
    plot_minimap,
    plot_stage_positions_on_image,
)
from fibsem.structures import BeamType, FibsemImage, ImageSettings

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Deliberately wide: a square test image would pass every assertion here whatever
# the code did.
_WIDE = (256, 768)  # (height, width), 3:1


@pytest.fixture(scope="module")
def wide_overview() -> FibsemImage:
    """A 3:1 image carrying enough metadata for positions to be reprojected onto it."""
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


class TestFigsizeForImage:
    def test_it_follows_the_image_aspect(self):
        width, height = figsize_for_image(_WIDE)
        assert width / height == pytest.approx(3.0)

    @pytest.mark.parametrize("shape", [(0, 0), (10,), ()])
    def test_a_degenerate_shape_does_not_raise(self, shape):
        """Rendering must not be the thing that reports a malformed image."""
        width, height = figsize_for_image(shape)
        assert width > 0 and height > 0

    @pytest.mark.parametrize("shape", [(1, 10_000), (10_000, 1)])
    def test_an_extreme_aspect_is_clamped(self, shape):
        """A ribbon of a mosaic still leaves room for a title."""
        width, height = figsize_for_image(shape)
        assert 0.2 <= height / width <= 5.0


class TestRenderersDefaultToTheImageShape:
    @pytest.mark.parametrize("render", [plot_minimap, plot_stage_positions_on_image])
    def test_the_default_figure_has_the_image_aspect(self, wide_overview, render):
        fig = render(wide_overview, positions=[])
        try:
            width, height = fig.get_size_inches()
            assert width / height == pytest.approx(3.0)
        finally:
            plt.close(fig)

    @pytest.mark.parametrize("render", [plot_minimap, plot_stage_positions_on_image])
    def test_an_explicit_figsize_still_wins(self, wide_overview, render):
        """Callers that know what page they are filling keep control."""
        fig = render(wide_overview, positions=[], figsize=(6, 6))
        try:
            assert tuple(fig.get_size_inches()) == (6, 6)
        finally:
            plt.close(fig)
