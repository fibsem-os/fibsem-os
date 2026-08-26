"""Several overviews of one view, drawn on one set of axes.

Two overviews of the same grid -- one re-acquired, or two covering different regions --
are two pictures of one thing. A page each leaves the reader to stitch them mentally,
with nothing saying which is current. Composited, a lamella that falls off the edge of
one but inside another is simply on the page.

What has to be true for that to be worth doing:

* an image is placed at the *ground it covers*, so a finer pixel size draws smaller;
* it is placed where it was *acquired*, from its own recorded stage position;
* the markers land in the same frame as the pixels -- which they do by construction
  here, because both go through `reproject_stage_positions_onto_image2`;
* images from **different views** are never composited, because they do not register and
  the result would look exactly as authoritative as a correct one.
"""

import copy
import os

import numpy as np
import pytest

from fibsem import utils
from fibsem.imaging.tiling.plotting import (
    compose_overview_extent,
    plot_overview_composite,
)
from fibsem.structures import BeamType, FibsemImage, ImageSettings

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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


@pytest.fixture
def overview(microscope) -> FibsemImage:
    image = FibsemImage.generate_blank_image(resolution=(600, 400), hfw=900e-6)
    image.data = np.zeros((400, 600), dtype=np.uint8)
    image.metadata.image_settings = ImageSettings(hfw=900e-6, beam_type=BeamType.ION)
    image.metadata.microscope_state = microscope.get_microscope_state(
        beam_type=BeamType.ION
    )
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    return image


def _moved(image: FibsemImage, dx: float = 0.0, finer: float = 1.0) -> FibsemImage:
    """A copy of *image* acquired *dx* metres away, at *finer* times the resolution."""
    other = copy.deepcopy(image)
    other.metadata.pixel_size.x = image.metadata.pixel_size.x / finer
    other.metadata.pixel_size.y = image.metadata.pixel_size.y / finer
    other.metadata.microscope_state.stage_position.x += dx
    return other


class TestPlacement:
    def test_a_finer_image_covers_less_ground(self, overview):
        """Placement is by ground covered, not by pixel count.

        An overview acquired at twice the resolution over the same field is drawn at
        half the linear size -- otherwise a detail image would claim the whole grid.
        """
        finer = _moved(overview, finer=2.0)
        extent = compose_overview_extent(finer, overview, centre=(0.0, 0.0))
        width = extent[1] - extent[0]
        assert width == pytest.approx(finer.data.shape[1] / 2.0)

    def test_an_image_at_the_same_scale_keeps_its_size(self, overview):
        same = _moved(overview)
        extent = compose_overview_extent(same, overview, centre=(0.0, 0.0))
        assert extent[1] - extent[0] == pytest.approx(same.data.shape[1])

    def test_y_runs_downwards(self, overview):
        """imshow was given origin="upper" for the reference, so the extent must agree.

        Getting this backwards flips a placed image vertically -- which on an overview of
        ice looks entirely plausible.
        """
        extent = compose_overview_extent(overview, overview, centre=(0.0, 0.0))
        assert extent[2] > extent[3]


class TestComposite:
    def test_the_offset_matches_the_stage_move(self, overview):
        """A metre on the stage is a metre on the page.

        Checked as a magnitude: the *sign* follows the recorded scan rotation, and is
        the reprojection's business rather than this function's. The markers go through
        the same reprojection, so they agree with the pixels either way -- which is the
        property worth having and the reason it is not duplicated here.
        """
        pixel_size = overview.metadata.pixel_size.x
        for dx_um in (0.0, 40.0, -25.0):
            other = _moved(overview, dx=dx_um * 1e-6)
            fig = plot_overview_composite(
                [overview, other], positions=[], show_names=False, show_scalebar=False
            )
            try:
                extent = fig.axes[0].images[1].get_extent()
                centre = (extent[0] + extent[1]) / 2.0
                offset_um = (centre - overview.data.shape[1] / 2.0) * pixel_size * 1e6
                assert abs(offset_um) == pytest.approx(abs(dx_um), abs=0.5)
            finally:
                plt.close(fig)

    def test_the_view_grows_to_hold_every_image(self, overview):
        """An overview off the edge of the first one is still on the page."""
        far = _moved(overview, dx=300e-6)
        fig = plot_overview_composite(
            [overview, far], positions=[], show_names=False, show_scalebar=False
        )
        try:
            ax = fig.axes[0]
            x0, x1 = ax.get_xlim()
            for image in ax.images:
                e = image.get_extent()
                assert x0 <= e[0] + 1e-6 and e[1] <= x1 + 1e-6
        finally:
            plt.close(fig)

    def test_one_image_is_just_that_image(self, overview):
        """The single-overview case must not be a special path that drifts."""
        fig = plot_overview_composite(
            [overview], positions=[], show_names=False, show_scalebar=False
        )
        try:
            ax = fig.axes[0]
            assert len(ax.images) == 1
            assert ax.get_xlim() == pytest.approx((-0.5, overview.data.shape[1] - 0.5))
        finally:
            plt.close(fig)

    def test_it_refuses_an_empty_list(self):
        with pytest.raises(ValueError):
            plot_overview_composite([], positions=[])

    def test_an_overview_with_no_stage_position_is_skipped_not_fatal(self, overview):
        """One unplaceable image must not cost the page.

        The map is the valuable thing; a file that cannot say where it was taken is
        dropped with a warning rather than taking the others down with it.
        """
        broken = _moved(overview)
        broken.metadata.microscope_state.stage_position = None
        fig = plot_overview_composite(
            [overview, broken], positions=[], show_names=False, show_scalebar=False
        )
        try:
            assert len(fig.axes[0].images) == 1
        finally:
            plt.close(fig)


class TestViewGrouping:
    """Only images that register may share a page."""

    def test_beam_type_separates_views(self, overview):
        from fibsem.applications.autolamella.tools.handoff_map import view_key

        electron = copy.deepcopy(overview)
        electron.metadata.image_settings.beam_type = BeamType.ELECTRON
        assert view_key(overview) != view_key(electron)

    def test_stage_orientation_separates_views(self, overview):
        from fibsem.applications.autolamella.tools.handoff_map import view_key

        tilted = copy.deepcopy(overview)
        tilted.metadata.microscope_state.stage_position.t += np.radians(20)
        assert view_key(overview) != view_key(tilted)

    def test_the_same_view_groups_together(self, overview):
        """A re-acquired overview is the same view, however far the stage moved."""
        from fibsem.applications.autolamella.tools.handoff_map import view_key

        assert view_key(overview) == view_key(_moved(overview, dx=500e-6))

    def test_a_degree_of_wobble_is_the_same_view(self, overview):
        """Stage tilt is read back with noise; a hundredth of a degree is not a view."""
        from fibsem.applications.autolamella.tools.handoff_map import view_key

        jittered = copy.deepcopy(overview)
        jittered.metadata.microscope_state.stage_position.t += np.radians(0.01)
        assert view_key(overview) == view_key(jittered)

    def test_the_label_names_the_beam_and_the_pose(self, overview):
        """No microscope here, so the label says what the image itself records."""
        from fibsem.applications.autolamella.tools.handoff_map import (
            view_key,
            view_label,
        )

        label = view_label(view_key(overview))
        assert "Ion beam" in label
        assert "deg" in label
