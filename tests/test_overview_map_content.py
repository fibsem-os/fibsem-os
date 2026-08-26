"""What the overview map says, as opposed to how it is drawn.

Companion to `test_overview_plot_rendering.py`, which covers the rendering. This covers
the three things that decide the map's *content*: which overview it is drawn on, which
positions are singled out, and whether it says where it came from.

The behaviours here replace ones that were wrong rather than merely absent:

- both the dialog and the PDF report globbed for overviews themselves, disagreed, and the
  dialog then rendered `filenames[-1]` with no way to pick another;
- a position was drawn in the grid colour if its *name* contained "Grid";
- a lamella a human had flagged defective was drawn exactly like a good one.
"""

import os

import numpy as np
import pytest

from fibsem import utils
from fibsem.imaging.tiling.plotting import (
    CURRENT_POSITION_COLOUR,
    GRID_POSITION_COLOUR,
    POSITION_COLOURS,
    plot_minimap,
    plot_stage_positions_on_image,
)
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
)

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


@pytest.fixture(scope="module")
def overview(microscope) -> FibsemImage:
    image = FibsemImage.generate_blank_image(resolution=(512, 512), hfw=900e-6)
    image.data = np.zeros((512, 512), dtype=np.uint8)
    image.metadata.image_settings = ImageSettings(
        hfw=900e-6, beam_type=BeamType.ELECTRON
    )
    image.metadata.microscope_state = microscope.get_microscope_state(
        beam_type=BeamType.ELECTRON
    )
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    return image


def _at_centre(overview: FibsemImage, name: str) -> FibsemStagePosition:
    """A position that reprojects to the middle of the overview, so it is always drawn."""
    base = overview.metadata.microscope_state.stage_position
    pos = FibsemStagePosition(
        x=base.x,
        y=base.y,
        z=base.z,
        r=base.r,
        t=base.t,
        coordinate_system=base.coordinate_system,
    )
    pos.name = name
    return pos


def _marker_colours(fig) -> list:
    """The colours actually handed to the scatter, as hex or name strings."""
    from matplotlib.colors import to_hex

    out = []
    for coll in fig.axes[0].collections:
        for rgba in coll.get_facecolor():
            out.append(to_hex(rgba))
    return out


class TestFindingOverviews:
    """One place decides which overviews an experiment has."""

    @pytest.fixture
    def experiment(self, tmp_path):
        from fibsem.applications.autolamella.structures import Experiment

        exp = Experiment(path=tmp_path, name="find-overviews")
        os.makedirs(exp.path, exist_ok=True)
        for name in (
            "overview-image-10-00-00.tif",
            "overview-image-09-00-00.tiff",
            "overview-11-00-00.ome.tiff",
            "sem-image-10-00-00.tif",
            "ref_Polishing_final_ib.tif",
        ):
            open(os.path.join(exp.path, name), "w").close()
        return exp

    def test_beam_overviews_are_found_in_name_order(self, experiment):
        found = [os.path.basename(p) for p in experiment.find_overview_images()]
        assert found == [
            "overview-image-09-00-00.tiff",
            "overview-image-10-00-00.tif",
        ]

    def test_fluorescence_overviews_are_not_mixed_in(self, experiment):
        """`.ome.tiff` matches `*.tiff`, so excluding it has to be deliberate.

        They are a different view of the sample -- a different stage tilt, a different
        instrument -- so they cannot share a beam overview's axes.
        """
        beam = experiment.find_overview_images()
        assert not any(p.endswith(".ome.tiff") for p in beam)

        fm = [
            os.path.basename(p) for p in experiment.find_fluorescence_overview_images()
        ]
        assert fm == ["overview-11-00-00.ome.tiff"]

    def test_unrelated_images_are_ignored(self, experiment):
        """Reference images live in the same tree and must not be mistaken for maps.

        Matching is on the substring "overview", deliberately: the filename is the
        operator's to choose and only the default starts with it. So a file called
        `not-an-overview.tif` *would* match -- that is the glob working as specified,
        not a case worth defending against.
        """
        found = [os.path.basename(p) for p in experiment.find_overview_images()]
        assert "ref_Polishing_final_ib.tif" not in found
        assert "sem-image-10-00-00.tif" not in found

    def test_an_experiment_with_no_overviews_returns_empty(self, tmp_path):
        from fibsem.applications.autolamella.structures import Experiment

        exp = Experiment(path=tmp_path, name="empty")
        os.makedirs(exp.path, exist_ok=True)
        assert exp.find_overview_images() == []
        assert exp.find_fluorescence_overview_images() == []


class TestPerPositionColours:
    """A caller can single out individual positions."""

    def test_named_positions_override_the_default_colour(self, overview):
        fig = plot_minimap(
            overview,
            positions=[_at_centre(overview, "flagged")],
            color="cyan",
            colors={"flagged": "#d04040"},
        )
        try:
            assert "#d04040" in _marker_colours(fig)
        finally:
            plt.close(fig)

    def test_unnamed_positions_keep_the_default_colour(self, overview):
        """A caller with nothing to distinguish is unaffected by the new argument."""
        fig = plot_minimap(
            overview,
            positions=[_at_centre(overview, "ordinary")],
            color="cyan",
            colors={"someone-else": "#d04040"},
        )
        try:
            assert _marker_colours(fig) == ["#00ffff"]
        finally:
            plt.close(fig)

    def test_a_lamella_named_grid_is_not_drawn_as_a_grid_position(self, overview):
        """The behaviour this replaces tested `"Grid" in pt.name`.

        Petnames are generated, so "Grid" appearing in one is a matter of time -- and a
        lamella silently drawn in the grid colour is a lamella the recipient does not
        know is a lamella.
        """
        fig = plot_minimap(
            overview,
            positions=[_at_centre(overview, "Grid square four")],
            color="cyan",
        )
        try:
            colours = _marker_colours(fig)
            assert colours == ["#00ffff"]
            assert GRID_POSITION_COLOUR not in colours
        finally:
            plt.close(fig)

    def test_a_lamella_named_current_position_keeps_its_own_colour(self, overview):
        from matplotlib.colors import to_hex

        fig = plot_minimap(
            overview,
            positions=[_at_centre(overview, "Current Position of interest")],
            color="cyan",
        )
        try:
            assert to_hex(CURRENT_POSITION_COLOUR) not in _marker_colours(fig)
        finally:
            plt.close(fig)

    def test_real_grid_positions_still_get_the_grid_colour(self, overview):
        from matplotlib.colors import to_hex

        fig = plot_minimap(
            overview,
            positions=[],
            grid_positions=[_at_centre(overview, "anything at all")],
            color="cyan",
        )
        try:
            assert to_hex(GRID_POSITION_COLOUR) in _marker_colours(fig)
        finally:
            plt.close(fig)


class TestTheReportUsesTheSameRenderer:
    """`plot_stage_positions_on_image` is an adapter now, not a second implementation."""

    def test_the_colour_cycle_survives_the_collapse(self, overview):
        """What the statistics plots rely on: color=None means one colour per position."""
        from matplotlib.colors import to_hex

        positions = [_at_centre(overview, f"p{i}") for i in range(3)]
        fig = plot_stage_positions_on_image(overview, positions, color=None)
        try:
            colours = _marker_colours(fig)
            expected = [to_hex(POSITION_COLOURS[i]) for i in range(3)]
            assert colours == expected
        finally:
            plt.close(fig)

    def test_an_explicit_colour_applies_to_every_position(self, overview):
        positions = [_at_centre(overview, f"p{i}") for i in range(3)]
        fig = plot_stage_positions_on_image(overview, positions, color="cyan")
        try:
            assert set(_marker_colours(fig)) == {"#00ffff"}
        finally:
            plt.close(fig)
