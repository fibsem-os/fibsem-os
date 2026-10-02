"""The overview plot dialog and the PDF report draw positions the same way.

They were two renderers -- ``plot_minimap`` behind the dialog, and
``plot_stage_positions_on_image`` behind the report's overview page and the statistics
plots -- and they had drifted: different label sizes, different ways of drawing the
marker, descriptions in one only. ``plot_stage_positions_on_image`` is now an adapter
over ``plot_minimap``.

The merge also retired a name-sniff: ``plot_minimap`` coloured any position whose name
contained ``"Grid"`` red and drew the grid's radius around it. Grid and current
positions are now known by the list they were passed in.
"""

import os

import numpy as np
import pytest
from matplotlib.colors import to_hex

from fibsem import utils
from fibsem.imaging.tiling.plotting import (
    POSITION_COLOURS,
    plot_minimap,
    plot_stage_positions_on_image,
)
from fibsem.structures import BeamType, FibsemImage, FibsemStagePosition, ImageSettings

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SHAPE = (256, 768)


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
        resolution=(_SHAPE[1], _SHAPE[0]), hfw=900e-6
    )
    image.data = np.zeros(_SHAPE, dtype=np.uint8)
    image.metadata.image_settings = ImageSettings(
        hfw=900e-6, beam_type=BeamType.ELECTRON
    )
    image.metadata.microscope_state = microscope.get_microscope_state(
        beam_type=BeamType.ELECTRON
    )
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    return image


def _positions(image: FibsemImage, names):
    """Positions spread across the middle of the image, one per name."""
    base = image.metadata.microscope_state.stage_position
    out = []
    for i, name in enumerate(names):
        position = FibsemStagePosition(
            x=base.x + (i - len(names) // 2) * 60e-6,
            y=base.y,
            z=base.z,
            r=base.r,
            t=base.t,
        )
        position.name = name
        out.append(position)
    return out


def _pixels(fig) -> np.ndarray:
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return pixels


def _marker_colours(fig):
    """The marker colours, in the order the positions were drawn."""
    (markers,) = fig.axes[0].collections
    colours = markers.get_facecolors()
    if len(colours) == 0:
        colours = markers.get_edgecolors()
    return [to_hex(c) for c in colours]


def test_the_report_draws_what_the_dialog_draws(overview):
    """Same image, same positions, same settings: the same pixels."""
    positions = _positions(overview, ["01-a", "02-b", "03-c"])
    common = dict(color="cyan", show_scalebar=True, show_names=True)

    report = _pixels(plot_stage_positions_on_image(overview, positions, **common))
    dialog = _pixels(
        plot_minimap(overview, positions, fontsize=14, markersize=20, **common)
    )

    assert report.shape == dialog.shape
    assert np.array_equal(report, dialog)


def test_without_a_colour_each_position_gets_its_own(overview):
    """The statistics plots rely on this to tell tracks apart."""
    positions = _positions(overview, ["01-a", "02-b", "03-c"])
    fig = plot_stage_positions_on_image(overview, positions, color=None)
    try:
        assert _marker_colours(fig) == [to_hex(c) for c in POSITION_COLOURS[:3]]
    finally:
        plt.close(fig)


def test_a_lamella_named_grid_is_drawn_as_a_lamella(overview):
    """It used to be drawn red, as if it were a grid position."""
    positions = _positions(overview, ["Grid-01-lamella"])
    fig = plot_minimap(overview, positions, color="cyan", show_grid_radius=True)
    try:
        assert _marker_colours(fig) == [to_hex("cyan")]
        assert len(fig.axes[0].patches) == 0, "a grid radius was drawn round a lamella"
    finally:
        plt.close(fig)


def test_grid_and_current_positions_are_known_by_their_list(overview):
    lamella, grid, current = _positions(overview, ["01-a", "a", "b"])
    fig = plot_minimap(
        overview,
        [lamella],
        current_position=current,
        grid_positions=[grid],
        color="cyan",
    )
    try:
        assert _marker_colours(fig) == [
            to_hex("cyan"),
            to_hex("yellow"),
            to_hex("red"),
        ]
    finally:
        plt.close(fig)
