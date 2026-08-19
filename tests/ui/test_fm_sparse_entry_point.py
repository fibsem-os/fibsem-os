"""Reaching the sparse selection from the FM Overview tab, and taking its answer back.

Two halves. Finding the beam overviews, which happens off disk rather than through the
FIB/SEM tab -- that tab keeps pixels and a position, not the metadata a projection is
read from. And applying what comes back, which has to reach the *acquisition* rather than
only the drawing, or the grid on screen and the run would disagree.
"""
import os
from contextlib import contextmanager

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.fm.structures import OverviewParameters
from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings
from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget
from fibsem.ui.fm.widgets.fm_sparse_selection_dialog import (
    SparseSelection,
    beam_overviews_in,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def microscope(qapp):
    from fibsem.ui.fm.overview_app import build_microscope

    return build_microscope()


def save_beam_overview(microscope, directory, orientation, beam_type):
    pose = microscope.get_orientation(orientation)
    microscope.move_stage_absolute(
        FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
    )
    image = microscope.acquire_image(
        ImageSettings(
            hfw=900e-6, resolution=[1024, 1024], beam_type=beam_type, save=False
        )
    )
    image.save(os.path.join(directory, f"overview-image-{orientation}"))
    return image


@pytest.fixture(scope="module")
def experiment(microscope, tmp_path_factory):
    """An experiment directory with one beam overview per view, as a session leaves it."""
    directory = str(tmp_path_factory.mktemp("experiment"))
    save_beam_overview(microscope, directory, "SEM", BeamType.ELECTRON)
    save_beam_overview(microscope, directory, "MILLING", BeamType.ION)
    microscope.move_to_microscope("FM")
    return directory


# ── finding them ─────────────────────────────────────────────────────────


def test_it_finds_a_beam_overview_for_each_view(microscope, experiment):
    views = beam_overviews_in(experiment, microscope)

    assert {v.label for v in views} == {"SEM @ SEM", "FIB @ MILLING"}


def test_it_does_not_descend_into_the_tile_folders(microscope, experiment, tmp_path):
    """A run writes its tiles into a directory and the mosaic *beside* it, so the top
    level is exactly the mosaics -- and a hundred tiles are exactly what must not be
    offered as overviews to select on."""
    directory = str(tmp_path)
    save_beam_overview(microscope, directory, "MILLING", BeamType.ION)
    tiles = os.path.join(directory, "overview-image-MILLING")
    os.makedirs(tiles, exist_ok=True)
    save_beam_overview(microscope, tiles, "MILLING", BeamType.ION)
    microscope.move_to_microscope("FM")

    views = beam_overviews_in(directory, microscope)

    assert sum(len(images) for images in views.values()) == 1


def test_a_file_it_cannot_read_is_skipped_rather_than_fatal(
    microscope, experiment, tmp_path
):
    """The experiment folder holds more than overviews."""
    directory = str(tmp_path)
    save_beam_overview(microscope, directory, "MILLING", BeamType.ION)
    with open(os.path.join(directory, "not-an-image.tif"), "w") as handle:
        handle.write("nonsense")
    microscope.move_to_microscope("FM")

    assert beam_overviews_in(directory, microscope)


def test_an_image_with_no_beam_geometry_is_not_offered(microscope, tmp_path):
    """Fluorescence overviews live in the same experiment folder, and a region drawn on
    one could not be resolved -- there is no beam projection to read.

    The filter is exactly that question: keep what `BeamStageProjection.from_image` can
    read, rather than guessing from a filename.
    """
    directory = str(tmp_path)
    save_beam_overview(microscope, directory, "MILLING", BeamType.ION)

    # An image that records *where* it was taken but not the geometry it was taken
    # under. This is the case the projection check exists for, and the only one: a file
    # with neither -- a plain TIFF dropped in the folder -- is turned away by the
    # position check a few lines later, so testing with one proves nothing about this.
    # Acquired FM tiles carry no geometry at all (FIB-416), so it is a real shape.
    stray = save_beam_overview(microscope, directory, "MILLING", BeamType.ION)
    stray.metadata.hardware_geometry = None
    stray.save(os.path.join(directory, "no-geometry"))
    microscope.move_to_microscope("FM")

    views = beam_overviews_in(directory, microscope)

    assert sum(len(images) for images in views.values()) == 1, (
        "the readable overview, and not the one with no geometry beside it"
    )


@pytest.mark.parametrize("directory", [None, "", "/nowhere/at/all"])
def test_nowhere_to_look_is_not_an_error(microscope, directory):
    assert beam_overviews_in(directory, microscope) == {}


# ── the button ───────────────────────────────────────────────────────────


@pytest.fixture()
def widget(microscope, experiment):
    built = FMOverviewWidget(microscope)
    built.set_save_directory(experiment)
    return built


def test_the_button_is_hidden_until_the_feature_is_turned_on(widget):
    assert widget.button_select_ground.isVisibleTo(widget) is False

    widget.set_sparse_selection_enabled(True)

    assert widget.button_select_ground.isVisibleTo(widget) is True


@contextmanager
def dialog_never_opens():
    """Watch for the dialog rather than waiting to see whether it appears.

    It is modal, so a version that opened it here would *hang* the run rather than fail
    it -- which is a worse test than a wrong one, because a hang says nothing about which
    assertion was violated.
    """
    from fibsem.ui.fm.widgets import fm_sparse_selection_dialog as module

    opened = []
    original = module.FMSparseSelectionDialog.choose
    module.FMSparseSelectionDialog.choose = classmethod(
        lambda cls, *a, **k: opened.append(1)
    )
    try:
        yield opened
    finally:
        module.FMSparseSelectionDialog.choose = original


def test_the_flag_is_carried_past_the_button(widget):
    """A control nobody can see is not a feature that is off. Someone reaching the
    method another way must meet the same answer as someone looking for the button."""
    widget.set_sparse_selection_enabled(False)

    with dialog_never_opens() as opened:
        widget.select_ground_to_image()

    assert opened == []


def test_with_no_overviews_it_says_so_rather_than_opening_an_empty_dialog(
    microscope, tmp_path
):
    """A dialog offering nothing to select on is a dead end with a Cancel button."""
    built = FMOverviewWidget(microscope)
    built.set_save_directory(str(tmp_path))
    built.set_sparse_selection_enabled(True)

    with dialog_never_opens() as opened:
        built.select_ground_to_image()

    assert opened == []
    assert built.label_selection.isVisibleTo(built) is False


def test_it_does_open_when_there_is_something_to_select_on(widget):
    """The guard above has to be a guard, not the only path."""
    widget.set_sparse_selection_enabled(True)

    with dialog_never_opens() as opened:
        widget.select_ground_to_image()

    assert opened == [1]


# ── taking the answer back ───────────────────────────────────────────────


def selection(rows=4, cols=3, enabled=((0, 0), (1, 1))):
    mask = [[(r, c) in enabled for c in range(cols)] for r in range(rows)]
    return SparseSelection(
        parameters=OverviewParameters(rows=rows, cols=cols, overlap=0.1, tile_mask=mask),
        centre_position=FibsemStagePosition(
            x=1e-4, y=2e-4, z=0.0, r=0.0, t=-3.141592653589793
        ),
    )


def test_applying_a_selection_reaches_the_settings(widget):
    widget.apply_sparse_selection(selection(rows=4, cols=3))

    parameters = widget.settings_widget.parameters
    assert (parameters.rows, parameters.cols) == (4, 3)
    assert parameters.n_enabled_tiles == 2


def test_applying_a_selection_moves_where_the_run_will_be_centred(widget):
    """`_target` is what `_grid_centre` hands the runner, and what the drawn grid is
    placed against -- so the picture and the acquisition agree without either being told
    about this feature."""
    chosen = selection()

    widget.apply_sparse_selection(chosen)

    assert widget._grid_centre() == chosen.centre_position


def test_it_says_the_grid_came_from_a_selection_and_will_be_replaced(widget):
    """There is no restore: the regions end with the dialog, so a second selection starts
    over. Saying so is cheaper than letting someone find out."""
    widget.apply_sparse_selection(selection(rows=4, cols=3))

    assert widget.label_selection.isVisibleTo(widget) is True
    assert "2 of 12 tiles came from a selection" in widget.label_selection.text()
    assert "replaces it" in widget.label_selection.text()
