"""Where a fluorescence overview run lands, and who decides.

The FIB/SEM tab has had a Folder and a Name for as long as it has had a rebuilt tab; the
fluorescence one took a directory from its host and named every run `overview-HH-MM-SS`.
So the two tabs wrote to different places under different rules, and only one of them
could be pointed somewhere else.

Both now hold a base name that the run stamps with the time it started, defaulting to
`overview-image`. The stamp is not decoration: the stem is a *directory*, and two runs
sharing one land on each other -- the second overwrites the first's tiles and its mosaic,
and only the canvas, holding both in memory, still shows two.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
import sys

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication

from fibsem.ui.fm.widgets.fm_overview_settings_widget import FMOverviewSettingsWidget
from fibsem.ui.widgets.overview_acquisition_settings_widget import (
    DEFAULT_OVERVIEW_FILENAME,
    stamped_overview_name,
)

_app = QApplication.instance() or QApplication(sys.argv)

STAMPED = re.compile(r"^(?P<base>.+)-\d\d-\d\d-\d\d$")


@pytest.fixture
def settings():
    w = FMOverviewSettingsWidget()
    yield w
    w.deleteLater()


def base_of(name: str) -> str:
    match = STAMPED.match(name)
    assert match, f"{name!r} carries no time stamp"
    return match.group("base")


# ── the name ─────────────────────────────────────────────────────────────


def test_it_opens_on_the_same_default_as_the_fibsem_tab(settings):
    """One constant, not two spellings of it. The tabs write into the same experiment
    folder, and a reader globbing `*overview*` should find both runs alike."""
    assert settings.filename_edit.text() == DEFAULT_OVERVIEW_FILENAME
    assert base_of(settings.basename) == DEFAULT_OVERVIEW_FILENAME


def test_a_name_you_type_is_stamped_with_the_time(settings):
    """Applied to whatever the box says, not only to the default -- a memorable name
    invites being reused, so it is *more* prone to collision, not less."""
    settings.filename_edit.setText("grid1")

    assert base_of(settings.basename) == "grid1"


def test_an_empty_name_falls_back_rather_than_making_an_unnamed_folder(settings):
    """The stem becomes a directory name, so an empty box would ask `OverviewDestination`
    to create a folder called nothing at all."""
    settings.filename_edit.setText("   ")

    assert base_of(settings.basename) == DEFAULT_OVERVIEW_FILENAME


def test_the_panel_says_what_the_run_will_be_called(settings):
    """The box keeps showing the base, so nothing else on screen would reveal that a
    stamp is coming."""
    settings.filename_edit.setText("grid1")

    assert settings.label_destination.text() == "saved as grid1-HH-MM-SS"


def test_an_emptied_name_previews_the_fallback_and_not_a_bare_stamp(settings):
    """The preview has to agree with `basename`, which falls back -- otherwise the line
    promises `-HH-MM-SS` and the run produces `overview-image-HH-MM-SS`."""
    settings.filename_edit.setText("")

    assert settings.label_destination.text() == (
        f"saved as {DEFAULT_OVERVIEW_FILENAME}-HH-MM-SS"
    )


def test_both_tabs_stamp_through_the_same_helper():
    """Structural, and the point of moving it out of the FIB/SEM tab: two copies of this
    rule would drift, and the drift would only show up as files in the wrong place."""
    import fibsem.ui.widgets.overview_widget as beam

    assert beam.stamped_overview_name is stamped_overview_name
    assert base_of(stamped_overview_name("x")) == "x"


# ── the folder ───────────────────────────────────────────────────────────


def test_the_host_s_directory_lands_in_the_field(settings):
    """Being told a folder and choosing one are the same field. Held separately, an edit
    would be silently outranked by whatever the host set last."""
    settings.set_save_directory("/tmp/experiment-a")

    assert settings.path_edit.text() == "/tmp/experiment-a"
    assert settings.save_directory == "/tmp/experiment-a"


def test_loading_another_experiment_replaces_a_folder_you_chose(settings):
    """Deliberate, and what the FIB/SEM tab does: the folder belongs to the experiment
    that is open, so a run after switching must not still be pointed at the old one."""
    settings.set_save_directory("/tmp/experiment-a")
    settings.path_edit.setText("/tmp/somewhere-else")

    settings.set_save_directory("/tmp/experiment-b")

    assert settings.save_directory == "/tmp/experiment-b"


def test_no_folder_reads_as_none_rather_than_an_empty_path(settings):
    """`None` is what the tab tests for to decide a run is unsaved; an empty string
    would be a falsy path that still reached `makedirs`."""
    settings.set_save_directory(None)

    assert settings.save_directory is None


# ── and what the run actually claims ─────────────────────────────────────


@pytest.fixture(scope="module")
def microscope():
    from fibsem.ui.fm.overview_app import build_microscope

    return build_microscope()


@pytest.fixture
def widget(microscope, tmp_path):
    from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget

    built = FMOverviewWidget(microscope)
    built.set_save_directory(str(tmp_path))
    yield built
    built.deleteLater()


def test_a_run_claims_the_folder_and_the_name_the_panel_shows(widget, tmp_path):
    """The end of the chain, and the only part a user can check: the panel is not a
    display of where a run goes, it is where it goes."""
    widget.settings_widget.filename_edit.setText("grid1")

    destination = widget._claim_destination()

    assert os.path.dirname(destination.tiles_directory) == str(tmp_path)
    assert base_of(os.path.basename(destination.tiles_directory)) == "grid1"
    # The mosaic sits beside the tile directory under the same name, so both move
    # together when the name changes -- otherwise a renamed run splits in two.
    assert destination.mosaic_path == f"{destination.tiles_directory}.ome.tiff"


def test_two_runs_in_the_same_second_do_not_land_on_each_other(widget):
    """What the stamp cannot cover on its own: it is good to the second, and a
    single-tile overview at a short exposure takes far less."""
    widget.settings_widget.filename_edit.setText("grid1")

    first = widget._claim_destination()
    second = widget._claim_destination()

    assert first.tiles_directory != second.tiles_directory
    assert os.path.isdir(first.tiles_directory)
    assert os.path.isdir(second.tiles_directory)


def test_editing_the_folder_moves_where_the_run_goes(widget, tmp_path):
    """The reason the field is editable at all -- the host's default is a starting
    point, not a cage."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    widget.settings_widget.path_edit.setText(str(elsewhere))

    destination = widget._claim_destination()

    assert os.path.dirname(destination.tiles_directory) == str(elsewhere)


def test_with_no_folder_the_run_goes_ahead_unsaved(widget):
    """Unchanged by this panel, and worth pinning while it is being rewired: an overview
    you can see is worth more than no overview at all."""
    widget.settings_widget.set_save_directory(None)

    assert widget._claim_destination() is None
