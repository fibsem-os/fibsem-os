"""Where a fluorescence overview run lands, and who decides.

The FIB/SEM tab has had a Folder and a Name for as long as it has had a rebuilt tab; the
fluorescence one took a directory from its host and named every run `overview-HH-MM-SS`.
So the two tabs wrote to different places under different rules, and only one of them
could be pointed somewhere else.

Both now hold a base name that the imaging service stamps with the time the run
started, defaulting to `overview-image`. The stamp is not decoration: the stem is a
*directory*, and two runs sharing one land on each other -- the second overwrites the
first's tiles and its mosaic, and only the canvas, holding both in memory, still shows
two.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
import sys

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication

from fibsem.imaging.overview_session import FLUORESCENCE, OverviewSession
from fibsem.imaging.tiled import stamped_overview_name
from fibsem.structures import DEFAULT_OVERVIEW_FILENAME
from fibsem.ui.fm.widgets.fm_overview_settings_widget import FMOverviewSettingsWidget

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
    assert settings.basename == DEFAULT_OVERVIEW_FILENAME


def test_a_name_you_type_is_the_name_the_run_asks_for(settings):
    settings.filename_edit.setText("grid1")

    assert settings.basename == "grid1"


def test_an_empty_name_falls_back_rather_than_making_an_unnamed_folder(settings):
    """The stem becomes a directory name, so an empty box would ask `OverviewDestination`
    to create a folder called nothing at all."""
    settings.filename_edit.setText("   ")

    assert settings.basename == DEFAULT_OVERVIEW_FILENAME


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


def test_the_stamp_keeps_the_base_as_a_prefix():
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
    from tests.fixtures.fm_microscope import build_microscope

    return build_microscope()


@pytest.fixture
def widget(microscope, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QDialog

    from fibsem.ui.fm.widgets import fm_overview_widget as module
    from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget

    class _Dialog:
        def __init__(self, **kwargs):
            pass

        def exec_(self):
            return QDialog.Accepted

    class _Worker:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(module, "FMOverviewConfirmationDialog", _Dialog)
    monkeypatch.setattr(module, "FunctionWorker", _Worker)
    microscope.fm.objective.insert()
    built = FMOverviewWidget(OverviewSession(microscope, FLUORESCENCE))
    monkeypatch.setattr(built, "at_acquisition_orientation", lambda: True)
    built.set_save_directory(str(tmp_path))
    yield built
    built.deleteLater()


def _tiles_directory(widget, monkeypatch) -> str:
    """Run an overview from the tab, through the imaging service, and say where its
    tiles went. Cancelled at the first tile: where they go is decided before."""
    from fibsem.cancellation import OperationCancelledError
    from fibsem.fm import acquisition

    claimed = []

    class _Runner:
        def __init__(self, **kwargs):
            claimed.append(kwargs["save_directory"])

        def run_and_stitch(self):
            raise OperationCancelledError()

    monkeypatch.setattr(acquisition, "FMTiledAcquisitionRunner", _Runner)
    widget.acquire()
    widget._acquire_worker()
    widget._set_running(False)
    return claimed[0]


def test_a_run_claims_the_folder_and_the_name_the_panel_shows(
    widget, tmp_path, monkeypatch
):
    """The end of the chain, and the only part a user can check: the panel is not a
    display of where a run goes, it is where it goes."""
    widget.settings_widget.filename_edit.setText("grid1")

    tiles = _tiles_directory(widget, monkeypatch)

    assert os.path.dirname(tiles) == str(tmp_path)
    assert base_of(os.path.basename(tiles)) == "grid1"


def test_two_runs_in_the_same_second_do_not_land_on_each_other(widget, monkeypatch):
    """What the stamp cannot cover on its own: it is good to the second, and a
    single-tile overview at a short exposure takes far less."""
    widget.settings_widget.filename_edit.setText("grid1")

    first = _tiles_directory(widget, monkeypatch)
    second = _tiles_directory(widget, monkeypatch)

    assert first != second
    assert os.path.isdir(first)
    assert os.path.isdir(second)


def test_editing_the_folder_moves_where_the_run_goes(widget, tmp_path, monkeypatch):
    """The reason the field is editable at all -- the host's default is a starting
    point, not a cage."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    widget.settings_widget.path_edit.setText(str(elsewhere))

    tiles = _tiles_directory(widget, monkeypatch)

    assert os.path.dirname(tiles) == str(elsewhere)


def test_with_no_folder_the_run_saves_in_the_working_directory(widget, monkeypatch):
    """As the beam tab does: the imaging service always takes a path."""
    widget.settings_widget.set_save_directory(None)

    tiles = _tiles_directory(widget, monkeypatch)

    assert os.path.dirname(tiles) == os.getcwd()
