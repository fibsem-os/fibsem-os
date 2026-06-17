"""Offscreen tests for the Sample tab assembly contract (Phase 4 grid UI).

The Sample tab in AutoLamellaUI hosts a LoaderMagazineWidget (storage, top,
compustage only) above a SampleHolderWidget (working slot, always). Building
the full AutoLamellaUI requires a napari viewer, so these tests exercise the
same wiring contract the tab relies on at the widget level:

- the magazine is present only when the stage has a loader;
- the holder is always present;
- a magazine grid and the holder working slot share the same SampleGrid
  object once loaded, so refreshing the holder reflects magazine edits
  (the tab's magazine_changed -> holder.refresh view-sync);
- loading a grid into the working slot is reflected in the holder.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.microscopes._stage import _create_sample_stage  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as e:  # pragma: no cover - no Qt platform available
        pytest.skip(f"Qt unavailable: {e}")
    return app


def _compustage_microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope._stage = _create_sample_stage(microscope)  # loader + 1 working slot
    return microscope


def _build_widgets(microscope):
    """Replicate how _create_sample_tab assembles the two widgets."""
    from fibsem.ui.widgets.loader_magazine_widget import LoaderMagazineWidget
    from fibsem.ui.widgets.sample_holder_widget import SampleHolderWidget

    magazine = None
    if microscope._stage.loader is not None:
        magazine = LoaderMagazineWidget(microscope=microscope)
        magazine.set_microscope(microscope)

    holder = SampleHolderWidget(microscope=microscope)
    holder._auto_save = lambda *a, **k: None  # don't touch the shared config
    holder.set_holder(microscope._stage.holder)
    return magazine, holder


def test_magazine_present_only_with_loader(qapp):
    # compustage -> loader -> magazine built
    magazine, holder = _build_widgets(_compustage_microscope())
    assert magazine is not None
    assert holder is not None

    # non-compustage -> no loader -> no magazine, holder still present
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)  # no loader
    assert microscope._stage.loader is None
    magazine2, holder2 = _build_widgets(microscope)
    assert magazine2 is None
    assert holder2 is not None


def test_load_reflected_in_holder(qapp):
    microscope = _compustage_microscope()
    magazine, holder = _build_widgets(microscope)

    # name a magazine grid, then exchange it into the working slot
    row = magazine._list.itemWidget(magazine._list.item(0))
    row.name_edit.setText("grid-aspen")
    row._on_name_edited()

    from fibsem.applications.autolamella.structures import GridRecord
    from fibsem.applications.autolamella.workflows.tasks.grid_manager import (
        GridTaskManager,
    )

    GridTaskManager(microscope, experiment=None).ensure_loaded(
        GridRecord(name="grid-aspen")
    )
    holder.refresh()  # the tab does this after the exchange worker finishes

    slot = next(iter(microscope._stage.holder.slots.values()))
    assert slot.loaded_grid is not None
    assert slot.loaded_grid.name == "grid-aspen"


def test_view_sync_shared_grid_object(qapp):
    microscope = _compustage_microscope()
    magazine, holder = _build_widgets(microscope)

    # load a grid into the working slot
    row = magazine._list.itemWidget(magazine._list.item(0))
    row.name_edit.setText("grid-aspen")
    row._on_name_edited()
    from fibsem.applications.autolamella.structures import GridRecord
    from fibsem.applications.autolamella.workflows.tasks.grid_manager import (
        GridTaskManager,
    )
    GridTaskManager(microscope, experiment=None).ensure_loaded(
        GridRecord(name="grid-aspen")
    )

    # the magazine slot grid and the working-slot grid are the same object
    mag_slot = microscope._stage.loader.find_grid("grid-aspen")
    work_slot = next(iter(microscope._stage.holder.slots.values()))
    assert mag_slot.loaded_grid is work_slot.loaded_grid

    # editing the description via the magazine row mutates the shared object;
    # holder.refresh() (the view-sync the tab wires to magazine_changed) shows it
    row.desc_edit.setText("lamella candidate")
    row._on_desc_edited()
    holder.refresh()
    assert work_slot.loaded_grid.description == "lamella candidate"
